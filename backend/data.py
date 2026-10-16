"""
Market data access layer.

This is the ONLY module in the project that should import yfinance.
Every other module gets prices/volume through MarketDataService, which
keeps the rest of the codebase testable without hitting the network.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Iterable

import pandas as pd
import yfinance as yf

DEFAULT_BENCHMARK = "^GSPC"  # S&P 500
TRADING_DAYS_PER_YEAR = 252

# Freshness policy for latest ("current") prices: a price fetched from the provider is
# reused for at most this many seconds, then re-fetched. Deliberately SHORT and
# independent of market hours / holidays:
#   - while a market is open it bounds how stale a displayed price can be (the provider's
#     own data is already delayed by far more than this);
#   - when a market is closed (nights, weekends, exchange holidays) the price cannot
#     change, so a few redundant refreshes cost nothing;
#   - it collapses the burst of identical requests one dashboard view makes (create,
#     performance, risk, frontier, ...) into a single provider call, so every endpoint in
#     that burst also sees the SAME price snapshot.
# Per-exchange calendar logic is intentionally avoided (mixed-exchange portfolios, and the
# complexity is not worth it for a dashboard). Set to 0 to disable caching entirely.
LATEST_PRICE_TTL_SECONDS = 60.0


class MarketDataError(Exception):
    """Raised when requested market data cannot be retrieved or is invalid."""


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date

    @classmethod
    def last_n_years(cls, years: float, end: date | None = None) -> "DateRange":
        end = end or date.today()
        start = end - timedelta(days=int(years * 365.25))
        return cls(start=start, end=end)


@dataclass(frozen=True)
class LatestQuote:
    """A latest price together with the metadata needed to judge how current it is."""

    price: float
    price_date: date | None  # trading date of the bar the price came from (not "today" on weekends/holidays)
    fetched_at: datetime  # when WE obtained it from the provider (UTC)
    from_cache: bool = False


class MarketDataService:
    """
    Thin, validated wrapper around yfinance.

    All methods return pandas objects indexed by date, with tickers as
    columns where applicable. Missing/invalid tickers raise MarketDataError
    rather than silently returning NaNs.
    """

    def __init__(
        self,
        benchmark_ticker: str = DEFAULT_BENCHMARK,
        latest_price_ttl_seconds: float = LATEST_PRICE_TTL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.benchmark_ticker = benchmark_ticker
        self.latest_price_ttl_seconds = latest_price_ttl_seconds
        self._clock = clock  # injectable so freshness/expiry can be tested deterministically
        # normalized ticker -> (stored_at, quote). The API runs sync endpoints in a thread pool
        # against one shared service, so every access to this dict goes through _latest_lock.
        self._latest_cache: dict[str, tuple[float, LatestQuote]] = {}
        self._latest_lock = threading.Lock()

    def get_price_data(self, tickers: Iterable[str], start: date, end: date) -> pd.DataFrame:
        """
        Fetch adjusted close prices for one or more tickers.
        Returns a DataFrame: index = date, columns = ticker, values = adjusted close.
        """
        tickers = sorted(set(tickers))
        if not tickers:
            raise MarketDataError("No tickers provided.")

        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end + timedelta(days=1),  # yfinance end is exclusive
            auto_adjust=True,
            progress=False,
            group_by="ticker" if len(tickers) > 1 else "column",
        )

        if raw.empty:
            raise MarketDataError(
                f"No price data returned for {tickers} between {start} and {end}."
            )

        prices = self._extract_field(raw, tickers, field="Close")
        self._validate_tickers_present(prices, tickers)
        return prices

    def get_volume_data(self, tickers: Iterable[str], start: date, end: date) -> pd.DataFrame:
        """Fetch trading volume. Same shape/semantics as get_price_data."""
        tickers = sorted(set(tickers))
        if not tickers:
            raise MarketDataError("No tickers provided.")

        raw = yf.download(
            tickers=tickers,
            start=start,
            end=end + timedelta(days=1),
            auto_adjust=True,
            progress=False,
            group_by="ticker" if len(tickers) > 1 else "column",
        )

        if raw.empty:
            raise MarketDataError(
                f"No volume data returned for {tickers} between {start} and {end}."
            )

        volume = self._extract_field(raw, tickers, field="Volume")
        self._validate_tickers_present(volume, tickers)
        return volume

    def get_benchmark_data(self, start: date, end: date, ticker: str | None = None) -> pd.Series:
        """Fetch adjusted close prices for the benchmark (default: S&P 500)."""
        ticker = ticker or self.benchmark_ticker
        df = self.get_price_data([ticker], start, end)
        return df[ticker].rename(ticker)

    def get_latest_prices(self, tickers: Iterable[str]) -> dict[str, float]:
        """
        Fetch the most recent available close price for each ticker.
        Used for current portfolio valuation (not historical analysis).

        Prices are cached in memory for LATEST_PRICE_TTL_SECONDS (see its comment for the
        freshness policy). Failed or invalid provider responses are never cached, and an
        expired entry is never served as a fallback when the provider fails -- so this
        still acts as a validation gate (e.g. portfolio creation validates every holding and
        the benchmark through it before anything is written to the database).
        """
        return {t: q.price for t, q in self.get_latest_quotes(tickers).items()}

    def get_latest_quotes(self, tickers: Iterable[str]) -> dict[str, LatestQuote]:
        """
        Same as get_latest_prices, but each value also carries the price's trading date, the
        time it was fetched from the provider, and whether it was served from the cache --
        so a caller can show (or reason about) how current a price is.
        """
        tickers = sorted(set(tickers))
        if not tickers:
            raise MarketDataError("No tickers provided.")

        quotes: dict[str, LatestQuote] = {}
        missing: dict[str, str] = {}  # normalized key -> the ticker spelling to send to the provider
        for ticker in tickers:
            key = self._latest_key(ticker)
            cached = self._get_fresh_latest(key)
            if cached is not None:
                quotes[ticker] = replace(cached, from_cache=True)
            else:
                missing.setdefault(key, ticker)

        if missing:
            # Raises (and therefore caches nothing) if the provider fails or any ticker is invalid.
            # Concurrent cold requests may each fetch once; that is redundant but never incorrect.
            fetched = self._fetch_latest_quotes(list(missing.values()))
            for key, raw in missing.items():
                self._store_latest(key, fetched[raw])
            for ticker in tickers:
                if ticker not in quotes:
                    quotes[ticker] = fetched[missing[self._latest_key(ticker)]]

        return {t: quotes[t] for t in tickers}

    def clear_latest_price_cache(self) -> None:
        """Drop every cached latest price (explicit invalidation)."""
        with self._latest_lock:
            self._latest_cache.clear()

    # -- internal helpers -------------------------------------------------

    @staticmethod
    def _latest_key(ticker: str) -> str:
        """Normalized cache key, so 'aapl', ' AAPL ' and 'AAPL' share one entry."""
        return ticker.strip().upper()

    def _get_fresh_latest(self, key: str) -> LatestQuote | None:
        if self.latest_price_ttl_seconds <= 0:
            return None
        with self._latest_lock:
            entry = self._latest_cache.get(key)
        if entry is None:
            return None
        stored_at, quote = entry
        return quote if (self._clock() - stored_at) < self.latest_price_ttl_seconds else None

    def _store_latest(self, key: str, quote: LatestQuote) -> None:
        ttl = self.latest_price_ttl_seconds
        if ttl <= 0:
            return
        now = self._clock()
        with self._latest_lock:
            self._latest_cache[key] = (now, quote)
            # Evict expired entries so the cache cannot grow without bound.
            for stale in [k for k, (stored_at, _) in self._latest_cache.items() if now - stored_at >= ttl]:
                del self._latest_cache[stale]

    def _fetch_latest_quotes(self, tickers: list[str]) -> dict[str, LatestQuote]:
        """Fetch the latest close for each ticker straight from the provider (never touches the cache)."""
        end = date.today()
        start = end - timedelta(days=10)  # small buffer to cross weekends/holidays
        prices = self.get_price_data(tickers, start, end)
        filled = prices.ffill()
        fetched_at = datetime.now(timezone.utc)

        result: dict[str, LatestQuote] = {}
        for ticker in tickers:
            value = filled[ticker].iloc[-1]
            if pd.isna(value):
                raise MarketDataError(f"Could not retrieve a current price for '{ticker}'.")
            last_bar = prices[ticker].last_valid_index()
            price_date = last_bar.date() if hasattr(last_bar, "date") else last_bar
            result[ticker] = LatestQuote(price=float(value), price_date=price_date, fetched_at=fetched_at)
        return result

    @staticmethod
    def _extract_field(raw: pd.DataFrame, tickers: list[str], field: str) -> pd.DataFrame:
        """Normalize yfinance's differing column layouts for 1 vs N tickers."""
        if len(tickers) == 1:
            ticker = tickers[0]
            if field not in raw.columns:
                raise MarketDataError(f"Field '{field}' not found for '{ticker}'.")
            out = raw[[field]].copy()
            out.columns = [ticker]
            return out

        try:
            out = raw.xs(field, axis=1, level=1)
        except KeyError as exc:
            raise MarketDataError(f"Field '{field}' not found in downloaded data.") from exc
        return out

    @staticmethod
    def _validate_tickers_present(df: pd.DataFrame, tickers: list[str]) -> None:
        """Raise if a requested ticker is entirely missing/NaN (invalid symbol or delisting)."""
        missing = [t for t in tickers if t not in df.columns]
        if missing:
            raise MarketDataError(f"No data returned for ticker(s): {missing}")

        all_nan = [t for t in tickers if df[t].isna().all()]
        if all_nan:
            raise MarketDataError(f"Ticker(s) returned only missing data (likely invalid): {all_nan}")
