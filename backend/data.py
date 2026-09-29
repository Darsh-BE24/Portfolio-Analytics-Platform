"""
Market data access layer.

This is the ONLY module in the project that should import yfinance.
Every other module gets prices/volume through MarketDataService, which
keeps the rest of the codebase testable without hitting the network.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Iterable

import pandas as pd
import yfinance as yf

DEFAULT_BENCHMARK = "^GSPC"  # S&P 500
TRADING_DAYS_PER_YEAR = 252


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


class MarketDataService:
    """
    Thin, validated wrapper around yfinance.

    All methods return pandas objects indexed by date, with tickers as
    columns where applicable. Missing/invalid tickers raise MarketDataError
    rather than silently returning NaNs.
    """

    def __init__(self, benchmark_ticker: str = DEFAULT_BENCHMARK):
        self.benchmark_ticker = benchmark_ticker

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
        """
        tickers = sorted(set(tickers))
        end = date.today()
        start = end - timedelta(days=10)  # small buffer to cross weekends/holidays
        prices = self.get_price_data(tickers, start, end)
        latest = prices.ffill().iloc[-1]

        result: dict[str, float] = {}
        for ticker in tickers:
            value = latest[ticker]
            if pd.isna(value):
                raise MarketDataError(f"Could not retrieve a current price for '{ticker}'.")
            result[ticker] = float(value)
        return result

    # -- internal helpers -------------------------------------------------

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
