"""
Market data caching layer.

Checks the market_data table for previously-fetched prices before calling
out to yfinance via MarketDataService. This is what section 17 of the
project spec calls for: don't re-download the same historical data on
every request.

Deliberately kept separate from data.py: MarketDataService remains the
only module that imports yfinance and has zero database dependency, so it
stays trivially testable in isolation (as built in Milestone 1). This
module is the thing that combines the two concerns.

Caching strategy (a real, documented simplification): if EVERY ticker has
a cached row for EVERY business day in the requested range, the cache is
used and no network call happens at all. If ANYTHING is missing for ANY
ticker, the ENTIRE requested range is re-fetched fresh from yfinance for
every ticker, and any new (ticker, date) rows are written to the cache.
A more sophisticated cache would patch only the missing date gaps --
that's meaningfully more complex (handling holidays, partial-ticker gaps,
etc.) and was left out here in favor of a correct, simple, well-tested
strategy over a more complete but riskier one.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy.orm import Session

from backend.data import MarketDataService
from backend.models import MarketDataCache


def _cached_prices(db: Session, tickers: list[str], start: date, end: date) -> pd.DataFrame:
    """Return whatever is currently cached for these tickers in [start, end]. May be empty or partial."""
    rows = (
        db.query(MarketDataCache)
        .filter(MarketDataCache.ticker.in_(tickers))
        .filter(MarketDataCache.date >= start)
        .filter(MarketDataCache.date <= end)
        .all()
    )
    if not rows:
        return pd.DataFrame()

    records = [{"ticker": r.ticker, "date": r.date, "close": r.close_price} for r in rows]
    wide = pd.DataFrame(records).pivot(index="date", columns="ticker", values="close").sort_index()
    wide.index = pd.to_datetime(wide.index)
    return wide


def _is_cache_complete(cached: pd.DataFrame, tickers: list[str], expected_index: pd.DatetimeIndex) -> bool:
    """True if every requested ticker has a non-null value for every expected business day."""
    if cached.empty:
        return False
    if not set(tickers).issubset(set(cached.columns)):
        return False

    aligned = cached.reindex(expected_index)[tickers]
    return not aligned.isna().any().any()


def _store_prices(db: Session, prices: pd.DataFrame) -> None:
    """Insert any (ticker, date) rows not already present in the cache."""
    if prices.empty:
        return

    tickers = list(prices.columns)
    existing = {
        (r.ticker, r.date)
        for r in db.query(MarketDataCache.ticker, MarketDataCache.date)
        .filter(MarketDataCache.ticker.in_(tickers))
        .all()
    }

    new_rows = []
    for ticker in tickers:
        for ts, price in prices[ticker].items():
            if pd.isna(price):
                continue
            row_date = ts.date() if hasattr(ts, "date") else ts
            if (ticker, row_date) not in existing:
                new_rows.append(MarketDataCache(ticker=ticker, date=row_date, close_price=float(price)))

    if new_rows:
        db.add_all(new_rows)
        db.commit()


def get_price_data_cached(
    db: Session,
    market_data: MarketDataService,
    tickers: list[str],
    start: date,
    end: date,
) -> pd.DataFrame:
    """
    Cache-first price lookup: returns cached data if it fully covers the
    request, otherwise fetches fresh from yfinance and populates the cache
    for next time.

    IMPORTANT: MarketDataService.get_price_data() internally requests one
    extra trailing day from yfinance (to work around yfinance treating its
    own `end` parameter as exclusive) -- so its raw return value can include
    one calendar day beyond what was actually asked for here. That extra
    day is trimmed off below before storing or returning, specifically so
    that a cache HIT and a cache MISS for the exact same (tickers, start,
    end) always return identically-ranged data. Skipping this trim was a
    real bug caught during testing: the two code paths silently disagreed
    on the last included date.
    """
    expected_index = pd.bdate_range(start=start, end=end)
    cached = _cached_prices(db, tickers, start, end)

    if _is_cache_complete(cached, tickers, expected_index):
        return cached[tickers]

    fresh = market_data.get_price_data(tickers, start, end)
    fresh = fresh[(fresh.index.date >= start) & (fresh.index.date <= end)]
    _store_prices(db, fresh)
    return fresh


def get_benchmark_data_cached(
    db: Session,
    market_data: MarketDataService,
    ticker: str,
    start: date,
    end: date,
) -> pd.Series:
    """Same cache-first strategy as get_price_data_cached, for a single benchmark ticker."""
    df = get_price_data_cached(db, market_data, [ticker], start, end)
    return df[ticker].rename(ticker)
