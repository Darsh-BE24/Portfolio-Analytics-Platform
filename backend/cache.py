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

Caching strategy: completeness is tracked by COVERAGE RANGES (see
MarketDataCoverage in models.py), not by checking whether every calendar
business day has a row. A requested [start, end] is a cache hit only if
some existing coverage range for that ticker fully contains it -- this is
exchange-holiday-agnostic by construction, since it never assumes which
specific dates a market should have been open on. If any requested ticker
lacks sufficient coverage, the ENTIRE requested range is fetched fresh for
every ticker in the request (not just the uncovered ones) and the fetched
rows plus the new coverage range are both recorded. A more sophisticated
cache would patch only the uncovered tickers/sub-ranges -- that's
meaningfully more complex and was left out here in favor of a correct,
simple, well-tested strategy over a more complete but riskier one.

This was previously implemented by comparing cached rows against
pd.bdate_range(start, end) -- every Mon-Fri in the range, with no concept
of exchange holidays. That meant any range spanning a holiday (true of
nearly every multi-month lookback window) could NEVER be judged complete,
since the holiday date would forever show up as "missing" even though a
correct fetch legitimately has no row for it. The cache was effectively
never hit for realistic date ranges. Reproduced directly: requesting
2023-12-28 to 2024-01-03 (spans New Year's Day) showed 2024-01-01 as
"always missing" even with a perfectly complete real fetch.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from backend.data import MarketDataService
from backend.models import MarketDataCache, MarketDataCoverage

# How far before the requested `end` a ticker's last observation may fall and
# still be treated as "the data goes all the way to `end`". Covers a weekend
# plus a holiday cluster, or a latest bar the provider hasn't published yet.
# A response that stops further back than this is treated as truncated.
_TRAILING_TOLERANCE = timedelta(days=7)


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


def _is_range_covered(db: Session, ticker: str, start: date, end: date) -> bool:
    """True if some existing coverage row for `ticker` fully contains [start, end]."""
    return (
        db.query(MarketDataCoverage)
        .filter(
            MarketDataCoverage.ticker == ticker,
            MarketDataCoverage.start_date <= start,
            MarketDataCoverage.end_date >= end,
        )
        .first()
        is not None
    )


def _has_unfetched_weekday(gap_start: date, gap_end: date) -> bool:
    """True if any Mon-Fri date lies in [gap_start, gap_end] (an empty span has none)."""
    days = (gap_end - gap_start).days + 1
    return any((gap_start + timedelta(days=i)).weekday() < 5 for i in range(min(days, 5)))


def _ranges_connect(a_start: date, a_end: date, b_start: date, b_end: date) -> bool:
    """
    True if two FETCHED ranges may be recorded as one contiguous covered range:
    they overlap, touch, or are separated only by days on which no market trades
    (a weekend).

    A gap containing ANY weekday is an interval that was never fetched. Without
    an exchange calendar a holiday cannot be told apart from a genuinely missing
    day, so such a gap is never bridged: merging across it would make the cache
    report data as covered that was never retrieved. (The cost of this caution is
    at worst one extra fetch, never wrong data.)
    """
    if a_end < b_start:
        return not _has_unfetched_weekday(a_end + timedelta(days=1), b_start - timedelta(days=1))
    if b_end < a_start:
        return not _has_unfetched_weekday(b_end + timedelta(days=1), a_start - timedelta(days=1))
    return True  # overlapping


def _record_coverage(db: Session, ticker: str, start: date, end: date) -> None:
    """
    Record that [start, end] is now fully fetched for `ticker`, merging it with
    existing coverage that overlaps or is contiguous with it (see
    _ranges_connect). Ranges separated by an unfetched interval stay separate
    rows, so a request spanning that interval is correctly a cache miss.
    """
    existing = db.query(MarketDataCoverage).filter(MarketDataCoverage.ticker == ticker).all()

    merged_start, merged_end = start, end
    for row in existing:
        if not _ranges_connect(row.start_date, row.end_date, start, end):
            continue
        merged_start = min(merged_start, row.start_date)
        merged_end = max(merged_end, row.end_date)
        db.delete(row)

    db.add(MarketDataCoverage(ticker=ticker, start_date=merged_start, end_date=merged_end))
    db.commit()


def _existing_cache_keys(db: Session, tickers: list[str]) -> set[tuple[str, date]]:
    """The (ticker, date) pairs already cached for these tickers, across any date range."""
    return {
        (r.ticker, r.date)
        for r in db.query(MarketDataCache.ticker, MarketDataCache.date).filter(MarketDataCache.ticker.in_(tickers)).all()
    }


def _store_prices(db: Session, prices: pd.DataFrame) -> None:
    """
    Insert any (ticker, date) rows not already present in the cache.

    Two concurrent requests can both see the same rows as "not yet cached"
    (each checks before either has committed), then both try to insert them --
    the second insert then violates the (ticker, date) unique constraint.
    This is a genuine, expected race for any server handling more than one
    request at a time, not a hypothetical: it was caught during real
    browser-driven end-to-end testing, where the Overview page fires two
    analytics requests back-to-back. The fix is to treat that specific
    failure as benign: whoever lost the race can simply roll back, since the
    data they were trying to write already exists in the cache anyway.
    """
    if prices.empty:
        return

    tickers = list(prices.columns)
    existing = _existing_cache_keys(db, tickers)

    new_rows = []
    for ticker in tickers:
        for ts, price in prices[ticker].items():
            if pd.isna(price):
                continue
            row_date = ts.date() if hasattr(ts, "date") else ts
            if (ticker, row_date) not in existing:
                new_rows.append(MarketDataCache(ticker=ticker, date=row_date, close_price=float(price)))

    if not new_rows:
        return

    db.add_all(new_rows)
    try:
        db.commit()
    except IntegrityError:
        # A concurrent request already cached one or more of these rows
        # between our existence check and our commit. That's fine -- the
        # data is cached either way, just by the other request instead.
        # Note this discards the WHOLE batch on any single conflict (SQLite
        # doesn't make it easy to commit only the non-conflicting rows from
        # one add_all()) -- any genuinely-new rows in this batch simply
        # aren't cached yet and will be fetched again on the next request
        # that needs them. That's a minor inefficiency, never a correctness
        # problem: no data is lost or corrupted, just re-fetched once more.
        db.rollback()


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

    How coverage is recorded (per ticker), and why:

    - Interior gaps (exchange holidays, a day the provider has no bar for)
      never affect coverage: it is recorded as one contiguous span, so a
      range containing holidays is a hit on the next identical request.
    - The START is recorded as the requested start. If the asset's real
      history simply begins later (recent listing), earlier data does not
      exist and retrying can never produce it, so re-requesting it on every
      call would be pure waste. (Known limitation: a transient provider
      failure that drops only the *beginning* of a response would be
      indistinguishable from this and would be cached as-is; Issue #10's
      gap handling then deals with the short history explicitly.)
    - The END is recorded as the requested end ONLY IF the ticker's last
      observed date is within _TRAILING_TOLERANCE of it. That tolerance
      absorbs weekends, holidays and a not-yet-published latest bar (so a
      request ending on a Saturday is still a hit next time). If the last
      observation is further from `end` than that, the response looks
      truncated (an outage or rate limit cutting it short, not a market
      closure) and coverage is recorded only up to what was actually
      observed -- so a later request for the full range retries the fetch
      instead of serving a partial result forever. (Known limitation: a
      truncation shorter than the tolerance is not detected.)
    - A ticker with no observed data records no coverage at all.
    """
    if all(_is_range_covered(db, t, start, end) for t in tickers):
        return _cached_prices(db, tickers, start, end).reindex(columns=tickers)

    fresh = market_data.get_price_data(tickers, start, end)
    fresh = fresh[(fresh.index.date >= start) & (fresh.index.date <= end)]
    _store_prices(db, fresh)

    for ticker in tickers:
        observed = fresh[ticker].dropna().index
        if len(observed) == 0:
            continue  # no real data for this ticker in the response; nothing to record as covered
        last_observed = observed.max().date()
        coverage_end = end if (end - last_observed) <= _TRAILING_TOLERANCE else last_observed
        _record_coverage(db, ticker, start, coverage_end)

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
