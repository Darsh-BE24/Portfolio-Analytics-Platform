from datetime import date, timedelta

import numpy as np
import pandas as pd

import backend.cache as cache_module
from backend.cache import get_benchmark_data_cached, get_price_data_cached
from backend.models import MarketDataCache


class _CountingStubMarketData:
    """
    A fake market data service that counts how many times it was actually
    called -- used to verify the cache genuinely avoids redundant calls,
    rather than just checking the returned data looks right.
    """

    def __init__(self):
        self.call_count = 0

    def get_price_data(self, tickers, start, end):
        self.call_count += 1
        dates = pd.bdate_range(start=start, end=end)
        data = {}
        for i, t in enumerate(tickers):
            rng = np.random.RandomState(i)
            data[t] = 100 * (1 + pd.Series(rng.normal(0.0005, 0.01, len(dates)), index=dates)).cumprod()
        return pd.DataFrame(data, index=dates)


def test_first_call_hits_the_stub_and_populates_cache(db_session):
    stub = _CountingStubMarketData()
    start, end = date(2024, 1, 2), date(2024, 3, 1)

    result = get_price_data_cached(db_session, stub, ["AAPL", "MSFT"], start, end)

    assert stub.call_count == 1
    assert not result.empty
    cached_rows = db_session.query(MarketDataCache).count()
    assert cached_rows > 0


def test_second_call_same_range_is_a_cache_hit(db_session):
    stub = _CountingStubMarketData()
    start, end = date(2024, 1, 2), date(2024, 3, 1)

    get_price_data_cached(db_session, stub, ["AAPL", "MSFT"], start, end)
    assert stub.call_count == 1

    get_price_data_cached(db_session, stub, ["AAPL", "MSFT"], start, end)
    assert stub.call_count == 1  # NOT incremented -- served entirely from cache


def test_cache_returns_same_data_as_original_fetch(db_session):
    stub = _CountingStubMarketData()
    start, end = date(2024, 1, 2), date(2024, 3, 1)

    first = get_price_data_cached(db_session, stub, ["AAPL"], start, end)
    second = get_price_data_cached(db_session, stub, ["AAPL"], start, end)

    # Compare by date + value rather than raw Series equality: the cache
    # round-trips through the database's DATE column, which can carry a
    # different (but equivalent) datetime64 precision than the original
    # in-memory DataFrame -- a cosmetic difference, not a data error.
    first_by_date = {ts.date(): round(v, 6) for ts, v in first["AAPL"].items()}
    second_by_date = {ts.date(): round(v, 6) for ts, v in second["AAPL"].items()}
    assert first_by_date == second_by_date


def test_new_ticker_not_in_cache_triggers_a_fetch(db_session):
    stub = _CountingStubMarketData()
    start, end = date(2024, 1, 2), date(2024, 3, 1)

    get_price_data_cached(db_session, stub, ["AAPL"], start, end)
    assert stub.call_count == 1

    # NVDA was never cached -- must trigger another fetch even though
    # AAPL (requested alongside it) is already cached
    get_price_data_cached(db_session, stub, ["AAPL", "NVDA"], start, end)
    assert stub.call_count == 2


def test_wider_date_range_triggers_a_fresh_fetch(db_session):
    stub = _CountingStubMarketData()
    narrow_start, narrow_end = date(2024, 2, 1), date(2024, 3, 1)
    wide_start, wide_end = date(2024, 1, 1), date(2024, 3, 1)

    get_price_data_cached(db_session, stub, ["AAPL"], narrow_start, narrow_end)
    assert stub.call_count == 1

    # requesting a range that extends earlier than what's cached is a
    # cache miss -- the narrow cache doesn't cover the wider request
    get_price_data_cached(db_session, stub, ["AAPL"], wide_start, wide_end)
    assert stub.call_count == 2


def test_benchmark_cache_uses_same_underlying_mechanism(db_session):
    stub = _CountingStubMarketData()
    start, end = date(2024, 1, 2), date(2024, 3, 1)

    result = get_benchmark_data_cached(db_session, stub, "^GSPC", start, end)
    assert stub.call_count == 1
    assert result.name == "^GSPC"

    get_benchmark_data_cached(db_session, stub, "^GSPC", start, end)
    assert stub.call_count == 1  # second call is a cache hit


def test_store_prices_survives_a_concurrent_duplicate_insert(db_session, monkeypatch):
    """
    Regression test for a real bug caught during live, browser-driven
    end-to-end testing: the Overview page fires two analytics requests
    (performance + risk) back-to-back. Both hit the cache for the same
    tickers/range at nearly the same moment, both see "not yet cached"
    (each request's existence-check runs before the OTHER's commit lands),
    and both try to insert the same rows. The second insert then violated
    the (ticker, date) unique constraint and crashed the whole request --
    which is exactly the "CORS error" a real browser session hit, since the
    connection dropped before response headers could be sent.

    True thread-level concurrency is nondeterministic to test reliably, so
    the race is reproduced deterministically instead: a row is committed
    directly (simulating "a concurrent request already wrote this"), and
    _store_prices' own existence check is monkeypatched to report an empty
    result anyway -- exactly what it would have seen had it run a moment
    earlier, before that concurrent commit existed.
    """
    # Simulate a concurrent request having already committed this row.
    db_session.add(MarketDataCache(ticker="AAPL", date=date(2024, 1, 2), close_price=100.0))
    db_session.commit()

    # Force this call's existence check to (falsely) report nothing cached --
    # what it would have seen if it ran just before the commit above.
    monkeypatch.setattr(cache_module, "_existing_cache_keys", lambda db, tickers: set())

    prices = pd.DataFrame({"AAPL": [100.0]}, index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")]))

    # Must not raise, despite attempting to insert a row that already exists.
    cache_module._store_prices(db_session, prices)

    rows = db_session.query(MarketDataCache).filter_by(ticker="AAPL", date=date(2024, 1, 2)).all()
    assert len(rows) == 1  # exactly one row survives: no duplicate, no crash


def test_get_price_data_cached_still_works_after_a_lost_race(db_session, monkeypatch):
    """
    The failed writer must not leave the session broken for the rest of the
    request -- after losing the race, a subsequent read through the normal
    cache-first path should still return correct, complete data.
    """
    db_session.add(MarketDataCache(ticker="AAPL", date=date(2024, 1, 2), close_price=100.0))
    db_session.commit()
    monkeypatch.setattr(cache_module, "_existing_cache_keys", lambda db, tickers: set())

    prices = pd.DataFrame({"AAPL": [100.0]}, index=pd.DatetimeIndex([pd.Timestamp("2024-01-02")]))
    cache_module._store_prices(db_session, prices)  # loses the race, rolls back

    monkeypatch.undo()  # restore the real (correct) existence check
    stub = _CountingStubMarketData()
    result = get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 1, 2), date(2024, 1, 2))

    assert stub.call_count == 0  # served from cache, no fetch needed
    assert result["AAPL"].iloc[0] == 100.0


def test_cache_hit_and_cache_miss_return_identical_date_coverage(db_session):
    """
    Regression test for a real bug caught during manual testing:
    MarketDataService.get_price_data() internally requests one extra
    trailing day from the underlying data source (to compensate for
    yfinance's exclusive `end` convention). A cache MISS returning that raw,
    untrimmed data -- while a cache HIT correctly trims to [start, end] --
    would mean identical requests silently returned different date ranges
    depending on whether the cache happened to be warm. This must never
    happen: the two paths have to agree exactly.
    """

    class _StubWithTrailingDayBug:
        """Mimics MarketDataService's real behavior: returns one extra day past `end`."""

        def get_price_data(self, tickers, start, end):
            padded_end = end + timedelta(days=1)
            dates = pd.bdate_range(start=start, end=padded_end)
            return pd.DataFrame({t: 100.0 + i for i, t in enumerate(tickers)}, index=dates)

    stub = _StubWithTrailingDayBug()
    start, end = date(2024, 1, 2), date(2024, 1, 31)

    miss_result = get_price_data_cached(db_session, stub, ["AAPL"], start, end)
    hit_result = get_price_data_cached(db_session, stub, ["AAPL"], start, end)

    miss_dates = {d.date() for d in miss_result.index}
    hit_dates = {d.date() for d in hit_result.index}

    assert miss_dates == hit_dates
    assert max(miss_dates) <= end
    assert max(hit_dates) <= end
