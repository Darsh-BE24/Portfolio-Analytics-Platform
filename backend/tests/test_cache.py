from datetime import date, timedelta

import numpy as np
import pandas as pd

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
