from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

import backend.cache as cache_module
from backend.cache import get_benchmark_data_cached, get_price_data_cached
from backend.data import MarketDataError
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
    from backend.models import MarketDataCoverage

    # State left behind by a legitimate earlier fetch: the price row AND the
    # coverage record. (Raw rows alone are deliberately NOT a cache hit any
    # more -- see test_price_rows_without_a_coverage_record_are_not_a_hit.)
    db_session.add(MarketDataCache(ticker="AAPL", date=date(2024, 1, 2), close_price=100.0))
    db_session.add(MarketDataCoverage(ticker="AAPL", start_date=date(2024, 1, 2), end_date=date(2024, 1, 2)))
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


# ===========================================================================
# Issue #1: coverage-based caching (exchange holidays, truncation, extension)
# ===========================================================================

class _ExchangeStub:
    """
    A provider that behaves like a real exchange feed:
    - weekends and `holidays` are ABSENT rows (not NaN) -- nothing trades;
    - like MarketDataService, it returns one extra trailing day past `end`;
    - `truncate_after` cuts the response short (simulating an outage / rate limit);
    - `listings` maps ticker -> first trading day (NaN before it, as in a
      combined multi-ticker download);
    - `fail` makes every call raise MarketDataError.
    """

    def __init__(self, holidays=(), truncate_after=None, listings=None, fail=False):
        self.holidays = pd.DatetimeIndex(sorted(pd.Timestamp(h) for h in holidays))
        self.truncate_after = pd.Timestamp(truncate_after) if truncate_after else None
        self.listings = {t: pd.Timestamp(d) for t, d in (listings or {}).items()}
        self.fail = fail
        self.calls = []

    @property
    def call_count(self):
        return len(self.calls)

    def get_price_data(self, tickers, start, end):
        self.calls.append((tuple(tickers), start, end))
        if self.fail:
            raise MarketDataError("provider unavailable")
        days = pd.bdate_range(start=start, end=end + timedelta(days=1)).difference(self.holidays)
        if self.truncate_after is not None:
            days = days[days <= self.truncate_after]
        frame = pd.DataFrame(
            {t: 100.0 + i + np.arange(len(days)) * 0.1 for i, t in enumerate(tickers)}, index=days
        )
        for ticker, listed in self.listings.items():
            if ticker in frame.columns:
                frame.loc[frame.index < listed, ticker] = np.nan
        return frame


def _coverage_rows(db_session, ticker):
    from backend.models import MarketDataCoverage

    return (
        db_session.query(MarketDataCoverage)
        .filter_by(ticker=ticker)
        .order_by(MarketDataCoverage.start_date)
        .all()
    )


HOLIDAYS_2023_24 = ["2023-12-25", "2024-01-01", "2024-01-15"]  # Christmas, New Year, MLK Day


def test_range_containing_exchange_holidays_is_a_hit_on_the_second_request(db_session):
    # THE Issue #1 reproduction. Previously completeness was judged against
    # pd.bdate_range(), which counts every Mon-Fri -- including these three
    # market holidays, which a correct fetch legitimately has NO row for.
    # Those dates therefore looked "missing" forever and the range could never
    # be judged complete, so every identical request re-downloaded everything.
    stub = _ExchangeStub(holidays=HOLIDAYS_2023_24)
    start, end = date(2023, 12, 18), date(2024, 1, 19)

    first = get_price_data_cached(db_session, stub, ["AAPL", "MSFT"], start, end)
    second = get_price_data_cached(db_session, stub, ["AAPL", "MSFT"], start, end)

    # The premise: bdate_range really does include the holidays the data lacks.
    weekday_index = pd.bdate_range(start, end)
    for holiday in HOLIDAYS_2023_24:
        assert pd.Timestamp(holiday) in weekday_index
        assert pd.Timestamp(holiday) not in first.index

    assert stub.call_count == 1  # second request served entirely from cache
    assert {d.date() for d in first.index} == {d.date() for d in second.index}
    assert not (set(pd.to_datetime(HOLIDAYS_2023_24)) & set(second.index))  # no invented rows


def test_weekend_boundaries_do_not_cause_cache_misses(db_session):
    stub = _ExchangeStub()
    start, end = date(2024, 1, 7), date(2024, 2, 3)  # a Sunday through a Saturday

    get_price_data_cached(db_session, stub, ["AAPL"], start, end)
    get_price_data_cached(db_session, stub, ["AAPL"], start, end)

    assert stub.call_count == 1
    (row,) = _coverage_rows(db_session, "AAPL")
    assert (row.start_date, row.end_date) == (start, end)  # recorded as requested, not as "last Friday"


def test_a_sub_range_of_covered_data_is_a_hit_and_returns_matching_values(db_session):
    stub = _ExchangeStub(holidays=HOLIDAYS_2023_24)
    full = get_price_data_cached(db_session, stub, ["AAPL"], date(2023, 12, 18), date(2024, 3, 29))

    sub = get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 2, 1), date(2024, 2, 28))

    assert stub.call_count == 1
    assert sub.index.min() >= pd.Timestamp("2024-02-01") and sub.index.max() <= pd.Timestamp("2024-02-28")
    for ts, value in sub["AAPL"].items():
        assert value == pytest.approx(full.loc[ts, "AAPL"])


def test_uncovered_range_fetches_and_extension_merges_into_one_coverage_row(db_session):
    stub = _ExchangeStub()
    get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 1, 2), date(2024, 2, 28))
    assert stub.call_count == 1

    # Extends past the covered end -> must fetch.
    get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 1, 2), date(2024, 4, 30))
    assert stub.call_count == 2

    # Now the extended range -- and anything inside it -- is covered.
    get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 3, 1), date(2024, 4, 15))
    get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 1, 2), date(2024, 4, 30))
    assert stub.call_count == 2

    (row,) = _coverage_rows(db_session, "AAPL")  # merged, not two overlapping rows
    assert (row.start_date, row.end_date) == (date(2024, 1, 2), date(2024, 4, 30))


def test_extending_backwards_also_fetches_and_merges(db_session):
    stub = _ExchangeStub()
    get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 2, 1), date(2024, 2, 28))
    get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 1, 2), date(2024, 2, 28))
    assert stub.call_count == 2

    (row,) = _coverage_rows(db_session, "AAPL")
    assert (row.start_date, row.end_date) == (date(2024, 1, 2), date(2024, 2, 28))


def _spans(db_session, ticker):
    return [(r.start_date, r.end_date) for r in _coverage_rows(db_session, ticker)]


def test_ranges_are_never_merged_across_an_unfetched_interval(db_session):
    # Regression: two fetched ranges 4 calendar days apart used to be merged merely because the gap
    # was inside a 10-day "buffer". Thu Feb 1 and Fri Feb 2 (normal trading days) were NEVER fetched,
    # yet a request for the whole span was then served from cache with those days missing.
    stub = _ExchangeStub()
    get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 1, 2), date(2024, 1, 31))
    get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 2, 5), date(2024, 2, 28))
    assert stub.call_count == 2

    assert _spans(db_session, "AAPL") == [  # two rows: the gap is NOT claimed as covered
        (date(2024, 1, 2), date(2024, 1, 31)),
        (date(2024, 2, 5), date(2024, 2, 28)),
    ]

    spanning = get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 1, 2), date(2024, 2, 28))

    assert stub.call_count == 3  # a MISS: the unfetched interval is fetched, not assumed present
    assert pd.Timestamp("2024-02-01") in spanning.index and pd.Timestamp("2024-02-02") in spanning.index
    assert _spans(db_session, "AAPL") == [(date(2024, 1, 2), date(2024, 2, 28))]  # now genuinely contiguous

    get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 1, 2), date(2024, 2, 28))
    assert stub.call_count == 3  # and only now a hit


def test_ranges_separated_only_by_a_weekend_still_merge_into_one_coverage_row(db_session):
    # Fri Feb 2 .. Mon Feb 5: the only uncovered days are Sat/Sun, when nothing trades.
    stub = _ExchangeStub()
    get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 1, 2), date(2024, 2, 2))
    get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 2, 5), date(2024, 2, 28))
    assert stub.call_count == 2

    assert _spans(db_session, "AAPL") == [(date(2024, 1, 2), date(2024, 2, 28))]
    get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 1, 2), date(2024, 2, 28))
    assert stub.call_count == 2  # the union is a hit


def test_a_gap_containing_a_weekday_is_not_bridged_even_if_it_is_a_holiday(db_session):
    # Mon Jan 15 (MLK Day) is a weekday gap. Without an exchange calendar it cannot be told apart from
    # a missing day, so the safe behaviour is one extra fetch -- never wrong data.
    stub = _ExchangeStub(holidays=["2024-01-15"])
    get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 1, 2), date(2024, 1, 12))
    get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 1, 16), date(2024, 1, 31))
    assert len(_spans(db_session, "AAPL")) == 2

    result = get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 1, 2), date(2024, 1, 31))

    assert stub.call_count == 3
    assert pd.Timestamp("2024-01-15") not in result.index  # the holiday is correctly absent, not invented
    assert pd.Timestamp("2024-01-16") in result.index
    assert _spans(db_session, "AAPL") == [(date(2024, 1, 2), date(2024, 1, 31))]


def test_a_new_range_overlapping_two_existing_ranges_merges_all_three(db_session):
    stub = _ExchangeStub()
    get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 1, 2), date(2024, 1, 12))
    get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 1, 22), date(2024, 1, 31))
    assert len(_spans(db_session, "AAPL")) == 2

    get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 1, 8), date(2024, 1, 25))

    assert _spans(db_session, "AAPL") == [(date(2024, 1, 2), date(2024, 1, 31))]


def test_each_ticker_has_independent_coverage(db_session):
    stub = _ExchangeStub()
    start, end = date(2024, 1, 2), date(2024, 2, 28)

    get_price_data_cached(db_session, stub, ["AAPL"], start, end)
    get_price_data_cached(db_session, stub, ["AAPL", "MSFT"], start, end)  # MSFT uncovered -> fetch
    assert stub.call_count == 2

    get_price_data_cached(db_session, stub, ["MSFT"], start, end)  # MSFT now covered on its own
    assert stub.call_count == 2

    # Widening only AAPL must not widen MSFT's coverage.
    get_price_data_cached(db_session, stub, ["AAPL"], date(2023, 11, 1), end)
    assert [(r.start_date, r.end_date) for r in _coverage_rows(db_session, "AAPL")] == [(date(2023, 11, 1), end)]
    assert [(r.start_date, r.end_date) for r in _coverage_rows(db_session, "MSFT")] == [(start, end)]

    calls_before = stub.call_count
    get_price_data_cached(db_session, stub, ["MSFT"], start, end)
    assert stub.call_count == calls_before  # MSFT still a hit


def test_truncated_response_is_not_recorded_as_full_coverage_and_is_retried(db_session):
    start, end = date(2024, 1, 2), date(2024, 3, 1)  # end is a Friday
    truncated = _ExchangeStub(truncate_after="2024-02-09")  # response stops 3 weeks early

    get_price_data_cached(db_session, truncated, ["AAPL"], start, end)

    (row,) = _coverage_rows(db_session, "AAPL")
    assert row.end_date == date(2024, 2, 9)  # what was actually observed -- NOT the requested end
    assert row.end_date != end

    # Same request again: must NOT be served from the truncated cache.
    get_price_data_cached(db_session, truncated, ["AAPL"], start, end)
    assert truncated.call_count == 2

    # The provider recovers: full data arrives, coverage extends, then it's a hit.
    healthy = _ExchangeStub()
    get_price_data_cached(db_session, healthy, ["AAPL"], start, end)
    assert healthy.call_count == 1
    get_price_data_cached(db_session, healthy, ["AAPL"], start, end)
    assert healthy.call_count == 1

    (row,) = _coverage_rows(db_session, "AAPL")
    assert (row.start_date, row.end_date) == (start, end)


def test_trailing_tolerance_boundary(db_session):
    # Documents the methodology choice: a last observation up to 7 calendar
    # days before `end` is treated as "reaches end" (weekends/holiday clusters/
    # a not-yet-published bar); further back is treated as truncation.
    start, end = date(2024, 1, 2), date(2024, 3, 1)

    at_limit = _ExchangeStub(truncate_after="2024-02-23")  # exactly 7 days before end
    get_price_data_cached(db_session, at_limit, ["AAPL"], start, end)
    assert _coverage_rows(db_session, "AAPL")[0].end_date == end

    beyond = _ExchangeStub(truncate_after="2024-02-22")  # 8 days before end
    get_price_data_cached(db_session, beyond, ["MSFT"], start, end)
    assert _coverage_rows(db_session, "MSFT")[0].end_date == date(2024, 2, 22)


def test_failed_fetch_records_no_price_rows_and_no_coverage(db_session):
    from backend.models import MarketDataCoverage

    broken = _ExchangeStub(fail=True)
    with pytest.raises(MarketDataError):
        get_price_data_cached(db_session, broken, ["AAPL"], date(2024, 1, 2), date(2024, 2, 28))

    assert db_session.query(MarketDataCoverage).count() == 0
    assert db_session.query(MarketDataCache).count() == 0

    # A failure must not poison the key: the next attempt fetches normally.
    healthy = _ExchangeStub()
    get_price_data_cached(db_session, healthy, ["AAPL"], date(2024, 1, 2), date(2024, 2, 28))
    assert healthy.call_count == 1


def test_interior_missing_observation_is_neither_invented_nor_refetched(db_session):
    # A day the provider simply has no bar for (indistinguishable from a
    # holiday without an exchange calendar) must not be fabricated, and must
    # not cause the range to be refetched on every request.
    class _StubWithOneMissingBar(_ExchangeStub):
        def get_price_data(self, tickers, start, end):
            frame = super().get_price_data(tickers, start, end)
            frame.loc[pd.Timestamp("2024-02-14"), "AAPL"] = np.nan
            return frame

    stub = _StubWithOneMissingBar()
    start, end = date(2024, 1, 2), date(2024, 2, 28)

    first = get_price_data_cached(db_session, stub, ["AAPL", "MSFT"], start, end)
    second = get_price_data_cached(db_session, stub, ["AAPL", "MSFT"], start, end)

    assert stub.call_count == 1
    assert np.isnan(first.loc["2024-02-14", "AAPL"])
    assert np.isnan(second.loc["2024-02-14", "AAPL"])  # still missing, not back-filled by the cache
    assert not np.isnan(second.loc["2024-02-14", "MSFT"])


def test_late_listing_is_cached_and_not_refetched_for_the_nonexistent_history(db_session):
    stub = _ExchangeStub(listings={"NEWCO": "2024-02-15"})
    start, end = date(2024, 1, 2), date(2024, 3, 1)

    get_price_data_cached(db_session, stub, ["AAPL", "NEWCO"], start, end)
    get_price_data_cached(db_session, stub, ["AAPL", "NEWCO"], start, end)
    assert stub.call_count == 1  # no endless retry for data that cannot exist


def test_covered_sub_range_before_a_listing_returns_nan_not_a_crash(db_session):
    stub = _ExchangeStub(listings={"NEWCO": "2024-02-15"})
    get_price_data_cached(db_session, stub, ["AAPL", "NEWCO"], date(2024, 1, 2), date(2024, 3, 1))

    early = get_price_data_cached(db_session, stub, ["AAPL", "NEWCO"], date(2024, 1, 2), date(2024, 1, 31))

    assert stub.call_count == 1
    assert list(early.columns) == ["AAPL", "NEWCO"]
    assert early["NEWCO"].isna().all()  # honest NaN; Issue #10's gap handling governs what happens next
    assert early["AAPL"].notna().all()


def test_price_rows_without_a_coverage_record_are_not_a_hit(db_session):
    # Raw rows are not proof a range was fetched -- only a coverage record is.
    db_session.add_all(
        MarketDataCache(ticker="AAPL", date=d.date(), close_price=100.0)
        for d in pd.bdate_range("2024-01-02", "2024-01-31")
    )
    db_session.commit()

    stub = _ExchangeStub()
    get_price_data_cached(db_session, stub, ["AAPL"], date(2024, 1, 2), date(2024, 1, 31))
    assert stub.call_count == 1


def test_hit_and_miss_return_identical_data_across_holidays(db_session):
    stub = _ExchangeStub(holidays=HOLIDAYS_2023_24)
    args = (db_session, stub, ["AAPL", "MSFT"], date(2023, 12, 18), date(2024, 1, 19))

    miss = get_price_data_cached(*args)
    hit = get_price_data_cached(*args)

    assert stub.call_count == 1
    assert {(d.date(), round(v, 6)) for d, v in miss["AAPL"].items()} == {
        (d.date(), round(v, 6)) for d, v in hit["AAPL"].items()
    }
