"""
Tests for latest-price caching in MarketDataService (QA Issue #12).

Policy under test (see LATEST_PRICE_TTL_SECONDS in backend/data.py):
  - a latest price is reused for a short, explicit TTL, then re-fetched;
  - entries are keyed per normalized ticker;
  - provider failures / invalid tickers are NEVER cached, and an expired entry is NEVER
    served as a fallback when the provider fails (no silently-stale "current" price);
  - TTL <= 0 disables caching;
  - the historical price path is untouched;
  - the shared, thread-pool-served service stays correct under concurrent use.

get_latest_prices() is also the validation gate used by PortfolioService.create_portfolio
(every holding AND the benchmark are fetched through it BEFORE anything is committed), so
several tests pin that the cache cannot weaken that gate.

yfinance is replaced by a counting fake: no network access, fully deterministic.
"""

import dataclasses
import threading
import time
from datetime import date, datetime, timezone

import numpy as np
import pandas as pd
import pytest

from backend import data as data_module
from backend.data import LatestQuote, MarketDataError, MarketDataService

BARS = pd.bdate_range("2024-03-04", periods=5)  # Mon..Fri
INVALID = "ZZZINVALID"  # same convention as conftest.INVALID_TICKER: a symbol the provider returns only NaN for


class _FakeProvider:
    """Stands in for yfinance.download. Records every call; prices/failures are controllable."""

    def __init__(self, prices: dict[str, float]):
        self.prices = dict(prices)
        self.calls: list[tuple[str, ...]] = []
        self.kwargs: list[dict] = []
        self.error: Exception | None = None
        self.trailing_nan_bars = 0  # simulate "no bar yet" for the most recent N sessions

    def _close(self, ticker: str) -> pd.Series:
        last = self.prices.get(ticker, np.nan)  # unknown ticker -> all-NaN, like an invalid symbol
        close = pd.Series(last * np.linspace(0.97, 1.0, len(BARS)), index=BARS, dtype=float)
        if self.trailing_nan_bars:
            close.iloc[-self.trailing_nan_bars :] = np.nan
        return close

    def __call__(self, tickers=None, start=None, end=None, **kwargs):
        tickers = [tickers] if isinstance(tickers, str) else list(tickers)
        self.calls.append(tuple(tickers))
        self.kwargs.append({"start": start, "end": end, **kwargs})
        if self.error is not None:
            raise self.error
        if len(tickers) == 1:
            return pd.DataFrame({"Close": self._close(tickers[0])})
        cols = pd.MultiIndex.from_product([tickers, ["Close"]])
        return pd.DataFrame({(t, "Close"): self._close(t) for t in tickers}, index=BARS, columns=cols)


class _Clock:
    def __init__(self):
        self.now = 1_000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def provider(monkeypatch):
    fake = _FakeProvider({"AAPL": 230.0, "MSFT": 415.0, "GOOG": 170.0, "^GSPC": 5200.0})
    monkeypatch.setattr(data_module.yf, "download", fake)
    return fake


def _service(ttl: float = 60.0) -> tuple[MarketDataService, _Clock]:
    clock = _Clock()
    return MarketDataService(latest_price_ttl_seconds=ttl, clock=clock), clock


# ---------------------------------------------------------------------------
# First request hits the provider; repeats within the window do not.
# ---------------------------------------------------------------------------


def test_default_service_reuses_latest_prices_within_the_freshness_window(provider):
    # Default constructor == how the app wires it (PortfolioService() -> MarketDataService()).
    # Before the fix every call below hit the provider (2 calls for 2 requests).
    svc = MarketDataService()

    first = svc.get_latest_prices(["AAPL", "MSFT"])
    second = svc.get_latest_prices(["AAPL", "MSFT"])

    assert len(provider.calls) == 1
    assert first == second == {"AAPL": pytest.approx(230.0), "MSFT": pytest.approx(415.0)}


def test_first_request_calls_provider_once_with_the_sorted_unique_tickers(provider):
    svc, _ = _service()

    prices = svc.get_latest_prices(["MSFT", "AAPL", "MSFT"])

    assert provider.calls == [("AAPL", "MSFT")]  # identical to the pre-cache behaviour on a cold cache
    assert prices["AAPL"] == pytest.approx(230.0)
    assert prices["MSFT"] == pytest.approx(415.0)
    assert list(prices) == ["AAPL", "MSFT"]
    start, end = provider.kwargs[0]["start"], provider.kwargs[0]["end"]
    assert (end - start).days == 11  # 10-day lookback (+1 day: yfinance's end is exclusive)


def test_repeat_within_window_does_not_call_provider_and_serves_the_same_snapshot(provider):
    svc, clock = _service(ttl=60)
    first = svc.get_latest_quotes(["AAPL"])
    provider.prices["AAPL"] = 999.0  # the market "moves" at the provider

    clock.advance(59.9)
    again = svc.get_latest_quotes(["AAPL"])

    assert len(provider.calls) == 1
    assert again["AAPL"].price == pytest.approx(230.0)  # bounded staleness: the snapshot, not 999
    assert again["AAPL"].from_cache is True
    assert first["AAPL"].from_cache is False
    assert again["AAPL"].fetched_at == first["AAPL"].fetched_at  # freshness metadata travels with the price


# ---------------------------------------------------------------------------
# The real call patterns in service.py.
# ---------------------------------------------------------------------------


def test_creation_validation_call_primes_the_cache_for_the_endpoints_that_follow(provider):
    # create_portfolio:   get_latest_prices(holdings + [benchmark])
    # /performance, /risk, /efficient-frontier, GET /portfolio/{id}:   get_latest_prices(holdings)
    # Before the fix this sequence made 5 provider downloads; now it makes 1.
    svc, _ = _service()
    holdings = ["AAPL", "MSFT"]

    created = svc.get_latest_prices(holdings + ["^GSPC"])
    for _ in range(4):
        later = svc.get_latest_prices(holdings)

    assert len(provider.calls) == 1
    assert later == {"AAPL": created["AAPL"], "MSFT": created["MSFT"]}  # every endpoint sees the same snapshot


def test_a_failed_validation_batch_blocks_creation_every_time_and_caches_nothing(provider):
    # An invalid benchmark must keep failing portfolio creation: the cache must not turn
    # "validated before commit" into "validated once, then waved through".
    svc, _ = _service()

    for _ in range(3):
        with pytest.raises(MarketDataError):
            svc.get_latest_prices(["AAPL", "MSFT", INVALID])

    assert len(provider.calls) == 3  # re-validated against the provider each time
    assert svc._latest_cache == {}  # and not even the valid batch-mates were remembered


def test_valid_tickers_cached_earlier_do_not_mask_a_new_invalid_ticker_in_the_batch(provider):
    svc, _ = _service()
    svc.get_latest_prices(["AAPL", "MSFT"])  # cached

    with pytest.raises(MarketDataError):
        svc.get_latest_prices(["AAPL", "MSFT", INVALID])  # only the unknown ticker is fetched -> still rejected

    assert provider.calls[-1] == (INVALID,)
    assert "ZZZINVALID" not in svc._latest_cache


# ---------------------------------------------------------------------------
# Different tickers are separate entries; only missing ones are fetched.
# ---------------------------------------------------------------------------


def test_each_ticker_is_a_separate_cache_entry_and_only_missing_ones_are_fetched(provider):
    svc, _ = _service()

    svc.get_latest_prices(["AAPL"])
    svc.get_latest_prices(["MSFT"])  # different ticker -> its own provider call
    assert provider.calls == [("AAPL",), ("MSFT",)]

    svc.get_latest_prices(["AAPL", "MSFT"])  # both cached -> no call
    assert len(provider.calls) == 2

    prices = svc.get_latest_prices(["AAPL", "MSFT", "GOOG"])  # only GOOG is new
    assert provider.calls[-1] == ("GOOG",)
    assert prices == {"AAPL": pytest.approx(230.0), "MSFT": pytest.approx(415.0), "GOOG": pytest.approx(170.0)}


# ---------------------------------------------------------------------------
# Key normalization.
# ---------------------------------------------------------------------------


def test_cache_key_is_normalized_but_results_keep_the_callers_spelling(provider):
    svc, _ = _service()

    svc.get_latest_prices(["AAPL"])
    lower = svc.get_latest_prices(["aapl"])
    padded = svc.get_latest_prices([" AAPL "])

    assert len(provider.calls) == 1  # 'aapl' and ' AAPL ' reuse the 'AAPL' entry
    assert lower == {"aapl": pytest.approx(230.0)}
    assert padded == {" AAPL ": pytest.approx(230.0)}


def test_different_spellings_of_one_ticker_in_one_request_share_a_single_fetch(provider):
    svc, _ = _service()

    prices = svc.get_latest_prices(["AAPL", "aapl"])

    assert provider.calls == [("AAPL",)]
    assert prices["AAPL"] == prices["aapl"] == pytest.approx(230.0)


# ---------------------------------------------------------------------------
# Expiry.
# ---------------------------------------------------------------------------


def test_expired_entry_is_refetched_and_returns_the_new_price(provider):
    svc, clock = _service(ttl=60)
    svc.get_latest_prices(["AAPL"])

    clock.advance(59.999)
    svc.get_latest_prices(["AAPL"])
    assert len(provider.calls) == 1  # still fresh just before the TTL

    provider.prices["AAPL"] = 240.0
    clock.advance(0.001)  # age == TTL exactly -> expired
    refreshed = svc.get_latest_quotes(["AAPL"])

    assert len(provider.calls) == 2
    assert refreshed["AAPL"].price == pytest.approx(240.0)
    assert refreshed["AAPL"].from_cache is False


def test_expiry_is_per_ticker(provider):
    svc, clock = _service(ttl=60)
    svc.get_latest_prices(["AAPL"])
    clock.advance(40)
    svc.get_latest_prices(["MSFT"])
    clock.advance(30)  # AAPL is 70s old (expired), MSFT is 30s old (fresh)

    svc.get_latest_prices(["AAPL", "MSFT"])

    assert provider.calls[-1] == ("AAPL",)
    assert len(provider.calls) == 3


def test_expired_entries_are_evicted_so_the_cache_stays_bounded(provider):
    svc, clock = _service(ttl=60)
    svc.get_latest_prices(["AAPL"])
    svc.get_latest_prices(["MSFT"])
    clock.advance(120)

    svc.get_latest_prices(["GOOG"])  # storing GOOG sweeps out the two expired entries

    assert set(svc._latest_cache) == {"GOOG"}


# ---------------------------------------------------------------------------
# Failures are never cached, and never masked by stale data.
# ---------------------------------------------------------------------------


def test_provider_exception_is_not_cached_and_the_next_request_retries(provider):
    svc, _ = _service()
    provider.error = RuntimeError("yahoo is down")

    with pytest.raises(RuntimeError):
        svc.get_latest_prices(["AAPL"])

    provider.error = None  # provider recovers immediately
    prices = svc.get_latest_prices(["AAPL"])

    assert len(provider.calls) == 2  # the failure did not poison the cache
    assert prices["AAPL"] == pytest.approx(230.0)


def test_expired_price_is_never_served_when_the_provider_fails(provider):
    svc, clock = _service(ttl=60)
    svc.get_latest_prices(["AAPL"])
    clock.advance(61)  # now stale
    provider.error = RuntimeError("yahoo is down")

    with pytest.raises(RuntimeError):  # must fail loudly, not show a 61s-old price as current
        svc.get_latest_prices(["AAPL"])


# ---------------------------------------------------------------------------
# Caching can be disabled; behaviour is then exactly the old one.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ttl", [0, 0.0, -5])
def test_ttl_of_zero_or_less_disables_caching(provider, ttl):
    svc, _ = _service(ttl=ttl)

    svc.get_latest_prices(["AAPL"])
    provider.prices["AAPL"] = 235.0
    again = svc.get_latest_prices(["AAPL"])

    assert len(provider.calls) == 2
    assert again["AAPL"] == pytest.approx(235.0)  # always the provider's current value
    assert svc._latest_cache == {}  # nothing stored either


def test_clear_latest_price_cache_forces_a_refetch(provider):
    svc, _ = _service()
    svc.get_latest_prices(["AAPL"])

    svc.clear_latest_price_cache()
    svc.get_latest_prices(["AAPL"])

    assert len(provider.calls) == 2


def test_cached_and_uncached_services_return_identical_prices(provider):
    provider.trailing_nan_bars = 1  # include a missing most-recent bar, so forward-fill is exercised
    cached, _ = _service(ttl=60)
    uncached, _ = _service(ttl=0)
    tickers = ["AAPL", "MSFT", "GOOG"]

    cached.get_latest_prices(tickers)  # populate
    from_cache = cached.get_latest_prices(tickers)
    direct = uncached.get_latest_prices(tickers)

    assert from_cache == direct


# ---------------------------------------------------------------------------
# Returned values are never corrupted by the cache.
# ---------------------------------------------------------------------------


def test_mutating_a_returned_dict_cannot_corrupt_later_results(provider):
    svc, _ = _service()

    first = svc.get_latest_prices(["AAPL"])
    first["AAPL"] = -1.0
    first["EXTRA"] = 123.0
    again = svc.get_latest_prices(["AAPL"])

    assert again == {"AAPL": pytest.approx(230.0)}
    assert len(provider.calls) == 1


def test_cached_quote_objects_are_immutable(provider):
    svc, _ = _service()
    quote = svc.get_latest_quotes(["AAPL"])["AAPL"]

    with pytest.raises(dataclasses.FrozenInstanceError):
        quote.price = 1.0


# ---------------------------------------------------------------------------
# Weekends / holidays and freshness metadata.
# ---------------------------------------------------------------------------


def test_missing_recent_bars_use_last_real_close_and_report_its_trading_date(provider):
    # Long weekend / holiday: the last two sessions have no bar. That is not an error.
    provider.trailing_nan_bars = 2
    svc, _ = _service()

    quote = svc.get_latest_quotes(["AAPL"])["AAPL"]

    expected = 230.0 * np.linspace(0.97, 1.0, len(BARS))[-3]  # the 3rd-last bar is the last real one
    assert quote.price == pytest.approx(expected)
    assert quote.price_date == BARS[-3].date()  # surfaces that the price is NOT from "today"
    assert quote.price_date < BARS[-1].date()


def test_quote_records_when_it_was_fetched_in_utc(provider):
    svc, _ = _service()
    before = datetime.now(timezone.utc)

    quote = svc.get_latest_quotes(["AAPL"])["AAPL"]

    after = datetime.now(timezone.utc)
    assert quote.fetched_at.tzinfo is not None
    assert before <= quote.fetched_at <= after
    assert quote.price_date == BARS[-1].date()


def test_empty_ticker_list_still_raises_without_calling_the_provider(provider):
    svc, _ = _service()
    with pytest.raises(MarketDataError, match="No tickers"):
        svc.get_latest_prices([])
    assert provider.calls == []


# ---------------------------------------------------------------------------
# Concurrency: the API serves sync endpoints from a thread pool against ONE shared service.
# ---------------------------------------------------------------------------


def test_concurrent_cold_requests_all_get_correct_prices_and_fetch_at_most_once_per_thread(provider):
    svc = MarketDataService(latest_price_ttl_seconds=60)  # real clock
    expected = {"AAPL": 230.0, "GOOG": 170.0, "MSFT": 415.0}
    n_threads, n_calls = 12, 40
    barrier = threading.Barrier(n_threads)
    results, errors = [], []

    def worker():
        try:
            barrier.wait(timeout=10)  # release every thread at once against a cold cache
            for _ in range(n_calls):
                results.append(svc.get_latest_prices(["AAPL", "MSFT", "GOOG"]))
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors
    assert len(results) == n_threads * n_calls
    for r in results:
        assert r == {k: pytest.approx(v) for k, v in expected.items()}  # no torn / wrong values, ever
    # A cold stampede may fetch redundantly (benign), but never more than once per thread,
    # and once warm every later call is a hit.
    assert 1 <= len(provider.calls) <= n_threads
    assert set(svc._latest_cache) == set(expected)


def test_concurrent_use_while_entries_expire_and_are_evicted_stays_correct(provider):
    # A tiny TTL on a clock that keeps advancing makes entries expire and get evicted while
    # other threads are reading and writing the same dict.
    clock = _Clock()
    svc = MarketDataService(latest_price_ttl_seconds=1.0, clock=clock)
    universe = ["AAPL", "MSFT", "GOOG"]
    stop = threading.Event()
    errors, bad = [], []

    def ticker_thread():
        while not stop.is_set():
            clock.advance(0.4)

    def worker(offset: int):
        try:
            for i in range(300):
                subset = [universe[(offset + i) % 3], universe[(offset + i + 1) % 3]]
                got = svc.get_latest_prices(subset)
                for t, price in got.items():
                    if price != pytest.approx(provider.prices[t]):
                        bad.append((t, price))
        except Exception as exc:  # pragma: no cover - failure path (e.g. dict changed size during iteration)
            errors.append(exc)

    clock_thread = threading.Thread(target=ticker_thread)
    workers = [threading.Thread(target=worker, args=(k,)) for k in range(8)]
    clock_thread.start()
    for w in workers:
        w.start()
    for w in workers:
        w.join(timeout=60)
    stop.set()
    clock_thread.join(timeout=10)

    assert not errors
    assert not bad
    assert len(svc._latest_cache) <= len(universe)  # eviction kept it bounded


class _PausingDict(dict):
    """A dict whose next items() scan pauses after its first item until released.

    Lets a test force, deterministically, the interleaving a thread pool could produce:
    thread A is mid-way through scanning the cache while thread B tries to mutate it.
    """

    def __init__(self):
        super().__init__()
        self.armed = False
        self.paused = threading.Event()
        self.resume = threading.Event()

    def items(self):
        underlying = super().items()
        if not self.armed:
            return underlying
        self.armed = False

        def scan():
            first = True
            for kv in underlying:
                yield kv
                if first:
                    first = False
                    self.paused.set()
                    self.resume.wait(10)

        return scan()


def test_cache_mutation_during_an_eviction_scan_is_serialized_by_the_lock(provider):
    # Without the lock, thread B inserting while thread A is mid-scan makes A's scan fail with
    # "dictionary changed size during iteration". (The stress test above cannot catch that: the GIL
    # usually hides it. This one forces the interleaving.)
    svc, _ = _service(ttl=60)
    cache = _PausingDict()
    svc._latest_cache = cache
    quote = LatestQuote(price=1.0, price_date=None, fetched_at=datetime.now(timezone.utc))
    svc._store_latest("AAPL", quote)
    svc._store_latest("MSFT", quote)
    cache.armed = True
    errors = []

    def store(key):
        try:
            svc._store_latest(key, quote)
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    thread_a = threading.Thread(target=store, args=("NVDA",))
    thread_b = threading.Thread(target=store, args=("GOOG",))
    thread_a.start()
    assert cache.paused.wait(10)  # A is now mid-scan, holding the lock
    thread_b.start()  # B must wait for A; without a lock it would mutate the dict right now
    time.sleep(0.2)
    cache.resume.set()
    thread_a.join(timeout=10)
    thread_b.join(timeout=10)

    assert not errors
    assert set(cache) == {"AAPL", "MSFT", "NVDA", "GOOG"}


# ---------------------------------------------------------------------------
# Historical path is untouched by (and independent of) the latest-price cache.
# ---------------------------------------------------------------------------


def test_historical_price_fetches_do_not_use_or_populate_the_latest_price_cache(provider):
    svc, _ = _service()

    hist_1 = svc.get_price_data(["AAPL"], date(2024, 3, 4), date(2024, 3, 8))
    hist_2 = svc.get_price_data(["AAPL"], date(2024, 3, 4), date(2024, 3, 8))
    assert len(provider.calls) == 2  # historical fetch is still a plain provider call (DB cache lives in cache.py)
    pd.testing.assert_frame_equal(hist_1, hist_2)
    assert svc._latest_cache == {}  # a historical fetch must not masquerade as a latest price

    svc.get_latest_prices(["AAPL"])
    assert len(provider.calls) == 3  # ...and a latest price still needs its own fetch
    assert len(svc.get_price_data(["AAPL"], date(2024, 3, 4), date(2024, 3, 8))) == len(BARS)
    assert len(provider.calls) == 4  # latest-price cache does not short-circuit historical requests
