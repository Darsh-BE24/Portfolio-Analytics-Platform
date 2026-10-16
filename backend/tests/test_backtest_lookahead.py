"""
Issue #3: the backtest's "current" strategy must not use weights that leak the
test period's own returns.

Before the fix, service.run_backtest priced the user's share counts at TODAY's
live quote -- the END of the test window -- and used those weights as the
allocation held from the START of it. A holding that rallied during the test
therefore began with an inflated weight (the strategy "buys more of whatever
won"), biasing the result upward. The fix prices the same share counts with
prices known as of the split date.

These tests use a controlled market (seeded random walks, no network) in which
holding A TRIPLES between the split date and today, so the leak is large and
the expected answer is computable by plain arithmetic -- the expectations below
are deliberately derived independently of the code under test.
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from backend.backtest import (
    BacktestError,
    buy_and_hold_returns,
    compute_turnover,
    train_test_split,
)
from backend.optimization import minimum_volatility
from backend.performance import daily_returns
from backend.schemas import BacktestRequest, PortfolioCreateRequest
from backend.service import PortfolioService

CALENDAR = pd.bdate_range("2018-01-01", "2035-12-31")
QUANTITIES = {"A": 30.0, "B": 10.0}


def _walk(seed, drift=0.0004, vol=0.012):
    rng = np.random.RandomState(seed)
    return pd.Series(100 * np.cumprod(1 + rng.normal(drift, vol, len(CALENDAR))), index=CALENDAR)


def _pos(day):
    """Position of the last calendar row on or before `day`."""
    return int(CALENDAR.searchsorted(pd.Timestamp(day), side="right")) - 1


def _price(series, day):
    return float(series.iloc[_pos(day)])


def _expected_weights(series, quantities, as_of):
    values = {t: q * _price(series[t], as_of) for t, q in quantities.items()}
    total = sum(values.values())
    return {t: v / total for t, v in values.items()}


def _scenario(split):
    """A: unchanged up to the split date, then ramps to 3x by today. B and the benchmark: plain walks."""
    factor = np.interp(np.arange(len(CALENDAR)), [_pos(split), _pos(date.today())], [1.0, 3.0])
    return {
        "A": _walk(11) * factor,
        "B": _walk(22, drift=0.0003),
        "^GSPC": _walk(33, drift=0.0003, vol=0.009),
    }


class _ControlledMarket:
    """Deterministic stand-in for MarketDataService: prices come from fixed series."""

    def __init__(self, series):
        self.series = series

    def get_price_data(self, tickers, start, end):
        frame = pd.DataFrame({t: self.series[t] for t in tickers})
        return frame.loc[pd.Timestamp(start): pd.Timestamp(end) + pd.Timedelta(days=1)]

    def get_latest_prices(self, tickers):
        return {t: _price(self.series[t], date.today()) for t in tickers}


@pytest.fixture()
def split():
    return date.today() - timedelta(days=300)


@pytest.fixture()
def world(db_session, split):
    series = _scenario(split)
    market = _ControlledMarket(series)
    service = PortfolioService(market_data=market)
    created = service.create_portfolio(
        db_session,
        PortfolioCreateRequest(
            holdings=[{"ticker": t, "quantity": q} for t, q in QUANTITIES.items()],
            benchmark="^GSPC",
            lookback_years=2.0,
        ),
    )
    return service, created.id, market, series


def _backtest(db_session, world, split, **kwargs):
    service, portfolio_id, _, _ = world
    return service.run_backtest(db_session, portfolio_id, BacktestRequest(split_date=split, **kwargs))


# --- the reproduction -------------------------------------------------------------

def test_current_strategy_uses_split_date_weights_not_todays_live_weights(db_session, world, split):
    _, _, _, series = world
    split_weights = _expected_weights(series, QUANTITIES, split)
    live_weights = _expected_weights(series, QUANTITIES, date.today())

    # Sanity of the scenario itself: A's tripling makes the two allocations clearly different.
    assert live_weights["A"] - split_weights["A"] > 0.05

    current = _backtest(db_session, world, split).strategies["current"].weights

    assert current == pytest.approx(split_weights, abs=1e-9)
    assert current["A"] < live_weights["A"] - 0.05  # NOT today's live weights (the old behavior)


def test_leaking_todays_weights_really_would_inflate_the_current_strategy(db_session, world, split):
    # Shows the issue is a genuine bias, not stylistic: holding the test period's
    # winner at a weight that already reflects its win beats holding it at the
    # weight that was knowable at the start.
    _, _, _, series = world
    frame = pd.DataFrame({t: series[t] for t in QUANTITIES}).loc[
        pd.Timestamp(date.today() - timedelta(days=730)): pd.Timestamp(date.today())
    ]
    _, test_prices = train_test_split(frame, split)

    honest = buy_and_hold_returns(test_prices, _expected_weights(series, QUANTITIES, split))
    leaked = buy_and_hold_returns(test_prices, _expected_weights(series, QUANTITIES, date.today()))

    assert (1 + leaked).prod() > (1 + honest).prod() * 1.05


def test_current_strategy_result_matches_an_independent_buy_and_hold(db_session, world, split):
    _, _, _, series = world
    weights = _expected_weights(series, QUANTITIES, split)
    frame = pd.DataFrame({t: series[t] for t in QUANTITIES}).loc[
        pd.Timestamp(date.today() - timedelta(days=730)): pd.Timestamp(date.today())
    ]
    test = frame[frame.index > pd.Timestamp(split)]

    # independent re-derivation: positions sized at the test's first price, value tracked daily
    shares = {t: weights[t] / test[t].iloc[0] for t in QUANTITIES}
    value = sum(shares[t] * test[t] for t in QUANTITIES)
    expected_total = float(value.iloc[-1] / value.iloc[0])
    n_returns = len(test) - 1
    expected_cagr = expected_total ** (252 / n_returns) - 1

    metrics = _backtest(db_session, world, split).strategies["current"].metrics
    assert metrics["cagr"] == pytest.approx(expected_cagr, rel=1e-9)


# --- everything that must NOT have changed ------------------------------------------

def test_other_strategies_are_unaffected_by_the_fix(db_session, world, split):
    _, _, _, series = world
    result = _backtest(db_session, world, split)

    assert set(result.strategies) == {"current", "equal_weight", "min_volatility", "max_sharpe", "benchmark"}
    assert result.strategies["equal_weight"].weights == pytest.approx({"A": 0.5, "B": 0.5})
    assert result.strategies["benchmark"].weights is None

    # min-volatility depends on training data only -- re-derive it independently.
    frame = pd.DataFrame({t: series[t] for t in QUANTITIES}).loc[
        pd.Timestamp(date.today() - timedelta(days=730)): pd.Timestamp(date.today())
    ]
    train, _ = train_test_split(frame, split)
    expected = minimum_volatility(daily_returns(train))["weights"]
    assert result.strategies["min_volatility"].weights == pytest.approx(expected, abs=1e-6)


def test_other_endpoints_still_use_live_weights(db_session, world):
    # Only the backtest changes: "what is my portfolio now" is correctly live-priced.
    service, portfolio_id, _, series = world
    live = service.get_portfolio(db_session, portfolio_id)
    expected = _expected_weights(series, QUANTITIES, date.today())
    assert {p.ticker: p.weight for p in live.positions} == pytest.approx(expected)


# --- Issue #8 still applies, now measured from the split-date allocation -------------

def test_transaction_cost_is_charged_on_turnover_from_the_split_date_allocation(db_session, world, split):
    free = _backtest(db_session, world, split, transaction_cost_rate=0.0)
    costly = _backtest(db_session, world, split, transaction_cost_rate=0.02)

    # the "current" strategy is the starting point: it never pays to switch
    assert costly.strategies["current"].metrics["cagr"] == pytest.approx(free.strategies["current"].metrics["cagr"])

    current_w = free.strategies["current"].weights
    eq_w = free.strategies["equal_weight"].weights
    turnover = compute_turnover(current_w, eq_w)
    assert turnover > 0.2  # 75/25 -> 50/50: a real switch

    # Issue #8 wealth equation: wealth is scaled by (1 - turnover * cost) once, at entry,
    # so (1 + CAGR) scales by that factor ** (252 / n_returns).
    test_rows = CALENDAR[(CALENDAR > pd.Timestamp(split)) & (CALENDAR <= pd.Timestamp(date.today()))]
    n_returns = len(test_rows) - 1
    ratio = (1 + costly.strategies["equal_weight"].metrics["cagr"]) / (1 + free.strategies["equal_weight"].metrics["cagr"])
    assert ratio == pytest.approx((1 - turnover * 0.02) ** (252 / n_returns), rel=1e-9)


# --- boundary dates ------------------------------------------------------------------

def test_split_after_all_available_data_is_rejected(db_session, world):
    with pytest.raises(BacktestError, match="after split date"):
        _backtest(db_session, world, date.today())


def test_split_before_any_data_is_rejected(db_session, world):
    with pytest.raises(BacktestError, match="on or before split date"):
        _backtest(db_session, world, date.today() - timedelta(days=5000))


def test_split_date_on_a_weekend_prices_holdings_at_the_prior_trading_day(db_session, world):
    _, _, _, series = world
    saturday = date.today() - timedelta(days=300)
    while saturday.weekday() != 5:
        saturday -= timedelta(days=1)
    friday, monday = saturday - timedelta(days=1), saturday + timedelta(days=2)

    # the scenario's ramp starts at an arbitrary split; re-anchor so the weekend is the split
    series = _scenario(saturday)
    world[2].series = series
    current = _backtest(db_session, world, saturday).strategies["current"].weights

    assert current == pytest.approx(_expected_weights(series, QUANTITIES, friday), abs=1e-9)
    assert current != pytest.approx(_expected_weights(series, QUANTITIES, monday), abs=1e-6)


def test_split_date_that_is_itself_a_trading_day_includes_that_days_price(db_session, world):
    split_day = date.today() - timedelta(days=299)
    while split_day.weekday() >= 5:
        split_day -= timedelta(days=1)

    series = _scenario(split_day)
    world[2].series = series
    current = _backtest(db_session, world, split_day).strategies["current"].weights

    assert current == pytest.approx(_expected_weights(series, QUANTITIES, split_day), abs=1e-9)


# --- Issue #10 interplay: gaps right at the split date --------------------------------

def test_short_gap_ending_at_the_split_date_uses_the_last_available_price(db_session, world, split):
    _, _, market, series = world
    i = _pos(split)
    gapped = series["B"].copy()
    gapped.iloc[i - 2: i + 1] = np.nan  # 3 missing observations ending at the split date
    market.series = {**series, "B": gapped}

    current = _backtest(db_session, world, split).strategies["current"].weights

    last_known_b = float(series["B"].iloc[i - 3])
    values = {"A": QUANTITIES["A"] * float(series["A"].iloc[i]), "B": QUANTITIES["B"] * last_known_b}
    total = sum(values.values())
    assert current == pytest.approx({t: v / total for t, v in values.items()}, abs=1e-9)


def test_long_gap_ending_at_the_split_date_is_rejected_naming_the_asset(db_session, world, split):
    _, _, market, series = world
    i = _pos(split)
    gapped = series["B"].copy()
    gapped.iloc[i - 5: i + 1] = np.nan  # 6 missing observations: beyond the 5-observation policy
    market.series = {**series, "B": gapped}

    with pytest.raises(BacktestError, match="'B'"):
        _backtest(db_session, world, split)


def test_holding_with_no_price_before_the_split_date_is_rejected_naming_the_asset(db_session, world, split):
    _, _, market, series = world
    late = series["B"].copy()
    late.iloc[: _pos(split) + 1] = np.nan  # B only starts trading after the split date
    market.series = {**series, "B": late}

    with pytest.raises(BacktestError, match="'B'.*split date"):
        _backtest(db_session, world, split)


# --- the helper in isolation ----------------------------------------------------------

def _frame(rows):
    return pd.DataFrame(rows, index=pd.bdate_range("2024-01-02", periods=len(next(iter(rows.values())))))


def test_helper_weights_are_quantity_times_last_price_normalized():
    from backend.backtest import holdings_weights_as_of

    prices = _frame({"A": [10.0, 11.0, 12.0], "B": [50.0, 50.0, 40.0]})
    weights = holdings_weights_as_of(prices, {"A": 5.0, "B": 2.0})

    assert weights == pytest.approx({"A": 60 / 140, "B": 80 / 140})
    assert sum(weights.values()) == pytest.approx(1.0)


def test_helper_works_with_a_single_row_of_history():
    from backend.backtest import holdings_weights_as_of

    weights = holdings_weights_as_of(_frame({"A": [10.0], "B": [30.0]}), {"A": 3.0, "B": 1.0})
    assert weights == pytest.approx({"A": 0.5, "B": 0.5})


def test_helper_gap_boundary_is_exactly_the_five_observation_policy():
    from backend.backtest import holdings_weights_as_of

    ok = _frame({"A": [10.0] * 7, "B": [20.0, 21.0] + [np.nan] * 5})      # 5 stale rows: allowed
    assert holdings_weights_as_of(ok, {"A": 1.0, "B": 1.0})["B"] == pytest.approx(21 / 31)

    too_stale = _frame({"A": [10.0] * 8, "B": [20.0, 21.0] + [np.nan] * 6})  # 6 stale rows: rejected
    with pytest.raises(BacktestError, match="'B'"):
        holdings_weights_as_of(too_stale, {"A": 1.0, "B": 1.0})


@pytest.mark.parametrize(
    "prices, holdings, message",
    [
        (pd.DataFrame(), {"A": 1.0}, "No price data"),
        (_frame({"A": [10.0, 11.0]}), {}, "No holdings"),
        (_frame({"A": [10.0, 11.0]}), {"A": 1.0, "ZZZ": 1.0}, "ZZZ"),
        (_frame({"A": [10.0, 11.0]}), {"A": 0.0}, "'A'"),
        (_frame({"A": [10.0, 11.0]}), {"A": -2.0}, "'A'"),
        (_frame({"A": [10.0, 0.0]}), {"A": 1.0}, "'A'"),
        (_frame({"A": [10.0, -5.0]}), {"A": 1.0}, "'A'"),
        (_frame({"A": [np.nan, np.nan]}), {"A": 1.0}, "'A'"),
    ],
)
def test_helper_rejects_values_that_cannot_be_priced(prices, holdings, message):
    from backend.backtest import holdings_weights_as_of

    with pytest.raises(BacktestError, match=message):
        holdings_weights_as_of(prices, holdings)


# --- API surface -----------------------------------------------------------------------

def test_api_backtest_with_unusable_split_dates_returns_400(client):
    created = client.post(
        "/portfolio",
        json={"holdings": [{"ticker": "AAPL", "quantity": 10}, {"ticker": "MSFT", "quantity": 5}], "lookback_years": 3.0},
    ).json()

    too_early = client.post(f"/portfolio/{created['id']}/backtest", json={"split_date": "2000-01-01"})
    too_late = client.post(
        f"/portfolio/{created['id']}/backtest", json={"split_date": (date.today() + timedelta(days=5)).isoformat()}
    )

    assert too_early.status_code == 400 and "split date" in too_early.json()["detail"]
    assert too_late.status_code == 400 and "split date" in too_late.json()["detail"]
