"""
Regression tests for partial / gappy historical market data (QA Issue #10).

Intended treatment of incomplete data (what these tests pin down):

1. Weekends and exchange holidays are simply ABSENT dates, not missing data.
   Nothing is flagged, and the return across a holiday is the true multi-day move.
2. A SHORT gap in one asset's prices (an isolated missing observation, a
   different holiday calendar, a late data print) is bridged by carrying the
   last observed price forward. Compounded wealth then stays exactly equal to
   P_last / P_first, and the other assets' real returns on those days are kept.
3. A LEADING gap (asset not yet trading / listed later) is legitimate partial
   history, not a provider failure. A fixed-weight portfolio return only
   exists once EVERY holding has a price, so analysis uses the common window.
4. A LONG interior/trailing gap, or an asset with fewer than 2 prices, is
   genuinely insufficient data and must raise a clear error naming the asset.
   It must never silently become NaN or silently drop days of returns.
5. Too few observations for a covariance/volatility estimate must raise
   rather than return NaN.

All data below is deterministic and hand-checkable; no network access.
"""

import numpy as np
import pandas as pd
import pytest

from backend import data as data_module
from backend.backtest import BacktestError, buy_and_hold_returns, run_backtest, train_test_split
from backend.data import MarketDataService
from backend.optimization import minimum_volatility
from backend.performance import (
    PerformanceError,
    cagr,
    cumulative_returns,
    daily_returns,
    portfolio_returns,
    sharpe_ratio,
    sortino_ratio,
)
from backend.risk import (
    RiskError,
    annualized_volatility,
    beta,
    covariance_matrix,
    current_drawdown,
    historical_cvar,
    historical_var,
    max_drawdown,
    portfolio_volatility,
)

NAN = np.nan


def _bdays(n: int, start: str = "2024-01-02") -> pd.DatetimeIndex:
    return pd.bdate_range(start, periods=n)


def _grow(n: int, rate: float, start: float = 100.0) -> np.ndarray:
    return start * (1 + rate) ** np.arange(n)


# ---------------------------------------------------------------------------
# daily_returns: weekends / holidays are NOT gaps
# ---------------------------------------------------------------------------


def test_daily_returns_complete_data_known_values():
    prices = pd.Series([100.0, 110.0, 121.0], index=_bdays(3))
    returns = daily_returns(prices)
    assert returns.tolist() == pytest.approx([0.10, 0.10])


def test_holiday_dates_absent_from_all_assets_are_not_treated_as_missing_data():
    # Mon 2024-01-15 (MLK day) is absent for every asset: no NaN, no error.
    dates = pd.to_datetime(["2024-01-12", "2024-01-16", "2024-01-17"])
    prices = pd.DataFrame({"A": [100.0, 102.0, 104.04], "B": [50.0, 50.5, 51.005]}, index=dates)

    returns = daily_returns(prices)

    assert returns.shape == (2, 2)
    assert not returns.isna().any().any()
    # Fri -> Tue is ONE observation: the true move across the long weekend.
    assert returns["A"].tolist() == pytest.approx([0.02, 0.02])
    assert returns["B"].tolist() == pytest.approx([0.01, 0.01])
    assert list(returns.index) == [pd.Timestamp("2024-01-16"), pd.Timestamp("2024-01-17")]


# ---------------------------------------------------------------------------
# daily_returns: isolated / short interior gaps are carried forward
# ---------------------------------------------------------------------------


def test_isolated_missing_price_preserves_total_compounded_return_single_series():
    prices = pd.Series([100.0, 110.0, NAN, 133.1], index=_bdays(4))

    returns = daily_returns(prices)

    assert returns.tolist() == pytest.approx([0.10, 0.0, 0.21])
    # Wealth must still end at P_last / P_first = 1.331 (old behaviour lost the gap move: 1.10).
    assert (1 + returns).prod() == pytest.approx(133.1 / 100.0)


def test_isolated_missing_price_in_one_asset_keeps_other_assets_real_returns():
    prices = pd.DataFrame(
        {
            "A": [100.0, 110.0, NAN, 133.1, 146.41],
            "B": [50.0, 51.0, 52.02, 53.0604, 54.121608],
        },
        index=_bdays(5),
    )

    returns = daily_returns(prices)

    assert returns.shape == (4, 2)
    assert not returns.isna().any().any()
    # B traded normally on A's gap day: its +2% on that day must NOT be discarded.
    assert returns["B"].tolist() == pytest.approx([0.02, 0.02, 0.02, 0.02])
    assert returns["A"].tolist() == pytest.approx([0.10, 0.0, 0.21, 0.10])
    assert (1 + returns["A"]).prod() == pytest.approx(146.41 / 100.0)


def test_series_input_gap_is_carried_forward_and_name_preserved():
    # e.g. a benchmark series with one missing print
    prices = pd.Series([100.0, NAN, 105.0], index=_bdays(3), name="^GSPC")
    returns = daily_returns(prices)
    assert isinstance(returns, pd.Series)
    assert returns.name == "^GSPC"
    assert returns.tolist() == pytest.approx([0.0, 0.05])


# ---------------------------------------------------------------------------
# daily_returns: leading gap = legitimately later start -> common window
# ---------------------------------------------------------------------------


def test_asset_that_starts_trading_later_limits_analysis_to_common_window():
    n = 8
    dates = _bdays(n)
    b = _grow(n, 0.02)
    b[:3] = NAN  # B has no price before dates[3]
    prices = pd.DataFrame({"A": _grow(n, 0.01), "B": b}, index=dates)

    returns = daily_returns(prices)

    # First return where BOTH assets have a return = B's 2nd price, dates[4].
    assert returns.index[0] == dates[4]
    assert len(returns) == n - 4
    assert not returns.isna().any().any()
    assert returns["A"].tolist() == pytest.approx([0.01] * (n - 4))
    assert returns["B"].tolist() == pytest.approx([0.02] * (n - 4))


# ---------------------------------------------------------------------------
# daily_returns: trailing gaps
# ---------------------------------------------------------------------------


def test_short_trailing_gap_is_carried_forward_as_zero_return():
    n = 6
    b = _grow(n, 0.02)
    b[-2:] = NAN  # B's last two prints missing (e.g. today's bar not yet published)
    prices = pd.DataFrame({"A": _grow(n, 0.01), "B": b}, index=_bdays(n))

    returns = daily_returns(prices)

    assert len(returns) == n - 1
    assert not returns.isna().any().any()
    assert returns["B"].iloc[-2:].tolist() == pytest.approx([0.0, 0.0])
    assert returns["A"].tolist() == pytest.approx([0.01] * (n - 1))


def test_long_trailing_gap_raises_naming_the_asset():
    n = 30
    b = _grow(n, 0.02)
    b[-10:] = NAN  # B's data ends 10 trading days early (e.g. delisted)
    prices = pd.DataFrame({"A": _grow(n, 0.01), "B": b}, index=_bdays(n))

    with pytest.raises(PerformanceError, match=r"'B'.*10 consecutive"):
        daily_returns(prices)


# ---------------------------------------------------------------------------
# daily_returns: gap-length boundary and configurability
# ---------------------------------------------------------------------------


def _frame_with_interior_gap(run_length: int, n: int = 16) -> pd.DataFrame:
    b = _grow(n, 0.02)
    b[3 : 3 + run_length] = NAN
    return pd.DataFrame({"A": _grow(n, 0.01), "B": b}, index=_bdays(n))


def test_interior_gap_of_exactly_five_observations_is_carried_forward():
    returns = daily_returns(_frame_with_interior_gap(5))
    assert not returns.isna().any().any()
    assert len(returns) == 15


def test_interior_gap_of_six_observations_raises():
    with pytest.raises(PerformanceError, match=r"'B'.*6 consecutive"):
        daily_returns(_frame_with_interior_gap(6))


def test_max_gap_parameter_is_respected():
    frame = _frame_with_interior_gap(2)
    with pytest.raises(PerformanceError, match="2 consecutive"):
        daily_returns(frame, max_gap=1)
    with pytest.raises(PerformanceError, match="1 consecutive"):
        daily_returns(_frame_with_interior_gap(1), max_gap=0)  # 0 = never carry forward
    assert not daily_returns(frame, max_gap=2).isna().any().any()


# ---------------------------------------------------------------------------
# daily_returns: insufficient data per asset
# ---------------------------------------------------------------------------


def test_asset_with_a_single_price_raises_naming_the_asset():
    prices = pd.DataFrame({"A": _grow(5, 0.01), "B": [NAN, NAN, NAN, NAN, 100.0]}, index=_bdays(5))
    with pytest.raises(PerformanceError, match=r"need >= 2.*'B'"):
        daily_returns(prices)


def test_asset_with_no_prices_in_window_raises_naming_the_asset():
    prices = pd.DataFrame({"A": _grow(5, 0.01), "B": [NAN] * 5}, index=_bdays(5))
    with pytest.raises(PerformanceError, match="'B'"):
        daily_returns(prices)


# ---------------------------------------------------------------------------
# End-to-end financial results on gappy data (daily_returns -> portfolio -> metrics)
# ---------------------------------------------------------------------------


def test_late_start_asset_cagr_is_annualized_over_the_actual_common_window():
    n, late = 300, 60
    b = _grow(n, 0.02)
    b[:late] = NAN
    prices = pd.DataFrame({"A": _grow(n, 0.01), "B": b}, index=_bdays(n))
    weights = {"A": 0.5, "B": 0.5}

    port = portfolio_returns(daily_returns(prices), weights)

    # Common window: returns from B's 2nd price onward = n - late - 1 observations,
    # each exactly 0.5*1% + 0.5*2% = 1.5%.
    assert len(port) == n - late - 1
    assert port.isna().sum() == 0
    assert port.tolist() == pytest.approx([0.015] * (n - late - 1))
    assert cumulative_returns(port).iloc[-1] == pytest.approx(1.015 ** (n - late - 1) - 1)
    # 239 days at 1.5% per day, annualized over those same 239 days = 1.015^252 - 1.
    assert cagr(port) == pytest.approx(1.015**252 - 1)


def test_metrics_on_gappy_data_equal_hand_computed_values_with_prices_carried_forward():
    n = 300
    rng = np.random.RandomState(11)
    dates = _bdays(n)
    prices = pd.DataFrame(
        {t: 100 * np.cumprod(1 + rng.normal(0.0005, 0.01 + i * 0.004, n)) for i, t in enumerate("ABC")},
        index=dates,
    )
    bench_prices = pd.Series(100 * np.cumprod(1 + rng.normal(0.0004, 0.012, n)), index=dates)
    weights = {"A": 0.5, "B": 0.3, "C": 0.2}

    gappy = prices.copy()
    gappy.iloc[100, 1] = NAN  # isolated missing observation in B
    gappy.iloc[200:203, 2] = NAN  # 3-day gap in C
    gappy.iloc[-2:, 0] = NAN  # A's last two prints not yet available

    # --- independent expectation: carry prices forward, then plain numpy ---
    filled = gappy.ffill().to_numpy()
    exp_assets = filled[1:] / filled[:-1] - 1
    w = np.array([weights[t] for t in "ABC"])
    exp_port = exp_assets @ w
    exp_bench = bench_prices.pct_change().dropna().to_numpy()
    wealth = np.cumprod(1 + exp_port)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1

    ar = daily_returns(gappy)
    port = portfolio_returns(ar, weights)
    bench = daily_returns(bench_prices)

    assert len(port) == n - 1
    assert port.to_numpy() == pytest.approx(exp_port)
    assert cumulative_returns(port).iloc[-1] == pytest.approx(wealth[-1] - 1)
    assert annualized_volatility(port) == pytest.approx(np.std(exp_port, ddof=1) * np.sqrt(252))
    assert sharpe_ratio(port) == pytest.approx(exp_port.mean() / np.std(exp_port, ddof=1) * np.sqrt(252))
    downside = exp_port[exp_port < 0]
    assert sortino_ratio(port) == pytest.approx(exp_port.mean() / np.sqrt((downside**2).mean()) * np.sqrt(252))
    assert max_drawdown(port) == pytest.approx(drawdown.min())
    assert current_drawdown(port) == pytest.approx(drawdown[-1])  # trailing gap must not make this NaN
    var95 = np.percentile(exp_port, 5)
    assert historical_var(port) == pytest.approx(var95)
    assert historical_cvar(port) == pytest.approx(exp_port[exp_port <= var95].mean())
    assert beta(port, bench) == pytest.approx(np.cov(exp_port, exp_bench)[0, 1] / np.var(exp_bench, ddof=1))
    assert portfolio_volatility(ar, weights) == pytest.approx(
        np.sqrt(w @ np.cov(exp_assets.T) @ w) * np.sqrt(252)
    )


def test_late_start_asset_risk_metrics_are_finite_and_use_only_common_window():
    n, late = 200, 50
    rng = np.random.RandomState(5)
    dates = _bdays(n)
    prices = pd.DataFrame(
        {t: 100 * np.cumprod(1 + rng.normal(0.0004, 0.01 + i * 0.004, n)) for i, t in enumerate("AB")},
        index=dates,
    )
    prices.iloc[:late, 1] = NAN
    weights = {"A": 0.6, "B": 0.4}

    port = portfolio_returns(daily_returns(prices), weights)

    common = prices.iloc[late:]  # B's first price onward: fully populated
    exp_assets = common.to_numpy()[1:] / common.to_numpy()[:-1] - 1
    exp_port = exp_assets @ np.array([0.6, 0.4])
    var95 = np.percentile(exp_port, 5)

    assert historical_var(port) == pytest.approx(var95)  # old code: NaN
    assert historical_cvar(port) == pytest.approx(exp_port[exp_port <= var95].mean())  # old code: raised
    assert max_drawdown(port) < 0


def test_portfolio_returns_refuses_missing_asset_returns_instead_of_propagating_nan():
    asset_returns = pd.DataFrame({"A": [0.01, 0.02, 0.03], "B": [NAN, 0.01, 0.02]})
    with pytest.raises(PerformanceError, match=r"missing.*'B'"):
        portfolio_returns(asset_returns, {"A": 0.5, "B": 0.5})


def test_optimizer_inputs_for_late_start_asset_use_the_same_window_for_every_asset():
    n, late = 260, 80
    rng = np.random.RandomState(21)
    dates = _bdays(n)
    prices = pd.DataFrame(
        {t: 100 * np.cumprod(1 + rng.normal(0.0004, 0.008 + i * 0.006, n)) for i, t in enumerate("ABC")},
        index=dates,
    )
    gappy = prices.copy()
    gappy.iloc[:late, 1] = NAN

    got = minimum_volatility(daily_returns(gappy))["weights"]
    # Expected: optimize on the fully-populated sub-window starting at B's first price.
    expected = minimum_volatility(daily_returns(prices.iloc[late:]))["weights"]

    for t in "ABC":
        assert got[t] == pytest.approx(expected[t], abs=1e-6)


# ---------------------------------------------------------------------------
# Insufficient observations: clear errors, never silent NaN
# ---------------------------------------------------------------------------


def test_single_price_observation_raises():
    with pytest.raises(PerformanceError, match="need >= 2"):
        daily_returns(pd.DataFrame({"A": [100.0], "B": [50.0]}, index=_bdays(1)))


def test_covariance_matrix_with_single_return_observation_raises_not_nan():
    one_row = pd.DataFrame({"A": [0.01], "B": [0.02]})
    with pytest.raises(RiskError, match="observations"):
        covariance_matrix(one_row)


def test_portfolio_volatility_with_single_return_observation_raises_not_nan():
    one_row = pd.DataFrame({"A": [0.01], "B": [0.02]})
    with pytest.raises(RiskError, match="observations"):
        portfolio_volatility(one_row, {"A": 0.5, "B": 0.5})


def test_optimizer_with_single_return_observation_fails_with_clear_message():
    one_row = pd.DataFrame({"A": [0.01], "B": [0.02]})
    with pytest.raises(RiskError, match="observations"):
        minimum_volatility(one_row)


def test_beta_uses_only_overlapping_dates():
    dates = _bdays(6)
    bench = pd.Series([0.01, -0.02, 0.03, 0.005], index=dates[[0, 1, 3, 5]])
    port = pd.Series([0.02, -0.04, 0.06, 0.01, 0.50], index=dates[[0, 1, 3, 5, 4]]).sort_index()
    # Portfolio = 2 x benchmark on the 4 shared dates; the extra date (idx 4) must be ignored.
    assert beta(port, bench) == pytest.approx(2.0)


def test_beta_with_no_overlapping_dates_raises():
    a = pd.Series([0.01, 0.02], index=_bdays(2))
    b = pd.Series([0.01, 0.02], index=_bdays(2, start="2025-03-03"))
    with pytest.raises(RiskError, match="no overlapping"):
        beta(a, b)


# ---------------------------------------------------------------------------
# Backtesting on gappy data
# ---------------------------------------------------------------------------


def test_buy_and_hold_isolated_gap_preserves_terminal_wealth():
    # 50/50: A has a missing print on day 2, B is flat. Value path: 1.0, 1.05, 1.05(carried), 1.105
    prices = pd.DataFrame(
        {"A": [100.0, 110.0, NAN, 121.0], "B": [100.0, 100.0, 100.0, 100.0]}, index=_bdays(4)
    )
    returns = buy_and_hold_returns(prices, {"A": 0.5, "B": 0.5})

    assert len(returns) == 3
    assert returns.iloc[0] == pytest.approx(0.05)
    assert returns.iloc[1] == pytest.approx(0.0)
    assert (1 + returns).prod() == pytest.approx(1.105)  # old code lost the gap: 1.05


def test_buy_and_hold_missing_start_price_raises_instead_of_returning_empty_series():
    prices = pd.DataFrame({"A": [100.0, 101.0, 102.0], "B": [NAN, 50.0, 51.0]}, index=_bdays(3))
    with pytest.raises(BacktestError, match="'B'"):
        buy_and_hold_returns(prices, {"A": 0.5, "B": 0.5})


def test_buy_and_hold_long_gap_raises():
    n = 20
    b = _grow(n, 0.01)
    b[5:15] = NAN
    prices = pd.DataFrame({"A": _grow(n, 0.01), "B": b}, index=_bdays(n))
    with pytest.raises(BacktestError, match=r"'B'.*10 consecutive"):
        buy_and_hold_returns(prices, {"A": 0.5, "B": 0.5})


def _random_prices(n=500, tickers="ABC", seed=3) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = _bdays(n, start="2022-01-03")
    return pd.DataFrame(
        {
            t: 100 * np.cumprod(1 + rng.normal(0.0003 + i * 0.0002, 0.01 + i * 0.005, n))
            for i, t in enumerate(tickers)
        },
        index=dates,
    )


def _benchmark(index, seed=4) -> pd.Series:
    return pd.Series(100 * np.cumprod(1 + np.random.RandomState(seed).normal(0.0004, 0.012, len(index))), index=index)


def test_run_backtest_gap_inside_test_window_does_not_change_terminal_results():
    prices = _random_prices()
    bench = _benchmark(prices.index)
    current = {"A": 0.5, "B": 0.3, "C": 0.2}
    split = prices.index[400]

    clean = run_backtest(prices, bench, split, current, transaction_cost_rate=0.0)

    gappy = prices.copy()
    gappy.iloc[420:424, 1] = NAN  # 4-day gap in B inside the test window
    gappy.iloc[450, 2] = NAN  # isolated missing print in C
    got = run_backtest(gappy, bench, split, current, transaction_cost_rate=0.0)

    # Carrying prices forward only changes the path inside the gap, never the terminal wealth.
    for name in ["current", "equal_weight", "min_volatility", "max_sharpe"]:
        assert (1 + got[name]["test_returns"]).prod() == pytest.approx(
            (1 + clean[name]["test_returns"]).prod(), rel=1e-9
        ), name
        assert got[name]["weights"] == clean[name]["weights"]


def test_run_backtest_asset_listed_during_training_window_is_optimized_on_common_history():
    prices = _random_prices()
    bench = _benchmark(prices.index)
    current = {"A": 0.5, "B": 0.3, "C": 0.2}
    split = prices.index[400]
    late = 150  # B first trades inside the training window

    gappy = prices.copy()
    gappy.iloc[:late, 1] = NAN
    got = run_backtest(gappy, bench, split, current, transaction_cost_rate=0.0)

    train, _ = train_test_split(prices, split)
    expected = minimum_volatility(daily_returns(train.iloc[late:]))["weights"]

    assert sum(got["min_volatility"]["weights"].values()) == pytest.approx(1.0)
    for t in "ABC":
        assert got["min_volatility"]["weights"][t] == pytest.approx(expected[t], abs=1e-6)
    for result in got.values():
        assert not result["test_returns"].isna().any()
        assert all(np.isfinite(v) for v in result["metrics"].values() if v == v)


def test_run_backtest_asset_that_starts_after_the_split_date_raises_clearly():
    prices = _random_prices()
    bench = _benchmark(prices.index)
    gappy = prices.copy()
    gappy.iloc[:420, 1] = NAN  # B has NO training data (split is at index 400)

    with pytest.raises(PerformanceError, match="'B'"):
        run_backtest(gappy, bench, prices.index[400], {"A": 0.5, "B": 0.3, "C": 0.2})


# ---------------------------------------------------------------------------
# Data layer: partial data is passed through faithfully (not rejected, not altered)
# ---------------------------------------------------------------------------


def _multi_frame(close_by_ticker: dict[str, list[float]], index) -> pd.DataFrame:
    fields = ["Open", "High", "Low", "Close", "Volume"]
    data = {}
    for t, close in close_by_ticker.items():
        c = pd.Series(close, index=index, dtype=float)
        data[(t, "Open")] = c
        data[(t, "High")] = c
        data[(t, "Low")] = c
        data[(t, "Close")] = c
        data[(t, "Volume")] = 1_000_000.0
    return pd.DataFrame(data, index=index, columns=pd.MultiIndex.from_product([list(close_by_ticker), fields]))


def test_data_layer_keeps_late_listed_ticker_with_leading_nan(monkeypatch):
    idx = _bdays(6)
    frame = _multi_frame({"AAA": [10, 11, 12, 13, 14, 15], "BBB": [NAN, NAN, 50, 51, 52, 53]}, idx)
    monkeypatch.setattr(data_module.yf, "download", lambda **kw: frame)

    prices = MarketDataService().get_price_data(["AAA", "BBB"], idx[0].date(), idx[-1].date())

    assert prices["BBB"].iloc[:2].isna().all()  # not rejected, not back-filled, not dropped
    assert prices["BBB"].iloc[2:].tolist() == [50, 51, 52, 53]
    assert prices["AAA"].tolist() == [10, 11, 12, 13, 14, 15]
    assert len(prices) == 6  # no rows silently removed


def test_data_layer_keeps_interior_gap_unfilled_and_analytics_carries_it_forward(monkeypatch):
    idx = _bdays(5)
    frame = _multi_frame({"AAA": [100, 110, NAN, 133.1, 146.41], "BBB": [50, 51, 52.02, 53.0604, 54.121608]}, idx)
    monkeypatch.setattr(data_module.yf, "download", lambda **kw: frame)

    prices = MarketDataService().get_price_data(["AAA", "BBB"], idx[0].date(), idx[-1].date())
    assert np.isnan(prices["AAA"].iloc[2])  # the data layer reports reality; it does not invent prices

    returns = daily_returns(prices)
    assert (1 + returns["AAA"]).prod() == pytest.approx(1.4641)


def test_data_layer_ticker_with_one_price_passes_but_analytics_rejects_it(monkeypatch):
    idx = _bdays(5)
    frame = _multi_frame({"AAA": [10, 11, 12, 13, 14], "BBB": [NAN, NAN, NAN, NAN, 50]}, idx)
    monkeypatch.setattr(data_module.yf, "download", lambda **kw: frame)

    prices = MarketDataService().get_price_data(["AAA", "BBB"], idx[0].date(), idx[-1].date())
    with pytest.raises(PerformanceError, match="'BBB'"):
        daily_returns(prices)


def test_data_layer_holiday_dates_are_not_nan_rows(monkeypatch):
    idx = pd.to_datetime(["2024-01-12", "2024-01-16", "2024-01-17"])  # MLK Monday absent
    frame = _multi_frame({"AAA": [100, 102, 104.04], "BBB": [50, 50.5, 51.005]}, idx)
    monkeypatch.setattr(data_module.yf, "download", lambda **kw: frame)

    prices = MarketDataService().get_price_data(["AAA", "BBB"], idx[0].date(), idx[-1].date())

    assert len(prices) == 3
    assert not prices.isna().any().any()
    assert daily_returns(prices)["AAA"].tolist() == pytest.approx([0.02, 0.02])


def test_latest_price_uses_last_valid_close_when_todays_bar_is_missing(monkeypatch):
    idx = _bdays(5)
    frame = _multi_frame({"AAA": [10, 11, 12, 13, 14], "BBB": [50, 51, 52, NAN, NAN]}, idx)
    monkeypatch.setattr(data_module.yf, "download", lambda **kw: frame)

    latest = MarketDataService().get_latest_prices(["AAA", "BBB"])

    assert latest["AAA"] == pytest.approx(14.0)
    assert latest["BBB"] == pytest.approx(52.0)  # last real close, not NaN
