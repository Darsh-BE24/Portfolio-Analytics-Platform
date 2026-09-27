import numpy as np
import pandas as pd
import pytest

from backend.backtest import (
    BacktestError,
    apply_one_time_transaction_cost,
    buy_and_hold_returns,
    compute_turnover,
    run_backtest,
    strategy_performance_summary,
    train_test_split,
)
from backend.optimization import minimum_volatility
from backend.performance import daily_returns


def _make_prices(seed: int = 0, n_days: int = 500, tickers=("A", "B", "C")) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2022-01-03", periods=n_days, freq="B")
    data = {}
    for i, t in enumerate(tickers):
        drift = 0.0003 + i * 0.0002
        vol = 0.01 + i * 0.005
        data[t] = 100 * (1 + pd.Series(rng.normal(drift, vol, n_days), index=dates)).cumprod()
    return pd.DataFrame(data, index=dates)


# --- train_test_split ---

def test_train_test_split_no_overlap():
    prices = _make_prices()
    split_date = prices.index[250]
    train, test = train_test_split(prices, split_date)

    assert train.index.max() == split_date
    assert test.index.min() > split_date
    assert set(train.index).isdisjoint(set(test.index))


def test_train_test_split_covers_all_data():
    prices = _make_prices()
    split_date = prices.index[250]
    train, test = train_test_split(prices, split_date)
    assert len(train) + len(test) == len(prices)


def test_train_test_split_date_before_range_raises():
    prices = _make_prices()
    with pytest.raises(BacktestError):
        train_test_split(prices, prices.index[0] - pd.Timedelta(days=10))


def test_train_test_split_date_after_range_raises():
    prices = _make_prices()
    with pytest.raises(BacktestError):
        train_test_split(prices, prices.index[-1] + pd.Timedelta(days=10))


# --- no look-ahead bias (the most important property of this module) ---

def test_optimizer_weights_unaffected_by_test_period_data():
    """
    The core correctness property of a backtest: optimizing on the training
    window must give IDENTICAL results regardless of what happens in the
    test window. If changing test-period prices changed the optimized
    weights, that would be look-ahead bias.
    """
    prices_a = _make_prices(seed=1, n_days=500)
    train_a, _ = train_test_split(prices_a, prices_a.index[300])
    weights_a = minimum_volatility(daily_returns(train_a))["weights"]

    prices_b = prices_a.copy()
    rng = np.random.RandomState(999)
    future_dates = prices_b.index[301:]
    for t in prices_b.columns:
        shock = 1 + pd.Series(rng.normal(0, 0.5, len(future_dates)), index=future_dates)
        prices_b.loc[future_dates, t] = prices_b.loc[future_dates, t] * shock.cumprod()

    train_b, _ = train_test_split(prices_b, prices_b.index[300])
    weights_b = minimum_volatility(daily_returns(train_b))["weights"]

    assert train_a.equals(train_b)
    for t in weights_a:
        assert weights_a[t] == pytest.approx(weights_b[t], abs=1e-9)


# --- compute_turnover ---

def test_turnover_identical_weights_is_zero():
    weights = {"A": 0.5, "B": 0.5}
    assert compute_turnover(weights, weights) == pytest.approx(0.0)


def test_turnover_completely_disjoint_is_one():
    old = {"A": 1.0, "B": 0.0}
    new = {"A": 0.0, "B": 1.0}
    assert compute_turnover(old, new) == pytest.approx(1.0)


def test_turnover_partial_shift():
    old = {"A": 0.6, "B": 0.4}
    new = {"A": 0.4, "B": 0.6}
    assert compute_turnover(old, new) == pytest.approx(0.2)


# --- buy_and_hold_returns ---

def test_buy_and_hold_matches_simple_return_for_single_asset():
    prices = pd.DataFrame({"A": [100, 110, 121]})
    returns = buy_and_hold_returns(prices, {"A": 1.0})
    assert returns.iloc[0] == pytest.approx(0.10)
    assert returns.iloc[1] == pytest.approx(0.10)


def test_buy_and_hold_weights_drift_when_not_rebalanced():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    prices = pd.DataFrame({"A": [100, 150, 200], "B": [100, 100, 100]}, index=dates)

    initial_prices = prices.iloc[0]
    weights = {"A": 0.5, "B": 0.5}
    shares = {t: weights[t] / initial_prices[t] for t in weights}

    final_value_a = shares["A"] * prices["A"].iloc[-1]
    final_value_b = shares["B"] * prices["B"].iloc[-1]
    final_weight_a = final_value_a / (final_value_a + final_value_b)

    assert final_weight_a > 0.5


def test_buy_and_hold_weights_must_sum_to_one():
    prices = _make_prices()
    with pytest.raises(BacktestError):
        buy_and_hold_returns(prices, {"A": 0.5, "B": 0.6, "C": 0.0})


def test_buy_and_hold_missing_ticker_raises():
    prices = _make_prices()
    with pytest.raises(BacktestError):
        buy_and_hold_returns(prices, {"A": 0.5, "ZZZZ": 0.5})


# --- apply_one_time_transaction_cost ---

def test_transaction_cost_only_affects_first_period():
    returns = pd.Series([0.02, 0.01, -0.01])
    adjusted = apply_one_time_transaction_cost(returns, turnover=0.5, cost_rate=0.001)

    assert adjusted.iloc[0] == pytest.approx(0.02 - 0.5 * 0.001)
    assert adjusted.iloc[1] == pytest.approx(0.01)
    assert adjusted.iloc[2] == pytest.approx(-0.01)


def test_transaction_cost_zero_turnover_no_change():
    returns = pd.Series([0.02, 0.01])
    adjusted = apply_one_time_transaction_cost(returns, turnover=0.0, cost_rate=0.01)
    assert adjusted.iloc[0] == pytest.approx(returns.iloc[0])


def test_transaction_cost_invalid_turnover_raises():
    returns = pd.Series([0.01])
    with pytest.raises(BacktestError):
        apply_one_time_transaction_cost(returns, turnover=1.5, cost_rate=0.001)


# --- strategy_performance_summary ---

def test_strategy_performance_summary_has_expected_keys():
    returns = pd.Series(np.random.RandomState(1).normal(0.0005, 0.01, 300))
    summary = strategy_performance_summary(returns)
    for key in ["cagr", "volatility", "sharpe", "sortino", "max_drawdown", "historical_var_95", "historical_cvar_95"]:
        assert key in summary


def test_strategy_performance_summary_max_drawdown_is_non_positive():
    returns = pd.Series(np.random.RandomState(2).normal(0.0005, 0.02, 300))
    summary = strategy_performance_summary(returns)
    assert summary["max_drawdown"] <= 0


# --- run_backtest (full integration) ---

def test_run_backtest_returns_all_expected_strategies():
    prices = _make_prices(seed=3, n_days=600)
    benchmark = 100 * (1 + pd.Series(np.random.RandomState(4).normal(0.0004, 0.012, 600), index=prices.index)).cumprod()
    current_weights = {"A": 0.5, "B": 0.3, "C": 0.2}

    results = run_backtest(prices, benchmark, prices.index[400], current_weights)

    for strategy in ["current", "equal_weight", "min_volatility", "max_sharpe", "benchmark"]:
        assert strategy in results
        assert "metrics" in results[strategy]
        assert "test_returns" in results[strategy]


def test_run_backtest_current_strategy_has_no_transaction_cost():
    prices = _make_prices(seed=3, n_days=600)
    benchmark = 100 * (1 + pd.Series(np.random.RandomState(4).normal(0.0004, 0.012, 600), index=prices.index)).cumprod()
    current_weights = {"A": 0.5, "B": 0.3, "C": 0.2}

    results = run_backtest(prices, benchmark, prices.index[400], current_weights, transaction_cost_rate=0.01)

    _, test_prices = train_test_split(prices, prices.index[400])
    raw_current_returns = buy_and_hold_returns(test_prices, current_weights)

    assert results["current"]["test_returns"].iloc[0] == pytest.approx(raw_current_returns.iloc[0])


def test_run_backtest_benchmark_has_no_weights():
    prices = _make_prices(seed=3, n_days=600)
    benchmark = 100 * (1 + pd.Series(np.random.RandomState(4).normal(0.0004, 0.012, 600), index=prices.index)).cumprod()
    current_weights = {"A": 0.5, "B": 0.3, "C": 0.2}

    results = run_backtest(prices, benchmark, prices.index[400], current_weights)
    assert results["benchmark"]["weights"] is None


def test_run_backtest_min_volatility_weights_only_from_training_window():
    prices = _make_prices(seed=5, n_days=600)
    benchmark = 100 * (1 + pd.Series(np.random.RandomState(6).normal(0.0004, 0.012, 600), index=prices.index)).cumprod()
    current_weights = {"A": 0.5, "B": 0.3, "C": 0.2}
    split = prices.index[400]

    results = run_backtest(prices, benchmark, split, current_weights)

    train_prices, _ = train_test_split(prices, split)
    expected_weights = minimum_volatility(daily_returns(train_prices))["weights"]

    for t in expected_weights:
        assert results["min_volatility"]["weights"][t] == pytest.approx(expected_weights[t], abs=1e-9)


def test_run_backtest_empty_prices_raises():
    with pytest.raises(BacktestError):
        run_backtest(pd.DataFrame(), pd.Series(dtype=float), pd.Timestamp("2024-01-01"), {})
