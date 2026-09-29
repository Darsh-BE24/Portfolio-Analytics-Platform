import numpy as np
import pandas as pd
import pytest

from backend.performance import (
    PerformanceError,
    cagr,
    compare_to_benchmark,
    cumulative_returns,
    daily_returns,
    portfolio_returns,
    rolling_returns,
    rolling_sharpe,
    sharpe_ratio,
    sortino_ratio,
)


# --- daily_returns ---

def test_daily_returns_known_values():
    prices = pd.Series([100, 110, 99])  # +10%, then -10%
    returns = daily_returns(prices)
    assert returns.iloc[0] == pytest.approx(0.10)
    assert returns.iloc[1] == pytest.approx(-0.10)


def test_daily_returns_empty_raises():
    with pytest.raises(PerformanceError):
        daily_returns(pd.Series(dtype=float))


def test_daily_returns_single_price_raises():
    with pytest.raises(PerformanceError):
        daily_returns(pd.Series([100.0]))


# --- portfolio_returns ---

def test_portfolio_returns_weighted_average():
    asset_returns = pd.DataFrame({"A": [0.10, 0.00], "B": [0.00, 0.10]})
    weights = {"A": 0.5, "B": 0.5}
    result = portfolio_returns(asset_returns, weights)
    assert result.iloc[0] == pytest.approx(0.05)
    assert result.iloc[1] == pytest.approx(0.05)


def test_portfolio_returns_weights_must_sum_to_one():
    asset_returns = pd.DataFrame({"A": [0.10], "B": [0.05]})
    with pytest.raises(PerformanceError):
        portfolio_returns(asset_returns, {"A": 0.5, "B": 0.6})


def test_portfolio_returns_missing_asset_raises():
    asset_returns = pd.DataFrame({"A": [0.10]})
    with pytest.raises(PerformanceError):
        portfolio_returns(asset_returns, {"A": 0.5, "B": 0.5})


# --- cumulative_returns ---

def test_cumulative_returns_known_values():
    returns = pd.Series([0.10, 0.10])  # 10% then 10%
    cum = cumulative_returns(returns)
    # (1.10 * 1.10) - 1 = 0.21
    assert cum.iloc[-1] == pytest.approx(0.21)


def test_cumulative_returns_empty_raises():
    with pytest.raises(PerformanceError):
        cumulative_returns(pd.Series(dtype=float))


# --- cagr ---

def test_cagr_known_value_one_year():
    # constant 0.01 daily return for exactly 252 days
    returns = pd.Series([0.01] * 252)
    result = cagr(returns, periods_per_year=252)
    expected = (1.01 ** 252) - 1
    assert result == pytest.approx(expected)


def test_cagr_two_years_annualizes_down():
    # constant daily return over 504 days (2 years) -- CAGR should be the
    # per-year growth rate, not the 2-year total growth rate
    returns = pd.Series([0.01] * 504)
    result = cagr(returns, periods_per_year=252)
    one_year_growth = 1.01 ** 252
    assert result == pytest.approx(one_year_growth - 1)


def test_cagr_total_loss_raises():
    returns = pd.Series([-1.0] * 10)  # -100% in one period -> zero value
    with pytest.raises(PerformanceError):
        cagr(returns)


def test_cagr_empty_raises():
    with pytest.raises(PerformanceError):
        cagr(pd.Series(dtype=float))


# --- sharpe_ratio ---

def test_sharpe_ratio_known_value():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.015])
    result = sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252)
    expected = (returns.mean() / returns.std()) * np.sqrt(252)
    assert result == pytest.approx(expected)


def test_sharpe_ratio_zero_volatility_raises():
    returns = pd.Series([0.01, 0.01, 0.01])
    with pytest.raises(PerformanceError):
        sharpe_ratio(returns)


def test_sharpe_ratio_empty_raises():
    with pytest.raises(PerformanceError):
        sharpe_ratio(pd.Series(dtype=float))


def test_sharpe_ratio_higher_return_gives_higher_sharpe():
    low_return = pd.Series([0.001, -0.001, 0.002, -0.002, 0.0015])
    high_return = low_return + 0.01  # shift every return up, same volatility
    assert sharpe_ratio(high_return) > sharpe_ratio(low_return)


# --- sortino_ratio ---

def test_sortino_ratio_known_value():
    returns = pd.Series([0.01, -0.01, 0.02, -0.02, 0.015])
    result = sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252)

    downside = returns[returns < 0]
    downside_dev = np.sqrt((downside ** 2).mean())
    expected = (returns.mean() / downside_dev) * np.sqrt(252)
    assert result == pytest.approx(expected)


def test_sortino_ratio_no_downside_raises():
    returns = pd.Series([0.01, 0.02, 0.03])  # all positive, no downside
    with pytest.raises(PerformanceError):
        sortino_ratio(returns)


def test_sortino_ignores_upside_volatility():
    # Sortino should not penalize a series for having large POSITIVE swings,
    # only negative ones -- this distinguishes it from Sharpe.
    steady = pd.Series([0.01, -0.005, 0.01, -0.005, 0.01])
    volatile_upside = pd.Series([0.01, -0.005, 0.05, -0.005, 0.08])  # bigger upside, same downside

    # Sharpe would likely be affected differently since total std differs,
    # but Sortino's downside deviation should be identical for both (same
    # negative values), so the ratio comparison should favor the higher-mean series.
    assert sortino_ratio(volatile_upside) > sortino_ratio(steady)


# --- rolling_returns / rolling_sharpe ---

def test_rolling_returns_length_and_window_validity():
    returns = pd.Series([0.01] * 10)
    result = rolling_returns(returns, window=3)
    assert len(result) == len(returns)
    # first 2 values are NaN (not enough data for the window yet)
    assert result.iloc[:2].isna().all()
    # 3rd value: compound of the first 3 returns
    assert result.iloc[2] == pytest.approx((1.01 ** 3) - 1)


def test_rolling_returns_window_larger_than_data_raises():
    returns = pd.Series([0.01, 0.02])
    with pytest.raises(PerformanceError):
        rolling_returns(returns, window=5)


def test_rolling_returns_invalid_window_raises():
    returns = pd.Series([0.01, 0.02, 0.03])
    with pytest.raises(PerformanceError):
        rolling_returns(returns, window=0)


def test_rolling_sharpe_produces_series_same_length():
    returns = pd.Series(np.random.RandomState(42).normal(0.001, 0.01, 100))
    result = rolling_sharpe(returns, window=20)
    assert len(result) == len(returns)


# --- compare_to_benchmark ---

def test_compare_to_benchmark_aligns_and_computes_cumulative():
    dates = pd.date_range("2024-01-01", periods=3)
    portfolio = pd.Series([0.01, 0.02, -0.01], index=dates)
    benchmark = pd.Series([0.005, 0.01, 0.0], index=dates)

    result = compare_to_benchmark(portfolio, benchmark)

    assert "portfolio_cumulative" in result.columns
    assert "benchmark_cumulative" in result.columns
    assert result["portfolio_cumulative"].iloc[-1] == pytest.approx(
        cumulative_returns(portfolio).iloc[-1]
    )


def test_compare_to_benchmark_no_overlap_raises():
    portfolio = pd.Series([0.01], index=pd.date_range("2024-01-01", periods=1))
    benchmark = pd.Series([0.01], index=pd.date_range("2025-01-01", periods=1))
    with pytest.raises(PerformanceError):
        compare_to_benchmark(portfolio, benchmark)
