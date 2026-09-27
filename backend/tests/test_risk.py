import numpy as np
import pandas as pd
import pytest

from backend.risk import (
    RiskError,
    beta,
    concentration_metrics,
    correlation_matrix,
    covariance_matrix,
    current_drawdown,
    drawdown_periods,
    drawdown_series,
    max_drawdown,
    portfolio_volatility,
    rolling_volatility,
    wealth_index,
    annualized_volatility,
)


# --- covariance / volatility ---

def test_covariance_matrix_diagonal_matches_variance():
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.03, 0.0], "B": [0.02, 0.01, -0.01, 0.03]})
    cov = covariance_matrix(returns, annualize=False)
    assert cov.loc["A", "A"] == pytest.approx(returns["A"].var())
    assert cov.loc["B", "B"] == pytest.approx(returns["B"].var())


def test_covariance_matrix_empty_raises():
    with pytest.raises(RiskError):
        covariance_matrix(pd.DataFrame())


def test_portfolio_volatility_single_asset_matches_annualized_std():
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.03, -0.01, 0.02]})
    result = portfolio_volatility(returns, {"A": 1.0}, periods_per_year=252)
    expected = returns["A"].std() * np.sqrt(252)
    assert result == pytest.approx(expected)


def test_portfolio_volatility_diversification_reduces_risk():
    # two perfectly anti-correlated assets with equal volatility --
    # an equal-weight combination should have LOWER volatility than either alone
    np.random.seed(1)
    a = pd.Series(np.random.normal(0, 0.02, 200))
    b = -a  # perfectly anti-correlated, same magnitude
    returns = pd.DataFrame({"A": a, "B": b})

    vol_a_alone = portfolio_volatility(returns, {"A": 1.0, "B": 0.0})
    vol_combined = portfolio_volatility(returns, {"A": 0.5, "B": 0.5})

    assert vol_combined < vol_a_alone


def test_portfolio_volatility_weights_must_sum_to_one():
    returns = pd.DataFrame({"A": [0.01, 0.02], "B": [0.01, -0.01]})
    with pytest.raises(RiskError):
        portfolio_volatility(returns, {"A": 0.5, "B": 0.6})


def test_annualized_volatility_known_value():
    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    result = annualized_volatility(returns, periods_per_year=252)
    assert result == pytest.approx(returns.std() * np.sqrt(252))


def test_rolling_volatility_window_too_large_raises():
    returns = pd.Series([0.01, 0.02])
    with pytest.raises(RiskError):
        rolling_volatility(returns, window=5)


# --- beta ---

def test_beta_matches_benchmark_exactly():
    # a portfolio that IS the benchmark should have beta == 1
    benchmark = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    result = beta(benchmark, benchmark)
    assert result == pytest.approx(1.0)


def test_beta_double_benchmark_is_two():
    benchmark = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02])
    portfolio = benchmark * 2
    result = beta(portfolio, benchmark)
    assert result == pytest.approx(2.0)


def test_beta_zero_benchmark_variance_raises():
    benchmark = pd.Series([0.01, 0.01, 0.01])
    portfolio = pd.Series([0.02, -0.01, 0.03])
    with pytest.raises(RiskError):
        beta(portfolio, benchmark)


def test_beta_no_overlap_raises():
    portfolio = pd.Series([0.01], index=pd.date_range("2024-01-01", periods=1))
    benchmark = pd.Series([0.01], index=pd.date_range("2025-01-01", periods=1))
    with pytest.raises(RiskError):
        beta(portfolio, benchmark)


# --- drawdown ---

def test_wealth_index_known_values():
    returns = pd.Series([0.10, -0.10])
    wealth = wealth_index(returns, initial_value=100)
    assert wealth.iloc[0] == pytest.approx(110.0)
    assert wealth.iloc[1] == pytest.approx(99.0)


def test_drawdown_series_simple_case():
    # up 10%, then down enough to be 10% below the new peak
    returns = pd.Series([0.10, -0.10])
    dd = drawdown_series(returns)
    # peak after day 1 = 1.10; day 2 wealth = 1.10 * 0.90 = 0.99
    # drawdown = (0.99 - 1.10) / 1.10
    expected_day2 = (0.99 - 1.10) / 1.10
    assert dd.iloc[1] == pytest.approx(expected_day2)


def test_max_drawdown_never_positive():
    returns = pd.Series([0.05, 0.03, -0.10, 0.02, -0.15, 0.20])
    result = max_drawdown(returns)
    assert result <= 0


def test_current_drawdown_zero_at_new_high():
    # monotonically increasing returns -- always at a new peak, so
    # current drawdown should be exactly 0
    returns = pd.Series([0.01, 0.02, 0.01, 0.03])
    result = current_drawdown(returns)
    assert result == pytest.approx(0.0)


def test_drawdown_periods_identifies_recovery():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    # up, down, down, up past old peak (recovered), flat
    returns = pd.Series([0.10, -0.05, -0.05, 0.15, 0.0, 0.0], index=dates)
    periods = drawdown_periods(returns)

    assert len(periods) >= 1
    first = periods[0]
    assert first["depth"] < 0
    assert first["start"] is not None
    assert first["trough"] is not None


def test_drawdown_periods_unrecovered_has_none_end():
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    returns = pd.Series([0.10, -0.20, -0.05], index=dates)  # never recovers
    periods = drawdown_periods(returns)

    assert len(periods) == 1
    assert periods[0]["end"] is None
    assert periods[0]["recovery_days"] is None


def test_drawdown_periods_no_drawdown_returns_empty():
    returns = pd.Series([0.01, 0.02, 0.03])  # monotonically up, never underwater
    assert drawdown_periods(returns) == []


# --- correlation ---

def test_correlation_matrix_diagonal_is_one():
    returns = pd.DataFrame({"A": [0.01, -0.02, 0.03], "B": [0.02, 0.01, -0.01]})
    corr = correlation_matrix(returns)
    assert corr.loc["A", "A"] == pytest.approx(1.0)
    assert corr.loc["B", "B"] == pytest.approx(1.0)


def test_correlation_matrix_perfectly_correlated_assets():
    a = pd.Series([0.01, -0.02, 0.03, 0.01])
    returns = pd.DataFrame({"A": a, "B": a * 2})  # B is a scaled copy of A
    corr = correlation_matrix(returns)
    assert corr.loc["A", "B"] == pytest.approx(1.0)


# --- concentration ---

def test_concentration_equal_weights_four_holdings():
    weights = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
    result = concentration_metrics(weights)
    assert result["hhi"] == pytest.approx(4 * 0.25 ** 2)
    assert result["effective_number_of_holdings"] == pytest.approx(4.0)
    assert result["largest_position"] == pytest.approx(0.25)
    assert result["top_3_concentration"] == pytest.approx(0.75)


def test_concentration_single_holding_is_fully_concentrated():
    weights = {"A": 1.0}
    result = concentration_metrics(weights)
    assert result["hhi"] == pytest.approx(1.0)
    assert result["effective_number_of_holdings"] == pytest.approx(1.0)


def test_concentration_weights_must_sum_to_one():
    with pytest.raises(RiskError):
        concentration_metrics({"A": 0.5, "B": 0.6})


def test_concentration_empty_raises():
    with pytest.raises(RiskError):
        concentration_metrics({})
