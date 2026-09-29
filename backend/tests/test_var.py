import numpy as np
import pandas as pd
import pytest

from backend.risk import (
    RiskError,
    historical_cvar,
    historical_var,
    monte_carlo_var,
    parametric_cvar,
    parametric_var,
)


# --- historical VaR ---

def test_historical_var_known_percentile():
    # 100 evenly spaced returns from -0.10 to +0.09 -- 5th percentile is
    # easy to hand-check
    returns = pd.Series(np.linspace(-0.10, 0.09, 100))
    result = historical_var(returns, confidence=0.95)
    expected = float(np.percentile(returns, 5))
    assert result == pytest.approx(expected)


def test_historical_var_is_negative_for_typical_returns():
    np.random.seed(0)
    returns = pd.Series(np.random.normal(0.0005, 0.02, 500))
    result = historical_var(returns, confidence=0.95)
    assert result < 0


def test_historical_var_higher_confidence_is_more_extreme():
    np.random.seed(0)
    returns = pd.Series(np.random.normal(0.0005, 0.02, 500))
    var_95 = historical_var(returns, confidence=0.95)
    var_99 = historical_var(returns, confidence=0.99)
    assert var_99 < var_95  # 99% VaR should be a bigger (more negative) loss


def test_historical_var_invalid_confidence_raises():
    returns = pd.Series([0.01, -0.01, 0.02])
    with pytest.raises(RiskError):
        historical_var(returns, confidence=1.5)
    with pytest.raises(RiskError):
        historical_var(returns, confidence=0)


def test_historical_var_empty_raises():
    with pytest.raises(RiskError):
        historical_var(pd.Series(dtype=float))


def test_historical_var_horizon_scaling():
    returns = pd.Series(np.linspace(-0.10, 0.09, 100))
    one_day = historical_var(returns, confidence=0.95, horizon_days=1)
    ten_day = historical_var(returns, confidence=0.95, horizon_days=10)
    assert ten_day == pytest.approx(one_day * np.sqrt(10))


# --- parametric VaR ---

def test_parametric_var_known_value():
    from scipy.stats import norm

    returns = pd.Series([0.01, -0.02, 0.03, -0.01, 0.02, 0.00, -0.015])
    result = parametric_var(returns, confidence=0.95)

    mean, std = returns.mean(), returns.std()
    expected = mean + norm.ppf(0.05) * std
    assert result == pytest.approx(expected)


def test_parametric_var_invalid_confidence_raises():
    returns = pd.Series([0.01, -0.01, 0.02])
    with pytest.raises(RiskError):
        parametric_var(returns, confidence=-0.1)


# --- Monte Carlo VaR ---

def test_monte_carlo_var_reproducible_with_seed():
    returns = pd.Series(np.random.RandomState(1).normal(0.001, 0.02, 300))
    result1 = monte_carlo_var(returns, confidence=0.95, n_simulations=5000, random_seed=42)
    result2 = monte_carlo_var(returns, confidence=0.95, n_simulations=5000, random_seed=42)
    assert result1 == result2


def test_monte_carlo_var_close_to_parametric_var_with_many_simulations():
    # Monte Carlo VaR draws from a Normal(mean, std) fit to the data, so
    # with enough simulations it should converge close to parametric VaR
    # (which uses the same normal assumption analytically).
    returns = pd.Series(np.random.RandomState(2).normal(0.0005, 0.015, 500))
    mc = monte_carlo_var(returns, confidence=0.95, n_simulations=50_000, random_seed=1)
    param = parametric_var(returns, confidence=0.95)
    assert mc == pytest.approx(param, abs=0.002)


def test_monte_carlo_var_invalid_simulations_raises():
    returns = pd.Series([0.01, -0.01, 0.02])
    with pytest.raises(RiskError):
        monte_carlo_var(returns, n_simulations=0)


def test_monte_carlo_var_empty_raises():
    with pytest.raises(RiskError):
        monte_carlo_var(pd.Series(dtype=float))


# --- CVaR ---

def test_historical_cvar_is_more_extreme_than_var():
    np.random.seed(3)
    returns = pd.Series(np.random.normal(0.0005, 0.02, 500))
    var_95 = historical_var(returns, confidence=0.95)
    cvar_95 = historical_cvar(returns, confidence=0.95)
    # CVaR averages the tail beyond VaR, so it should be <= VaR (more negative)
    assert cvar_95 <= var_95


def test_historical_cvar_known_value():
    returns = pd.Series(np.linspace(-0.10, 0.09, 100))
    var_95 = float(np.percentile(returns, 5))
    expected_cvar = returns[returns <= var_95].mean()
    result = historical_cvar(returns, confidence=0.95)
    assert result == pytest.approx(expected_cvar)


def test_historical_cvar_empty_raises():
    with pytest.raises(RiskError):
        historical_cvar(pd.Series(dtype=float))


def test_parametric_cvar_more_extreme_than_parametric_var():
    returns = pd.Series(np.random.RandomState(4).normal(0.0005, 0.02, 500))
    var_95 = parametric_var(returns, confidence=0.95)
    cvar_95 = parametric_cvar(returns, confidence=0.95)
    assert cvar_95 <= var_95


def test_parametric_cvar_invalid_confidence_raises():
    returns = pd.Series([0.01, -0.01, 0.02])
    with pytest.raises(RiskError):
        parametric_cvar(returns, confidence=1.0)
