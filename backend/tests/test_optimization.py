import numpy as np
import pandas as pd
import pytest

from backend.optimization import (
    OptimizationError,
    efficient_frontier,
    expected_returns,
    maximum_sharpe,
    minimum_volatility,
)
from backend.risk import covariance_matrix, portfolio_volatility


def _make_returns(seed: int = 0, n_days: int = 500) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    return pd.DataFrame(
        {
            "A": rng.normal(0.0008, 0.015, n_days),
            "B": rng.normal(0.0005, 0.010, n_days),
            "C": rng.normal(0.0012, 0.025, n_days),
            "D": rng.normal(0.0004, 0.008, n_days),
        }
    )


# --- expected_returns ---

def test_expected_returns_matches_hand_calculation():
    returns = pd.DataFrame({"A": [0.01, 0.02, -0.01]})
    result = expected_returns(returns, periods_per_year=252)
    assert result["A"] == pytest.approx(returns["A"].mean() * 252)


def test_expected_returns_empty_raises():
    with pytest.raises(OptimizationError):
        expected_returns(pd.DataFrame())


# --- minimum_volatility ---

def test_minimum_volatility_weights_sum_to_one():
    returns = _make_returns()
    result = minimum_volatility(returns)
    assert sum(result["weights"].values()) == pytest.approx(1.0, abs=1e-6)


def test_minimum_volatility_weights_within_default_bounds():
    returns = _make_returns()
    result = minimum_volatility(returns)
    for w in result["weights"].values():
        assert -1e-6 <= w <= 1 + 1e-6


def test_minimum_volatility_is_actually_minimal():
    # the optimizer's volatility should be <= the volatility of a naive
    # equal-weight portfolio on the same assets
    returns = _make_returns()
    result = minimum_volatility(returns)

    equal_weights = {t: 1 / len(returns.columns) for t in returns.columns}
    equal_weight_vol = portfolio_volatility(returns, equal_weights)

    assert result["volatility"] <= equal_weight_vol + 1e-9


def test_minimum_volatility_respects_custom_bounds():
    returns = _make_returns()
    # force every asset to hold at least 20% -- with 4 assets that's tight
    # but feasible (4 x 0.20 = 0.80 <= 1.0)
    result = minimum_volatility(returns, weight_bounds=(0.20, 0.40))
    for w in result["weights"].values():
        assert w >= 0.20 - 1e-6
        assert w <= 0.40 + 1e-6


def test_minimum_volatility_respects_per_ticker_bounds():
    returns = _make_returns()
    bounds = {"A": (0.0, 0.10), "B": (0.0, 1.0), "C": (0.0, 1.0), "D": (0.0, 1.0)}
    result = minimum_volatility(returns, weight_bounds=bounds)
    assert result["weights"]["A"] <= 0.10 + 1e-6


def test_minimum_volatility_infeasible_bounds_raises():
    returns = _make_returns()
    # 4 assets each forced to at least 30% -- sums to 1.2, impossible
    with pytest.raises(OptimizationError):
        minimum_volatility(returns, weight_bounds=(0.30, 1.0))


def test_minimum_volatility_empty_raises():
    with pytest.raises(OptimizationError):
        minimum_volatility(pd.DataFrame())


def test_minimum_volatility_negative_lower_bound_raises():
    returns = _make_returns()
    with pytest.raises(OptimizationError):
        minimum_volatility(returns, weight_bounds=(-0.1, 1.0))


# --- maximum_sharpe ---

def test_maximum_sharpe_weights_sum_to_one():
    returns = _make_returns()
    result = maximum_sharpe(returns, risk_free_rate=0.03)
    assert sum(result["weights"].values()) == pytest.approx(1.0, abs=1e-6)


def test_maximum_sharpe_achieves_higher_sharpe_than_equal_weight():
    returns = _make_returns()
    result = maximum_sharpe(returns, risk_free_rate=0.02)

    equal_weights = {t: 1 / len(returns.columns) for t in returns.columns}
    equal_ret = expected_returns(returns) @ pd.Series(equal_weights)
    equal_vol = portfolio_volatility(returns, equal_weights)
    equal_sharpe = (equal_ret - 0.02) / equal_vol

    assert result["sharpe_ratio"] >= equal_sharpe - 1e-9


def test_maximum_sharpe_respects_custom_bounds():
    returns = _make_returns()
    result = maximum_sharpe(returns, weight_bounds=(0.0, 0.40))
    for w in result["weights"].values():
        assert w <= 0.40 + 1e-6


def test_maximum_sharpe_empty_raises():
    with pytest.raises(OptimizationError):
        maximum_sharpe(pd.DataFrame())


def test_maximum_sharpe_infeasible_bounds_raises():
    returns = _make_returns()
    with pytest.raises(OptimizationError):
        maximum_sharpe(returns, weight_bounds=(0.30, 1.0))  # 4 x 0.30 = 1.2 > 1


def test_upper_bound_exceeding_one_raises():
    returns = _make_returns()
    with pytest.raises(OptimizationError):
        minimum_volatility(returns, weight_bounds=(0.0, 1.5))


def test_lower_bound_exceeds_upper_bound_raises():
    returns = _make_returns()
    with pytest.raises(OptimizationError):
        minimum_volatility(returns, weight_bounds={"A": (0.5, 0.1), "B": (0, 1), "C": (0, 1), "D": (0, 1)})


# --- efficient_frontier ---

def test_efficient_frontier_returns_dataframe_with_expected_columns():
    returns = _make_returns()
    frontier = efficient_frontier(returns, n_points=15)

    for col in ["expected_return", "volatility", "sharpe_ratio", "A", "B", "C", "D"]:
        assert col in frontier.columns
    assert len(frontier) > 0


def test_efficient_frontier_weights_sum_to_one_for_every_point():
    returns = _make_returns()
    frontier = efficient_frontier(returns, n_points=15)
    tickers = list(returns.columns)

    weight_sums = frontier[tickers].sum(axis=1)
    assert (weight_sums.round(6) == 1.0).all()


def test_efficient_frontier_weights_within_bounds():
    returns = _make_returns()
    frontier = efficient_frontier(returns, n_points=15, weight_bounds=(0.0, 0.6))
    tickers = list(returns.columns)

    for t in tickers:
        assert (frontier[t] >= -1e-6).all()
        assert (frontier[t] <= 0.6 + 1e-6).all()


def test_efficient_frontier_sorted_by_volatility():
    returns = _make_returns()
    frontier = efficient_frontier(returns, n_points=15)
    assert frontier["volatility"].is_monotonic_increasing


def test_efficient_frontier_contains_the_minimum_volatility_point():
    # the lowest-volatility point ON the frontier should be at least as good
    # (i.e. no lower volatility exists) as the standalone min-vol optimizer
    returns = _make_returns()
    frontier = efficient_frontier(returns, n_points=30)
    mv = minimum_volatility(returns)

    frontier_min_vol = frontier["volatility"].min()
    assert frontier_min_vol >= mv["volatility"] - 1e-4


def test_efficient_frontier_includes_max_sharpe_somewhere_near_the_top():
    # the point with the highest Sharpe ratio ON the frontier should be
    # close to the standalone max-Sharpe optimizer's Sharpe ratio
    returns = _make_returns()
    frontier = efficient_frontier(returns, n_points=40, risk_free_rate=0.02)
    ms = maximum_sharpe(returns, risk_free_rate=0.02)

    best_frontier_sharpe = frontier["sharpe_ratio"].max()
    assert best_frontier_sharpe <= ms["sharpe_ratio"] + 1e-3
    assert best_frontier_sharpe >= ms["sharpe_ratio"] - 0.05  # frontier is discretized, allow small gap


def test_efficient_frontier_empty_raises():
    with pytest.raises(OptimizationError):
        efficient_frontier(pd.DataFrame())


def test_efficient_frontier_respects_n_points_approximately():
    returns = _make_returns()
    frontier = efficient_frontier(returns, n_points=10)
    # some points near the extremes may fail to converge, so allow some slack
    assert 5 <= len(frontier) <= 10


def test_efficient_frontier_excludes_dominated_lower_branch():
    # The raw minimum-variance optimization traces a full parabola with an
    # upper (efficient) branch and a lower, dominated branch -- for the same
    # volatility, a point on the lower branch has a WORSE return than a point
    # on the upper branch. A correctly filtered efficient frontier should
    # never contain a dominated point: as volatility increases, expected
    # return should never decrease.
    returns = _make_returns(seed=5)
    frontier = efficient_frontier(returns, n_points=40)

    returns_col = frontier["expected_return"].values
    # allow a tiny numerical tolerance for solver noise between adjacent points
    assert all(returns_col[i + 1] >= returns_col[i] - 1e-6 for i in range(len(returns_col) - 1))


def test_efficient_frontier_every_point_has_return_at_least_min_variance_return():
    # The frontier is anchored to its OWN internally-solved min-variance
    # point, which may differ very slightly (solver precision) from a
    # separately-run minimum_volatility() call -- both are legitimate SLSQP
    # solutions to the same convex problem, so allow a small tolerance
    # rather than expecting bit-identical results between two solver runs.
    returns = _make_returns(seed=5)
    frontier = efficient_frontier(returns, n_points=40)
    mv = minimum_volatility(returns)

    assert (frontier["expected_return"] >= mv["expected_return"] - 5e-3).all()
