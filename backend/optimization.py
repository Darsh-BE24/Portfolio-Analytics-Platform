"""
Portfolio optimization: minimum volatility and maximum Sharpe portfolios.

Uses scipy.optimize.minimize (SLSQP) to solve constrained quadratic/nonlinear
optimization problems over portfolio weights. Both optimizers share the same
constraint structure:

    Sum(w_i) = 1        (fully invested, no leverage/cash)
    lower_i <= w_i <= upper_i   (per-asset bounds, configurable)

Expected returns and covariance are estimated from HISTORICAL data (simple
mean and sample covariance, annualized). This is a standard but real
simplification -- historical mean returns are a notoriously noisy estimator
of *future* expected returns, which is why max-Sharpe portfolios often look
more concentrated/unstable than investors expect. Worth knowing, not hidden.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from backend.performance import TRADING_DAYS_PER_YEAR
from backend.risk import covariance_matrix


class OptimizationError(Exception):
    """Raised when a portfolio optimization problem cannot be solved or set up."""


WeightBounds = tuple[float, float] | dict[str, tuple[float, float]]


def expected_returns(asset_returns: pd.DataFrame, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """
    Annualized expected return per asset, estimated as the historical mean
    daily return scaled up by periods_per_year.

    This is the simplest possible estimator (no shrinkage, no CAPM, no
    factor model) -- appropriate for this milestone, but a known source of
    estimation error in mean-variance optimization generally.
    """
    if asset_returns.empty:
        raise OptimizationError("Cannot estimate expected returns from empty return data.")
    return asset_returns.mean() * periods_per_year


def _resolve_bounds(tickers: list[str], weight_bounds: WeightBounds) -> list[tuple[float, float]]:
    """
    Normalize weight_bounds into a per-ticker list of (min, max) tuples.
    Accepts either a single (min, max) applied to every asset, or a dict
    of per-ticker bounds (any ticker not in the dict falls back to (0, 1)).
    """
    if isinstance(weight_bounds, dict):
        resolved = [weight_bounds.get(t, (0.0, 1.0)) for t in tickers]
    else:
        resolved = [weight_bounds for _ in tickers]

    for ticker, (low, high) in zip(tickers, resolved):
        if low < 0:
            raise OptimizationError(f"Lower weight bound for '{ticker}' cannot be negative (got {low}); short selling is not supported.")
        if high > 1:
            raise OptimizationError(f"Upper weight bound for '{ticker}' cannot exceed 1.0 (got {high}).")
        if low > high:
            raise OptimizationError(f"Lower bound ({low}) exceeds upper bound ({high}) for '{ticker}'.")

    min_total = sum(b[0] for b in resolved)
    max_total = sum(b[1] for b in resolved)
    if min_total > 1.0:
        raise OptimizationError(
            f"Infeasible constraints: minimum weights sum to {min_total:.2f} > 1.0. "
            "Loosen the lower bounds."
        )
    if max_total < 1.0:
        raise OptimizationError(
            f"Infeasible constraints: maximum weights sum to {max_total:.2f} < 1.0. "
            "Loosen the upper bounds."
        )

    return resolved


def _portfolio_stats(weights: np.ndarray, mean_returns: pd.Series, cov: pd.DataFrame) -> tuple[float, float]:
    """Return (expected_return, volatility) for a given weight vector."""
    ret = float(weights @ mean_returns.values)
    variance = float(weights @ cov.values @ weights)
    vol = float(np.sqrt(max(variance, 0.0)))
    return ret, vol


def _build_result(weights: np.ndarray, tickers: list[str], mean_returns: pd.Series, cov: pd.DataFrame, risk_free_rate: float) -> dict:
    ret, vol = _portfolio_stats(weights, mean_returns, cov)
    sharpe = (ret - risk_free_rate) / vol if vol > 0 else float("nan")
    return {
        "weights": {t: float(w) for t, w in zip(tickers, weights)},
        "expected_return": ret,
        "volatility": vol,
        "sharpe_ratio": sharpe,
    }


def minimum_volatility(
    asset_returns: pd.DataFrame,
    weight_bounds: WeightBounds = (0.0, 1.0),
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """
    Solve for the portfolio with the lowest possible volatility:

        minimize   w^T Sigma w
        subject to Sum(w_i) = 1
                   lower_i <= w_i <= upper_i

    Returns a dict with the optimal weights, expected return, volatility,
    and Sharpe ratio (reported for context, not part of the objective).
    """
    if asset_returns.empty:
        raise OptimizationError("Cannot optimize an empty return dataset.")

    tickers = list(asset_returns.columns)
    n = len(tickers)
    bounds = _resolve_bounds(tickers, weight_bounds)

    mean_returns = expected_returns(asset_returns, periods_per_year)
    cov = covariance_matrix(asset_returns, annualize=True)

    initial_guess = np.repeat(1 / n, n)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    result = minimize(
        lambda w: w @ cov.values @ w,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise OptimizationError(f"Minimum volatility optimization failed to converge: {result.message}")

    return _build_result(result.x, tickers, mean_returns, cov, risk_free_rate)


def maximum_sharpe(
    asset_returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    weight_bounds: WeightBounds = (0.0, 1.0),
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """
    Solve for the portfolio that maximizes the Sharpe ratio:

        maximize   (w^T mu - R_f) / sqrt(w^T Sigma w)
        subject to Sum(w_i) = 1
                   lower_i <= w_i <= upper_i

    scipy.optimize only minimizes, so internally this minimizes the
    NEGATIVE Sharpe ratio, which is mathematically equivalent.
    """
    if asset_returns.empty:
        raise OptimizationError("Cannot optimize an empty return dataset.")

    tickers = list(asset_returns.columns)
    n = len(tickers)
    bounds = _resolve_bounds(tickers, weight_bounds)

    mean_returns = expected_returns(asset_returns, periods_per_year)
    cov = covariance_matrix(asset_returns, annualize=True)

    def negative_sharpe(w: np.ndarray) -> float:
        ret, vol = _portfolio_stats(w, mean_returns, cov)
        if vol == 0:
            return np.inf
        return -(ret - risk_free_rate) / vol

    initial_guess = np.repeat(1 / n, n)
    constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1},)

    result = minimize(
        negative_sharpe,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if not result.success:
        raise OptimizationError(f"Maximum Sharpe optimization failed to converge: {result.message}")

    return _build_result(result.x, tickers, mean_returns, cov, risk_free_rate)


def efficient_frontier(
    asset_returns: pd.DataFrame,
    risk_free_rate: float = 0.0,
    n_points: int = 50,
    weight_bounds: WeightBounds = (0.0, 1.0),
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.DataFrame:
    """
    Construct the efficient frontier: for a range of target returns spanning
    what the assets can achieve, solve for the MINIMUM-VOLATILITY portfolio
    that achieves each target return exactly:

        minimize   w^T Sigma w
        subject to Sum(w_i) = 1
                   w^T mu = target_return
                   lower_i <= w_i <= upper_i

    This is the standard mean-variance frontier construction -- solving an
    optimization problem at each target return -- rather than randomly
    sampling portfolios and hoping enough of them land near the frontier.
    Random sampling is noisier and never actually touches the true frontier;
    this approach gives exact points on it.

    The target-return range starts at the TRUE global minimum-variance
    portfolio's return (solved for directly, not approximated from a grid)
    and runs up to the highest individual asset's expected return. Starting
    exactly at the minimum-variance return -- rather than at the lowest
    asset return -- is what keeps this frontier free of the "dominated"
    lower branch of the mean-variance parabola (the half where a lower
    return is achieved for the same volatility, which no rational investor
    would ever choose).

    Returns a DataFrame with one row per achieved target return, sorted by
    volatility ascending, with columns: expected_return, volatility,
    sharpe_ratio, and one column per ticker holding that portfolio's weight.

    Some target returns near the top end may fail to converge (especially
    with tight weight_bounds) -- these are silently skipped rather than
    raising, since a partial frontier is still useful. An error is only
    raised if NO target return converges at all.
    """
    if asset_returns.empty:
        raise OptimizationError("Cannot build an efficient frontier from empty return data.")

    tickers = list(asset_returns.columns)
    n = len(tickers)
    bounds = _resolve_bounds(tickers, weight_bounds)

    mean_returns = expected_returns(asset_returns, periods_per_year)
    cov = covariance_matrix(asset_returns, annualize=True)
    initial_guess = np.repeat(1 / n, n)

    # First, solve for the TRUE global minimum-variance portfolio (no target
    # return constraint at all -- just fully invested and within bounds).
    # Its return anchors the low end of the target-return range below.
    # Anchoring here (rather than starting the range at mean_returns.min())
    # is what keeps this function from ever generating the dominated lower
    # branch of the mean-variance parabola in the first place -- no
    # post-hoc filtering needed, and no risk of it silently regressing.
    min_var_result = minimize(
        lambda w: w @ cov.values @ w,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=({"type": "eq", "fun": lambda w: np.sum(w) - 1},),
    )
    if not min_var_result.success:
        raise OptimizationError(
            f"Could not solve for the minimum-variance portfolio to anchor the frontier: {min_var_result.message}"
        )
    min_var_return = float(min_var_result.x @ mean_returns.values)

    target_returns = np.linspace(min_var_return, mean_returns.max(), n_points)

    records = []
    for target in target_returns:
        constraints = (
            {"type": "eq", "fun": lambda w: np.sum(w) - 1},
            {"type": "eq", "fun": lambda w, target=target: w @ mean_returns.values - target},
        )

        result = minimize(
            lambda w: w @ cov.values @ w,
            initial_guess,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
        )

        if not result.success:
            continue

        ret, vol = _portfolio_stats(result.x, mean_returns, cov)
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else float("nan")

        row = {"expected_return": ret, "volatility": vol, "sharpe_ratio": sharpe}
        row.update({t: float(w) for t, w in zip(tickers, result.x)})
        records.append(row)

    if not records:
        raise OptimizationError(
            "Could not construct an efficient frontier: no target return converged. "
            "This usually means weight_bounds are too restrictive."
        )

    frontier = pd.DataFrame(records).sort_values("volatility").reset_index(drop=True)
    return frontier
