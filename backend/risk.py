"""
Risk analytics: volatility, beta, drawdown, correlation, concentration.

Pure calculation logic on pandas Series/DataFrames of returns -- no
network or database dependency, fully unit testable with fixed inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.performance import TRADING_DAYS_PER_YEAR


class RiskError(Exception):
    """Raised when a risk calculation cannot be meaningfully computed."""


def covariance_matrix(asset_returns: pd.DataFrame, annualize: bool = True) -> pd.DataFrame:
    """
    Covariance matrix of asset returns:

        Sigma = Cov(R)

    Annualized by multiplying by periods_per_year, since daily covariance
    scales linearly with time (unlike volatility, which scales with sqrt(time)).
    """
    if asset_returns.empty:
        raise RiskError("Cannot compute covariance matrix from empty return data.")

    cov = asset_returns.cov()
    return cov * TRADING_DAYS_PER_YEAR if annualize else cov


def portfolio_volatility(
    asset_returns: pd.DataFrame,
    weights: dict[str, float],
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualized portfolio volatility from the covariance matrix:

        sigma_p = sqrt(w^T Sigma w)

    Uses DAILY covariance internally, then annualizes the final result by
    sqrt(periods_per_year) -- equivalent to annualizing the covariance
    matrix first, but avoids computing a full annualized matrix just to
    reduce it to a scalar.
    """
    missing = [t for t in weights if t not in asset_returns.columns]
    if missing:
        raise RiskError(f"No return data for weighted holding(s): {missing}")

    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise RiskError(f"Weights must sum to 1.0, got {weight_sum:.6f}")

    aligned = asset_returns[list(weights.keys())]
    w = np.array([weights[t] for t in aligned.columns])
    daily_cov = aligned.cov().values

    variance = w.T @ daily_cov @ w
    if variance < 0:
        raise RiskError("Computed negative variance -- covariance matrix is not valid (check input data).")

    return float(np.sqrt(variance) * np.sqrt(periods_per_year))


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Annualized volatility of a single return series (e.g. portfolio returns):

        sigma_annual = sigma_daily * sqrt(periods_per_year)
    """
    if returns.empty:
        raise RiskError("Cannot compute volatility from empty return series.")
    return float(returns.std() * np.sqrt(periods_per_year))


def rolling_volatility(returns: pd.Series, window: int, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """Rolling annualized volatility over a trailing window of `window` periods."""
    if window <= 0:
        raise RiskError(f"Rolling window must be positive, got {window}.")
    if len(returns) < window:
        raise RiskError(f"Not enough data ({len(returns)} periods) for a {window}-period rolling window.")

    return returns.rolling(window).std() * np.sqrt(periods_per_year)


def beta(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """
    Portfolio beta relative to a benchmark:

        beta_p = Cov(R_p, R_m) / Var(R_m)

    Measures how much the portfolio moves relative to the benchmark
    (beta > 1: more volatile than the benchmark; beta < 1: less volatile).
    Series are aligned on shared dates before computing.
    """
    aligned = pd.concat(
        [portfolio_returns.rename("portfolio"), benchmark_returns.rename("benchmark")],
        axis=1,
        join="inner",
    )
    if aligned.empty:
        raise RiskError("Portfolio and benchmark return series share no overlapping dates.")
    if len(aligned) < 2:
        raise RiskError("Need at least 2 overlapping observations to compute beta.")

    benchmark_variance = aligned["benchmark"].var()
    if benchmark_variance == 0:
        raise RiskError("Cannot compute beta: benchmark has zero variance.")

    covariance = aligned["portfolio"].cov(aligned["benchmark"])
    return float(covariance / benchmark_variance)


def wealth_index(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """Cumulative wealth from a starting value of `initial_value`, growing by each period's return."""
    if returns.empty:
        raise RiskError("Cannot compute wealth index from empty return series.")
    return initial_value * (1 + returns).cumprod()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """
    Running drawdown series:

        Peak_t = max(V_1, ..., V_t)
        Drawdown_t = (V_t - Peak_t) / Peak_t

    Always <= 0. A value of -0.20 means the portfolio is currently 20%
    below its highest-ever value.
    """
    wealth = wealth_index(returns)
    running_peak = wealth.cummax()
    return (wealth - running_peak) / running_peak


def current_drawdown(returns: pd.Series) -> float:
    """Drawdown as of the most recent period in the series."""
    return float(drawdown_series(returns).iloc[-1])


def max_drawdown(returns: pd.Series) -> float:
    """The single worst (most negative) drawdown over the whole period."""
    return float(drawdown_series(returns).min())


def drawdown_periods(returns: pd.Series) -> list[dict]:
    """
    Identify each distinct underwater period (a contiguous stretch where
    drawdown < 0), and report:
    - start: date the drawdown began (last date at a peak, i.e. dd == 0)
    - trough: date of the worst point in that period
    - end: date the portfolio recovered back to its prior peak (None if
      the period has not yet recovered by the end of the data)
    - depth: the maximum drawdown reached in that period
    - peak_to_trough_days / recovery_days: durations in calendar days
      (only meaningful if the index is date-like)
    """
    dd = drawdown_series(returns)
    is_underwater = dd < 0

    if not is_underwater.any():
        return []

    group_id = (is_underwater != is_underwater.shift()).cumsum()
    periods: list[dict] = []

    for _, group in dd.groupby(group_id):
        if not is_underwater.loc[group.index[0]]:
            continue

        start_date = group.index[0]
        trough_date = group.idxmin()
        trough_value = float(group.min())
        end_date = group.index[-1]
        recovered = dd.loc[end_date] == 0

        try:
            peak_to_trough_days = (trough_date - start_date).days
            recovery_days = (end_date - trough_date).days if recovered else None
        except TypeError:
            peak_to_trough_days = None
            recovery_days = None

        periods.append(
            {
                "start": start_date,
                "trough": trough_date,
                "end": end_date if recovered else None,
                "depth": trough_value,
                "peak_to_trough_days": peak_to_trough_days,
                "recovery_days": recovery_days,
            }
        )

    return periods


def correlation_matrix(asset_returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of asset returns."""
    if asset_returns.empty:
        raise RiskError("Cannot compute correlation matrix from empty return data.")
    return asset_returns.corr()


def concentration_metrics(weights: dict[str, float]) -> dict:
    """
    Portfolio concentration risk metrics:
    - largest_position: the single biggest weight
    - top_3_concentration: sum of the 3 largest weights (or fewer, if the
      portfolio has fewer than 3 holdings)
    - hhi: Herfindahl-Hirschman Index = Sum(w_i^2). Ranges from 1/N (perfectly
      diversified across N holdings) to 1 (fully concentrated in one holding).
    - effective_number_of_holdings: 1 / HHI. A rough measure of how many
      "equally-weighted" positions the portfolio's diversification is
      equivalent to -- e.g. an HHI implying 2.5 effective holdings means
      the portfolio is about as diversified as 2.5 equal-weighted positions,
      even if it technically holds 10 names.
    """
    if not weights:
        raise RiskError("Cannot compute concentration metrics for an empty portfolio.")

    weight_values = sorted(weights.values(), reverse=True)
    weight_sum = sum(weight_values)
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise RiskError(f"Weights must sum to 1.0, got {weight_sum:.6f}")

    hhi = sum(w ** 2 for w in weight_values)

    return {
        "largest_position": weight_values[0],
        "top_3_concentration": sum(weight_values[:3]),
        "hhi": hhi,
        "effective_number_of_holdings": 1 / hhi if hhi > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Value at Risk (VaR) and Conditional VaR (CVaR / Expected Shortfall)
#
# All VaR/CVaR functions here return a NEGATIVE decimal return -- e.g. a
# result of -0.024 means "a 2.4% loss at this confidence level / horizon",
# matching the convention that VaR represents a return threshold, not a
# dollar amount. Multiply by portfolio value separately if a dollar figure
# is needed.
# ---------------------------------------------------------------------------


def _validate_confidence(confidence: float) -> None:
    if not 0 < confidence < 1:
        raise RiskError(f"Confidence must be between 0 and 1 (exclusive), got {confidence}.")


def historical_var(returns: pd.Series, confidence: float = 0.95, horizon_days: int = 1) -> float:
    """
    Historical VaR: uses the empirical distribution of past returns directly,
    with no assumption about their shape (e.g. no normality assumption).

    The (1 - confidence) percentile of the historical return distribution
    is the VaR threshold -- e.g. at 95% confidence, this is the 5th
    percentile: the loss level exceeded only 5% of the time historically.

    horizon_days scales the 1-day VaR by sqrt(horizon_days), the standard
    (if approximate) square-root-of-time scaling rule -- it assumes returns
    are independent and identically distributed across days, which is a
    simplification worth knowing about, not a guarantee.
    """
    if returns.empty:
        raise RiskError("Cannot compute VaR from empty return series.")
    _validate_confidence(confidence)

    one_day_var = float(np.percentile(returns, (1 - confidence) * 100))
    return one_day_var * np.sqrt(horizon_days)


def parametric_var(returns: pd.Series, confidence: float = 0.95, horizon_days: int = 1) -> float:
    """
    Parametric (variance-covariance) VaR: assumes returns are normally
    distributed, and computes VaR from the mean and standard deviation
    of the historical sample rather than its empirical percentiles.

        VaR = mu + z * sigma

    where z is the standard normal quantile at (1 - confidence) -- a
    negative number (e.g. -1.645 at 95% confidence).
    """
    if returns.empty:
        raise RiskError("Cannot compute VaR from empty return series.")
    _validate_confidence(confidence)

    from scipy.stats import norm

    mean = returns.mean()
    std = returns.std()
    z = norm.ppf(1 - confidence)

    one_day_var = float(mean + z * std)
    return one_day_var * np.sqrt(horizon_days)


def monte_carlo_var(
    returns: pd.Series,
    confidence: float = 0.95,
    horizon_days: int = 1,
    n_simulations: int = 10_000,
    random_seed: int | None = None,
) -> float:
    """
    Monte Carlo VaR:
        1. Estimate the return distribution's mean and volatility from history.
        2. Generate `n_simulations` random draws from a Normal(mean, std)
           distribution -- simulated future daily returns.
        3. Compute the (1 - confidence) percentile of the simulated returns.

    This is a normal-distribution Monte Carlo simulation (not a bootstrap
    of historical returns), so its results will closely resemble parametric
    VaR when the sample is large, but the approach generalizes more easily
    to more complex distributional assumptions later if needed.

    random_seed makes results reproducible for testing; leave as None for
    genuinely random simulations in production use.
    """
    if returns.empty:
        raise RiskError("Cannot compute VaR from empty return series.")
    _validate_confidence(confidence)
    if n_simulations <= 0:
        raise RiskError(f"n_simulations must be positive, got {n_simulations}.")

    mean = returns.mean()
    std = returns.std()

    rng = np.random.default_rng(random_seed)
    simulated_returns = rng.normal(mean, std, n_simulations)

    one_day_var = float(np.percentile(simulated_returns, (1 - confidence) * 100))
    return one_day_var * np.sqrt(horizon_days)


def historical_cvar(returns: pd.Series, confidence: float = 0.95, horizon_days: int = 1) -> float:
    """
    Historical CVaR (Conditional VaR / Expected Shortfall): the AVERAGE
    loss among returns that fall at or beyond the historical VaR threshold.

    CVaR is always at least as extreme as VaR (more negative), since it
    answers "given that we're in the bad tail, how bad is it on average?"
    rather than just "where does the tail begin?"
    """
    if returns.empty:
        raise RiskError("Cannot compute CVaR from empty return series.")
    _validate_confidence(confidence)

    one_day_var = float(np.percentile(returns, (1 - confidence) * 100))
    tail_returns = returns[returns <= one_day_var]

    if tail_returns.empty:
        raise RiskError("No return observations fall within the VaR tail; cannot compute CVaR.")

    one_day_cvar = float(tail_returns.mean())
    return one_day_cvar * np.sqrt(horizon_days)


def parametric_cvar(returns: pd.Series, confidence: float = 0.95, horizon_days: int = 1) -> float:
    """
    Parametric CVaR under a normal-distribution assumption, using the
    closed-form expected shortfall formula for a normal distribution:

        CVaR = mu - sigma * phi(z) / (1 - confidence)

    where phi is the standard normal PDF and z is the standard normal
    quantile at (1 - confidence).
    """
    if returns.empty:
        raise RiskError("Cannot compute CVaR from empty return series.")
    _validate_confidence(confidence)

    from scipy.stats import norm

    mean = returns.mean()
    std = returns.std()
    z = norm.ppf(1 - confidence)

    one_day_cvar = float(mean - std * norm.pdf(z) / (1 - confidence))
    return one_day_cvar * np.sqrt(horizon_days)
