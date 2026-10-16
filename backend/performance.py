"""
Return and performance analytics.

Pure calculation logic operating on pandas Series/DataFrames of prices or
returns -- no yfinance or database dependency, so every function here is
testable with fixed, hand-checkable inputs.

Conventions used throughout this module:
- "returns" always means SIMPLE daily returns (R_t = P_t / P_(t-1) - 1),
  not log returns, unless a function name says otherwise.
- Annualization uses TRADING_DAYS_PER_YEAR = 252 by default, but every
  function accepts periods_per_year as a parameter so this is never
  silently hard-coded where it matters.
- A daily risk-free rate is derived by dividing the ANNUAL risk-free rate
  by periods_per_year. This is a simplification (it ignores compounding
  within the year) that is standard practice for daily Sharpe/Sortino
  calculations, but worth knowing if you ever need more precision.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252

# Longest run of consecutive missing price observations (for one asset) that is
# bridged by carrying the last observed price forward. Short gaps are normal
# (a different holiday calendar, one late/missed print); a longer run means the
# data is genuinely incomplete (delisting, provider outage) and is rejected
# rather than silently flat-lined.
MAX_FFILL_GAP = 5


class PerformanceError(Exception):
    """Raised when a performance calculation cannot be meaningfully computed."""


def _validate_periods_per_year(periods_per_year: int) -> None:
    """
    periods_per_year is used as a divisor (e.g. risk_free_rate / periods_per_year)
    in several functions below. A non-positive value isn't just semantically
    meaningless -- passed straight through, it previously caused a raw
    ZeroDivisionError instead of a clean, catchable PerformanceError. Every
    function that divides by periods_per_year calls this first.
    """
    if periods_per_year <= 0:
        raise PerformanceError(f"periods_per_year must be positive, got {periods_per_year}.")


def _first_excess_gap(is_missing: np.ndarray, first_valid: int, max_gap: int) -> tuple[int, int] | None:
    """(start position, length) of the first run of missing values after `first_valid` longer than max_gap."""
    n = len(is_missing)
    i = first_valid
    while i < n:
        if is_missing[i]:
            j = i
            while j < n and is_missing[j]:
                j += 1
            if j - i > max_gap:
                return i, j - i
            i = j
        else:
            i += 1
    return None


def fill_price_gaps(prices: pd.DataFrame | pd.Series, max_gap: int = MAX_FFILL_GAP) -> pd.DataFrame | pd.Series:
    """
    Bridge short gaps in price data by carrying the last observed price forward
    (last-observation-carried-forward, LOCF).

    - Weekends / exchange holidays are NOT gaps: those dates are simply absent
      from the index and are left alone.
    - A run of up to `max_gap` consecutive missing prices for an asset AFTER its
      first valid price is filled. The asset's return on those days is 0 and the
      next observed price realizes the whole move, so the compounded return is
      exactly P_last / P_first -- no part of the move is lost.
    - LEADING missing values (before the asset's first price) are left as NaN: the
      asset was not trading yet, and no price can be invented for it.
    - A longer run (interior or trailing) raises PerformanceError naming the asset,
      instead of silently flat-lining it or dropping days of returns.
    """
    if max_gap < 0:
        raise PerformanceError(f"max_gap cannot be negative, got {max_gap}.")

    is_series = isinstance(prices, pd.Series)
    frame = prices.to_frame() if is_series else prices

    for position, col in enumerate(frame.columns):
        missing = frame.iloc[:, position].isna().to_numpy()
        if missing.all():
            continue  # no data at all for this asset; the caller reports that
        first_valid = int(np.argmin(missing))
        gap = _first_excess_gap(missing, first_valid, max_gap)
        if gap is not None:
            start, length = gap
            when = frame.index[start]
            when = when.date() if hasattr(when, "date") else when
            name = prices.name if is_series and prices.name is not None else col
            raise PerformanceError(
                f"Price data for '{name}' has {length} consecutive missing observations starting {when}, "
                f"more than the {max_gap} that can be carried forward safely. The data looks incomplete "
                "(e.g. delisting or a provider outage), so returns cannot be computed reliably."
            )

    filled = frame.ffill(limit=max_gap) if max_gap > 0 else frame.copy()
    if is_series:
        result = filled.iloc[:, 0]
        result.name = prices.name
        return result
    return filled


def daily_returns(prices: pd.DataFrame, max_gap: int = MAX_FFILL_GAP) -> pd.DataFrame:
    """
    Compute simple daily returns from a price DataFrame/Series.

        R_t = P_t / P_(t-1) - 1

    The first row is dropped (there is no prior price to compare against).

    Handling of incomplete data (explicit, and independent of the pandas
    version -- pandas changed pct_change's default gap filling in 3.0):

    - Weekends/holidays are absent dates, not missing data; the return across
      them is the true multi-day move.
    - Gaps of up to `max_gap` consecutive missing prices are bridged by carrying
      the last price forward (see fill_price_gaps); longer gaps raise.
    - An asset that starts trading later (leading NaNs) is legitimate partial
      history. A fixed-weight portfolio return is only defined when EVERY asset
      has a return, so the result covers the common window: from the first date
      on which all assets have a return, to the end of the data.
    - An asset with fewer than 2 valid prices raises, naming the asset.
    """
    if prices.empty:
        raise PerformanceError("Cannot compute returns from empty price data.")

    is_series = isinstance(prices, pd.Series)
    frame = prices.to_frame() if is_series else prices

    sparse = [c for c in frame.columns if frame[c].notna().sum() < 2]
    if sparse:
        detail = "" if is_series else " for: " + ", ".join(f"'{c}'" for c in sparse)
        raise PerformanceError(
            f"Not enough price observations to compute returns (need >= 2){detail}."
        )

    filled = fill_price_gaps(frame, max_gap)
    returns = (filled / filled.shift(1) - 1.0).iloc[1:]
    returns = returns.dropna(how="any")  # common window across all assets

    if returns.empty:
        raise PerformanceError(
            "The assets share no common window with at least 2 consecutive price observations; "
            "cannot compute returns."
        )

    if is_series:
        result = returns.iloc[:, 0]
        result.name = prices.name
        return result
    return returns


def portfolio_returns(asset_returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """
    Combine per-asset returns into a single portfolio return series using
    FIXED weights:

        R_p = Sum(w_i * R_i)

    Note: this assumes weights are held constant over the period (i.e. the
    portfolio is rebalanced back to these weights every period). It does
    NOT model weight drift from a true buy-and-hold position -- that's a
    deliberate simplification for this milestone, revisited in backtesting.
    """
    missing = [t for t in weights if t not in asset_returns.columns]
    if missing:
        raise PerformanceError(f"No return data for weighted holding(s): {missing}")

    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise PerformanceError(f"Weights must sum to 1.0, got {weight_sum:.6f}")

    aligned = asset_returns[list(weights.keys())]

    # A fixed-weight portfolio return needs a return for EVERY holding on each date.
    # Refuse NaNs rather than let them silently propagate into NaN portfolio rows
    # (which would then distort CAGR's period count, VaR, drawdown, ...).
    has_gaps = [t for t in aligned.columns if aligned[t].isna().any()]
    if has_gaps:
        raise PerformanceError(
            "Return data has missing values for: "
            + ", ".join(f"'{t}'" for t in has_gaps)
            + ". Compute asset returns with daily_returns(), which bridges short price gaps and "
            "restricts to the common history window."
        )

    weight_vector = np.array([weights[t] for t in aligned.columns])
    return aligned.dot(weight_vector).rename("portfolio")


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """
    Compute the running cumulative return series:

        R_cumulative(t) = Product(1 + R_1...R_t) - 1

    Returns a series the same length as the input, where the value at each
    date is the total compounded return from the start through that date.
    """
    if returns.empty:
        raise PerformanceError("Cannot compute cumulative returns from empty series.")
    return (1 + returns).cumprod() - 1


def cagr(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Compound Annual Growth Rate, derived from a return series:

        CAGR = (V_T / V_0)^(1/Y) - 1

    where V_T / V_0 is the total compound growth factor implied by the
    return series, and Y is the number of years spanned (n_periods / periods_per_year).
    """
    if returns.empty:
        raise PerformanceError("Cannot compute CAGR from empty return series.")
    _validate_periods_per_year(periods_per_year)

    growth_factor = (1 + returns).prod()
    years = len(returns) / periods_per_year

    if growth_factor <= 0:
        raise PerformanceError(
            f"Cannot compute CAGR: cumulative growth factor is non-positive ({growth_factor:.4f}), "
            "implying a total loss greater than 100%."
        )

    return growth_factor ** (1 / years) - 1


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """
    Alias for CAGR, kept as a separate name because 'annualized return' and
    'CAGR' are used interchangeably in finance and the spec asks for both.
    """
    return cagr(returns, periods_per_year)


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualized Sharpe ratio:

        Sharpe = (R_p - R_f) / sigma_p

    Computed from daily data then annualized by sqrt(periods_per_year),
    which is the standard convention for converting a daily Sharpe ratio
    to an annual one (since return scales linearly with time but volatility
    scales with sqrt(time)).

    risk_free_rate is an ANNUAL rate (e.g. 0.04 for 4%); it is converted
    to a per-period rate internally.
    """
    if returns.empty:
        raise PerformanceError("Cannot compute Sharpe ratio from empty return series.")
    _validate_periods_per_year(periods_per_year)

    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf

    std = returns.std()
    if std == 0 or np.isnan(std):
        raise PerformanceError(
            "Cannot compute Sharpe ratio: return series has zero volatility."
        )

    daily_sharpe = excess_returns.mean() / std
    return daily_sharpe * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """
    Annualized Sortino ratio -- like Sharpe, but penalizes only DOWNSIDE
    volatility (returns below the risk-free rate), rather than total
    volatility. This better reflects investor risk aversion, since upside
    volatility isn't something investors want to "penalize" a strategy for.

        Sortino = (R_p - R_f) / downside_deviation
    """
    if returns.empty:
        raise PerformanceError("Cannot compute Sortino ratio from empty return series.")
    _validate_periods_per_year(periods_per_year)

    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf

    downside_returns = excess_returns[excess_returns < 0]
    if downside_returns.empty:
        raise PerformanceError(
            "Cannot compute Sortino ratio: no downside returns in this period "
            "(every period outperformed the risk-free rate)."
        )

    downside_deviation = np.sqrt((downside_returns ** 2).mean())
    if downside_deviation == 0:
        raise PerformanceError("Cannot compute Sortino ratio: downside deviation is zero.")

    daily_sortino = excess_returns.mean() / downside_deviation
    return daily_sortino * np.sqrt(periods_per_year)


def rolling_returns(returns: pd.Series, window: int) -> pd.Series:
    """
    Rolling cumulative return over a trailing window of `window` periods.
    E.g. window=21 gives an approximate rolling 1-month return series.
    """
    if window <= 0:
        raise PerformanceError(f"Rolling window must be positive, got {window}.")
    if len(returns) < window:
        raise PerformanceError(
            f"Not enough data ({len(returns)} periods) for a {window}-period rolling window."
        )

    return returns.rolling(window).apply(lambda r: (1 + r).prod() - 1, raw=True)


def rolling_sharpe(
    returns: pd.Series,
    window: int,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """
    Rolling annualized Sharpe ratio over a trailing window of `window` periods.
    Useful for seeing how risk-adjusted performance has changed over time,
    rather than a single number for the whole history.
    """
    if window <= 0:
        raise PerformanceError(f"Rolling window must be positive, got {window}.")
    if len(returns) < window:
        raise PerformanceError(
            f"Not enough data ({len(returns)} periods) for a {window}-period rolling window."
        )
    _validate_periods_per_year(periods_per_year)

    daily_rf = risk_free_rate / periods_per_year
    excess = returns - daily_rf

    rolling_mean = excess.rolling(window).mean()
    rolling_std = returns.rolling(window).std()

    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = (rolling_mean / rolling_std) * np.sqrt(periods_per_year)

    return sharpe.replace([np.inf, -np.inf], np.nan)


def compare_to_benchmark(
    portfolio_returns_series: pd.Series,
    benchmark_returns_series: pd.Series,
) -> pd.DataFrame:
    """
    Align portfolio and benchmark return series on shared dates and return
    a DataFrame of cumulative returns for both, for side-by-side plotting
    or comparison. Dates present in only one series are dropped, since a
    comparison needs matching timestamps.
    """
    aligned = pd.concat(
        [portfolio_returns_series.rename("portfolio"), benchmark_returns_series.rename("benchmark")],
        axis=1,
        join="inner",
    )

    if aligned.empty:
        raise PerformanceError("Portfolio and benchmark return series share no overlapping dates.")

    return pd.DataFrame(
        {
            "portfolio_cumulative": cumulative_returns(aligned["portfolio"]),
            "benchmark_cumulative": cumulative_returns(aligned["benchmark"]),
        }
    )
