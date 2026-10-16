"""
Backtesting: train/test split, out-of-sample strategy comparison, and
transaction costs.

The core discipline this module enforces: the optimizer NEVER sees test-
period data. Expected returns and covariance are estimated exclusively from
the training window, weights are optimized once on that window, then frozen
and carried forward unchanged into the test window. This is what "no
look-ahead bias" means in practice -- it's enforced structurally here (the
optimizer functions are only ever given the training DataFrame slice), not
just asserted in a docstring.

Strategies are simulated as BUY-AND-HOLD over the test period (weights are
frozen, not rebalanced daily), so their actual weights drift naturally as
underlying asset prices diverge -- this is more realistic than the fixed
continuously-rebalanced weights used in performance.portfolio_returns(),
which is a different (and also valid, but distinct) assumption used
elsewhere in this project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.optimization import maximum_sharpe, minimum_volatility, WeightBounds
from backend.performance import (
    MAX_FFILL_GAP,
    PerformanceError,
    cagr,
    daily_returns,
    fill_price_gaps,
    sharpe_ratio,
    sortino_ratio,
)
from backend.risk import historical_cvar, historical_var, max_drawdown


class BacktestError(Exception):
    """Raised when a backtest cannot be constructed or run."""


def train_test_split(prices: pd.DataFrame, split_date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a price DataFrame into a training window (everything up to and
    including split_date) and a test window (everything strictly after).

    The strict inequality on the test side is deliberate: split_date itself
    belongs only to training, so there is no single day counted in both
    windows, which would otherwise leak one day of test information into
    the optimizer via that shared boundary point.
    """
    split_date = pd.Timestamp(split_date)

    train = prices[prices.index <= split_date]
    test = prices[prices.index > split_date]

    if train.empty:
        raise BacktestError(f"No price data on or before split date {split_date.date()}.")
    if test.empty:
        raise BacktestError(f"No price data after split date {split_date.date()}.")

    return train, test


def compute_turnover(old_weights: dict[str, float], new_weights: dict[str, float]) -> float:
    """
    Portfolio turnover between two weight allocations:

        Turnover = 0.5 * Sum(|w_new_i - w_old_i|)

    The 0.5 factor is standard convention -- it avoids double-counting a
    trade (selling out of one asset AND buying into another both contribute
    to the raw sum, but represent one "unit" of rebalancing activity).
    Turnover ranges from 0 (identical allocations) to 1 (completely
    non-overlapping allocations).
    """
    tickers = set(old_weights) | set(new_weights)
    diffs = [abs(new_weights.get(t, 0.0) - old_weights.get(t, 0.0)) for t in tickers]
    return 0.5 * sum(diffs)


def holdings_weights_as_of(prices_to_date: pd.DataFrame, holdings: dict[str, float]) -> dict[str, float]:
    """
    Weights of a set of SHARE holdings, valued with the last prices observed in
    `prices_to_date` -- i.e. as of the end of that window (for a backtest, the split date).

        w_i = q_i * P_i(split) / Sum_j( q_j * P_j(split) )

    Why this exists (Issue #3): the "current" strategy asks "what if I had held my
    current positions through the test period?". The share counts q_i are the only
    thing the data model knows about the user's holdings (there is no transaction
    history), so they are the input. But the WEIGHTS must be derived from prices
    that were knowable when the test period begins. Valuing the same shares at
    TODAY's price (the end of the test window) bakes the test period's own returns
    into the starting allocation: a stock that tripled during the test gets a
    tripled weight at the start, so the strategy "buys more of whatever won" and
    its backtest return is biased upward. Pricing at the split date gives the
    "current" strategy the same information set the optimized strategies get
    (everything up to and including the split date), and it is also the correct
    "from" allocation for the turnover charged when switching into them.

    The last observed price is used only if it is at most MAX_FFILL_GAP rows old
    (the same short-gap policy as Issue #10). A holding with no price, a stale
    price, or a non-positive price as of the split date cannot be valued, and
    raises BacktestError naming it -- rather than being silently skipped (which
    would shift every other weight) or priced with a months-old quote.
    """
    if not holdings:
        raise BacktestError("No holdings to value as of the split date.")
    if prices_to_date.empty:
        raise BacktestError("No price data on or before the split date to value the holdings with.")

    missing = [t for t in holdings if t not in prices_to_date.columns]
    if missing:
        raise BacktestError(f"No price data for holding(s): {missing}")

    values: dict[str, float] = {}
    for ticker, quantity in holdings.items():
        if not np.isfinite(quantity) or quantity <= 0:
            raise BacktestError(f"Quantity for '{ticker}' must be positive, got {quantity}.")

        series = prices_to_date[ticker]
        last_valid = series.last_valid_index()
        if last_valid is None:
            raise BacktestError(
                f"No price for '{ticker}' on or before the split date; cannot value the holding as of the split date."
            )

        stale_rows = len(series) - 1 - int(series.index.get_loc(last_valid))
        if stale_rows > MAX_FFILL_GAP:
            raise BacktestError(
                f"The last price for '{ticker}' is {stale_rows} observations before the split date, more than the "
                f"{MAX_FFILL_GAP} that can be carried forward safely; cannot value the holding as of the split date."
            )

        price = float(series.loc[last_valid])
        if not np.isfinite(price) or price <= 0:
            raise BacktestError(f"Non-positive price for '{ticker}' as of the split date: {price}.")

        values[ticker] = quantity * price

    total = sum(values.values())
    return {t: v / total for t, v in values.items()}


def buy_and_hold_returns(prices: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """
    Simulate a buy-and-hold strategy: weights are used to establish an
    initial allocation, then NEVER rebalanced. Portfolio value is tracked
    as the sum of (shares held x price) each day, where share counts are
    fixed after day 1. Actual weights drift away from the initial targets
    as asset prices diverge -- this drift is real and expected, not a bug.
    """
    missing = [t for t in weights if t not in prices.columns]
    if missing:
        raise BacktestError(f"No price data for weighted holding(s): {missing}")

    weight_sum = sum(weights.values())
    if not np.isclose(weight_sum, 1.0, atol=1e-6):
        raise BacktestError(f"Weights must sum to 1.0, got {weight_sum:.6f}")

    if prices.empty or len(prices) < 2:
        raise BacktestError("Need at least 2 price observations to simulate buy-and-hold returns.")

    tickers = list(weights.keys())

    # Short gaps in a held asset's prices are bridged by carrying the last price forward
    # (so the portfolio's moves across the gap are not lost); longer gaps are rejected.
    try:
        prices = fill_price_gaps(prices[tickers])
    except PerformanceError as exc:
        raise BacktestError(str(exc)) from exc

    initial_prices = prices.iloc[0]

    no_start_price = [t for t in tickers if pd.isna(initial_prices[t])]
    if no_start_price:
        raise BacktestError(
            "No price on the first day of the simulation window for: "
            + ", ".join(f"'{t}'" for t in no_start_price)
            + "; cannot establish the initial positions."
        )

    if (initial_prices <= 0).any():
        raise BacktestError("Cannot simulate buy-and-hold from a non-positive starting price.")

    shares = {t: weights[t] / initial_prices[t] for t in tickers}
    portfolio_value = sum(shares[t] * prices[t] for t in tickers)

    return portfolio_value.pct_change().dropna().rename("portfolio")


def apply_one_time_transaction_cost(returns: pd.Series, turnover: float, cost_rate: float) -> pd.Series:
    """
    Charge a ONE-TIME transaction cost against portfolio WEALTH at the start
    of the test window, modeling the cost of rebalancing into a new
    allocation immediately before the test period begins:

        Cost            = Turnover x Transaction Cost Rate   (fraction of wealth)
        Wealth after    = Wealth before x (1 - Cost)
        Net 1st return  = (1 - Cost) x (1 + Gross 1st return) - 1

    The cost is paid out of wealth at entry (before the first period's
    price move), so the reduced wealth is what then compounds. This is NOT
    the same as subtracting the cost from the gross return: that would
    effectively charge the cost on end-of-period wealth and drop the
    Cost x Return cross term (it overstates wealth after a gain and
    understates it after a loss). For example: $1000 wealth, 100% turnover,
    1% cost, then two consecutive +10% periods. Correct: 1000 x 0.99 x 1.10
    x 1.10 = $1197.90. The old (incorrect) subtraction approach gave $1199.00
    instead -- a real, verified discrepancy, not a rounding difference.

    This is a simplification: it assumes one rebalance at the start of the
    test window and none thereafter (consistent with the buy-and-hold, no-
    further-rebalancing assumption used in buy_and_hold_returns).
    """
    if not 0 <= turnover <= 1:
        raise BacktestError(f"Turnover must be between 0 and 1, got {turnover}.")
    if cost_rate < 0:
        raise BacktestError(f"Transaction cost rate cannot be negative, got {cost_rate}.")
    if returns.empty:
        raise BacktestError("Cannot apply transaction costs to an empty return series.")

    cost = turnover * cost_rate
    adjusted = returns.copy()
    adjusted.iloc[0] = (1.0 - cost) * (1.0 + adjusted.iloc[0]) - 1.0
    return adjusted


def strategy_performance_summary(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    var_confidence: float = 0.95,
) -> dict:
    """
    Standard set of performance/risk metrics for a single strategy's test-
    period return series: CAGR, volatility, Sharpe, Sortino, max drawdown,
    historical VaR, and historical CVaR.
    """
    if returns.empty:
        raise BacktestError("Cannot summarize an empty return series.")

    summary = {
        "cagr": cagr(returns),
        "volatility": float(returns.std() * np.sqrt(252)),
        "max_drawdown": max_drawdown(returns),
    }

    try:
        summary["sharpe"] = sharpe_ratio(returns, risk_free_rate=risk_free_rate)
    except Exception:
        summary["sharpe"] = float("nan")

    try:
        summary["sortino"] = sortino_ratio(returns, risk_free_rate=risk_free_rate)
    except Exception:
        summary["sortino"] = float("nan")

    summary["historical_var_95"] = historical_var(returns, confidence=var_confidence)
    summary["historical_cvar_95"] = historical_cvar(returns, confidence=var_confidence)

    return summary


def run_backtest(
    prices: pd.DataFrame,
    benchmark_prices: pd.Series,
    split_date,
    current_weights: dict[str, float],
    risk_free_rate: float = 0.0,
    transaction_cost_rate: float = 0.001,
    weight_bounds: WeightBounds = (0.0, 1.0),
) -> dict:
    """
    Full out-of-sample backtest workflow:

        1. Split prices into train / test at split_date.
        2. Using ONLY training data, optimize minimum-volatility and
           maximum-Sharpe portfolios.
        3. Freeze all strategy weights (current, equal-weight, min-vol,
           max-Sharpe) -- none are re-optimized using test data.
        4. Simulate each strategy buy-and-hold over the TEST period,
           applying a one-time transaction cost based on turnover from
           the current allocation into that strategy's weights.
        5. Compute performance/risk metrics for each strategy over the
           test period, plus the benchmark (no weights, no transaction cost).

    `current_weights` must be the weights of the user's current holdings valued
    as of the SPLIT DATE (see holdings_weights_as_of), not today's live weights:
    this function cannot tell the difference, but weights derived from later
    prices leak the test period's returns into the starting allocation
    (look-ahead bias -- Issue #3). The "current" strategy is a counterfactual
    ("what if I had held my current positions throughout the test period"),
    not the user's actual historical performance -- the data model has no
    transaction history to reconstruct that from.

    Returns a dict keyed by strategy name ("current", "equal_weight",
    "min_volatility", "max_sharpe", "benchmark"), each containing
    {"weights": ..., "test_returns": pd.Series, "metrics": {...}}.
    ("benchmark" has weights=None, since it isn't a weighted combination
    of these assets.)
    """
    if prices.empty:
        raise BacktestError("Cannot backtest an empty price dataset.")

    train_prices, test_prices = train_test_split(prices, split_date)
    train_returns = daily_returns(train_prices)

    tickers = list(prices.columns)
    equal_weights = {t: 1 / len(tickers) for t in tickers}

    min_vol_result = minimum_volatility(train_returns, weight_bounds=weight_bounds, risk_free_rate=risk_free_rate)
    max_sharpe_result = maximum_sharpe(train_returns, risk_free_rate=risk_free_rate, weight_bounds=weight_bounds)

    strategies = {
        "current": current_weights,
        "equal_weight": equal_weights,
        "min_volatility": min_vol_result["weights"],
        "max_sharpe": max_sharpe_result["weights"],
    }

    results: dict = {}

    for name, weights in strategies.items():
        test_returns = buy_and_hold_returns(test_prices, weights)

        # "current" incurs no turnover cost -- the investor already holds it.
        # Every other strategy pays the cost of switching from current -> it.
        if name != "current":
            turnover = compute_turnover(current_weights, weights)
            test_returns = apply_one_time_transaction_cost(test_returns, turnover, transaction_cost_rate)

        results[name] = {
            "weights": weights,
            "test_returns": test_returns,
            "metrics": strategy_performance_summary(test_returns, risk_free_rate=risk_free_rate),
        }

    _, test_benchmark = train_test_split(benchmark_prices.to_frame("benchmark"), split_date)
    benchmark_returns = daily_returns(test_benchmark["benchmark"])
    results["benchmark"] = {
        "weights": None,
        "test_returns": benchmark_returns,
        "metrics": strategy_performance_summary(benchmark_returns, risk_free_rate=risk_free_rate),
    }

    return results
