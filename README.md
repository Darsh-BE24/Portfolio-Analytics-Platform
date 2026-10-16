# Financial Portfolio Analytics & Optimization Platform

A quantitative finance project: portfolio valuation, performance/risk analytics,
optimization, efficient frontier, and backtesting -- built as a Python analytics
engine first, with FastAPI + React + PostgreSQL layered on top.

## Status: Milestone 2 -- Returns, CAGR, Sharpe, Sortino, rolling metrics

### What's implemented

- `backend/portfolio.py` -- `Portfolio` class: input validation, position
  valuation, and weight calculation. Pure Python/dataclasses, no network
  dependency, fully unit tested.
- `backend/data.py` -- `MarketDataService`: wraps yfinance for historical
  prices, volume, benchmark data, and latest prices. This is the only
  module in the project that imports yfinance.
- `backend/performance.py` -- daily/cumulative returns, portfolio returns
  from fixed weights, CAGR, annualized Sharpe & Sortino ratios, rolling
  returns/Sharpe, and portfolio-vs-benchmark comparison. Pure pandas/numpy
  logic, no network dependency, fully unit tested with hand-checkable values.

### Project structure

```
backend/
├── __init__.py
├── data.py              # yfinance wrapper (market data)
├── portfolio.py          # position valuation & weights (pure logic)
├── performance.py        # returns, CAGR, Sharpe, Sortino, rolling metrics
├── requirements.txt
└── tests/
    ├── __init__.py
    ├── test_portfolio.py     # 18 tests, no network needed
    ├── test_data.py          # 7 tests, mocked yfinance, no network needed
    └── test_performance.py   # 25 tests, no network needed
```

## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt
```

## Run tests

From the project root:

```bash
python -m pytest backend/tests/ -v
```

Expected: all tests pass, no network access required (yfinance calls are
mocked in `test_data.py`).

## Try it with live data

Requires internet access to Yahoo Finance:

```python
from backend.data import MarketDataService
from backend.portfolio import Portfolio

portfolio = Portfolio({"AAPL": 10, "MSFT": 5, "NVDA": 8, "AMZN": 4})

svc = MarketDataService()
prices = svc.get_latest_prices(portfolio.tickers)

for v in portfolio.value(prices):
    print(f"{v.ticker:<6}{v.price:<10.2f}{v.quantity:<10.0f}{v.value:<12.2f}{v.weight:.2%}")

print(f"Portfolio Value: ${portfolio.total_value(prices):,.2f}")
```

## Design notes

- **`portfolio.py` never imports yfinance.** Financial calculations are kept
  separate from data-fetching so they can be unit tested with fixed inputs,
  without network calls or API flakiness.
- **Missing data is never silently dropped.** A missing price for a holding,
  or an invalid/delisted ticker, raises an explicit error rather than being
  skipped -- silently dropping a position would understate portfolio value
  and distort every other holding's weight.
- **Inputs are normalized at the boundary.** Tickers are upper-cased and
  stripped of whitespace in `Portfolio.__init__`, so every method downstream
  can assume clean data.
- **Partial / gappy price data is handled explicitly in `daily_returns()`**
  (the backtester's buy-and-hold simulation uses the same rule), rather than
  depending on pandas' version-specific `pct_change` gap filling:
  - Weekends and exchange holidays are absent dates, not missing data. The
    return across them is the true multi-day move.
  - A gap of up to `MAX_FFILL_GAP` (5) consecutive missing prices for one asset
    is bridged by carrying the last price forward: 0% on the missing days, and
    the next observed price realizes the full move, so no return is lost.
  - An asset that starts trading later (leading gap) is valid partial history.
    Because a fixed-weight portfolio return needs every holding, analysis uses
    the **common window** -- from the first date all assets have a return. Note
    this shortens the window used for CAGR/volatility/VaR/optimization.
  - A longer interior or trailing gap, or an asset with fewer than 2 prices,
    raises a `PerformanceError` naming the asset (never a silent NaN).
  - Covariance/volatility with fewer than 2 overlapping observations raises a
    `RiskError`.
- **Latest (current) prices are cached briefly, in memory.**
  `MarketDataService.get_latest_prices()` reuses a fetched price for
  `LATEST_PRICE_TTL_SECONDS` (60s), per normalized ticker, instead of hitting
  the provider on every request (one dashboard view used to trigger ~5
  identical downloads, and could show different prices on different tabs).
  - The TTL is deliberately short and **not market-hours aware**: open markets
    get bounded staleness, closed markets (nights, weekends, holidays) cannot
    change anyway, and no per-exchange calendar logic is needed.
  - Provider errors and invalid tickers are never cached, and an expired price
    is never served as a fallback if the provider fails. Portfolio creation
    validates every holding and the benchmark through this call before writing
    to the database; the cache cannot weaken that gate.
  - `get_latest_quotes()` also returns each price's trading date and fetch time,
    so "last close from Friday" is distinguishable from a fresh quote.
  - Pass `latest_price_ttl_seconds=0` to disable. This cache is separate from
    the DB-backed historical cache (`cache.py`), which has no fetch timestamps
    and never refreshes existing rows, so it cannot safely hold current prices.

- **The historical price cache tracks coverage ranges, not business days.**
  `cache.py` records, per ticker, which date ranges were actually fetched
  (`MarketDataCoverage`); a request is a cache hit only if a recorded range
  contains it. It used to compare cached rows against every Mon-Fri in the
  range, so any range spanning an exchange holiday (nearly every multi-month
  lookback) could never be judged complete and was re-downloaded each time.
  No holiday calendar is needed, so it is correct for any exchange.
  - Interior gaps (holidays, a day with no bar) do not affect coverage.
  - A response whose last observation is more than 7 days before the requested
    end looks truncated (outage / rate limit, not a market closure): coverage
    is recorded only up to what was observed, so the next request retries.
    A truncation shorter than 7 days is not detected.
  - An asset whose history simply starts later than the requested start is
    cached as-is (retrying cannot produce data that does not exist); a transient
    failure that dropped only the *start* of a response would look the same.
  - A failed fetch records nothing. Extending a range refetches the whole
    request (not just the new days) and merges coverage into one row.
  - Existing databases pick up the new table on startup (`create_all`). Rows
    cached before this change have no coverage record, so each range is
    refetched once and then cached normally.
- **The backtest's "current" strategy is a what-if, valued at the split date.**
  It answers "what if I had held my current share counts throughout the test
  period?" -- not "what did my portfolio actually do": the data model has no
  transaction history, so past holdings are unknowable. The weights are the
  share counts valued at the last prices known on or before the split date
  (`backtest.holdings_weights_as_of`), the same information the optimized
  strategies get. They were previously valued at today's live price, which is
  the end of the test window, so a holding that rallied during the test started
  with an inflated weight and the strategy was credited for gains it could not
  have known about. Other endpoints still use live weights ("my portfolio now").
  - A holding with no price, a price older than `MAX_FFILL_GAP` observations,
    or a non-positive price at the split date raises a `BacktestError` naming
    it, instead of being skipped or priced with a stale quote.
  - The split-date allocation is also the "from" side of the turnover used for
    the transaction cost of switching into the other strategies.
  - Positions are sized at the first test day's prices, exactly like every
    other strategy, so all strategies share the same one-day execution lag.

### Try Milestone 2 with live data

```python
from backend.data import MarketDataService
from backend.performance import daily_returns, portfolio_returns, cagr, sharpe_ratio, sortino_ratio
from datetime import date, timedelta

svc = MarketDataService()
weights = {"AAPL": 0.4, "MSFT": 0.3, "NVDA": 0.2, "AMZN": 0.1}
prices = svc.get_price_data(weights.keys(), date.today() - timedelta(days=365*3), date.today())

asset_returns = daily_returns(prices)
port_returns = portfolio_returns(asset_returns, weights)

print(f"CAGR:   {cagr(port_returns):.2%}")
print(f"Sharpe: {sharpe_ratio(port_returns, risk_free_rate=0.04):.2f}")
print(f"Sortino:{sortino_ratio(port_returns, risk_free_rate=0.04):.2f}")
```

### Design notes specific to performance.py

- **Fixed-weight assumption:** `portfolio_returns()` combines asset returns
  using constant weights, which implicitly assumes the portfolio is
  rebalanced back to those weights every period. It does not model natural
  weight drift from a true buy-and-hold position -- that's addressed later
  in backtesting (Milestone 6), where rebalancing frequency becomes explicit.
- **Sharpe vs. Sortino:** Sharpe penalizes total volatility (including
  upside swings); Sortino only penalizes downside deviation. A strategy
  with big positive outliers and small, consistent losses will show a much
  higher Sortino than Sharpe -- that gap is a genuine signal, not a bug.
- **Annual risk-free rate is converted to a daily rate** by dividing by
  `periods_per_year` (default 252). This ignores intra-year compounding of
  the risk-free rate, a standard simplification for daily-frequency Sharpe/
  Sortino calculations.
- **Zero-volatility and no-downside edge cases raise explicitly** rather
  than returning `inf`/`nan` silently -- a `PerformanceError` with a message
  is far easier to debug than a silent `NaN` propagating into later results.

## Next milestone

`risk.py` -- portfolio volatility (via the covariance matrix), beta,
drawdown (current, max, recovery period), correlation matrix, and
concentration risk (HHI, effective number of holdings).
