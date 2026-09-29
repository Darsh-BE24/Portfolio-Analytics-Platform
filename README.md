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
