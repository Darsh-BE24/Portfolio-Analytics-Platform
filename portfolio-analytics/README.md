# Financial Portfolio Analytics & Optimization Platform

A quantitative finance project: portfolio valuation, performance/risk analytics,
optimization, efficient frontier, and backtesting -- built as a Python analytics
engine first, with FastAPI + React + PostgreSQL layered on top.

## Status: Milestone 1 -- Portfolio input, market data, valuation, weights

### What's implemented

- `backend/portfolio.py` -- `Portfolio` class: input validation, position
  valuation, and weight calculation. Pure Python/dataclasses, no network
  dependency, fully unit tested.
- `backend/data.py` -- `MarketDataService`: wraps yfinance for historical
  prices, volume, benchmark data, and latest prices. This is the only
  module in the project that imports yfinance.

### Project structure

```
backend/
├── __init__.py
├── data.py            # yfinance wrapper (market data)
├── portfolio.py        # position valuation & weights (pure logic)
├── requirements.txt
└── tests/
    ├── __init__.py
    ├── test_portfolio.py   # 18 tests, no network needed
    └── test_data.py        # 7 tests, mocked yfinance, no network needed
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

## Next milestone

`performance.py` -- daily/cumulative returns, CAGR, Sharpe ratio, Sortino
ratio, rolling performance metrics, portfolio vs. benchmark comparison.
