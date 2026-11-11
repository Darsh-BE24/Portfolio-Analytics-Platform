# Financial Portfolio Analytics & Optimization Platform (Indian Market)

A quantitative finance project for Indian equities (NSE by default): portfolio
valuation, performance and risk analytics, optimization, the efficient frontier,
and backtesting. It is a Python analytics engine first, with a FastAPI backend,
a React dashboard, and SQLite/PostgreSQL storage layered on top.

## Features

- **Portfolio valuation** -- positions, weights and total value in rupees from live Yahoo Finance prices.
- **Performance** -- returns, CAGR, annualized Sharpe and Sortino ratios, rolling returns/Sharpe, comparison with a benchmark (Nifty 50 by default).
- **Risk** -- volatility, beta, drawdown, Value at Risk, correlation matrix, concentration (HHI, effective number of holdings).
- **Optimization** -- minimum-volatility and maximum-Sharpe portfolios, and the efficient frontier.
- **Backtesting** -- train/test split with no look-ahead: current, equal-weight, min-volatility, max-Sharpe and benchmark strategies, with transaction costs.
- **Web dashboard** -- five tabs (Overview, Performance, Risk, Optimization, Backtest) with Plotly charts.

## Project structure

```
backend/
├── main.py              # FastAPI app: thin routes, error translation, CORS
├── service.py           # orchestrates the engine, the cache and the database
├── schemas.py           # request/response models (pydantic)
├── market.py            # market settings: benchmark, ticker suffix, trading days
├── data.py              # yfinance wrapper (the only module that imports yfinance)
├── cache.py             # database-backed historical price cache
├── db.py, models.py     # SQLAlchemy engine and tables
├── portfolio.py         # position valuation and weights (pure logic)
├── performance.py       # returns, CAGR, Sharpe, Sortino, rolling metrics
├── risk.py              # volatility, beta, drawdown, VaR, correlation, concentration
├── optimization.py      # min-volatility, max-Sharpe, efficient frontier
├── backtest.py          # train/test split, strategy comparison, transaction costs
├── requirements.txt
└── tests/               # pytest suite, no network needed (yfinance is mocked)

frontend/
├── src/
│   ├── App.jsx, main.jsx
│   ├── market.js        # market settings (mirrors backend/market.py)
│   ├── api/             # API client
│   ├── components/      # form, metric tiles, plots, loading/error wrapper
│   ├── hooks/           # data-fetching hooks
│   ├── tabs/            # Overview, Performance, Risk, Optimization, Backtest
│   └── utils/           # holdings validation, number/currency formatting
└── package.json         # Vite + React + vitest

smoke_test.py            # end-to-end check against a running backend (live data)
```

## Setup

Requires Python 3.9 or newer and Node.js 18 or newer.

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1
pip install -r backend/requirements.txt

cd frontend
npm install
```

## Run the app

Start the backend from the project root (virtual environment active):

```bash
python -m uvicorn backend.main:app --reload
```

The API runs at http://127.0.0.1:8000 (interactive docs at `/docs`). In a second
terminal, start the frontend:

```bash
cd frontend
npm run dev
```

Open the address it prints (http://localhost:5173). Add your holdings, for
example `RELIANCE`, `TCS`, `HDFCBANK`, `INFY`, and click create.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./portfolio_analytics.db` | Use PostgreSQL with e.g. `postgresql://user:password@localhost:5432/portfolio_analytics` |
| `CORS_ORIGINS` | the local dev server | Comma-separated list of allowed frontend origins for a deployed frontend |

### API endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/portfolio` | Create a portfolio (validates every holding and the benchmark against live prices) |
| GET | `/portfolio/{id}` | Portfolio with current values and weights |
| GET | `/portfolio/{id}/performance` | CAGR, Sharpe, Sortino, benchmark comparison |
| GET | `/portfolio/{id}/risk` | Volatility, beta, drawdown, VaR, concentration |
| GET | `/portfolio/{id}/correlation` | Correlation matrix |
| POST | `/portfolio/{id}/optimize` | Minimum-volatility or maximum-Sharpe weights |
| POST | `/portfolio/{id}/efficient-frontier` | Efficient frontier points |
| POST | `/portfolio/{id}/backtest` | Out-of-sample strategy comparison |
| GET | `/health` | Health check |

## Run the tests

Backend, from the project root (no network access needed):

```bash
python -m pytest backend
```

Frontend:

```bash
cd frontend
npm test
npm run build
```

### End-to-end smoke test (live data)

With the backend running, in another terminal (same virtual environment):

```bash
python smoke_test.py
```

It creates a portfolio from live Yahoo Finance data, calls every endpoint, checks
the error cases (unknown ticker, duplicate holding, unknown portfolio) and prints
PASS or FAIL for each check. It needs internet access and uses only the standard
library. Options: `--base-url`, `--holdings "RELIANCE:10,TCS:5"`, `--risk-free 0.07`.

## Try it from Python (live data)

```python
from datetime import date, timedelta

from backend.data import MarketDataService
from backend.performance import daily_returns, portfolio_returns, cagr, sharpe_ratio, sortino_ratio
from backend.portfolio import Portfolio

portfolio = Portfolio({"RELIANCE.NS": 10, "TCS.NS": 5, "HDFCBANK.NS": 8, "INFY.NS": 4})
svc = MarketDataService()
prices = svc.get_latest_prices(portfolio.tickers)

for v in portfolio.value(prices):
    print(f"{v.ticker:<14}{v.price:<10.2f}{v.quantity:<8.0f}{v.value:<12.2f}{v.weight:.2%}")
print(f"Portfolio value: Rs {portfolio.total_value(prices):,.2f}")

weights = {"RELIANCE.NS": 0.4, "TCS.NS": 0.3, "HDFCBANK.NS": 0.2, "INFY.NS": 0.1}
history = svc.get_price_data(weights.keys(), date.today() - timedelta(days=365 * 3), date.today())
returns = portfolio_returns(daily_returns(history), weights)

print(f"CAGR:    {cagr(returns):.2%}")
print(f"Sharpe:  {sharpe_ratio(returns, risk_free_rate=0.07):.2f}")
print(f"Sortino: {sortino_ratio(returns, risk_free_rate=0.07):.2f}")
```

## Design notes

### The Indian market

The platform targets NSE. Everything market-specific lives in `backend/market.py`
(and `frontend/src/market.js`), so changing markets means editing one file.

- **Tickers.** Bare symbols get the NSE suffix at the API boundary: `RELIANCE`
  becomes `RELIANCE.NS`. Indices (`^NSEI`) and symbols that already carry a
  suffix (`RELIANCE.BO` for BSE) are left alone. Symbols with `&` or `-`
  (`M&M`, `BAJAJ-AUTO`) are valid. The same stock typed two ways is rejected
  as a duplicate.
- **Benchmark.** The default is the Nifty 50 (`^NSEI`). `^BSESN` (Sensex) or an
  ETF such as `NIFTYBEES` also work.
- **Annualization.** Returns and risk use `TRADING_DAYS_PER_YEAR = 248`. NSE
  trades roughly 245-250 days a year, so this is an approximation; every module
  reads the same constant.
- **Display.** Values show in rupees with lakh/crore grouping. The risk-free rate
  field defaults to 7% (about the 10-year G-sec yield) and is meant to be edited.
- **Older data.** Portfolios saved before the switch keep their old `^GSPC`
  benchmark and US tickers. Delete `portfolio_analytics.db` (or those rows) to
  start clean.

### Architecture

- **`portfolio.py` never imports yfinance.** Financial calculations are kept
  separate from data fetching, so they are unit tested with fixed inputs, without
  network calls or API flakiness. `data.py` is the only module that imports
  yfinance.
- **Missing data is never silently dropped.** A missing price for a holding, or
  an invalid or delisted ticker, raises an explicit error. Silently dropping a
  position would understate portfolio value and distort every other weight.
- **Inputs are normalized at the boundary.** Tickers are stripped, upper-cased and
  suffixed once, so every method downstream can assume clean data.

### Performance and risk

- **Partial or gappy price data is handled explicitly** in `daily_returns()` (the
  backtester uses the same rule), instead of relying on pandas' version-specific
  gap filling:
  - Weekends and exchange holidays are absent dates, not missing data. The
    return across them is the true multi-day move.
  - A gap of up to `MAX_FFILL_GAP` (5) consecutive missing prices for one asset
    is bridged by carrying the last price forward.
  - An asset that starts trading later is valid partial history. Because a
    fixed-weight portfolio return needs every holding, analysis uses the
    **common window** -- from the first date all assets have a return. This
    shortens the window used for CAGR, volatility, VaR and optimization.
  - A longer interior or trailing gap, or an asset with fewer than 2 prices,
    raises a `PerformanceError` naming the asset (never a silent NaN).
  - Covariance or volatility with fewer than 2 overlapping observations raises a
    `RiskError`.
- **Fixed-weight assumption.** `portfolio_returns()` combines asset returns using
  constant weights, which assumes the portfolio is rebalanced to those weights
  every period. It does not model natural weight drift; the backtester models a
  true buy-and-hold position.
- **Sharpe vs. Sortino.** Sharpe penalizes total volatility; Sortino penalizes
  only downside deviation. A large gap between them is a real signal, not a bug.
- **The annual risk-free rate is converted to a daily rate** by dividing by
  `periods_per_year` (default `TRADING_DAYS_PER_YEAR`). This ignores intra-year
  compounding, a standard simplification for daily Sharpe and Sortino.
- **Zero-volatility and no-downside edge cases raise explicitly** rather than
  returning `inf` or `nan`.

### Caching

- **Latest prices are cached briefly, in memory.**
  `MarketDataService.get_latest_prices()` reuses a fetched price for
  `LATEST_PRICE_TTL_SECONDS` (60 s) per normalized ticker, so one dashboard view
  does not trigger several identical downloads or show different prices on
  different tabs.
  - The TTL is short and **not market-hours aware**: open markets get bounded
    staleness, and closed markets cannot change anyway.
  - Provider errors and invalid tickers are never cached, and an expired price is
    never served as a fallback. Portfolio creation validates every holding and
    the benchmark through this call before writing to the database.
  - `get_latest_quotes()` also returns each price's trading date and fetch time.
  - Pass `latest_price_ttl_seconds=0` to disable.
- **The historical price cache tracks coverage ranges, not business days.**
  `cache.py` records which date ranges were actually fetched per ticker; a request
  is a cache hit only if a recorded range contains it. No holiday calendar is
  needed, so it is correct for any exchange.
  - Interior gaps (holidays, a day with no bar) do not affect coverage.
  - A response whose last observation is more than 7 days before the requested
    end looks truncated (outage or rate limit): coverage is recorded only up to
    what was observed, so the next request retries.
  - An asset whose history starts later than the requested start is cached as-is.
  - A failed fetch records nothing. Extending a range refetches the whole request
    and merges coverage into one row.
  - Existing databases pick up the coverage table on startup.

### Backtesting

- **The optimizer never sees test-period data.** Expected returns and covariance
  are estimated only from data before the split date.
- **The "current" strategy is a what-if, valued at the split date.** It answers
  "what if I had held my current share counts throughout the test period?" -- not
  "what did my portfolio actually do", because there is no transaction history.
  Weights are the share counts valued at the last prices known on or before the
  split date (`backtest.holdings_weights_as_of`), the same information the
  optimized strategies get. Valuing them at today's price would credit the
  strategy for gains it could not have known about. Other endpoints still use
  live weights ("my portfolio now").
  - A holding with no price, a price older than `MAX_FFILL_GAP` observations, or
    a non-positive price at the split date raises a `BacktestError` naming it.
  - The split-date allocation is also the "from" side of the turnover used for
    the transaction cost of switching into the other strategies.
  - Positions are sized at the first test day's prices, like every other
    strategy, so all strategies share the same one-day execution lag.

## Known limitations

- There is no authentication; the database has a users table, but nothing uses it.
- `TRADING_DAYS_PER_YEAR` is an approximation, not an exact NSE holiday calendar.
- Market data comes from Yahoo Finance through yfinance, so availability and
  history for individual Indian symbols depend on that provider.
- The backtest "current" strategy is a what-if (see above), not a record of real
  past holdings.
- `backend/tests/test_backtest_lookahead.py` builds its scenario around today's
  date; its thresholds are set to hold for any date through 2034.
