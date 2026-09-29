"""
Service layer: orchestrates calls into the analytics engine
(portfolio.py, performance.py, risk.py, optimization.py, backtest.py),
the market data cache (cache.py), and the database (models.py).

This module contains NO financial calculations of its own -- only
persistence, fetching/caching, and wiring results into API response
shapes. If you find yourself writing a formula here, it belongs in one
of the analytics modules instead.

Portfolios are now persisted in the database (Milestone 9) rather than
held in an in-memory dict. Each analytics call loads the portfolio's
holdings from the database, fetches prices through the cache-first
layer, and recomputes current weights from LIVE latest prices -- weights
are intentionally never frozen/stored, since a portfolio's weights should
always reflect current market values, not a snapshot from creation time.
"""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from sqlalchemy.orm import Session

from backend.backtest import run_backtest, train_test_split
from backend.cache import get_benchmark_data_cached, get_price_data_cached
from backend.data import MarketDataService
from backend.models import PortfolioHolding, PortfolioModel
from backend.optimization import efficient_frontier as compute_efficient_frontier
from backend.optimization import expected_returns, maximum_sharpe, minimum_volatility
from backend.performance import (
    cagr,
    compare_to_benchmark,
    cumulative_returns,
    daily_returns,
    portfolio_returns,
    rolling_sharpe,
    sharpe_ratio,
    sortino_ratio,
)
from backend.portfolio import Portfolio
from backend.risk import (
    beta,
    concentration_metrics,
    correlation_matrix,
    current_drawdown,
    drawdown_periods,
    drawdown_series,
    historical_cvar,
    historical_var,
    max_drawdown,
    monte_carlo_var,
    parametric_cvar,
    parametric_var,
    portfolio_volatility,
    rolling_volatility,
)
from backend.schemas import (
    BacktestRequest,
    BacktestResponse,
    CorrelationResponse,
    DrawdownPeriodOut,
    EfficientFrontierRequest,
    EfficientFrontierResponse,
    FrontierPointOut,
    OptimizeRequest,
    OptimizeResponse,
    PerformanceResponse,
    PortfolioCreateRequest,
    PortfolioResponse,
    PositionOut,
    RiskResponse,
    StrategyResultOut,
    TimeSeriesPointOut,
)

ROLLING_WINDOW_DAYS = 63  # ~3 months of trading days


def _to_points(series: pd.Series) -> list[TimeSeriesPointOut]:
    """
    Convert a date-indexed pandas Series into API time-series points.
    NaN values (e.g. the leading rows of a rolling window, before enough
    data exists to fill it) are dropped rather than serialized -- JSON has
    no clean representation for NaN, and a chart has nothing to plot there.
    """
    clean = series.dropna()
    return [TimeSeriesPointOut(date=ts.date(), value=float(v)) for ts, v in clean.items()]


class PortfolioNotFoundError(Exception):
    """Raised when a requested portfolio ID does not exist."""


class ServiceError(Exception):
    """Raised for service-layer problems not covered by a more specific exception."""


class PortfolioService:
    def __init__(self, market_data: MarketDataService | None = None):
        self.market_data = market_data or MarketDataService()

    # -- internal helpers ---------------------------------------------------

    def _load_row(self, db: Session, portfolio_id: str) -> PortfolioModel:
        row = db.get(PortfolioModel, portfolio_id)
        if row is None:
            raise PortfolioNotFoundError(f"No portfolio found with id '{portfolio_id}'.")
        return row

    def _holdings_dict(self, row: PortfolioModel) -> dict[str, float]:
        return {h.ticker: h.quantity for h in row.holdings}

    def _date_range(self, row: PortfolioModel) -> tuple[date, date]:
        end = date.today()
        start = end - timedelta(days=int(row.lookback_years * 365.25))
        return start, end

    def _asset_prices(self, db: Session, row: PortfolioModel) -> pd.DataFrame:
        tickers = list(self._holdings_dict(row).keys())
        start, end = self._date_range(row)
        return get_price_data_cached(db, self.market_data, tickers, start, end)

    def _benchmark_prices(self, db: Session, row: PortfolioModel) -> pd.Series:
        start, end = self._date_range(row)
        return get_benchmark_data_cached(db, self.market_data, row.benchmark_ticker, start, end)

    def _current_weights(self, db: Session, row: PortfolioModel) -> dict[str, float]:
        holdings = self._holdings_dict(row)
        portfolio = Portfolio(holdings)
        latest_prices = self.market_data.get_latest_prices(portfolio.tickers)
        return portfolio.weights(latest_prices)

    # -- portfolio CRUD -------------------------------------------------------

    def create_portfolio(self, db: Session, request: PortfolioCreateRequest) -> PortfolioResponse:
        holdings_dict = {h.ticker: h.quantity for h in request.holdings}
        portfolio = Portfolio(holdings_dict)
        portfolio.validate()

        row = PortfolioModel(benchmark_ticker=request.benchmark, lookback_years=request.lookback_years)
        row.holdings = [
            PortfolioHolding(ticker=ticker, quantity=qty) for ticker, qty in portfolio.holdings.items()
        ]
        db.add(row)
        db.commit()
        db.refresh(row)

        latest_prices = self.market_data.get_latest_prices(portfolio.tickers)
        valuation = portfolio.value(latest_prices)

        return PortfolioResponse(
            id=row.id,
            created_at=row.created_at,
            benchmark=row.benchmark_ticker,
            positions=[
                PositionOut(ticker=v.ticker, price=v.price, quantity=v.quantity, value=v.value, weight=v.weight)
                for v in valuation
            ],
            total_value=portfolio.total_value(latest_prices),
        )

    def get_portfolio(self, db: Session, portfolio_id: str) -> PortfolioResponse:
        row = self._load_row(db, portfolio_id)
        holdings = self._holdings_dict(row)
        portfolio = Portfolio(holdings)
        latest_prices = self.market_data.get_latest_prices(portfolio.tickers)
        valuation = portfolio.value(latest_prices)

        return PortfolioResponse(
            id=row.id,
            created_at=row.created_at,
            benchmark=row.benchmark_ticker,
            positions=[
                PositionOut(ticker=v.ticker, price=v.price, quantity=v.quantity, value=v.value, weight=v.weight)
                for v in valuation
            ],
            total_value=portfolio.total_value(latest_prices),
        )

    # -- analytics ------------------------------------------------------------

    def get_performance(self, db: Session, portfolio_id: str, risk_free_rate: float = 0.0) -> PerformanceResponse:
        row = self._load_row(db, portfolio_id)
        asset_returns = daily_returns(self._asset_prices(db, row))
        weights = self._current_weights(db, row)
        port_returns = portfolio_returns(asset_returns, weights)
        bench_returns = daily_returns(self._benchmark_prices(db, row))

        comparison = compare_to_benchmark(port_returns, bench_returns)

        return PerformanceResponse(
            cagr=cagr(port_returns),
            cumulative_return=float(cumulative_returns(port_returns).iloc[-1]),
            sharpe_ratio=sharpe_ratio(port_returns, risk_free_rate=risk_free_rate),
            sortino_ratio=sortino_ratio(port_returns, risk_free_rate=risk_free_rate),
            risk_free_rate=risk_free_rate,
            cumulative_return_series=_to_points(comparison["portfolio_cumulative"]),
            benchmark_cumulative_return_series=_to_points(comparison["benchmark_cumulative"]),
        )

    def get_risk(
        self, db: Session, portfolio_id: str, risk_free_rate: float = 0.0, var_confidence: float = 0.95
    ) -> RiskResponse:
        row = self._load_row(db, portfolio_id)
        asset_returns = daily_returns(self._asset_prices(db, row))
        weights = self._current_weights(db, row)
        port_returns = portfolio_returns(asset_returns, weights)
        bench_returns = daily_returns(self._benchmark_prices(db, row))
        concentration = concentration_metrics(weights)

        periods = [
            DrawdownPeriodOut(
                start=p["start"].date(),
                trough=p["trough"].date(),
                end=p["end"].date() if p["end"] is not None else None,
                depth=p["depth"],
                peak_to_trough_days=p["peak_to_trough_days"],
                recovery_days=p["recovery_days"],
            )
            for p in drawdown_periods(port_returns)
        ]

        return RiskResponse(
            annualized_volatility=portfolio_volatility(asset_returns, weights),
            beta=beta(port_returns, bench_returns),
            max_drawdown=max_drawdown(port_returns),
            current_drawdown=current_drawdown(port_returns),
            drawdown_periods=periods,
            drawdown_series=_to_points(drawdown_series(port_returns)),
            rolling_volatility=_to_points(rolling_volatility(port_returns, window=ROLLING_WINDOW_DAYS)),
            rolling_sharpe=_to_points(
                rolling_sharpe(port_returns, window=ROLLING_WINDOW_DAYS, risk_free_rate=risk_free_rate)
            ),
            historical_var_95=historical_var(port_returns, confidence=var_confidence),
            parametric_var_95=parametric_var(port_returns, confidence=var_confidence),
            monte_carlo_var_95=monte_carlo_var(port_returns, confidence=var_confidence, random_seed=42),
            historical_cvar_95=historical_cvar(port_returns, confidence=var_confidence),
            parametric_cvar_95=parametric_cvar(port_returns, confidence=var_confidence),
            largest_position=concentration["largest_position"],
            top_3_concentration=concentration["top_3_concentration"],
            hhi=concentration["hhi"],
            effective_number_of_holdings=concentration["effective_number_of_holdings"],
        )

    def get_correlation(self, db: Session, portfolio_id: str) -> CorrelationResponse:
        row = self._load_row(db, portfolio_id)
        asset_returns = daily_returns(self._asset_prices(db, row))
        corr = correlation_matrix(asset_returns)

        return CorrelationResponse(tickers=list(corr.columns), matrix=corr.values.round(6).tolist())

    def optimize(self, db: Session, portfolio_id: str, request: OptimizeRequest) -> OptimizeResponse:
        row = self._load_row(db, portfolio_id)
        asset_returns = daily_returns(self._asset_prices(db, row))

        if request.method == "min_volatility":
            result = minimum_volatility(
                asset_returns, weight_bounds=request.weight_bounds, risk_free_rate=request.risk_free_rate
            )
        else:
            result = maximum_sharpe(
                asset_returns, risk_free_rate=request.risk_free_rate, weight_bounds=request.weight_bounds
            )

        return OptimizeResponse(**result)

    def get_efficient_frontier(
        self, db: Session, portfolio_id: str, request: EfficientFrontierRequest
    ) -> EfficientFrontierResponse:
        row = self._load_row(db, portfolio_id)
        asset_returns = daily_returns(self._asset_prices(db, row))
        weights = self._current_weights(db, row)
        tickers = list(asset_returns.columns)

        frontier_df = compute_efficient_frontier(
            asset_returns,
            risk_free_rate=request.risk_free_rate,
            n_points=request.n_points,
            weight_bounds=request.weight_bounds,
        )

        def _row_to_point(r) -> FrontierPointOut:
            return FrontierPointOut(
                expected_return=r["expected_return"],
                volatility=r["volatility"],
                sharpe_ratio=r["sharpe_ratio"],
                weights={t: r[t] for t in tickers},
            )

        points = [_row_to_point(r) for _, r in frontier_df.iterrows()]

        mean_returns = expected_returns(asset_returns)
        current_return = float(sum(mean_returns[t] * weights.get(t, 0.0) for t in tickers))
        current_vol = portfolio_volatility(asset_returns, weights)
        current_sharpe = (current_return - request.risk_free_rate) / current_vol if current_vol > 0 else float("nan")
        current_point = FrontierPointOut(
            expected_return=current_return, volatility=current_vol, sharpe_ratio=current_sharpe, weights=weights
        )

        min_vol_result = minimum_volatility(
            asset_returns, weight_bounds=request.weight_bounds, risk_free_rate=request.risk_free_rate
        )
        max_sharpe_result = maximum_sharpe(
            asset_returns, risk_free_rate=request.risk_free_rate, weight_bounds=request.weight_bounds
        )

        return EfficientFrontierResponse(
            points=points,
            current_portfolio=current_point,
            min_volatility_portfolio=FrontierPointOut(**min_vol_result),
            max_sharpe_portfolio=FrontierPointOut(**max_sharpe_result),
        )

    def run_backtest(self, db: Session, portfolio_id: str, request: BacktestRequest) -> BacktestResponse:
        row = self._load_row(db, portfolio_id)
        asset_prices = self._asset_prices(db, row)
        benchmark_prices = self._benchmark_prices(db, row)
        weights = self._current_weights(db, row)

        train_prices, test_prices = train_test_split(asset_prices, request.split_date)

        results = run_backtest(
            asset_prices,
            benchmark_prices,
            request.split_date,
            current_weights=weights,
            risk_free_rate=request.risk_free_rate,
            transaction_cost_rate=request.transaction_cost_rate,
            weight_bounds=request.weight_bounds,
        )

        strategies = {
            name: StrategyResultOut(weights=r["weights"], metrics=r["metrics"]) for name, r in results.items()
        }

        return BacktestResponse(
            train_start=train_prices.index.min().date(),
            train_end=train_prices.index.max().date(),
            test_start=test_prices.index.min().date(),
            test_end=test_prices.index.max().date(),
            strategies=strategies,
        )
