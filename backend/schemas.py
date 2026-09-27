"""
Request/response schemas for the FastAPI layer.

These are pure data-shape definitions -- no financial calculations live
here. That logic stays in performance.py / risk.py / optimization.py /
backtest.py, called from service.py.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field, field_validator


class HoldingIn(BaseModel):
    ticker: str
    quantity: float = Field(gt=0, description="Must be positive; short positions are not supported.")


class PortfolioCreateRequest(BaseModel):
    holdings: list[HoldingIn]
    benchmark: str = "^GSPC"
    lookback_years: float = Field(default=3.0, gt=0, le=10)

    @field_validator("holdings")
    @classmethod
    def holdings_not_empty(cls, v: list[HoldingIn]) -> list[HoldingIn]:
        if not v:
            raise ValueError("Portfolio must contain at least one holding.")
        return v


class PositionOut(BaseModel):
    ticker: str
    price: float
    quantity: float
    value: float
    weight: float


class PortfolioResponse(BaseModel):
    id: str
    created_at: datetime
    benchmark: str
    positions: list[PositionOut]
    total_value: float


class PerformanceResponse(BaseModel):
    cagr: float
    cumulative_return: float
    sharpe_ratio: float
    sortino_ratio: float
    risk_free_rate: float


class DrawdownPeriodOut(BaseModel):
    start: date
    trough: date
    end: date | None
    depth: float
    peak_to_trough_days: int | None
    recovery_days: int | None


class RiskResponse(BaseModel):
    annualized_volatility: float
    beta: float
    max_drawdown: float
    current_drawdown: float
    drawdown_periods: list[DrawdownPeriodOut]
    historical_var_95: float
    parametric_var_95: float
    monte_carlo_var_95: float
    historical_cvar_95: float
    parametric_cvar_95: float
    largest_position: float
    top_3_concentration: float
    hhi: float
    effective_number_of_holdings: float


class CorrelationResponse(BaseModel):
    tickers: list[str]
    matrix: list[list[float]]


WeightBoundsIn = tuple[float, float] | dict[str, tuple[float, float]]


class OptimizeRequest(BaseModel):
    method: str = Field(pattern="^(min_volatility|max_sharpe)$")
    risk_free_rate: float = 0.0
    weight_bounds: WeightBoundsIn = (0.0, 1.0)


class OptimizeResponse(BaseModel):
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float


class EfficientFrontierRequest(BaseModel):
    risk_free_rate: float = 0.0
    n_points: int = Field(default=30, ge=5, le=200)
    weight_bounds: WeightBoundsIn = (0.0, 1.0)


class FrontierPointOut(BaseModel):
    expected_return: float
    volatility: float
    sharpe_ratio: float
    weights: dict[str, float]


class EfficientFrontierResponse(BaseModel):
    points: list[FrontierPointOut]
    current_portfolio: FrontierPointOut
    min_volatility_portfolio: FrontierPointOut
    max_sharpe_portfolio: FrontierPointOut


class BacktestRequest(BaseModel):
    split_date: date
    risk_free_rate: float = 0.0
    transaction_cost_rate: float = Field(default=0.001, ge=0)
    weight_bounds: WeightBoundsIn = (0.0, 1.0)


class StrategyResultOut(BaseModel):
    weights: dict[str, float] | None
    metrics: dict[str, float]


class BacktestResponse(BaseModel):
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    strategies: dict[str, StrategyResultOut]
