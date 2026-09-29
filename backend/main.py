"""
FastAPI application.

Routes are intentionally thin: parse/validate the request (via Pydantic
schemas), obtain a request-scoped database session (via Depends(get_db)),
call into the service layer, translate exceptions into HTTP responses.
No financial calculations happen in this file -- see
service.py -> {performance,risk,optimization,backtest}.py for that.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import os

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from backend.backtest import BacktestError
from backend.data import MarketDataError
from backend.db import get_db, init_db
from backend.optimization import OptimizationError
from backend.performance import PerformanceError
from backend.portfolio import PortfolioValidationError
from backend.risk import RiskError
from backend.schemas import (
    BacktestRequest,
    BacktestResponse,
    CorrelationResponse,
    EfficientFrontierRequest,
    EfficientFrontierResponse,
    OptimizeRequest,
    OptimizeResponse,
    PerformanceResponse,
    PortfolioCreateRequest,
    PortfolioResponse,
    RiskResponse,
)
from backend.service import PortfolioNotFoundError, PortfolioService, ServiceError


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(title="Portfolio Analytics API", version="0.3.0", lifespan=lifespan)

# CORS: lets the React dev server (a different origin than this API) call
# these endpoints from the browser. Origins are configurable via the
# CORS_ORIGINS env var (comma-separated) so a deployed frontend URL can be
# added later without a code change. Defaults cover Vite's dev server.
_default_origins = "http://localhost:5173,http://127.0.0.1:5173"
_allowed_origins = [o.strip() for o in os.environ.get("CORS_ORIGINS", _default_origins).split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

service = PortfolioService()


# Every exception type that represents a "the request was bad" condition,
# as opposed to an unexpected server bug. Mapped to 400 Bad Request.
_VALIDATION_ERRORS = (
    PortfolioValidationError,
    MarketDataError,
    PerformanceError,
    RiskError,
    OptimizationError,
    BacktestError,
    ServiceError,
    ValueError,
)


def _handle(fn, *args, **kwargs):
    """Run a service-layer call and translate its exceptions into the appropriate HTTP error."""
    try:
        return fn(*args, **kwargs)
    except PortfolioNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except _VALIDATION_ERRORS as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/portfolio", response_model=PortfolioResponse)
def create_portfolio(request: PortfolioCreateRequest, db: Session = Depends(get_db)) -> PortfolioResponse:
    return _handle(service.create_portfolio, db, request)


@app.get("/portfolio/{portfolio_id}", response_model=PortfolioResponse)
def get_portfolio(portfolio_id: str, db: Session = Depends(get_db)) -> PortfolioResponse:
    return _handle(service.get_portfolio, db, portfolio_id)


@app.get("/portfolio/{portfolio_id}/performance", response_model=PerformanceResponse)
def get_performance(
    portfolio_id: str, risk_free_rate: float = 0.0, db: Session = Depends(get_db)
) -> PerformanceResponse:
    return _handle(service.get_performance, db, portfolio_id, risk_free_rate)


@app.get("/portfolio/{portfolio_id}/risk", response_model=RiskResponse)
def get_risk(
    portfolio_id: str, risk_free_rate: float = 0.0, var_confidence: float = 0.95, db: Session = Depends(get_db)
) -> RiskResponse:
    return _handle(service.get_risk, db, portfolio_id, risk_free_rate, var_confidence)


@app.get("/portfolio/{portfolio_id}/correlation", response_model=CorrelationResponse)
def get_correlation(portfolio_id: str, db: Session = Depends(get_db)) -> CorrelationResponse:
    return _handle(service.get_correlation, db, portfolio_id)


@app.post("/portfolio/{portfolio_id}/optimize", response_model=OptimizeResponse)
def optimize_portfolio(
    portfolio_id: str, request: OptimizeRequest, db: Session = Depends(get_db)
) -> OptimizeResponse:
    return _handle(service.optimize, db, portfolio_id, request)


# NOTE: this is a POST, not the GET the original spec sketch suggested --
# efficient frontier configuration (n_points, per-ticker weight_bounds
# dicts) doesn't fit cleanly into query parameters, so it's a request body
# instead. This is a deliberate, documented deviation, not an oversight.
@app.post("/portfolio/{portfolio_id}/efficient-frontier", response_model=EfficientFrontierResponse)
def get_efficient_frontier(
    portfolio_id: str, request: EfficientFrontierRequest, db: Session = Depends(get_db)
) -> EfficientFrontierResponse:
    return _handle(service.get_efficient_frontier, db, portfolio_id, request)


@app.post("/portfolio/{portfolio_id}/backtest", response_model=BacktestResponse)
def backtest_portfolio(
    portfolio_id: str, request: BacktestRequest, db: Session = Depends(get_db)
) -> BacktestResponse:
    return _handle(service.run_backtest, db, portfolio_id, request)


@app.get("/health")
def health_check() -> dict:
    return {"status": "ok"}
