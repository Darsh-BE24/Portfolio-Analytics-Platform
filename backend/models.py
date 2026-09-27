"""
SQLAlchemy ORM models.

Tables:
    users               -- registered users (no auth implemented in this
                            milestone; portfolios can exist without an owner)
    portfolios          -- one row per created portfolio (metadata only --
                            benchmark, lookback window -- not holdings)
    portfolio_holdings  -- one row per (portfolio, ticker, quantity)
    market_data         -- cached daily close prices, keyed by (ticker, date),
                            checked before calling yfinance (see cache.py)
"""

from __future__ import annotations

import uuid
from datetime import date, datetime, timezone

from sqlalchemy import Date, DateTime, Float, ForeignKey, String, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.db import Base


def _uuid() -> str:
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    username: Mapped[str] = mapped_column(String, unique=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    portfolios: Mapped[list["PortfolioModel"]] = relationship(back_populates="owner")


class PortfolioModel(Base):
    __tablename__ = "portfolios"

    id: Mapped[str] = mapped_column(String, primary_key=True, default=_uuid)
    user_id: Mapped[str | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    benchmark_ticker: Mapped[str] = mapped_column(String, nullable=False, default="^GSPC")
    lookback_years: Mapped[float] = mapped_column(Float, nullable=False, default=3.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow)

    owner: Mapped[User | None] = relationship(back_populates="portfolios")
    holdings: Mapped[list["PortfolioHolding"]] = relationship(
        back_populates="portfolio", cascade="all, delete-orphan"
    )


class PortfolioHolding(Base):
    __tablename__ = "portfolio_holdings"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    portfolio_id: Mapped[str] = mapped_column(ForeignKey("portfolios.id"), nullable=False)
    ticker: Mapped[str] = mapped_column(String, nullable=False)
    quantity: Mapped[float] = mapped_column(Float, nullable=False)

    portfolio: Mapped[PortfolioModel] = relationship(back_populates="holdings")


class MarketDataCache(Base):
    """
    Cached daily close prices. One row per (ticker, date). Checked by
    cache.py before falling back to a live yfinance call, so repeat
    requests for the same historical range don't re-download data
    that's already known.
    """

    __tablename__ = "market_data"
    __table_args__ = (UniqueConstraint("ticker", "date", name="uq_market_data_ticker_date"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String, nullable=False, index=True)
    date: Mapped[date] = mapped_column(Date, nullable=False, index=True)
    close_price: Mapped[float] = mapped_column(Float, nullable=False)
