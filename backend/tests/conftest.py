"""
Shared test fixtures.

IMPORTANT: DATABASE_URL is overridden to an in-memory SQLite database
BEFORE backend.db (and anything that imports it) gets loaded anywhere in
the test session. This prevents the test suite from creating a stray
portfolio_analytics.db file on disk every time it runs.

_fake_download generates deterministic, seeded price data for ANY tickers
and date range yfinance.download is called with -- this lets the full
FastAPI test suite exercise real end-to-end flows (create portfolio ->
fetch prices -> run analytics) without any network access. The seed is
derived from the ticker symbol, so the same ticker always produces the
same price path across calls within a test run.
"""

import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from backend import data as data_module
from backend.db import Base, get_db


def _fake_download(tickers=None, start=None, end=None, **kwargs):
    if isinstance(tickers, str):
        tickers = [tickers]

    dates = pd.bdate_range(start=start, end=end)
    if len(dates) == 0:
        dates = pd.bdate_range(start=start, periods=5)

    fields = ["Open", "High", "Low", "Close", "Volume"]

    def _series_for(ticker: str) -> pd.Series:
        seed = abs(hash(ticker)) % (2**32)
        rng = np.random.RandomState(seed)
        return 100 * (1 + pd.Series(rng.normal(0.0005, 0.015, len(dates)), index=dates)).cumprod()

    if len(tickers) == 1:
        t = tickers[0]
        close = _series_for(t)
        return pd.DataFrame(
            {"Open": close, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": 1_000_000},
            index=dates,
        )

    columns = pd.MultiIndex.from_product([tickers, fields])
    data = {}
    for t in tickers:
        close = _series_for(t)
        data[(t, "Open")] = close
        data[(t, "High")] = close * 1.01
        data[(t, "Low")] = close * 0.99
        data[(t, "Close")] = close
        data[(t, "Volume")] = 1_000_000
    return pd.DataFrame(data, index=dates, columns=columns)


@pytest.fixture(autouse=True)
def mock_yfinance(monkeypatch):
    monkeypatch.setattr(data_module.yf, "download", _fake_download)


@pytest.fixture()
def test_engine():
    """A fresh in-memory SQLite database, isolated per test."""
    import backend.models  # noqa: F401 -- registers tables on Base before create_all

    engine = create_engine("sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    yield engine
    engine.dispose()


@pytest.fixture()
def db_session(test_engine):
    """A session bound directly to the per-test engine, for tests that hit the DB without going through the API."""
    TestingSessionLocal = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(test_engine):
    """A TestClient wired to the per-test in-memory database via dependency override."""
    from backend.main import app

    TestingSessionLocal = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)

    def override_get_db():
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
