"""
Database engine and session configuration.

Defaults to a local SQLite file so the project runs with zero setup --
no Postgres server required just to try it out or run the test suite.
For real use, point DATABASE_URL at a Postgres instance, e.g.:

    export DATABASE_URL=postgresql://user:password@localhost:5432/portfolio_analytics

SQLAlchemy's ORM layer (models.py) is database-agnostic -- switching from
SQLite to Postgres requires no code changes, only this environment variable.
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./portfolio_analytics.db")

# SQLite requires this flag for use across multiple threads (FastAPI's
# TestClient and Uvicorn both may hand requests to different threads);
# Postgres has no such restriction, so the flag is omitted for it.
_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(DATABASE_URL, connect_args=_connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """FastAPI dependency: yields a request-scoped session, always closed after the request."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Create all tables that don't already exist. Safe to call repeatedly.
    For a production system this would be replaced by a migration tool
    (e.g. Alembic) so schema changes are versioned -- out of scope here,
    but worth knowing this is the simplified version of that.
    """
    from backend import models  # noqa: F401 -- ensures models are registered on Base before create_all

    Base.metadata.create_all(bind=engine)
