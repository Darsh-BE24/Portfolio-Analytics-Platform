"""
Tests for MarketDataService.

These monkeypatch yfinance.download so they run with no network access --
important for CI, and for anyone running this suite offline.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from backend import data as data_module
from backend.data import MarketDataError, MarketDataService


def _make_single_ticker_frame(n_days: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
    return pd.DataFrame(
        {
            "Open": np.linspace(100, 110, n_days),
            "High": np.linspace(101, 111, n_days),
            "Low": np.linspace(99, 109, n_days),
            "Close": np.linspace(100, 110, n_days),
            "Volume": np.linspace(1_000_000, 1_100_000, n_days),
        },
        index=idx,
    )


def _make_multi_ticker_frame(tickers: list[str], n_days: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
    fields = ["Open", "High", "Low", "Close", "Volume"]
    columns = pd.MultiIndex.from_product([tickers, fields])
    data = {}
    for i, ticker in enumerate(tickers):
        base = 100 + i * 50
        for field in fields:
            data[(ticker, field)] = np.linspace(base, base + 10, n_days)
    return pd.DataFrame(data, index=idx, columns=columns)


def test_get_price_data_single_ticker(monkeypatch):
    frame = _make_single_ticker_frame()
    monkeypatch.setattr(data_module.yf, "download", lambda **kwargs: frame)

    svc = MarketDataService()
    prices = svc.get_price_data(["AAPL"], date(2024, 1, 1), date(2024, 1, 15))

    assert list(prices.columns) == ["AAPL"]
    assert prices["AAPL"].iloc[0] == pytest.approx(100.0)


def test_get_price_data_multi_ticker(monkeypatch):
    tickers = ["AAPL", "MSFT"]
    frame = _make_multi_ticker_frame(tickers)
    monkeypatch.setattr(data_module.yf, "download", lambda **kwargs: frame)

    svc = MarketDataService()
    prices = svc.get_price_data(tickers, date(2024, 1, 1), date(2024, 1, 15))

    assert set(prices.columns) == set(tickers)
    assert prices["AAPL"].iloc[0] == pytest.approx(100.0)
    assert prices["MSFT"].iloc[0] == pytest.approx(150.0)


def test_get_price_data_empty_raises(monkeypatch):
    monkeypatch.setattr(data_module.yf, "download", lambda **kwargs: pd.DataFrame())

    svc = MarketDataService()
    with pytest.raises(MarketDataError):
        svc.get_price_data(["FAKETICKER"], date(2024, 1, 1), date(2024, 1, 15))


def test_get_price_data_all_nan_ticker_raises(monkeypatch):
    tickers = ["AAPL", "FAKETICKER"]
    frame = _make_multi_ticker_frame(tickers)
    frame[("FAKETICKER", "Close")] = np.nan  # simulate an invalid ticker
    monkeypatch.setattr(data_module.yf, "download", lambda **kwargs: frame)

    svc = MarketDataService()
    with pytest.raises(MarketDataError):
        svc.get_price_data(tickers, date(2024, 1, 1), date(2024, 1, 15))


def test_get_price_data_no_tickers_raises():
    svc = MarketDataService()
    with pytest.raises(MarketDataError):
        svc.get_price_data([], date(2024, 1, 1), date(2024, 1, 15))


def test_get_benchmark_data_uses_default_ticker(monkeypatch):
    frame = _make_single_ticker_frame()
    monkeypatch.setattr(data_module.yf, "download", lambda **kwargs: frame)

    svc = MarketDataService()
    benchmark = svc.get_benchmark_data(date(2024, 1, 1), date(2024, 1, 15))

    assert benchmark.name == "^GSPC"
    assert benchmark.iloc[0] == pytest.approx(100.0)


def test_get_latest_prices_returns_most_recent_close(monkeypatch):
    frame = _make_single_ticker_frame()
    monkeypatch.setattr(data_module.yf, "download", lambda **kwargs: frame)

    svc = MarketDataService()
    latest = svc.get_latest_prices(["AAPL"])

    assert latest["AAPL"] == pytest.approx(frame["Close"].iloc[-1])
