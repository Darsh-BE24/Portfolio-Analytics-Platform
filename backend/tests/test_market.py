"""
Market configuration tests: ticker normalization for the Indian market, and the
guarantee that every module annualizes with the same trading-days constant.
"""

import numpy as np
import pandas as pd
import pytest

from backend import backtest, data, optimization, performance, risk
from backend.market import (
    DEFAULT_BENCHMARK,
    EXCHANGE_SUFFIX,
    TRADING_DAYS_PER_YEAR,
    normalize_ticker,
)


def test_defaults_target_nifty_50_on_nse():
    assert DEFAULT_BENCHMARK == "^NSEI"
    assert EXCHANGE_SUFFIX == ".NS"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("RELIANCE", "RELIANCE.NS"),  # bare symbol -> NSE
        ("  reliance  ", "RELIANCE.NS"),  # whitespace and case
        ("tcs", "TCS.NS"),
        ("NIFTYBEES", "NIFTYBEES.NS"),  # ETFs are plain symbols too
        ("M&M", "M&M.NS"),  # '&' is part of real NSE symbols
        ("BAJAJ-AUTO", "BAJAJ-AUTO.NS"),  # '-' is part of real NSE symbols
        ("RELIANCE.NS", "RELIANCE.NS"),  # explicit NSE suffix is kept
        ("reliance.bo", "RELIANCE.BO"),  # explicit BSE suffix is kept
        ("^NSEI", "^NSEI"),  # indices are never suffixed
        ("^bsesn", "^BSESN"),
    ],
)
def test_normalize_ticker(raw, expected):
    assert normalize_ticker(raw) == expected


@pytest.mark.parametrize("raw", ["RELIANCE", "reliance.bo", "^nsei", "M&M"])
def test_normalize_ticker_is_idempotent(raw):
    once = normalize_ticker(raw)
    assert normalize_ticker(once) == once


@pytest.mark.parametrize("raw", ["", "   "])
def test_empty_input_stays_empty_so_validation_can_report_it(raw):
    assert normalize_ticker(raw) == ""


def test_every_module_shares_the_one_trading_days_constant():
    for module in (performance, risk, optimization, backtest, data):
        assert module.TRADING_DAYS_PER_YEAR == TRADING_DAYS_PER_YEAR, module.__name__


def test_default_annualization_uses_the_constant():
    returns = pd.Series(np.random.RandomState(7).normal(0.0004, 0.01, 400))

    assert risk.annualized_volatility(returns) == pytest.approx(returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR))
    assert optimization.expected_returns(pd.DataFrame({"A": returns}))["A"] == pytest.approx(
        returns.mean() * TRADING_DAYS_PER_YEAR
    )
