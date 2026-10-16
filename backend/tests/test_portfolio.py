import pytest

from backend.portfolio import Portfolio, PortfolioValidationError

SAMPLE_HOLDINGS = {"AAPL": 10, "MSFT": 5, "NVDA": 8, "AMZN": 4}
SAMPLE_PRICES = {"AAPL": 230.0, "MSFT": 415.0, "NVDA": 118.0, "AMZN": 186.0}


# --- validate() ---

def test_valid_portfolio_does_not_raise():
    Portfolio({"AAPL": 10, "MSFT": 5}).validate()


def test_empty_portfolio_raises():
    with pytest.raises(PortfolioValidationError):
        Portfolio({}).validate()


def test_zero_quantity_raises():
    with pytest.raises(PortfolioValidationError):
        Portfolio({"AAPL": 0}).validate()


def test_negative_quantity_raises():
    with pytest.raises(PortfolioValidationError):
        Portfolio({"AAPL": -5}).validate()


def test_non_numeric_quantity_raises():
    with pytest.raises(PortfolioValidationError):
        Portfolio({"AAPL": "ten"}).validate()


def test_invalid_ticker_characters_raise():
    with pytest.raises(PortfolioValidationError):
        Portfolio({"$$$": 10}).validate()


def test_ticker_with_dot_is_valid():
    Portfolio({"BRK.B": 3}).validate()


def test_ticker_with_dash_is_valid():
    Portfolio({"BF-B": 3}).validate()


def test_ticker_with_caret_is_valid():
    Portfolio({"^GSPC": 1}).validate()


def test_lowercase_ticker_is_normalized():
    p = Portfolio({"aapl": 10})
    assert "AAPL" in p.holdings
    p.validate()


def test_ticker_whitespace_is_normalized():
    p = Portfolio({"  AAPL  ": 10})
    assert "AAPL" in p.holdings


def test_error_message_names_the_offending_ticker():
    with pytest.raises(PortfolioValidationError, match="AAPL"):
        Portfolio({"AAPL": -5}).validate()


# --- value() / total_value() / weights() ---

def test_position_values_are_price_times_quantity():
    valuation = Portfolio(SAMPLE_HOLDINGS).value(SAMPLE_PRICES)
    by_ticker = {v.ticker: v for v in valuation}

    assert by_ticker["AAPL"].value == pytest.approx(230.0 * 10)
    assert by_ticker["MSFT"].value == pytest.approx(415.0 * 5)
    assert by_ticker["NVDA"].value == pytest.approx(118.0 * 8)
    assert by_ticker["AMZN"].value == pytest.approx(186.0 * 4)


def test_weights_sum_to_one():
    weights = Portfolio(SAMPLE_HOLDINGS).weights(SAMPLE_PRICES)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_weights_are_proportional_to_value():
    valuation = Portfolio(SAMPLE_HOLDINGS).value(SAMPLE_PRICES)
    total = sum(v.value for v in valuation)
    for v in valuation:
        assert v.weight == pytest.approx(v.value / total)


def test_total_value_matches_sum_of_positions():
    p = Portfolio(SAMPLE_HOLDINGS)
    valuation = p.value(SAMPLE_PRICES)
    assert p.total_value(SAMPLE_PRICES) == pytest.approx(sum(v.value for v in valuation))


def test_single_asset_portfolio_has_weight_one():
    weights = Portfolio({"AAPL": 10}).weights({"AAPL": 230.0})
    assert weights["AAPL"] == pytest.approx(1.0)


def test_missing_price_raises():
    incomplete_prices = {"AAPL": 230.0, "MSFT": 415.0}  # NVDA, AMZN missing
    with pytest.raises(PortfolioValidationError):
        Portfolio(SAMPLE_HOLDINGS).value(incomplete_prices)


def test_value_raises_if_portfolio_invalid():
    # value() must call validate() internally -- a bad portfolio should
    # never reach the arithmetic, even if prices are provided.
    with pytest.raises(PortfolioValidationError):
        Portfolio({"AAPL": -5}).value({"AAPL": 230.0})
