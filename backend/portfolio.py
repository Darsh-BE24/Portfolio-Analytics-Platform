"""
Portfolio construction and valuation.

Pure calculation logic -- no yfinance imports here. Prices are passed in
(typically fetched via MarketDataService.get_latest_prices), which keeps
this module trivial to unit test with fixed, known inputs.

    Position Value_i = Price_i x Quantity_i
    Portfolio Value  = Sum(Position Value_i)
    Weight_i         = Position Value_i / Portfolio Value
"""

from dataclasses import dataclass


class PortfolioValidationError(Exception):
    """Raised when portfolio holdings fail validation."""


@dataclass(frozen=True)
class PositionValuation:
    ticker: str
    price: float
    quantity: float
    value: float
    weight: float


class Portfolio:
    """
    Represents a set of holdings and computes valuation/weights.

    Example:
        p = Portfolio({"AAPL": 10, "MSFT": 5, "NVDA": 8, "AMZN": 4})
        valuation = p.value(prices={"AAPL": 230.1, "MSFT": 415.2, "NVDA": 118.4, "AMZN": 186.3})
    """

    def __init__(self, holdings: dict[str, float]):
        # Normalize at the boundary so every method downstream can trust clean input.
        self.holdings = {ticker.upper().strip(): qty for ticker, qty in holdings.items()}

    def validate(self) -> None:
        """
        Validate holdings before any calculation is attempted.

        Checks:
        - portfolio is not empty
        - quantities are numeric and positive
        - tickers contain only characters real ticker symbols use
        """
        if not self.holdings:
            raise PortfolioValidationError("Portfolio has no holdings")

        valid_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ^-.1234567890"

        for ticker, quantity in self.holdings.items():
            if not isinstance(quantity, (int, float)):
                raise PortfolioValidationError(
                    f"Quantity for '{ticker}' must be a number, got {type(quantity)}"
                )

            if quantity <= 0:
                raise PortfolioValidationError(
                    f"Quantity for '{ticker}' must be positive, got {quantity}"
                )

            for char in ticker:
                if char not in valid_chars:
                    raise PortfolioValidationError(f"Invalid ticker symbol: '{ticker}'")

    @property
    def tickers(self) -> list[str]:
        return list(self.holdings.keys())

    def value(self, prices: dict[str, float]) -> list[PositionValuation]:
        """
        Compute position values and weights given a dict of current prices.

        Raises PortfolioValidationError if a price is missing for any holding.
        We never silently drop a position -- that would understate portfolio
        value and distort every other holding's weight, since weight is
        computed relative to the total.
        """
        self.validate()

        missing_prices = [t for t in self.holdings if t not in prices]
        if missing_prices:
            raise PortfolioValidationError(
                f"Missing price data for holding(s): {missing_prices}"
            )

        position_values: dict[str, float] = {
            ticker: prices[ticker] * qty for ticker, qty in self.holdings.items()
        }
        total_value = sum(position_values.values())

        if total_value <= 0:
            raise PortfolioValidationError("Total portfolio value is zero or negative.")

        return [
            PositionValuation(
                ticker=ticker,
                price=prices[ticker],
                quantity=qty,
                value=position_values[ticker],
                weight=position_values[ticker] / total_value,
            )
            for ticker, qty in self.holdings.items()
        ]

    def total_value(self, prices: dict[str, float]) -> float:
        return sum(v.value for v in self.value(prices))

    def weights(self, prices: dict[str, float]) -> dict[str, float]:
        return {v.ticker: v.weight for v in self.value(prices)}
