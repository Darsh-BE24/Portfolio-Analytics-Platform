from datetime import date

from backend.models import MarketDataCache, PortfolioHolding, PortfolioModel, User


def test_create_user(db_session):
    user = User(username="alice")
    db_session.add(user)
    db_session.commit()

    fetched = db_session.get(User, user.id)
    assert fetched.username == "alice"
    assert fetched.id is not None  # UUID default applied


def test_create_portfolio_with_holdings(db_session):
    portfolio = PortfolioModel(benchmark_ticker="^GSPC", lookback_years=3.0)
    portfolio.holdings = [
        PortfolioHolding(ticker="AAPL", quantity=10),
        PortfolioHolding(ticker="MSFT", quantity=5),
    ]
    db_session.add(portfolio)
    db_session.commit()

    fetched = db_session.get(PortfolioModel, portfolio.id)
    assert len(fetched.holdings) == 2
    tickers = {h.ticker for h in fetched.holdings}
    assert tickers == {"AAPL", "MSFT"}


def test_portfolio_holdings_cascade_delete(db_session):
    portfolio = PortfolioModel(benchmark_ticker="^GSPC", lookback_years=3.0)
    portfolio.holdings = [PortfolioHolding(ticker="AAPL", quantity=10)]
    db_session.add(portfolio)
    db_session.commit()
    portfolio_id = portfolio.id

    db_session.delete(portfolio)
    db_session.commit()

    remaining_holdings = (
        db_session.query(PortfolioHolding).filter(PortfolioHolding.portfolio_id == portfolio_id).all()
    )
    assert remaining_holdings == []


def test_portfolio_defaults_applied(db_session):
    portfolio = PortfolioModel()  # no explicit benchmark/lookback
    db_session.add(portfolio)
    db_session.commit()

    fetched = db_session.get(PortfolioModel, portfolio.id)
    assert fetched.benchmark_ticker == "^GSPC"
    assert fetched.lookback_years == 3.0
    assert fetched.created_at is not None


def test_market_data_cache_unique_constraint(db_session):
    row1 = MarketDataCache(ticker="AAPL", date=date(2024, 1, 2), close_price=185.0)
    db_session.add(row1)
    db_session.commit()

    fetched = (
        db_session.query(MarketDataCache)
        .filter(MarketDataCache.ticker == "AAPL", MarketDataCache.date == date(2024, 1, 2))
        .one()
    )
    assert fetched.close_price == 185.0


def test_portfolio_without_owner_is_allowed(db_session):
    # user_id is nullable -- portfolios can exist with no authenticated owner
    portfolio = PortfolioModel()
    db_session.add(portfolio)
    db_session.commit()

    fetched = db_session.get(PortfolioModel, portfolio.id)
    assert fetched.user_id is None
