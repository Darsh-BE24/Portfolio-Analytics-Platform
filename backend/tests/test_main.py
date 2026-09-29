import datetime

import pytest


def _create_portfolio(client, holdings=None, benchmark="^GSPC", lookback_years=2.0):
    holdings = holdings or [
        {"ticker": "AAPL", "quantity": 10},
        {"ticker": "MSFT", "quantity": 5},
        {"ticker": "NVDA", "quantity": 8},
        {"ticker": "AMZN", "quantity": 4},
    ]
    return client.post(
        "/portfolio",
        json={"holdings": holdings, "benchmark": benchmark, "lookback_years": lookback_years},
    )


# --- POST /portfolio ---

def test_create_portfolio_returns_200_with_positions(client):
    response = _create_portfolio(client)
    assert response.status_code == 200

    body = response.json()
    assert "id" in body
    assert len(body["positions"]) == 4
    assert body["total_value"] > 0

    weight_sum = sum(p["weight"] for p in body["positions"])
    assert weight_sum == pytest.approx(1.0, abs=1e-6)


def test_create_portfolio_invalid_ticker_returns_400(client):
    response = _create_portfolio(client, holdings=[{"ticker": "$$$", "quantity": 10}])
    assert response.status_code == 400


def test_create_portfolio_empty_holdings_returns_422(client):
    response = client.post("/portfolio", json={"holdings": [], "benchmark": "^GSPC"})
    assert response.status_code == 422


def test_create_portfolio_negative_quantity_returns_422(client):
    response = _create_portfolio(client, holdings=[{"ticker": "AAPL", "quantity": -5}])
    assert response.status_code == 422


# --- GET /portfolio/{id} ---

def test_get_portfolio_by_id(client):
    created = _create_portfolio(client).json()
    response = client.get(f"/portfolio/{created['id']}")
    assert response.status_code == 200
    assert response.json()["id"] == created["id"]


def test_get_portfolio_not_found_returns_404(client):
    response = client.get("/portfolio/does-not-exist")
    assert response.status_code == 404


def test_portfolio_persists_across_separate_requests(client):
    created = _create_portfolio(client).json()
    response_1 = client.get(f"/portfolio/{created['id']}")
    response_2 = client.get(f"/portfolio/{created['id']}")
    assert response_1.status_code == 200
    assert response_2.status_code == 200
    assert response_1.json()["id"] == response_2.json()["id"]


# --- GET /portfolio/{id}/performance ---

def test_get_performance(client):
    created = _create_portfolio(client).json()
    response = client.get(f"/portfolio/{created['id']}/performance", params={"risk_free_rate": 0.04})
    assert response.status_code == 200

    body = response.json()
    for key in ["cagr", "cumulative_return", "sharpe_ratio", "sortino_ratio"]:
        assert key in body


def test_get_performance_includes_time_series_for_portfolio_and_benchmark(client):
    created = _create_portfolio(client).json()
    body = client.get(f"/portfolio/{created['id']}/performance").json()

    portfolio_series = body["cumulative_return_series"]
    benchmark_series = body["benchmark_cumulative_return_series"]

    assert len(portfolio_series) > 0
    assert len(portfolio_series) == len(benchmark_series)  # aligned on shared dates
    assert set(portfolio_series[0].keys()) == {"date", "value"}

    # the series' final value should match the scalar cumulative_return field
    assert portfolio_series[-1]["value"] == pytest.approx(body["cumulative_return"])


def test_get_performance_not_found_returns_404(client):
    response = client.get("/portfolio/does-not-exist/performance")
    assert response.status_code == 404


# --- GET /portfolio/{id}/risk ---

def test_get_risk(client):
    created = _create_portfolio(client).json()
    response = client.get(f"/portfolio/{created['id']}/risk")
    assert response.status_code == 200

    body = response.json()
    for key in [
        "annualized_volatility", "beta", "max_drawdown", "current_drawdown",
        "historical_var_95", "parametric_var_95", "monte_carlo_var_95",
        "historical_cvar_95", "parametric_cvar_95", "hhi", "effective_number_of_holdings",
    ]:
        assert key in body
    assert body["max_drawdown"] <= 0


def test_get_risk_includes_drawdown_and_rolling_time_series(client):
    created = _create_portfolio(client, lookback_years=2.0).json()
    body = client.get(f"/portfolio/{created['id']}/risk").json()

    drawdown = body["drawdown_series"]
    rolling_vol = body["rolling_volatility"]
    rolling_sharpe = body["rolling_sharpe"]

    assert len(drawdown) > 0
    assert all(point["value"] <= 1e-9 for point in drawdown)  # drawdown is never positive

    # rolling series drop their leading NaN rows, so they're shorter than the full history
    assert 0 < len(rolling_vol) < len(drawdown)
    assert 0 < len(rolling_sharpe) < len(drawdown)
    assert all(point["value"] >= 0 for point in rolling_vol)  # volatility is never negative


# --- GET /portfolio/{id}/correlation ---

def test_get_correlation(client):
    created = _create_portfolio(client).json()
    response = client.get(f"/portfolio/{created['id']}/correlation")
    assert response.status_code == 200

    body = response.json()
    n = len(body["tickers"])
    assert len(body["matrix"]) == n
    assert all(len(row) == n for row in body["matrix"])
    for i in range(n):
        assert body["matrix"][i][i] == pytest.approx(1.0, abs=1e-6)


# --- POST /portfolio/{id}/optimize ---

def test_optimize_min_volatility(client):
    created = _create_portfolio(client).json()
    response = client.post(
        f"/portfolio/{created['id']}/optimize",
        json={"method": "min_volatility", "risk_free_rate": 0.04},
    )
    assert response.status_code == 200

    body = response.json()
    assert sum(body["weights"].values()) == pytest.approx(1.0, abs=1e-6)


def test_optimize_max_sharpe_with_custom_bounds(client):
    created = _create_portfolio(client).json()
    response = client.post(
        f"/portfolio/{created['id']}/optimize",
        json={"method": "max_sharpe", "risk_free_rate": 0.04, "weight_bounds": [0.05, 0.5]},
    )
    assert response.status_code == 200

    body = response.json()
    for w in body["weights"].values():
        assert 0.05 - 1e-6 <= w <= 0.5 + 1e-6


def test_optimize_invalid_method_returns_422(client):
    created = _create_portfolio(client).json()
    response = client.post(
        f"/portfolio/{created['id']}/optimize",
        json={"method": "not_a_real_method"},
    )
    assert response.status_code == 422


def test_optimize_infeasible_bounds_returns_400(client):
    created = _create_portfolio(client).json()
    response = client.post(
        f"/portfolio/{created['id']}/optimize",
        json={"method": "min_volatility", "weight_bounds": [0.4, 1.0]},
    )
    assert response.status_code == 400


def test_optimize_not_found_returns_404(client):
    response = client.post(
        "/portfolio/does-not-exist/optimize",
        json={"method": "min_volatility"},
    )
    assert response.status_code == 404


# --- POST /portfolio/{id}/efficient-frontier ---

def test_efficient_frontier(client):
    created = _create_portfolio(client).json()
    response = client.post(
        f"/portfolio/{created['id']}/efficient-frontier",
        json={"risk_free_rate": 0.04, "n_points": 15},
    )
    assert response.status_code == 200

    body = response.json()
    assert len(body["points"]) > 0
    assert "current_portfolio" in body
    assert "min_volatility_portfolio" in body
    assert "max_sharpe_portfolio" in body


# --- POST /portfolio/{id}/backtest ---

def test_backtest(client):
    created = _create_portfolio(client, lookback_years=3.0).json()

    split = (datetime.date.today() - datetime.timedelta(days=365)).isoformat()

    response = client.post(
        f"/portfolio/{created['id']}/backtest",
        json={"split_date": split, "risk_free_rate": 0.04, "transaction_cost_rate": 0.001},
    )
    assert response.status_code == 200

    body = response.json()
    for strategy in ["current", "equal_weight", "min_volatility", "max_sharpe", "benchmark"]:
        assert strategy in body["strategies"]


def test_backtest_not_found_returns_404(client):
    response = client.post(
        "/portfolio/does-not-exist/backtest",
        json={"split_date": "2024-01-01"},
    )
    assert response.status_code == 404


# --- health check ---

def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


# --- CORS (required for the React frontend to call this API from the browser) ---

def test_cors_allows_vite_dev_server_origin(client):
    response = client.get("/health", headers={"Origin": "http://localhost:5173"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_preflight_request_succeeds_for_allowed_origin(client):
    response = client.options(
        "/portfolio",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_cors_does_not_allow_unlisted_origin(client):
    response = client.get("/health", headers={"Origin": "http://evil.example.com"})
    assert "access-control-allow-origin" not in response.headers
