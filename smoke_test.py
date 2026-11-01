"""
End-to-end smoke test for the running Portfolio Analytics API (Indian market).

Start the backend first (from the project root, venv active):

    python -m uvicorn backend.main:app --reload

then, in another PowerShell window (same venv):

    python smoke_test.py
    python smoke_test.py --base-url http://127.0.0.1:8000 --holdings "RELIANCE:10,TCS:5,HDFCBANK:8,INFY:4"

It uses LIVE Yahoo Finance data through your running backend, creates one portfolio, calls every
endpoint, and prints PASS / FAIL for each check. Standard library only. Exit code 0 = all passed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import urllib.error
import urllib.request
from datetime import date, timedelta

RESULTS: list[tuple[bool, str, str]] = []


def record(ok: bool, name: str, detail: str = "") -> bool:
    RESULTS.append((ok, name, detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def call(base: str, method: str, path: str, body: dict | None = None, timeout: float = 180.0):
    """Return (status_code, parsed_json_or_text, seconds)."""
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(
        base + path, data=data, method=method, headers={"Content-Type": "application/json"} if data else {}
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, raw = response.status, response.read().decode()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read().decode()
    except urllib.error.URLError as exc:
        raise SystemExit(f"Cannot reach {base}: {exc.reason}\nIs the backend running? (python -m uvicorn backend.main:app)")
    elapsed = time.perf_counter() - started
    try:
        return status, json.loads(raw), elapsed
    except json.JSONDecodeError:
        return status, raw, elapsed


def finite(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--holdings", default="RELIANCE:10,TCS:5,HDFCBANK:8,INFY:4", help='"SYMBOL:QTY,SYMBOL:QTY,..."')
    parser.add_argument("--risk-free", type=float, default=0.07)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")

    holdings = []
    for part in args.holdings.split(","):
        symbol, qty = part.split(":")
        holdings.append({"ticker": symbol.strip(), "quantity": float(qty)})
    rf = args.risk_free

    # --- health ---
    status, body, _ = call(base, "GET", "/health")
    record(status == 200, "GET /health", f"status {status}")

    # --- create (benchmark omitted on purpose: exercises the Nifty 50 default) ---
    status, created, secs = call(base, "POST", "/portfolio", {"holdings": holdings, "lookback_years": 3})
    if not record(status == 200, "POST /portfolio (live Yahoo prices)", f"status {status}, {secs:.1f}s"):
        print("      response:", created)
        print("\nCannot continue without a portfolio.")
        return summarize()
    pid = created["id"]
    positions = created["positions"]
    record(created["benchmark"] == "^NSEI", "default benchmark is ^NSEI", f"got {created['benchmark']}")
    record(
        all(p["ticker"].endswith((".NS", ".BO")) for p in positions),
        "bare tickers were given an exchange suffix",
        ", ".join(p["ticker"] for p in positions),
    )
    record(abs(sum(p["weight"] for p in positions) - 1.0) < 1e-6, "weights sum to 1")
    record(created["total_value"] > 0 and all(finite(p["price"]) and p["price"] > 0 for p in positions),
           "prices are positive rupee values", "; ".join(f"{p['ticker']}={p['price']:.2f}" for p in positions))

    status, again, secs2 = call(base, "POST", "/portfolio", {"holdings": holdings, "lookback_years": 3})
    record(status == 200, "second POST /portfolio (latest-price cache)", f"{secs2:.1f}s vs {secs:.1f}s first time")

    status, fetched, _ = call(base, "GET", f"/portfolio/{pid}")
    record(status == 200 and fetched["id"] == pid, "GET /portfolio/{id}")

    # --- analytics ---
    status, perf, _ = call(base, "GET", f"/portfolio/{pid}/performance?risk_free_rate={rf}")
    ok = status == 200 and all(finite(perf.get(k)) for k in ("cagr", "sharpe_ratio", "sortino_ratio"))
    record(ok, "GET /performance", f"cagr={perf.get('cagr')}, sharpe={perf.get('sharpe_ratio')}" if status == 200 else str(perf))

    status, risk, _ = call(base, "GET", f"/portfolio/{pid}/risk?risk_free_rate={rf}&var_confidence=0.95")
    record(status == 200, "GET /risk", f"status {status}" + ("" if status == 200 else f": {risk}"))

    status, corr, _ = call(base, "GET", f"/portfolio/{pid}/correlation")
    n = len(corr["tickers"]) if status == 200 else 0
    ok = status == 200 and len(corr["matrix"]) == n and all(abs(corr["matrix"][i][i] - 1) < 1e-6 for i in range(n))
    record(ok, "GET /correlation (square, diagonal = 1)", f"{n} assets")

    for method in ("min_volatility", "max_sharpe"):
        status, opt, _ = call(base, "POST", f"/portfolio/{pid}/optimize", {"method": method, "risk_free_rate": rf})
        ok = status == 200 and abs(sum(opt["weights"].values()) - 1) < 1e-4 and finite(opt["volatility"])
        record(ok, f"POST /optimize {method}", "" if status == 200 else str(opt))

    status, fr, _ = call(base, "POST", f"/portfolio/{pid}/efficient-frontier", {"risk_free_rate": rf, "n_points": 20})
    if status == 200:
        pts = fr["points"]
        ok = len(pts) >= 2 and fr["min_volatility_portfolio"]["volatility"] <= fr["max_sharpe_portfolio"]["volatility"] + 1e-9
        record(ok, "POST /efficient-frontier", f"{len(pts)} points; min-vol <= max-Sharpe volatility")
    else:
        record(False, "POST /efficient-frontier", str(fr))

    split = (date.today() - timedelta(days=365)).isoformat()
    status, bt, _ = call(base, "POST", f"/portfolio/{pid}/backtest",
                         {"split_date": split, "risk_free_rate": rf, "transaction_cost_rate": 0.001})
    if status == 200:
        names = set(bt["strategies"])
        record({"current", "equal_weight", "min_volatility", "max_sharpe", "benchmark"} <= names,
               f"POST /backtest (split {split})", ", ".join(sorted(names)))
    else:
        record(False, f"POST /backtest (split {split})", str(bt))

    # --- error handling ---
    status, _, _ = call(base, "POST", "/portfolio", {"holdings": [{"ticker": "ZZZNOTAREALSTOCK", "quantity": 1}]})
    record(status == 400, "unknown ticker is rejected with 400", f"status {status}")
    status, dup, _ = call(base, "POST", "/portfolio",
                          {"holdings": [{"ticker": "reliance", "quantity": 1}, {"ticker": "RELIANCE.NS", "quantity": 1}]})
    record(status == 422 and "Duplicate" in json.dumps(dup), "reliance + RELIANCE.NS is rejected as a duplicate", f"status {status}")
    status, _, _ = call(base, "GET", "/portfolio/does-not-exist")
    record(status == 404, "unknown portfolio id gives 404", f"status {status}")

    return summarize()


def summarize() -> int:
    failed = [r for r in RESULTS if not r[0]]
    print(f"\n{len(RESULTS) - len(failed)} passed, {len(failed)} failed")
    for _, name, detail in failed:
        print(f"  FAILED: {name} {detail}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
