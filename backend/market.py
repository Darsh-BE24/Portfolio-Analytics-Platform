"""
Market configuration: the one place that says which market this app targets.

The platform targets Indian equities (NSE by default). Everything that used to
assume the US market -- the default benchmark, the number of trading days used
to annualize, how a bare ticker is turned into a Yahoo Finance symbol -- lives
here, so switching markets is a change to this file rather than a hunt through
the codebase.

No imports, no I/O: this module is safe to import from anywhere.
"""

from __future__ import annotations

# Yahoo Finance symbols: NSE equities end in ".NS", BSE equities in ".BO", and
# indices start with "^" (e.g. "^NSEI" = Nifty 50, "^BSESN" = BSE Sensex).
DEFAULT_BENCHMARK = "^NSEI"  # Nifty 50
EXCHANGE_SUFFIX = ".NS"  # appended to bare tickers; type ".BO" explicitly for BSE

# Trading days per year used to annualize returns and risk. NSE trades roughly
# 245-250 days a year (it varies with the holiday calendar); 248 is a round
# middle estimate. Every module imports this one constant, so changing it here
# changes annualization everywhere consistently.
TRADING_DAYS_PER_YEAR = 248

CURRENCY = "INR"


def normalize_ticker(raw: str) -> str:
    """
    Turn user input into the symbol the data provider expects.

    - surrounding whitespace is stripped and the symbol is upper-cased
    - indices (leading "^", e.g. "^NSEI") are left alone
    - symbols that already carry an exchange suffix (contain "." e.g.
      "RELIANCE.NS", "RELIANCE.BO") are left alone
    - any other bare symbol gets EXCHANGE_SUFFIX ("RELIANCE" -> "RELIANCE.NS")
    - an empty input stays empty, so validation can report it as missing
      instead of it silently becoming just ".NS"

    Idempotent: normalizing an already-normalized symbol returns it unchanged.
    """
    ticker = raw.strip().upper()
    if not ticker or ticker.startswith("^") or "." in ticker:
        return ticker
    return ticker + EXCHANGE_SUFFIX
