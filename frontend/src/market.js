/**
 * Market configuration for the frontend: the one place that says which market
 * the UI targets (mirrors backend/market.py). Indian equities, NSE by default.
 */

export const DEFAULT_BENCHMARK = "^NSEI"; // Nifty 50
export const EXCHANGE_SUFFIX = ".NS"; // appended to bare tickers; type ".BO" explicitly for BSE
export const CURRENCY = "INR";
export const LOCALE = "en-IN"; // lakh/crore digit grouping

// Default annual risk-free rate shown in the header, in percent. About the yield
// on the 10-year Government of India security in early October 2026 (the 91-day
// T-bill was nearer 5.5%). It is only a default: the user can change it.
export const DEFAULT_RISK_FREE_PERCENT = 7;

/**
 * Turn what the user typed into the symbol the API expects. Same rules as
 * normalize_ticker in backend/market.py: trim and upper-case; leave indices
 * ("^NSEI") and symbols that already carry an exchange suffix ("RELIANCE.BO")
 * alone; otherwise append the NSE suffix. Empty input stays empty.
 */
export function normalizeTicker(raw) {
  const ticker = String(raw ?? "").trim().toUpperCase();
  if (!ticker || ticker.startsWith("^") || ticker.includes(".")) return ticker;
  return ticker + EXCHANGE_SUFFIX;
}
