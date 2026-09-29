/**
 * Formatting helpers. Backend returns ratios as decimals (0.052 = 5.2%);
 * these convert them for display. All handle null/NaN safely so a missing
 * value renders as an em dash instead of "NaN%".
 */

const MISSING = "\u2014";

function isMissing(value) {
  return value === null || value === undefined || Number.isNaN(value) || !Number.isFinite(value);
}

export function formatPercent(value, digits = 2) {
  if (isMissing(value)) return MISSING;
  return `${(value * 100).toFixed(digits)}%`;
}

export function formatSignedPercent(value, digits = 2) {
  if (isMissing(value)) return MISSING;
  const pct = (value * 100).toFixed(digits);
  return value > 0 ? `+${pct}%` : `${pct}%`;
}

export function formatNumber(value, digits = 2) {
  if (isMissing(value)) return MISSING;
  return value.toFixed(digits);
}

export function formatCurrency(value, digits = 2) {
  if (isMissing(value)) return MISSING;
  return value.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/** CSS class for coloring a signed figure. Zero and missing values stay neutral. */
export function signClass(value) {
  if (isMissing(value) || value === 0) return "";
  return value > 0 ? "positive" : "negative";
}

/**
 * The most recent single-period return, derived from a cumulative return
 * series (each point is total return since the start, as a decimal):
 *
 *   r_t = (1 + C_t) / (1 + C_{t-1}) - 1
 *
 * Returns null when fewer than two points exist.
 */
export function latestPeriodReturn(cumulativeSeries) {
  if (!Array.isArray(cumulativeSeries) || cumulativeSeries.length < 2) return null;
  const last = cumulativeSeries[cumulativeSeries.length - 1].value;
  const prev = cumulativeSeries[cumulativeSeries.length - 2].value;
  if (1 + prev === 0) return null;
  return (1 + last) / (1 + prev) - 1;
}
