/**
 * Parses the per-position min/max limits typed as percentages and checks
 * they are feasible for a portfolio of `holdingCount` positions.
 *
 * Mirrors the backend's rules so users see a plain-language message
 * immediately instead of a round trip ending in a 400:
 *   - every weight must lie in [0, 100]%
 *   - min <= max
 *   - holdingCount * min <= 100%   (the minimums must fit in a full portfolio)
 *   - holdingCount * max >= 100%   (the maximums must be able to fill it)
 *
 * @returns {{bounds: [number, number] | null, error: string | null}}
 *          bounds are decimals (0.05 = 5%), ready to send to the API.
 */
export function parseWeightBounds(minPercent, maxPercent, holdingCount) {
  const min = Number(String(minPercent).trim());
  const max = Number(String(maxPercent).trim());

  if (String(minPercent).trim() === "" || String(maxPercent).trim() === "" || !Number.isFinite(min) || !Number.isFinite(max)) {
    return { bounds: null, error: "Enter both limits as numbers, such as 5 and 40." };
  }
  if (min < 0 || max > 100) {
    return { bounds: null, error: "Limits must be between 0% and 100%." };
  }
  if (min > max) {
    return { bounds: null, error: "The minimum can't be higher than the maximum." };
  }
  if (holdingCount * min > 100) {
    return {
      bounds: null,
      error: `${holdingCount} positions at ${min}% each already exceed 100%. Lower the minimum.`,
    };
  }
  if (holdingCount * max < 100) {
    return {
      bounds: null,
      error: `${holdingCount} positions capped at ${max}% each can't add up to 100%. Raise the maximum.`,
    };
  }

  return { bounds: [min / 100, max / 100], error: null };
}
