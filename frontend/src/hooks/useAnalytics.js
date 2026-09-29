import { api } from "../api/client";
import { useResource } from "./useResource";

/**
 * One hook per backend endpoint. Cache keys live here so tabs that need the
 * same data (Overview and Performance both use the performance response)
 * share a single request instead of each making their own.
 */

export function usePerformance(portfolioId, riskFreeRate) {
  return useResource(
    portfolioId ? `performance:${portfolioId}:${riskFreeRate}` : null,
    () => api.getPerformance(portfolioId, riskFreeRate),
  );
}

export function useRisk(portfolioId, riskFreeRate) {
  return useResource(
    portfolioId ? `risk:${portfolioId}:${riskFreeRate}` : null,
    () => api.getRisk(portfolioId, riskFreeRate),
  );
}

export function useCorrelation(portfolioId) {
  return useResource(
    portfolioId ? `correlation:${portfolioId}` : null,
    () => api.getCorrelation(portfolioId),
  );
}

/** `bounds` is [min, max] as decimals, e.g. [0.05, 0.4]. */
export function useEfficientFrontier(portfolioId, riskFreeRate, bounds) {
  return useResource(
    portfolioId ? `frontier:${portfolioId}:${riskFreeRate}:${bounds[0]}:${bounds[1]}` : null,
    () =>
      api.getEfficientFrontier(portfolioId, {
        risk_free_rate: riskFreeRate,
        n_points: 40,
        weight_bounds: bounds,
      }),
  );
}

/**
 * `params` is null until the user runs a backtest, which keeps the hook idle
 * (no request) rather than firing an expensive computation on tab open.
 */
export function useBacktest(portfolioId, riskFreeRate, params) {
  const key =
    portfolioId && params
      ? `backtest:${portfolioId}:${riskFreeRate}:${params.splitDate}:${params.costRate}`
      : null;

  return useResource(key, () =>
    api.runBacktest(portfolioId, {
      split_date: params.splitDate,
      risk_free_rate: riskFreeRate,
      transaction_cost_rate: params.costRate,
    }),
  );
}
