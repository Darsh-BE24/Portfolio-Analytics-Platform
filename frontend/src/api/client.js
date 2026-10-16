/**
 * API client: the only module that talks to the FastAPI backend.
 * Components never call fetch() directly.
 */

const BASE_URL = (import.meta.env?.VITE_API_BASE_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(message, status) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * FastAPI reports errors in two shapes:
 *   - our own 400/404 responses:  { "detail": "Some readable message" }
 *   - Pydantic 422 validation:    { "detail": [ { "loc": [...], "msg": "..." }, ... ] }
 * Both are flattened into one readable string here.
 */
export function extractErrorMessage(body, fallback) {
  const detail = body?.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail) && detail.length > 0) {
    return detail
      .map((item) => {
        const field = Array.isArray(item.loc) ? item.loc.filter((p) => p !== "body").join(".") : "";
        return field ? `${field}: ${item.msg}` : item.msg;
      })
      .join("; ");
  }
  return fallback;
}

async function request(path, { method = "GET", body, params } = {}) {
  let url = `${BASE_URL}${path}`;
  if (params) {
    const query = new URLSearchParams();
    Object.entries(params).forEach(([key, value]) => {
      if (value !== undefined && value !== null) query.set(key, String(value));
    });
    const qs = query.toString();
    if (qs) url += `?${qs}`;
  }

  let response;
  try {
    response = await fetch(url, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
    });
  } catch {
    throw new ApiError(
      `Could not reach the analytics server at ${BASE_URL}. Check that the backend is running.`,
      0,
    );
  }

  let payload = null;
  try {
    payload = await response.json();
  } catch {
    // non-JSON body; handled below
  }

  if (!response.ok) {
    throw new ApiError(
      extractErrorMessage(payload, `Request failed with status ${response.status}.`),
      response.status,
    );
  }

  return payload;
}

export const api = {
  createPortfolio: (payload) => request("/portfolio", { method: "POST", body: payload }),
  getPortfolio: (id) => request(`/portfolio/${id}`),
  getPerformance: (id, riskFreeRate) =>
    request(`/portfolio/${id}/performance`, { params: { risk_free_rate: riskFreeRate } }),
  getRisk: (id, riskFreeRate) =>
    request(`/portfolio/${id}/risk`, { params: { risk_free_rate: riskFreeRate } }),
  getCorrelation: (id) => request(`/portfolio/${id}/correlation`),
  optimize: (id, payload) => request(`/portfolio/${id}/optimize`, { method: "POST", body: payload }),
  getEfficientFrontier: (id, payload) =>
    request(`/portfolio/${id}/efficient-frontier`, { method: "POST", body: payload }),
  runBacktest: (id, payload) => request(`/portfolio/${id}/backtest`, { method: "POST", body: payload }),
};
