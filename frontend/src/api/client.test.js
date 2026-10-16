import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { api, ApiError, extractErrorMessage } from "./client";

function mockFetch(response) {
  const fn = vi.fn().mockResolvedValue(response);
  vi.stubGlobal("fetch", fn);
  return fn;
}

function jsonResponse(body, status = 200) {
  return { ok: status >= 200 && status < 300, status, json: async () => body };
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("extractErrorMessage", () => {
  it("returns a string detail as-is (our own 400/404 responses)", () => {
    expect(extractErrorMessage({ detail: "No portfolio found." }, "fallback")).toBe("No portfolio found.");
  });

  it("flattens Pydantic 422 validation arrays", () => {
    const body = {
      detail: [
        { loc: ["body", "holdings", 0, "quantity"], msg: "Input should be greater than 0" },
        { loc: ["body", "lookback_years"], msg: "Input should be less than or equal to 10" },
      ],
    };
    expect(extractErrorMessage(body, "fallback")).toBe(
      "holdings.0.quantity: Input should be greater than 0; lookback_years: Input should be less than or equal to 10",
    );
  });

  it("falls back when the body has no usable detail", () => {
    expect(extractErrorMessage(null, "fallback")).toBe("fallback");
    expect(extractErrorMessage({}, "fallback")).toBe("fallback");
    expect(extractErrorMessage({ detail: [] }, "fallback")).toBe("fallback");
  });
});

describe("api requests", () => {
  beforeEach(() => {
    vi.stubGlobal("fetch", vi.fn());
  });

  it("POSTs JSON to create a portfolio", async () => {
    const fetchMock = mockFetch(jsonResponse({ id: "abc" }));
    const payload = { holdings: [{ ticker: "AAPL", quantity: 1 }], benchmark: "^GSPC", lookback_years: 3 };

    const result = await api.createPortfolio(payload);

    expect(result).toEqual({ id: "abc" });
    const [url, options] = fetchMock.mock.calls[0];
    expect(url).toMatch(/\/portfolio$/);
    expect(options.method).toBe("POST");
    expect(options.headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(options.body)).toEqual(payload);
  });

  it("sends the risk-free rate as a query parameter", async () => {
    const fetchMock = mockFetch(jsonResponse({}));
    await api.getPerformance("p1", 0.04);
    expect(fetchMock.mock.calls[0][0]).toMatch(/\/portfolio\/p1\/performance\?risk_free_rate=0\.04$/);
  });

  it("throws an ApiError carrying the server's message and status", async () => {
    mockFetch(jsonResponse({ detail: "No portfolio found with id 'x'." }, 404));

    await expect(api.getPortfolio("x")).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
      message: "No portfolio found with id 'x'.",
    });
  });

  it("reports an unreachable server in plain language", async () => {
    vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));

    const error = await api.getPortfolio("x").catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error.status).toBe(0);
    expect(error.message).toMatch(/Could not reach the analytics server/);
  });

  it("falls back to a status message when an error body isn't JSON", async () => {
    mockFetch({ ok: false, status: 500, json: async () => { throw new Error("not json"); } });
    await expect(api.getPortfolio("x")).rejects.toMatchObject({ message: "Request failed with status 500." });
  });
});
