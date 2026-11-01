import { describe, expect, it } from "vitest";
import { CURRENCY, DEFAULT_BENCHMARK, DEFAULT_RISK_FREE_PERCENT, EXCHANGE_SUFFIX, LOCALE, normalizeTicker } from "./market";

describe("market defaults", () => {
  it("targets Nifty 50 on NSE, in rupees", () => {
    expect(DEFAULT_BENCHMARK).toBe("^NSEI");
    expect(EXCHANGE_SUFFIX).toBe(".NS");
    expect(CURRENCY).toBe("INR");
    expect(LOCALE).toBe("en-IN");
  });

  it("uses a default risk-free rate that the header field accepts (0-20%)", () => {
    expect(DEFAULT_RISK_FREE_PERCENT).toBeGreaterThanOrEqual(0);
    expect(DEFAULT_RISK_FREE_PERCENT).toBeLessThanOrEqual(20);
  });
});

// These cases mirror backend/tests/test_market.py so the two normalizers cannot drift apart.
describe("normalizeTicker", () => {
  it.each([
    ["RELIANCE", "RELIANCE.NS"],
    ["  reliance  ", "RELIANCE.NS"],
    ["tcs", "TCS.NS"],
    ["NIFTYBEES", "NIFTYBEES.NS"],
    ["M&M", "M&M.NS"],
    ["BAJAJ-AUTO", "BAJAJ-AUTO.NS"],
    ["RELIANCE.NS", "RELIANCE.NS"],
    ["reliance.bo", "RELIANCE.BO"],
    ["^NSEI", "^NSEI"],
    ["^bsesn", "^BSESN"],
  ])("normalizes %j to %j", (raw, expected) => {
    expect(normalizeTicker(raw)).toBe(expected);
  });

  it("is idempotent", () => {
    for (const raw of ["RELIANCE", "reliance.bo", "^nsei", "M&M"]) {
      const once = normalizeTicker(raw);
      expect(normalizeTicker(once)).toBe(once);
    }
  });

  it("leaves empty input empty so the form can report it", () => {
    expect(normalizeTicker("")).toBe("");
    expect(normalizeTicker("   ")).toBe("");
    expect(normalizeTicker(undefined)).toBe("");
  });
});
