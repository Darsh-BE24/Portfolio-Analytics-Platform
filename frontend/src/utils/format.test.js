import { describe, expect, it } from "vitest";
import {
  formatCurrency,
  formatNumber,
  formatPercent,
  formatSignedPercent,
  latestPeriodReturn,
  signClass,
} from "./format";

describe("formatPercent", () => {
  it("converts a decimal ratio to a percentage", () => {
    expect(formatPercent(0.052)).toBe("5.20%");
  });

  it("respects the digits argument", () => {
    expect(formatPercent(0.052, 1)).toBe("5.2%");
  });

  it("keeps the sign on negative values", () => {
    expect(formatPercent(-0.2134)).toBe("-21.34%");
  });

  it.each([null, undefined, NaN, Infinity])("renders %s as an em dash", (value) => {
    expect(formatPercent(value)).toBe("\u2014");
  });
});

describe("formatSignedPercent", () => {
  it("adds an explicit plus for gains", () => {
    expect(formatSignedPercent(0.052)).toBe("+5.20%");
  });

  it("does not double up the sign on losses", () => {
    expect(formatSignedPercent(-0.052)).toBe("-5.20%");
  });

  it("shows zero without a sign", () => {
    expect(formatSignedPercent(0)).toBe("0.00%");
  });
});

describe("formatNumber and formatCurrency", () => {
  it("formats plain numbers", () => {
    expect(formatNumber(1.23456)).toBe("1.23");
    expect(formatNumber(1.23456, 3)).toBe("1.235");
  });

  it("formats US currency with thousands separators", () => {
    expect(formatCurrency(6130.5)).toBe("$6,130.50");
  });

  it("handles missing values", () => {
    expect(formatNumber(null)).toBe("\u2014");
    expect(formatCurrency(undefined)).toBe("\u2014");
  });
});

describe("signClass", () => {
  it("classifies by sign", () => {
    expect(signClass(0.01)).toBe("positive");
    expect(signClass(-0.01)).toBe("negative");
  });

  it("leaves zero and missing values neutral", () => {
    expect(signClass(0)).toBe("");
    expect(signClass(null)).toBe("");
    expect(signClass(NaN)).toBe("");
  });
});

describe("latestPeriodReturn", () => {
  it("derives the last single-period return from cumulative returns", () => {
    // Cumulative +10% then +21%: the last period grew 1.21 / 1.10 - 1 = 10%
    const series = [
      { date: "2024-01-02", value: 0.1 },
      { date: "2024-01-03", value: 0.21 },
    ];
    expect(latestPeriodReturn(series)).toBeCloseTo(0.1, 10);
  });

  it("only looks at the last two points", () => {
    const series = [
      { date: "a", value: 0.5 },
      { date: "b", value: 0.0 },
      { date: "c", value: 0.05 },
    ];
    expect(latestPeriodReturn(series)).toBeCloseTo(0.05, 10);
  });

  it("returns null when there are fewer than two points", () => {
    expect(latestPeriodReturn([{ date: "a", value: 0.1 }])).toBeNull();
    expect(latestPeriodReturn([])).toBeNull();
    expect(latestPeriodReturn(undefined)).toBeNull();
  });

  it("returns null instead of dividing by zero after a total loss", () => {
    const series = [
      { date: "a", value: -1 },
      { date: "b", value: 0.5 },
    ];
    expect(latestPeriodReturn(series)).toBeNull();
  });
});
