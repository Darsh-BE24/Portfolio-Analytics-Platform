import { describe, expect, it } from "vitest";
import { parseWeightBounds } from "./bounds";
import { validateHoldings } from "./holdings";

describe("validateHoldings", () => {
  it("accepts valid rows, upper-cases tickers and adds the NSE suffix", () => {
    const { holdings, errors } = validateHoldings([
      { ticker: " reliance ", quantity: "10" },
      { ticker: "TCS", quantity: "2.5" },
    ]);
    expect(errors).toEqual([]);
    expect(holdings).toEqual([
      { ticker: "RELIANCE.NS", quantity: 10 },
      { ticker: "TCS.NS", quantity: 2.5 },
    ]);
  });

  it("keeps an explicit exchange suffix and never suffixes an index", () => {
    const { holdings, errors } = validateHoldings([
      { ticker: "reliance.bo", quantity: "1" },
      { ticker: "TCS.NS", quantity: "1" },
      { ticker: "^NSEI", quantity: "1" },
    ]);
    expect(errors).toEqual([]);
    expect(holdings.map((h) => h.ticker)).toEqual(["RELIANCE.BO", "TCS.NS", "^NSEI"]);
  });

  it("accepts NSE symbols that contain & and -", () => {
    const { holdings, errors } = validateHoldings([
      { ticker: "m&m", quantity: "1" },
      { ticker: "BAJAJ-AUTO", quantity: "1" },
    ]);
    expect(errors).toEqual([]);
    expect(holdings.map((h) => h.ticker)).toEqual(["M&M.NS", "BAJAJ-AUTO.NS"]);
  });

  it("accepts tickers containing . ^ and -", () => {
    const { errors } = validateHoldings([
      { ticker: "BRK.B", quantity: "1" },
      { ticker: "BF-B", quantity: "1" },
      { ticker: "^GSPC", quantity: "1" },
    ]);
    expect(errors).toEqual([]);
  });

  it("rejects duplicate tickers instead of silently dropping one", () => {
    const { holdings, errors } = validateHoldings([
      { ticker: "RELIANCE", quantity: "10" },
      { ticker: "reliance", quantity: "5" },
    ]);
    expect(errors).toHaveLength(1);
    expect(errors[0]).toMatch(/RELIANCE\.NS appears more than once/);
    expect(holdings).toEqual([]);
  });

  it("treats a bare symbol and the same symbol with .NS as duplicates", () => {
    const { errors } = validateHoldings([
      { ticker: "TCS", quantity: "10" },
      { ticker: "tcs.ns", quantity: "5" },
    ]);
    expect(errors).toHaveLength(1);
    expect(errors[0]).toMatch(/TCS\.NS appears more than once/);
  });

  it("does not treat the NSE and BSE listings as duplicates", () => {
    const { errors } = validateHoldings([
      { ticker: "TCS.NS", quantity: "10" },
      { ticker: "TCS.BO", quantity: "5" },
    ]);
    expect(errors).toEqual([]);
  });

  it("rejects zero and negative quantities", () => {
    expect(validateHoldings([{ ticker: "AAPL", quantity: "0" }]).errors[0]).toMatch(/greater than zero/);
    expect(validateHoldings([{ ticker: "AAPL", quantity: "-3" }]).errors[0]).toMatch(/greater than zero/);
  });

  it("rejects a missing or non-numeric quantity", () => {
    expect(validateHoldings([{ ticker: "AAPL", quantity: "" }]).errors[0]).toMatch(/needs a quantity/);
    expect(validateHoldings([{ ticker: "AAPL", quantity: "ten" }]).errors[0]).toMatch(/needs a quantity/);
  });

  it("rejects a quantity without a ticker", () => {
    expect(validateHoldings([{ ticker: "", quantity: "5" }]).errors[0]).toMatch(/needs a ticker/);
  });

  it("rejects tickers with illegal characters", () => {
    expect(validateHoldings([{ ticker: "$$$", quantity: "1" }]).errors[0]).toMatch(/"\$\$\$" isn't a valid ticker/);
  });

  it("flags fully empty rows", () => {
    expect(validateHoldings([{ ticker: "", quantity: "" }]).errors[0]).toMatch(/Row 1 is empty/);
  });

  it("reports every problem at once, not just the first", () => {
    const { errors } = validateHoldings([
      { ticker: "AAPL", quantity: "-1" },
      { ticker: "", quantity: "5" },
    ]);
    expect(errors).toHaveLength(2);
  });
});

describe("parseWeightBounds", () => {
  it("converts percentages to decimals", () => {
    expect(parseWeightBounds("5", "40", 4)).toEqual({ bounds: [0.05, 0.4], error: null });
  });

  it("accepts the unconstrained default", () => {
    expect(parseWeightBounds("0", "100", 4).bounds).toEqual([0, 1]);
  });

  it("rejects non-numeric and empty input", () => {
    expect(parseWeightBounds("abc", "40", 4).error).toMatch(/as numbers/);
    expect(parseWeightBounds("", "40", 4).error).toMatch(/as numbers/);
  });

  it("rejects values outside 0-100", () => {
    expect(parseWeightBounds("-5", "40", 4).error).toMatch(/between 0% and 100%/);
    expect(parseWeightBounds("0", "120", 4).error).toMatch(/between 0% and 100%/);
  });

  it("rejects a minimum above the maximum", () => {
    expect(parseWeightBounds("50", "20", 4).error).toMatch(/can't be higher/);
  });

  it("rejects minimums that cannot fit in the portfolio", () => {
    // 4 positions x 30% = 120% > 100%
    expect(parseWeightBounds("30", "100", 4).error).toMatch(/exceed 100%/);
  });

  it("rejects maximums that cannot fill the portfolio", () => {
    // 4 positions x 20% = 80% < 100%
    expect(parseWeightBounds("0", "20", 4).error).toMatch(/can't add up to 100%/);
  });

  it("accepts the exact boundary where limits just fill the portfolio", () => {
    // 4 x 25% = exactly 100%
    expect(parseWeightBounds("25", "25", 4).error).toBeNull();
  });
});
