/**
 * Validates the raw rows typed into the portfolio form and converts them to
 * the payload shape the API expects.
 *
 * Duplicate tickers are rejected here, on purpose: if a list of holdings were
 * naively converted to a dictionary, the second entry would silently
 * overwrite the first and the user would never know a position vanished.
 *
 * @param {{ticker: string, quantity: string|number}[]} rows
 * @returns {{holdings: {ticker: string, quantity: number}[], errors: string[]}}
 */
export function validateHoldings(rows) {
  const errors = [];
  const holdings = [];
  const seen = new Map();

  rows.forEach((row, index) => {
    const position = index + 1;
    const ticker = String(row.ticker ?? "").trim().toUpperCase();
    const rawQuantity = String(row.quantity ?? "").trim();

    if (!ticker && !rawQuantity) {
      errors.push(`Row ${position} is empty. Fill it in or remove it.`);
      return;
    }
    if (!ticker) {
      errors.push(`Row ${position} needs a ticker.`);
      return;
    }
    if (!/^[A-Z0-9.^-]+$/.test(ticker)) {
      errors.push(`"${ticker}" isn't a valid ticker. Use letters, digits, and . ^ - only.`);
      return;
    }

    const quantity = Number(rawQuantity);
    if (!rawQuantity || !Number.isFinite(quantity)) {
      errors.push(`${ticker} needs a quantity.`);
      return;
    }
    if (quantity <= 0) {
      errors.push(`${ticker} quantity must be greater than zero.`);
      return;
    }

    if (seen.has(ticker)) {
      errors.push(`${ticker} appears more than once. Combine those rows into one.`);
      return;
    }

    seen.set(ticker, quantity);
    holdings.push({ ticker, quantity });
  });

  return { holdings: errors.length === 0 ? holdings : [], errors };
}
