import { useState } from "react";
import { validateHoldings } from "../utils/holdings";

const DEFAULT_ROWS = [
  { ticker: "AAPL", quantity: "10" },
  { ticker: "MSFT", quantity: "5" },
  { ticker: "NVDA", quantity: "8" },
  { ticker: "AMZN", quantity: "4" },
];

/**
 * Collects holdings + benchmark + lookback window and hands a validated
 * payload to `onSubmit`. Server-side failures arrive through `serverError`.
 */
export default function PortfolioForm({ onSubmit, busy = false, serverError = null }) {
  const [rows, setRows] = useState(DEFAULT_ROWS);
  const [benchmark, setBenchmark] = useState("^GSPC");
  const [lookbackYears, setLookbackYears] = useState("3");
  const [errors, setErrors] = useState([]);

  function updateRow(index, field, value) {
    setRows((current) => current.map((row, i) => (i === index ? { ...row, [field]: value } : row)));
  }

  function addRow() {
    setRows((current) => [...current, { ticker: "", quantity: "" }]);
  }

  function removeRow(index) {
    setRows((current) => (current.length > 1 ? current.filter((_, i) => i !== index) : current));
  }

  function handleSubmit(event) {
    event.preventDefault();

    const { holdings, errors: validationErrors } = validateHoldings(rows);
    const years = Number(lookbackYears);
    const allErrors = [...validationErrors];

    if (!Number.isFinite(years) || years <= 0 || years > 10) {
      allErrors.push("History must be between 0 and 10 years.");
    }
    if (!benchmark.trim()) {
      allErrors.push("Enter a benchmark ticker, such as ^GSPC.");
    }

    setErrors(allErrors);
    if (allErrors.length > 0) return;

    onSubmit({
      holdings,
      benchmark: benchmark.trim().toUpperCase(),
      lookback_years: years,
    });
  }

  return (
    <form className="portfolio-form" onSubmit={handleSubmit} noValidate>
      <fieldset>
        <legend>Holdings</legend>
        <div className="holdings-grid">
          <span className="grid-head">Ticker</span>
          <span className="grid-head">Shares</span>
          <span />
          {rows.map((row, index) => (
            <div className="holding-row" key={index}>
              <input
                aria-label={`Ticker, row ${index + 1}`}
                className="num"
                value={row.ticker}
                onChange={(e) => updateRow(index, "ticker", e.target.value)}
                autoCapitalize="characters"
                spellCheck={false}
              />
              <input
                aria-label={`Shares, row ${index + 1}`}
                className="num"
                inputMode="decimal"
                value={row.quantity}
                onChange={(e) => updateRow(index, "quantity", e.target.value)}
              />
              <button
                type="button"
                className="icon-button"
                onClick={() => removeRow(index)}
                disabled={rows.length === 1}
                aria-label={`Remove ${row.ticker || `row ${index + 1}`}`}
              >
                &times;
              </button>
            </div>
          ))}
        </div>
        <button type="button" className="link-button" onClick={addRow}>
          Add a holding
        </button>
      </fieldset>

      <div className="field-pair">
        <label>
          Benchmark
          <input className="num" value={benchmark} onChange={(e) => setBenchmark(e.target.value)} spellCheck={false} />
        </label>
        <label>
          History (years)
          <input
            className="num"
            inputMode="decimal"
            value={lookbackYears}
            onChange={(e) => setLookbackYears(e.target.value)}
          />
        </label>
      </div>

      {errors.length > 0 ? (
        <ul className="form-errors" role="alert">
          {errors.map((message) => (
            <li key={message}>{message}</li>
          ))}
        </ul>
      ) : null}

      {serverError ? (
        <p className="form-errors" role="alert">
          {serverError}
        </p>
      ) : null}

      <button type="submit" className="primary" disabled={busy}>
        {busy ? "Analyzing\u2026" : "Analyze portfolio"}
      </button>
    </form>
  );
}
