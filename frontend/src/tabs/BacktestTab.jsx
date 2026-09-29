import { useState } from "react";
import Resource from "../components/Resource";
import { useBacktest } from "../hooks/useAnalytics";
import { formatNumber, formatPercent, formatSignedPercent, signClass } from "../utils/format";

const STRATEGY_LABELS = {
  current: "Current portfolio",
  equal_weight: "Equal weight",
  min_volatility: "Minimum volatility",
  max_sharpe: "Maximum Sharpe",
  benchmark: "Benchmark",
};

const STRATEGY_ORDER = ["current", "equal_weight", "min_volatility", "max_sharpe", "benchmark"];

function isoDateYearsAgo(years) {
  const d = new Date();
  d.setFullYear(d.getFullYear() - years);
  return d.toISOString().slice(0, 10);
}

/** Percent input (e.g. "0.10") to the decimal the API expects (0.001). Null when invalid. */
export function parseCostPercent(text) {
  const trimmed = String(text).trim();
  if (trimmed === "") return null;
  const value = Number(trimmed);
  if (!Number.isFinite(value) || value < 0 || value > 5) return null;
  return value / 100;
}

export default function BacktestTab({ portfolio, riskFreeRate }) {
  const [splitDate, setSplitDate] = useState(isoDateYearsAgo(1));
  const [costInput, setCostInput] = useState("0.10");
  const [params, setParams] = useState(null);
  const [inputError, setInputError] = useState(null);

  const result = useBacktest(portfolio.id, riskFreeRate, params);

  function run(event) {
    event.preventDefault();
    const costRate = parseCostPercent(costInput);
    if (!splitDate) {
      setInputError("Choose a date to split the history into training and test periods.");
      return;
    }
    if (costRate === null) {
      setInputError("Transaction cost must be a number between 0 and 5 (percent).");
      return;
    }
    setInputError(null);
    setParams({ splitDate, costRate });
  }

  return (
    <div className="tab-body">
      <section aria-labelledby="bt-setup-heading">
        <h2 id="bt-setup-heading">Out-of-sample test</h2>
        <p className="muted note">
          The optimizer sees only data up to the split date. Its weights are then frozen and carried
          through the test period it never saw, so the results show how the strategy would have held up
          rather than how well it fit the past.
        </p>
        <form className="inline-form" onSubmit={run} noValidate>
          <label>
            Split date
            <input type="date" className="num" value={splitDate} onChange={(e) => setSplitDate(e.target.value)} />
          </label>
          <label>
            Transaction cost (%)
            <input className="num" inputMode="decimal" value={costInput} onChange={(e) => setCostInput(e.target.value)} />
          </label>
          <button type="submit" className="secondary">Run backtest</button>
        </form>
        {inputError ? <p className="form-errors" role="alert">{inputError}</p> : null}
      </section>

      {params === null ? null : (
        <Resource state={result} loadingLabel={"Running the backtest\u2026"}>
          {(bt) => (
            <>
              <section aria-labelledby="bt-results-heading">
                <h2 id="bt-results-heading">Test-period results</h2>
                <p className="muted">
                  Trained <span className="num">{bt.train_start}</span> to <span className="num">{bt.train_end}</span>.
                  Tested <span className="num">{bt.test_start}</span> to <span className="num">{bt.test_end}</span>.
                </p>
                <div className="table-scroll">
                  <table className="data-table">
                    <thead>
                      <tr>
                        <th scope="col">Strategy</th>
                        <th scope="col" className="right">CAGR</th>
                        <th scope="col" className="right">Volatility</th>
                        <th scope="col" className="right">Sharpe</th>
                        <th scope="col" className="right">Sortino</th>
                        <th scope="col" className="right">Max drawdown</th>
                        <th scope="col" className="right">VaR 95%</th>
                        <th scope="col" className="right">CVaR 95%</th>
                      </tr>
                    </thead>
                    <tbody>
                      {STRATEGY_ORDER.filter((key) => bt.strategies[key]).map((key) => {
                        const m = bt.strategies[key].metrics;
                        return (
                          <tr key={key} className={key === "current" ? "highlight" : ""}>
                            <th scope="row">{key === "benchmark" ? portfolio.benchmark : STRATEGY_LABELS[key]}</th>
                            <td className={`num right ${signClass(m.cagr)}`}>{formatSignedPercent(m.cagr)}</td>
                            <td className="num right">{formatPercent(m.volatility)}</td>
                            <td className="num right">{formatNumber(m.sharpe)}</td>
                            <td className="num right">{formatNumber(m.sortino)}</td>
                            <td className="num right negative">{formatPercent(m.max_drawdown)}</td>
                            <td className="num right negative">{formatPercent(m.historical_var_95)}</td>
                            <td className="num right negative">{formatPercent(m.historical_cvar_95)}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <p className="muted note">
                  Switching from your current holdings into each alternative is charged{" "}
                  <span className="num">{formatNumber(params.costRate * 100, 2)}%</span> of the amount traded.
                </p>
              </section>

              <section aria-labelledby="bt-weights-heading">
                <h2 id="bt-weights-heading">Weights used</h2>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th scope="col">Ticker</th>
                      {STRATEGY_ORDER.filter((k) => bt.strategies[k]?.weights).map((k) => (
                        <th scope="col" className="right" key={k}>{STRATEGY_LABELS[k]}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {Object.keys(bt.strategies.current.weights).map((ticker) => (
                      <tr key={ticker}>
                        <th scope="row" className="num">{ticker}</th>
                        {STRATEGY_ORDER.filter((k) => bt.strategies[k]?.weights).map((k) => (
                          <td className="num right" key={k}>{formatPercent(bt.strategies[k].weights[ticker] ?? 0, 1)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            </>
          )}
        </Resource>
      )}
    </div>
  );
}
