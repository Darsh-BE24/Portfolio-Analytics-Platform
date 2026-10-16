import { useState } from "react";
import Plot from "../components/Plot";
import Resource from "../components/Resource";
import { useEfficientFrontier } from "../hooks/useAnalytics";
import { colors } from "../theme";
import { formatNumber, formatPercent } from "../utils/format";
import { parseWeightBounds } from "../utils/bounds";

const pct = (values) => values.map((v) => v * 100);

export default function OptimizationTab({ portfolio, riskFreeRate }) {
  const [minInput, setMinInput] = useState("0");
  const [maxInput, setMaxInput] = useState("100");
  const [bounds, setBounds] = useState([0, 1]);
  const [boundsError, setBoundsError] = useState(null);

  const frontier = useEfficientFrontier(portfolio.id, riskFreeRate, bounds);

  function applyLimits(event) {
    event.preventDefault();
    const result = parseWeightBounds(minInput, maxInput, portfolio.positions.length);
    setBoundsError(result.error);
    if (result.bounds) setBounds(result.bounds);
  }

  return (
    <div className="tab-body">
      <section aria-labelledby="limits-heading">
        <h2 id="limits-heading">Position limits</h2>
        <form className="inline-form" onSubmit={applyLimits} noValidate>
          <label>
            Minimum per position (%)
            <input className="num" inputMode="decimal" value={minInput} onChange={(e) => setMinInput(e.target.value)} />
          </label>
          <label>
            Maximum per position (%)
            <input className="num" inputMode="decimal" value={maxInput} onChange={(e) => setMaxInput(e.target.value)} />
          </label>
          <button type="submit" className="secondary">Recalculate</button>
        </form>
        {boundsError ? (
          <p className="form-errors" role="alert">{boundsError}</p>
        ) : (
          <p className="muted note">
            Limits keep the optimizer from piling into one or two names. Try 5 and 40 to see the difference.
          </p>
        )}
      </section>

      <Resource state={frontier} loadingLabel={"Solving the efficient frontier\u2026"}>
        {(f) => {
          const tickers = Object.keys(f.current_portfolio.weights);
          const rows = [
            { key: "current", label: "Current portfolio", point: f.current_portfolio },
            { key: "min", label: "Minimum volatility", point: f.min_volatility_portfolio },
            { key: "max", label: "Maximum Sharpe", point: f.max_sharpe_portfolio },
          ];

          const marker = (point, name, color, symbol, size) => ({
            x: [point.volatility * 100],
            y: [point.expected_return * 100],
            name,
            type: "scatter",
            mode: "markers",
            marker: { color, symbol, size, line: { color: colors.ink, width: 1.5 } },
            hovertemplate: `${name}<br>Risk %{x:.2f}%<br>Return %{y:.2f}%<extra></extra>`,
          });

          return (
            <>
              <section aria-labelledby="frontier-heading">
                <h2 id="frontier-heading">Efficient frontier</h2>
                <Plot
                  label="Scatter chart of the efficient frontier with current, minimum volatility, and maximum Sharpe portfolios marked"
                  height={400}
                  data={[
                    {
                      x: pct(f.points.map((p) => p.volatility)),
                      y: pct(f.points.map((p) => p.expected_return)),
                      name: "Efficient frontier",
                      type: "scatter",
                      mode: "lines",
                      line: { color: colors.benchmark, width: 2 },
                      hovertemplate: "Risk %{x:.2f}%<br>Return %{y:.2f}%<extra></extra>",
                    },
                    marker(f.min_volatility_portfolio, "Minimum volatility", colors.info, "circle", 11),
                    marker(f.max_sharpe_portfolio, "Maximum Sharpe", colors.text, "star", 14),
                    marker(f.current_portfolio, "Your portfolio", colors.accent, "diamond", 14),
                  ]}
                  layout={{
                    hovermode: "closest",
                    xaxis: { ticksuffix: "%", title: { text: "Volatility (annual)" } },
                    yaxis: { ticksuffix: "%", title: { text: "Expected return (annual)" } },
                  }}
                />
                <p className="muted note">
                  Expected return is each holding&rsquo;s historical average, a noisy guide to the future.
                  The Backtest tab shows how these allocations held up on data the optimizer never saw.
                </p>
              </section>

              <section aria-labelledby="compare-heading">
                <h2 id="compare-heading">Current against optimized</h2>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th scope="col">Portfolio</th>
                      <th scope="col" className="right">Expected return</th>
                      <th scope="col" className="right">Volatility</th>
                      <th scope="col" className="right">Sharpe ratio</th>
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map(({ key, label, point }) => (
                      <tr key={key} className={key === "current" ? "highlight" : ""}>
                        <th scope="row">{label}</th>
                        <td className="num right">{formatPercent(point.expected_return)}</td>
                        <td className="num right">{formatPercent(point.volatility)}</td>
                        <td className="num right">{formatNumber(point.sharpe_ratio)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>

              <section aria-labelledby="alloc-heading">
                <h2 id="alloc-heading">Allocation</h2>
                <table className="data-table">
                  <thead>
                    <tr>
                      <th scope="col">Ticker</th>
                      {rows.map(({ key, label }) => (
                        <th scope="col" className="right" key={key}>{label}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {tickers.map((ticker) => (
                      <tr key={ticker}>
                        <th scope="row" className="num">{ticker}</th>
                        {rows.map(({ key, point }) => (
                          <td className="num right" key={key}>{formatPercent(point.weights[ticker] ?? 0, 1)}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </section>
            </>
          );
        }}
      </Resource>
    </div>
  );
}
