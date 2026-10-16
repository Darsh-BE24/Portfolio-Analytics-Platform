import Metric from "../components/Metric";
import Plot from "../components/Plot";
import Resource from "../components/Resource";
import { useCorrelation, useRisk } from "../hooks/useAnalytics";
import { colors } from "../theme";
import { formatNumber, formatPercent, signClass } from "../utils/format";

const dates = (series) => series.map((p) => p.date);
const percents = (series) => series.map((p) => p.value * 100);

function worstDrawdowns(periods, count = 5) {
  return [...periods].sort((a, b) => a.depth - b.depth).slice(0, count);
}

export default function RiskTab({ portfolio, riskFreeRate }) {
  const risk = useRisk(portfolio.id, riskFreeRate);
  const correlation = useCorrelation(portfolio.id);

  return (
    <div className="tab-body">
      <Resource state={risk}>
        {(r) => (
          <>
            <section aria-labelledby="risk-summary-heading">
              <h2 id="risk-summary-heading">Volatility and drawdown</h2>
              <dl className="metrics">
                <Metric label="Volatility (annual)" value={formatPercent(r.annualized_volatility)} />
                <Metric label="Beta" value={formatNumber(r.beta)} hint={`vs ${portfolio.benchmark}`} />
                <Metric label="Max drawdown" value={formatPercent(r.max_drawdown)} tone={signClass(r.max_drawdown)} />
                <Metric label="Current drawdown" value={formatPercent(r.current_drawdown)} tone={signClass(r.current_drawdown)} />
              </dl>
            </section>

            <section aria-labelledby="var-heading">
              <h2 id="var-heading">Value at risk</h2>
              <table className="data-table">
                <thead>
                  <tr>
                    <th scope="col">Method</th>
                    <th scope="col" className="right">VaR (95%, 1 day)</th>
                    <th scope="col" className="right">CVaR (95%, 1 day)</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <th scope="row">Historical</th>
                    <td className="num right negative">{formatPercent(r.historical_var_95)}</td>
                    <td className="num right negative">{formatPercent(r.historical_cvar_95)}</td>
                  </tr>
                  <tr>
                    <th scope="row">Parametric (normal)</th>
                    <td className="num right negative">{formatPercent(r.parametric_var_95)}</td>
                    <td className="num right negative">{formatPercent(r.parametric_cvar_95)}</td>
                  </tr>
                  <tr>
                    <th scope="row">Monte Carlo</th>
                    <td className="num right negative">{formatPercent(r.monte_carlo_var_95)}</td>
                    <td className="num right">{formatPercent(null)}</td>
                  </tr>
                </tbody>
              </table>
              <p className="muted note">
                VaR is the loss a single day should stay within 95% of the time. CVaR is the average loss on
                the days that break through it. Monte Carlo draws from a normal distribution, so it lands
                close to the parametric figure and understates fat-tailed losses.
              </p>
            </section>

            <section aria-labelledby="dd-heading">
              <h2 id="dd-heading">Drawdown</h2>
              <Plot
                label="Area chart of portfolio drawdown from its running peak"
                data={[
                  {
                    x: dates(r.drawdown_series),
                    y: percents(r.drawdown_series),
                    type: "scatter",
                    mode: "lines",
                    fill: "tozeroy",
                    name: "Drawdown",
                    line: { color: colors.negative, width: 1.5 },
                    fillcolor: "rgba(226,88,107,0.22)",
                    hovertemplate: "%{y:.2f}%",
                  },
                ]}
                layout={{ yaxis: { ticksuffix: "%", title: { text: "From running peak" } }, showlegend: false }}
              />
              {r.drawdown_periods.length > 0 ? (
                <table className="data-table">
                  <caption>Deepest drawdowns</caption>
                  <thead>
                    <tr>
                      <th scope="col">Began</th>
                      <th scope="col">Trough</th>
                      <th scope="col">Recovered</th>
                      <th scope="col" className="right">Depth</th>
                      <th scope="col" className="right">Days to trough</th>
                      <th scope="col" className="right">Days to recover</th>
                    </tr>
                  </thead>
                  <tbody>
                    {worstDrawdowns(r.drawdown_periods).map((p) => (
                      <tr key={`${p.start}-${p.trough}`}>
                        <td className="num">{p.start}</td>
                        <td className="num">{p.trough}</td>
                        <td className="num">{p.end ?? "Not yet"}</td>
                        <td className="num right negative">{formatPercent(p.depth)}</td>
                        <td className="num right">{p.peak_to_trough_days ?? "\u2014"}</td>
                        <td className="num right">{p.recovery_days ?? "\u2014"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              ) : null}
            </section>

            <section aria-labelledby="rolling-heading">
              <h2 id="rolling-heading">Rolling risk (63-day window)</h2>
              <div className="chart-pair">
                <div>
                  <h3>Volatility</h3>
                  <Plot
                    label="Line chart of 63-day rolling annualized volatility"
                    height={240}
                    data={[
                      {
                        x: dates(r.rolling_volatility),
                        y: percents(r.rolling_volatility),
                        type: "scatter",
                        mode: "lines",
                        line: { color: colors.info, width: 1.75 },
                        hovertemplate: "%{y:.2f}%",
                      },
                    ]}
                    layout={{ yaxis: { ticksuffix: "%" }, showlegend: false }}
                  />
                </div>
                <div>
                  <h3>Sharpe ratio</h3>
                  <Plot
                    label="Line chart of 63-day rolling Sharpe ratio"
                    height={240}
                    data={[
                      {
                        x: dates(r.rolling_sharpe),
                        y: r.rolling_sharpe.map((p) => p.value),
                        type: "scatter",
                        mode: "lines",
                        line: { color: colors.info, width: 1.75 },
                        hovertemplate: "%{y:.2f}",
                      },
                    ]}
                    layout={{ showlegend: false }}
                  />
                </div>
              </div>
            </section>

            <section aria-labelledby="conc-heading">
              <h2 id="conc-heading">Concentration</h2>
              <dl className="metrics">
                <Metric label="Largest position" value={formatPercent(r.largest_position, 1)} />
                <Metric label="Top three" value={formatPercent(r.top_3_concentration, 1)} />
                <Metric label="HHI" value={formatNumber(r.hhi, 3)} hint="1 / holdings when evenly split" />
                <Metric label="Effective holdings" value={formatNumber(r.effective_number_of_holdings, 1)} />
              </dl>
            </section>
          </>
        )}
      </Resource>

      <section aria-labelledby="corr-heading">
        <h2 id="corr-heading">Correlation</h2>
        <Resource state={correlation}>
          {(c) => (
            <Plot
              label="Heatmap of daily return correlations between holdings"
              height={Math.max(260, c.tickers.length * 64 + 60)}
              data={[
                {
                  type: "heatmap",
                  z: c.matrix,
                  x: c.tickers,
                  y: c.tickers,
                  zmin: -1,
                  zmax: 1,
                  colorscale: [
                    [0, colors.negative],
                    [0.5, colors.panel],
                    [1, colors.info],
                  ],
                  texttemplate: "%{z:.2f}",
                  showscale: false,
                  hovertemplate: "%{y} and %{x}: %{z:.2f}<extra></extra>",
                },
              ]}
              layout={{ hovermode: "closest", yaxis: { autorange: "reversed" }, margin: { l: 64, r: 16, t: 12, b: 48 } }}
            />
          )}
        </Resource>
      </section>
    </div>
  );
}
