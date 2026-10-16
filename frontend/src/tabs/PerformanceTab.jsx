import Metric from "../components/Metric";
import Plot from "../components/Plot";
import Resource from "../components/Resource";
import { usePerformance } from "../hooks/useAnalytics";
import { colors } from "../theme";
import { formatNumber, formatSignedPercent, signClass } from "../utils/format";

const toPercent = (series) => series.map((point) => point.value * 100);
const toDates = (series) => series.map((point) => point.date);

export default function PerformanceTab({ portfolio, riskFreeRate }) {
  const performance = usePerformance(portfolio.id, riskFreeRate);

  return (
    <div className="tab-body">
      <Resource state={performance}>
        {(perf) => (
          <>
            <section aria-labelledby="perf-summary-heading">
              <h2 id="perf-summary-heading">Return and risk-adjusted return</h2>
              <dl className="metrics">
                <Metric label="Total return" value={formatSignedPercent(perf.cumulative_return)} tone={signClass(perf.cumulative_return)} />
                <Metric label="CAGR" value={formatSignedPercent(perf.cagr)} tone={signClass(perf.cagr)} />
                <Metric label="Sharpe ratio" value={formatNumber(perf.sharpe_ratio)} />
                <Metric label="Sortino ratio" value={formatNumber(perf.sortino_ratio)} />
              </dl>
            </section>

            <section aria-labelledby="perf-chart-heading">
              <h2 id="perf-chart-heading">Cumulative return against {portfolio.benchmark}</h2>
              <Plot
                label="Line chart of cumulative return for the portfolio and its benchmark"
                height={360}
                data={[
                  {
                    x: toDates(perf.benchmark_cumulative_return_series),
                    y: toPercent(perf.benchmark_cumulative_return_series),
                    name: portfolio.benchmark,
                    type: "scatter",
                    mode: "lines",
                    line: { color: colors.benchmark, width: 1.5 },
                    hovertemplate: "%{y:.2f}%",
                  },
                  {
                    x: toDates(perf.cumulative_return_series),
                    y: toPercent(perf.cumulative_return_series),
                    name: "Your portfolio",
                    type: "scatter",
                    mode: "lines",
                    line: { color: colors.accent, width: 2.25 },
                    hovertemplate: "%{y:.2f}%",
                  },
                ]}
                layout={{ yaxis: { ticksuffix: "%", title: { text: "Cumulative return" } } }}
              />
              <p className="muted note">
                Assumes today&rsquo;s weights were held constant across the whole period, and rebalanced
                daily. Risk-free rate for Sharpe and Sortino: <span className="num">{formatNumber(perf.risk_free_rate * 100, 1)}%</span>.
              </p>
            </section>
          </>
        )}
      </Resource>
    </div>
  );
}
