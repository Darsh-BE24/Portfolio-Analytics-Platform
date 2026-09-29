import Metric from "../components/Metric";
import Resource from "../components/Resource";
import { usePerformance, useRisk } from "../hooks/useAnalytics";
import {
  formatCurrency,
  formatNumber,
  formatPercent,
  formatSignedPercent,
  latestPeriodReturn,
  signClass,
} from "../utils/format";

export default function OverviewTab({ portfolio, riskFreeRate }) {
  const performance = usePerformance(portfolio.id, riskFreeRate);
  const risk = useRisk(portfolio.id, riskFreeRate);

  return (
    <div className="tab-body">
      <section aria-labelledby="value-heading">
        <h2 id="value-heading">Portfolio value</h2>
        <p className="headline-figure num">{formatCurrency(portfolio.total_value)}</p>
        <p className="muted">
          Priced at the latest close. Benchmark: <span className="num">{portfolio.benchmark}</span>
        </p>
      </section>

      <section aria-labelledby="summary-heading">
        <h2 id="summary-heading">Summary</h2>
        <Resource state={performance}>
          {(perf) => (
            <Resource state={risk}>
              {(r) => {
                const lastDay = latestPeriodReturn(perf.cumulative_return_series);
                return (
                  <dl className="metrics">
                    <Metric label="Latest day" value={formatSignedPercent(lastDay)} tone={signClass(lastDay)} />
                    <Metric label="CAGR" value={formatSignedPercent(perf.cagr)} tone={signClass(perf.cagr)} />
                    <Metric label="Sharpe ratio" value={formatNumber(perf.sharpe_ratio)} />
                    <Metric label="Volatility (annual)" value={formatPercent(r.annualized_volatility)} />
                    <Metric label="Max drawdown" value={formatPercent(r.max_drawdown)} tone={signClass(r.max_drawdown)} />
                    <Metric label="Beta" value={formatNumber(r.beta)} hint={`vs ${portfolio.benchmark}`} />
                  </dl>
                );
              }}
            </Resource>
          )}
        </Resource>
      </section>

      <section aria-labelledby="positions-heading">
        <h2 id="positions-heading">Positions</h2>
        <table className="data-table">
          <thead>
            <tr>
              <th scope="col">Ticker</th>
              <th scope="col" className="right">Price</th>
              <th scope="col" className="right">Shares</th>
              <th scope="col" className="right">Value</th>
              <th scope="col">Weight</th>
            </tr>
          </thead>
          <tbody>
            {portfolio.positions.map((position) => (
              <tr key={position.ticker}>
                <th scope="row" className="num">{position.ticker}</th>
                <td className="num right">{formatCurrency(position.price)}</td>
                <td className="num right">{position.quantity}</td>
                <td className="num right">{formatCurrency(position.value)}</td>
                <td>
                  <div className="weight-cell">
                    <span className="weight-bar" aria-hidden="true">
                      <span style={{ width: `${position.weight * 100}%` }} />
                    </span>
                    <span className="num">{formatPercent(position.weight, 1)}</span>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}
