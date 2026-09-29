import { useState } from "react";
import { api } from "./api/client";
import PortfolioForm from "./components/PortfolioForm";
import { clearResourceCache } from "./hooks/useResource";
import BacktestTab from "./tabs/BacktestTab";
import OptimizationTab from "./tabs/OptimizationTab";
import OverviewTab from "./tabs/OverviewTab";
import PerformanceTab from "./tabs/PerformanceTab";
import RiskTab from "./tabs/RiskTab";

const TABS = [
  { id: "overview", label: "Overview", Component: OverviewTab },
  { id: "performance", label: "Performance", Component: PerformanceTab },
  { id: "risk", label: "Risk", Component: RiskTab },
  { id: "optimization", label: "Optimization", Component: OptimizationTab },
  { id: "backtest", label: "Backtest", Component: BacktestTab },
];

/** "4" (percent) -> 0.04. Returns null when the text isn't a rate between 0 and 20. */
export function parseRiskFreeRate(text) {
  const trimmed = String(text).trim();
  if (trimmed === "") return null;
  const value = Number(trimmed);
  if (!Number.isFinite(value) || value < 0 || value > 20) return null;
  return value / 100;
}

export default function App() {
  const [portfolio, setPortfolio] = useState(null);
  const [busy, setBusy] = useState(false);
  const [serverError, setServerError] = useState(null);

  const [riskFreeInput, setRiskFreeInput] = useState("4");
  const [riskFreeRate, setRiskFreeRate] = useState(0.04);
  const [riskFreeError, setRiskFreeError] = useState(null);

  const [activeTab, setActiveTab] = useState("overview");
  // Tabs mount the first time they're opened and then stay mounted (hidden),
  // so inputs like position limits survive switching away and back.
  const [visited, setVisited] = useState(new Set(["overview"]));

  async function handleCreate(payload) {
    setBusy(true);
    setServerError(null);
    try {
      const created = await api.createPortfolio(payload);
      clearResourceCache();
      setPortfolio(created);
      setActiveTab("overview");
      setVisited(new Set(["overview"]));
    } catch (error) {
      setServerError(error.message);
    } finally {
      setBusy(false);
    }
  }

  function commitRiskFree() {
    const parsed = parseRiskFreeRate(riskFreeInput);
    if (parsed === null) {
      setRiskFreeError("Enter a rate between 0 and 20.");
      return;
    }
    setRiskFreeError(null);
    setRiskFreeRate(parsed);
  }

  function selectTab(id) {
    setActiveTab(id);
    setVisited((current) => new Set(current).add(id));
  }

  return (
    <div className="app">
      <aside className="rail">
        <h1 className="app-name">Portfolio analytics</h1>
        <PortfolioForm onSubmit={handleCreate} busy={busy} serverError={serverError} />

        <form
          className="rate-field"
          onSubmit={(event) => {
            event.preventDefault();
            commitRiskFree();
          }}
        >
          <label>
            Risk-free rate (%)
            <input
              className="num"
              inputMode="decimal"
              value={riskFreeInput}
              onChange={(e) => setRiskFreeInput(e.target.value)}
              onBlur={commitRiskFree}
            />
          </label>
          {riskFreeError ? (
            <p className="form-errors" role="alert">{riskFreeError}</p>
          ) : (
            <p className="muted note">Used for Sharpe, Sortino, and optimization.</p>
          )}
        </form>
      </aside>

      <main className="main">
        {portfolio === null ? (
          <div className="empty-state">
            <h2>Start with your holdings</h2>
            <p>
              Enter tickers and share counts on the left, then choose Analyze portfolio. You&rsquo;ll get
              performance, risk, an efficient frontier, and an out-of-sample backtest.
            </p>
          </div>
        ) : (
          <>
            <div role="tablist" aria-label="Analysis sections" className="tabs">
              {TABS.map(({ id, label }) => (
                <button
                  key={id}
                  role="tab"
                  id={`tab-${id}`}
                  aria-selected={activeTab === id}
                  aria-controls={`panel-${id}`}
                  onClick={() => selectTab(id)}
                >
                  {label}
                </button>
              ))}
            </div>

            {TABS.map(({ id, Component }) =>
              visited.has(id) ? (
                <div
                  key={`${portfolio.id}-${id}`}
                  role="tabpanel"
                  id={`panel-${id}`}
                  aria-labelledby={`tab-${id}`}
                  hidden={activeTab !== id}
                >
                  <Component portfolio={portfolio} riskFreeRate={riskFreeRate} />
                </div>
              ) : null,
            )}
          </>
        )}
      </main>
    </div>
  );
}
