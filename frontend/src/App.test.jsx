import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App, { parseRiskFreeRate } from "./App";
import { api } from "./api/client";
import { clearResourceCache } from "./hooks/useResource";

// Plotly can't render in jsdom. This stub mirrors the real component's accessibility
// contract (an img role named by `label`), so tests also verify charts get descriptions.
vi.mock("./components/Plot", () => ({
  default: ({ label }) => <div role="img" aria-label={label} />,
}));

vi.mock("./api/client", () => ({
  api: {
    createPortfolio: vi.fn(),
    getPerformance: vi.fn(),
    getRisk: vi.fn(),
    getCorrelation: vi.fn(),
    getEfficientFrontier: vi.fn(),
    runBacktest: vi.fn(),
  },
}));

const portfolio = {
  id: "p1",
  created_at: "2024-06-01T00:00:00Z",
  benchmark: "^GSPC",
  total_value: 3000,
  positions: [
    { ticker: "AAPL", price: 200, quantity: 10, value: 2000, weight: 2 / 3 },
    { ticker: "MSFT", price: 400, quantity: 2.5, value: 1000, weight: 1 / 3 },
  ],
};

const performance = {
  cagr: 0.12,
  cumulative_return: 0.3,
  sharpe_ratio: 0.9,
  sortino_ratio: 1.1,
  risk_free_rate: 0.04,
  cumulative_return_series: [
    { date: "2024-01-02", value: 0.1 },
    { date: "2024-01-03", value: 0.21 },
  ],
  benchmark_cumulative_return_series: [
    { date: "2024-01-02", value: 0.05 },
    { date: "2024-01-03", value: 0.06 },
  ],
};

const risk = {
  annualized_volatility: 0.18,
  beta: 1.1,
  max_drawdown: -0.2,
  current_drawdown: -0.05,
  drawdown_periods: [
    { start: "2024-01-02", trough: "2024-01-10", end: null, depth: -0.2, peak_to_trough_days: 8, recovery_days: null },
  ],
  drawdown_series: [
    { date: "2024-01-02", value: 0 },
    { date: "2024-01-10", value: -0.2 },
  ],
  rolling_volatility: [{ date: "2024-01-10", value: 0.2 }],
  rolling_sharpe: [{ date: "2024-01-10", value: 0.8 }],
  historical_var_95: -0.02,
  parametric_var_95: -0.019,
  monte_carlo_var_95: -0.0195,
  historical_cvar_95: -0.03,
  parametric_cvar_95: -0.028,
  largest_position: 2 / 3,
  top_3_concentration: 1,
  hhi: 0.556,
  effective_number_of_holdings: 1.8,
};

const correlation = { tickers: ["AAPL", "MSFT"], matrix: [[1, 0.5], [0.5, 1]] };

const point = (ret, vol, sharpe, weights) => ({ expected_return: ret, volatility: vol, sharpe_ratio: sharpe, weights });
const frontier = {
  points: [point(0.08, 0.15, 0.5, { AAPL: 0.3, MSFT: 0.7 }), point(0.14, 0.22, 0.6, { AAPL: 0.8, MSFT: 0.2 })],
  current_portfolio: point(0.1, 0.2, 0.3, { AAPL: 2 / 3, MSFT: 1 / 3 }),
  min_volatility_portfolio: point(0.08, 0.15, 0.5, { AAPL: 0.3, MSFT: 0.7 }),
  max_sharpe_portfolio: point(0.14, 0.22, 0.6, { AAPL: 0.8, MSFT: 0.2 }),
};

const backtest = {
  train_start: "2021-01-04",
  train_end: "2023-06-01",
  test_start: "2023-06-02",
  test_end: "2024-06-01",
  strategies: Object.fromEntries(
    ["current", "equal_weight", "min_volatility", "max_sharpe", "benchmark"].map((key) => [
      key,
      {
        weights: key === "benchmark" ? null : { AAPL: 0.5, MSFT: 0.5 },
        metrics: {
          cagr: 0.1, volatility: 0.16, sharpe: 0.7, sortino: 0.8, max_drawdown: -0.12,
          historical_var_95: -0.018, historical_cvar_95: -0.025,
        },
      },
    ]),
  ),
};

beforeEach(() => {
  clearResourceCache();
  vi.clearAllMocks();
  api.createPortfolio.mockResolvedValue(portfolio);
  api.getPerformance.mockResolvedValue(performance);
  api.getRisk.mockResolvedValue(risk);
  api.getCorrelation.mockResolvedValue(correlation);
  api.getEfficientFrontier.mockResolvedValue(frontier);
  api.runBacktest.mockResolvedValue(backtest);
});

async function analyze() {
  await userEvent.click(screen.getByRole("button", { name: "Analyze portfolio" }));
  await screen.findByRole("tablist");
}

describe("parseRiskFreeRate", () => {
  it("converts a percent to a decimal", () => {
    expect(parseRiskFreeRate("4")).toBeCloseTo(0.04);
    expect(parseRiskFreeRate(" 4.5 ")).toBeCloseTo(0.045);
    expect(parseRiskFreeRate("0")).toBe(0);
  });

  it.each(["", "abc", "-1", "25"])("rejects %j", (text) => {
    expect(parseRiskFreeRate(text)).toBeNull();
  });
});

describe("App", () => {
  it("invites the user to enter holdings before anything is analyzed", () => {
    render(<App />);
    expect(screen.getByRole("heading", { name: "Start with your holdings" })).toBeInTheDocument();
    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
  });

  it("creates the portfolio and shows the overview", async () => {
    render(<App />);
    await analyze();

    expect(api.createPortfolio).toHaveBeenCalledTimes(1);
    expect(screen.getByText("$3,000.00")).toBeInTheDocument();
    expect(await screen.findByText("+12.00%")).toBeInTheDocument(); // CAGR
    expect(screen.getByText("+10.00%")).toBeInTheDocument(); // latest day, derived from the series
    expect(screen.getByText("66.7%")).toBeInTheDocument(); // AAPL weight
  });

  it("shows the server's message when portfolio creation fails", async () => {
    api.createPortfolio.mockRejectedValue(new Error("No data returned for ticker(s): ['ZZZZ']"));
    render(<App />);

    await userEvent.click(screen.getByRole("button", { name: "Analyze portfolio" }));

    expect(await screen.findByText(/No data returned for ticker/)).toBeInTheDocument();
    expect(screen.queryByRole("tablist")).not.toBeInTheDocument();
  });

  it("shows an error, not a blank panel, when analytics fail to load", async () => {
    api.getRisk.mockRejectedValue(new Error("Could not reach the analytics server."));
    render(<App />);
    await analyze();

    expect(await screen.findByText(/Couldn.t load this data/)).toBeInTheDocument();
    expect(screen.getByText(/Could not reach the analytics server/)).toBeInTheDocument();
  });

  it("loads risk data only when the Risk tab is opened", async () => {
    render(<App />);
    await analyze();
    await waitFor(() => expect(api.getRisk).toHaveBeenCalledTimes(1)); // Overview needs it too
    expect(api.getCorrelation).not.toHaveBeenCalled();

    await userEvent.click(screen.getByRole("tab", { name: "Risk" }));

    expect(await screen.findByText("Value at risk")).toBeInTheDocument();
    await waitFor(() => expect(api.getCorrelation).toHaveBeenCalledTimes(1));
    expect(api.getRisk).toHaveBeenCalledTimes(1); // shared cache: not fetched twice
    expect(await screen.findByRole("img", { name: /Heatmap of daily return correlations/ })).toBeInTheDocument();
  });

  it("lists the deepest drawdowns and marks unrecovered ones", async () => {
    render(<App />);
    await analyze();
    await userEvent.click(screen.getByRole("tab", { name: "Risk" }));

    expect(await screen.findByText("Deepest drawdowns")).toBeInTheDocument();
    expect(screen.getByText("Not yet")).toBeInTheDocument();
  });

  it("refetches with the new risk-free rate once the field loses focus", async () => {
    render(<App />);
    await analyze();
    await waitFor(() => expect(api.getPerformance).toHaveBeenCalledWith("p1", 0.04));

    const field = screen.getByLabelText("Risk-free rate (%)");
    await userEvent.clear(field);
    await userEvent.type(field, "5");
    expect(api.getPerformance).not.toHaveBeenCalledWith("p1", 0.05); // not on every keystroke

    await userEvent.tab();
    await waitFor(() => expect(api.getPerformance).toHaveBeenCalledWith("p1", 0.05));
  });

  it("rejects an invalid risk-free rate without refetching", async () => {
    render(<App />);
    await analyze();
    const callsBefore = api.getPerformance.mock.calls.length;

    const field = screen.getByLabelText("Risk-free rate (%)");
    await userEvent.clear(field);
    await userEvent.type(field, "abc");
    await userEvent.tab();

    expect(screen.getByText("Enter a rate between 0 and 20.")).toBeInTheDocument();
    expect(api.getPerformance.mock.calls.length).toBe(callsBefore);
  });
});

describe("Optimization tab", () => {
  async function openOptimization() {
    render(<App />);
    await analyze();
    await userEvent.click(screen.getByRole("tab", { name: "Optimization" }));
    await screen.findByText("Current against optimized");
  }

  it("shows the frontier chart and the comparison and allocation tables", async () => {
    await openOptimization();
    expect(screen.getByRole("img", { name: /Scatter chart of the efficient frontier/ })).toBeInTheDocument();
    expect(screen.getByRole("row", { name: /^Maximum Sharpe/ })).toBeInTheDocument();
    expect(api.getEfficientFrontier).toHaveBeenCalledWith("p1", {
      risk_free_rate: 0.04,
      n_points: 40,
      weight_bounds: [0, 1],
    });
  });

  it("applies valid position limits by refetching with them", async () => {
    await openOptimization();

    const panel = screen.getByRole("tabpanel");
    await userEvent.clear(within(panel).getByLabelText("Minimum per position (%)"));
    await userEvent.type(within(panel).getByLabelText("Minimum per position (%)"), "10");
    await userEvent.clear(within(panel).getByLabelText("Maximum per position (%)"));
    await userEvent.type(within(panel).getByLabelText("Maximum per position (%)"), "90");
    await userEvent.click(within(panel).getByRole("button", { name: "Recalculate" }));

    await waitFor(() =>
      expect(api.getEfficientFrontier).toHaveBeenLastCalledWith("p1", {
        risk_free_rate: 0.04,
        n_points: 40,
        weight_bounds: [0.1, 0.9],
      }),
    );
  });

  it("explains infeasible limits without calling the server", async () => {
    await openOptimization();
    const callsBefore = api.getEfficientFrontier.mock.calls.length;

    const panel = screen.getByRole("tabpanel");
    await userEvent.clear(within(panel).getByLabelText("Maximum per position (%)"));
    await userEvent.type(within(panel).getByLabelText("Maximum per position (%)"), "30");
    await userEvent.click(within(panel).getByRole("button", { name: "Recalculate" }));

    expect(await screen.findByText(/can't add up to 100%/)).toBeInTheDocument();
    expect(api.getEfficientFrontier.mock.calls.length).toBe(callsBefore);
  });

  it("keeps typed limits when switching to another tab and back", async () => {
    await openOptimization();

    const panel = screen.getByRole("tabpanel");
    await userEvent.clear(within(panel).getByLabelText("Maximum per position (%)"));
    await userEvent.type(within(panel).getByLabelText("Maximum per position (%)"), "80");

    await userEvent.click(screen.getByRole("tab", { name: "Performance" }));
    await userEvent.click(screen.getByRole("tab", { name: "Optimization" }));

    expect(within(screen.getByRole("tabpanel")).getByLabelText("Maximum per position (%)")).toHaveValue("80");
  });
});

describe("Backtest tab", () => {
  async function openBacktest() {
    render(<App />);
    await analyze();
    await userEvent.click(screen.getByRole("tab", { name: "Backtest" }));
  }

  it("does not run anything until asked", async () => {
    await openBacktest();
    expect(screen.getByText("Out-of-sample test")).toBeInTheDocument();
    expect(api.runBacktest).not.toHaveBeenCalled();
  });

  it("runs with the chosen split date and converts the cost percent to a rate", async () => {
    await openBacktest();

    const panel = screen.getByRole("tabpanel");
    const cost = within(panel).getByLabelText("Transaction cost (%)");
    await userEvent.clear(cost);
    await userEvent.type(cost, "0.25");
    await userEvent.click(within(panel).getByRole("button", { name: "Run backtest" }));

    await waitFor(() => expect(api.runBacktest).toHaveBeenCalledTimes(1));
    const [id, payload] = api.runBacktest.mock.calls[0];
    expect(id).toBe("p1");
    expect(payload.transaction_cost_rate).toBeCloseTo(0.0025);
    expect(payload.split_date).toMatch(/^\d{4}-\d{2}-\d{2}$/);
    expect(payload.risk_free_rate).toBe(0.04);

    expect(await screen.findByText("Test-period results")).toBeInTheDocument();
    expect(screen.getByText("2023-06-02")).toBeInTheDocument();
    expect(screen.getByRole("row", { name: /^Current portfolio/ })).toBeInTheDocument();
  });

  it("rejects an invalid transaction cost without calling the server", async () => {
    await openBacktest();

    const panel = screen.getByRole("tabpanel");
    const cost = within(panel).getByLabelText("Transaction cost (%)");
    await userEvent.clear(cost);
    await userEvent.type(cost, "-1");
    await userEvent.click(within(panel).getByRole("button", { name: "Run backtest" }));

    expect(await screen.findByText(/between 0 and 5/)).toBeInTheDocument();
    expect(api.runBacktest).not.toHaveBeenCalled();
  });
});
