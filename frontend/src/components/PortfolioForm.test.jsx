import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import PortfolioForm from "./PortfolioForm";

describe("PortfolioForm", () => {
  it("starts with four example Indian holdings", () => {
    render(<PortfolioForm onSubmit={() => {}} />);
    expect(screen.getByLabelText("Ticker, row 1")).toHaveValue("RELIANCE");
    expect(screen.getByLabelText("Ticker, row 4")).toHaveValue("INFY");
    expect(screen.getByLabelText("Shares, row 3")).toHaveValue("8");
  });

  it("submits a normalized payload for valid input", async () => {
    const onSubmit = vi.fn();
    render(<PortfolioForm onSubmit={onSubmit} />);

    await userEvent.click(screen.getByRole("button", { name: "Analyze portfolio" }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit).toHaveBeenCalledWith({
      holdings: [
        { ticker: "RELIANCE.NS", quantity: 10 },
        { ticker: "TCS.NS", quantity: 5 },
        { ticker: "HDFCBANK.NS", quantity: 8 },
        { ticker: "INFY.NS", quantity: 4 },
      ],
      benchmark: "^NSEI",
      lookback_years: 3,
    });
  });

  it("adds and removes holdings", async () => {
    render(<PortfolioForm onSubmit={() => {}} />);

    await userEvent.click(screen.getByRole("button", { name: "Add a holding" }));
    expect(screen.getByLabelText("Ticker, row 5")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Remove RELIANCE" }));
    expect(screen.queryByDisplayValue("RELIANCE")).not.toBeInTheDocument();
  });

  it("won't remove the last remaining row", async () => {
    render(<PortfolioForm onSubmit={() => {}} />);
    for (const ticker of ["RELIANCE", "TCS", "HDFCBANK"]) {
      await userEvent.click(screen.getByRole("button", { name: `Remove ${ticker}` }));
    }
    expect(screen.getByRole("button", { name: "Remove INFY" })).toBeDisabled();
  });

  it("blocks submission and lists problems when input is invalid", async () => {
    const onSubmit = vi.fn();
    render(<PortfolioForm onSubmit={onSubmit} />);

    await userEvent.clear(screen.getByLabelText("Shares, row 1"));
    await userEvent.type(screen.getByLabelText("Shares, row 1"), "-4");
    await userEvent.click(screen.getByRole("button", { name: "Analyze portfolio" }));

    expect(onSubmit).not.toHaveBeenCalled();
    const alert = screen.getByRole("alert");
    expect(within(alert).getByText(/RELIANCE\.NS quantity must be greater than zero/)).toBeInTheDocument();
  });

  it("catches duplicate tickers", async () => {
    const onSubmit = vi.fn();
    render(<PortfolioForm onSubmit={onSubmit} />);

    await userEvent.clear(screen.getByLabelText("Ticker, row 2"));
    await userEvent.type(screen.getByLabelText("Ticker, row 2"), "reliance");
    await userEvent.click(screen.getByRole("button", { name: "Analyze portfolio" }));

    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.getByText(/RELIANCE\.NS appears more than once/)).toBeInTheDocument();
  });

  it("sends a bare benchmark as an NSE symbol and keeps an index as typed", async () => {
    const onSubmit = vi.fn();
    render(<PortfolioForm onSubmit={onSubmit} />);

    await userEvent.clear(screen.getByLabelText("Benchmark"));
    await userEvent.type(screen.getByLabelText("Benchmark"), "niftybees");
    await userEvent.click(screen.getByRole("button", { name: "Analyze portfolio" }));
    expect(onSubmit.mock.calls[0][0].benchmark).toBe("NIFTYBEES.NS");

    await userEvent.clear(screen.getByLabelText("Benchmark"));
    await userEvent.type(screen.getByLabelText("Benchmark"), "^bsesn");
    await userEvent.click(screen.getByRole("button", { name: "Analyze portfolio" }));
    expect(onSubmit.mock.calls[1][0].benchmark).toBe("^BSESN");
  });

  it("rejects an out-of-range history length", async () => {
    const onSubmit = vi.fn();
    render(<PortfolioForm onSubmit={onSubmit} />);

    await userEvent.clear(screen.getByLabelText("History (years)"));
    await userEvent.type(screen.getByLabelText("History (years)"), "50");
    await userEvent.click(screen.getByRole("button", { name: "Analyze portfolio" }));

    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.getByText(/between 0 and 10 years/)).toBeInTheDocument();
  });

  it("shows a server error and disables the button while busy", () => {
    const { rerender } = render(<PortfolioForm onSubmit={() => {}} serverError="No data returned for ticker(s): ['ZZZZ']" />);
    expect(screen.getByText(/No data returned for ticker/)).toBeInTheDocument();

    rerender(<PortfolioForm onSubmit={() => {}} busy />);
    expect(screen.getByRole("button", { name: /Analyzing/ })).toBeDisabled();
  });
});
