import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import PortfolioForm from "./PortfolioForm";

describe("PortfolioForm", () => {
  it("starts with the four example holdings", () => {
    render(<PortfolioForm onSubmit={() => {}} />);
    expect(screen.getByLabelText("Ticker, row 1")).toHaveValue("AAPL");
    expect(screen.getByLabelText("Ticker, row 4")).toHaveValue("AMZN");
    expect(screen.getByLabelText("Shares, row 3")).toHaveValue("8");
  });

  it("submits a normalized payload for valid input", async () => {
    const onSubmit = vi.fn();
    render(<PortfolioForm onSubmit={onSubmit} />);

    await userEvent.click(screen.getByRole("button", { name: "Analyze portfolio" }));

    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit).toHaveBeenCalledWith({
      holdings: [
        { ticker: "AAPL", quantity: 10 },
        { ticker: "MSFT", quantity: 5 },
        { ticker: "NVDA", quantity: 8 },
        { ticker: "AMZN", quantity: 4 },
      ],
      benchmark: "^GSPC",
      lookback_years: 3,
    });
  });

  it("adds and removes holdings", async () => {
    render(<PortfolioForm onSubmit={() => {}} />);

    await userEvent.click(screen.getByRole("button", { name: "Add a holding" }));
    expect(screen.getByLabelText("Ticker, row 5")).toBeInTheDocument();

    await userEvent.click(screen.getByRole("button", { name: "Remove AAPL" }));
    expect(screen.queryByDisplayValue("AAPL")).not.toBeInTheDocument();
  });

  it("won't remove the last remaining row", async () => {
    render(<PortfolioForm onSubmit={() => {}} />);
    for (const ticker of ["AAPL", "MSFT", "NVDA"]) {
      await userEvent.click(screen.getByRole("button", { name: `Remove ${ticker}` }));
    }
    expect(screen.getByRole("button", { name: "Remove AMZN" })).toBeDisabled();
  });

  it("blocks submission and lists problems when input is invalid", async () => {
    const onSubmit = vi.fn();
    render(<PortfolioForm onSubmit={onSubmit} />);

    await userEvent.clear(screen.getByLabelText("Shares, row 1"));
    await userEvent.type(screen.getByLabelText("Shares, row 1"), "-4");
    await userEvent.click(screen.getByRole("button", { name: "Analyze portfolio" }));

    expect(onSubmit).not.toHaveBeenCalled();
    const alert = screen.getByRole("alert");
    expect(within(alert).getByText(/AAPL quantity must be greater than zero/)).toBeInTheDocument();
  });

  it("catches duplicate tickers", async () => {
    const onSubmit = vi.fn();
    render(<PortfolioForm onSubmit={onSubmit} />);

    await userEvent.clear(screen.getByLabelText("Ticker, row 2"));
    await userEvent.type(screen.getByLabelText("Ticker, row 2"), "aapl");
    await userEvent.click(screen.getByRole("button", { name: "Analyze portfolio" }));

    expect(onSubmit).not.toHaveBeenCalled();
    expect(screen.getByText(/AAPL appears more than once/)).toBeInTheDocument();
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
