import { renderHook, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { clearResourceCache, useResource } from "./useResource";

beforeEach(() => {
  clearResourceCache();
});

describe("useResource", () => {
  it("stays idle and never calls the fetcher when the key is null", () => {
    const fetcher = vi.fn();
    const { result } = renderHook(() => useResource(null, fetcher));
    expect(result.current.status).toBe("idle");
    expect(fetcher).not.toHaveBeenCalled();
  });

  it("moves from loading to success with the fetched data", async () => {
    const fetcher = vi.fn().mockResolvedValue({ value: 42 });
    const { result } = renderHook(() => useResource("k1", fetcher));

    expect(result.current.status).toBe("loading");
    await waitFor(() => expect(result.current.status).toBe("success"));
    expect(result.current.data).toEqual({ value: 42 });
  });

  it("serves a repeat request for the same key from cache", async () => {
    const fetcher = vi.fn().mockResolvedValue("data");

    const first = renderHook(() => useResource("shared", fetcher));
    await waitFor(() => expect(first.result.current.status).toBe("success"));
    first.unmount();

    const second = renderHook(() => useResource("shared", fetcher));
    expect(second.result.current.status).toBe("success");
    expect(second.result.current.data).toBe("data");
    expect(fetcher).toHaveBeenCalledTimes(1);
  });

  it("fetches again when the key changes", async () => {
    const fetcher = vi.fn().mockImplementation(async () => "x");
    const { result, rerender } = renderHook(({ k }) => useResource(k, fetcher), {
      initialProps: { k: "a" },
    });
    await waitFor(() => expect(result.current.status).toBe("success"));

    rerender({ k: "b" });
    await waitFor(() => expect(fetcher).toHaveBeenCalledTimes(2));
  });

  it("reports errors and does not cache them, so a revisit retries", async () => {
    const fetcher = vi
      .fn()
      .mockRejectedValueOnce(new Error("server down"))
      .mockResolvedValueOnce("recovered");

    const first = renderHook(() => useResource("flaky", fetcher));
    await waitFor(() => expect(first.result.current.status).toBe("error"));
    expect(first.result.current.error.message).toBe("server down");
    first.unmount();

    const second = renderHook(() => useResource("flaky", fetcher));
    await waitFor(() => expect(second.result.current.status).toBe("success"));
    expect(second.result.current.data).toBe("recovered");
    expect(fetcher).toHaveBeenCalledTimes(2);
  });
});
