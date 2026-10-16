import { useEffect, useState } from "react";

/**
 * Successful responses are cached by key so switching tabs does not refetch.
 * The key must encode EVERY input that affects the response (portfolio id,
 * risk-free rate, ...) -- that is what makes reusing a cached result safe.
 * Errors are deliberately not cached, so revisiting a tab retries.
 */
const cache = new Map();

export function clearResourceCache() {
  cache.clear();
}

/**
 * @param {string|null} key      cache key; pass null to stay idle (nothing to fetch yet)
 * @param {() => Promise<any>} fetcher
 * @returns {{status: "idle"|"loading"|"success"|"error", data?: any, error?: Error}}
 */
export function useResource(key, fetcher) {
  const [state, setState] = useState(() => {
    if (!key) return { status: "idle" };
    return cache.get(key) ?? { status: "loading" };
  });

  useEffect(() => {
    if (!key) {
      setState({ status: "idle" });
      return undefined;
    }

    const cached = cache.get(key);
    if (cached) {
      setState(cached);
      return undefined;
    }

    let cancelled = false;
    setState({ status: "loading" });

    fetcher()
      .then((data) => {
        const result = { status: "success", data };
        cache.set(key, result);
        if (!cancelled) setState(result);
      })
      .catch((error) => {
        if (!cancelled) setState({ status: "error", error });
      });

    return () => {
      cancelled = true;
    };
    // `fetcher` is intentionally omitted: `key` already encodes everything it depends on.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return state;
}
