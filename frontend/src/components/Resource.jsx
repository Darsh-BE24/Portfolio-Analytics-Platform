/**
 * Renders the right thing for each state of a useResource() result, so tabs
 * only describe the success case:
 *
 *   <Resource state={state}>{(data) => <Table rows={data.rows} />}</Resource>
 */
export default function Resource({ state, loadingLabel = "Loading\u2026", children }) {
  if (state.status === "idle" || state.status === "loading") {
    return (
      <p className="status" role="status">
        {loadingLabel}
      </p>
    );
  }

  if (state.status === "error") {
    return (
      <div className="status error" role="alert">
        <strong>Couldn&rsquo;t load this data.</strong> <span>{state.error?.message}</span>
      </div>
    );
  }

  return children(state.data);
}
