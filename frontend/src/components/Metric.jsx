/**
 * One labeled figure. Rendered inside a <dl className="metrics"> so screen
 * readers announce each as a term/definition pair. `tone` is "positive",
 * "negative", or empty -- see signClass() in utils/format.js.
 */
export default function Metric({ label, value, tone = "", hint }) {
  return (
    <div className="metric">
      <dt>{label}</dt>
      <dd className={`num ${tone}`.trim()}>{value}</dd>
      {hint ? <span className="hint">{hint}</span> : null}
    </div>
  );
}
