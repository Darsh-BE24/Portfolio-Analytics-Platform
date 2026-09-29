import { lazy, Suspense } from "react";
import { colors, fontStack } from "../theme";

/**
 * Plotly is several MB, so it is loaded on demand rather than bundled into
 * the initial page. `react-plotly.js/factory` and `plotly.js-dist-min` are
 * CommonJS packages; depending on the bundler they arrive either as the
 * export itself or wrapped in `{ default }`, so both shapes are handled.
 */
const LazyPlot = lazy(async () => {
  const [plotlyModule, factoryModule] = await Promise.all([
    import("plotly.js-dist-min"),
    import("react-plotly.js/factory"),
  ]);
  const Plotly = plotlyModule.default ?? plotlyModule;
  const factory = factoryModule.default?.default ?? factoryModule.default ?? factoryModule;
  return { default: factory(Plotly) };
});

const BASE_LAYOUT = {
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: fontStack, color: colors.muted, size: 12 },
  margin: { l: 56, r: 16, t: 12, b: 40 },
  legend: { orientation: "h", y: 1.12, x: 0, font: { color: colors.text } },
  hovermode: "x unified",
  xaxis: { gridcolor: colors.hairline, zerolinecolor: colors.hairline, linecolor: colors.hairline },
  yaxis: { gridcolor: colors.hairline, zerolinecolor: colors.hairline, linecolor: colors.hairline },
};

/** Shallow-merge axis objects too, so callers can override just `title` or `tickformat`. */
function mergeLayout(overrides = {}) {
  return {
    ...BASE_LAYOUT,
    ...overrides,
    xaxis: { ...BASE_LAYOUT.xaxis, ...overrides.xaxis },
    yaxis: { ...BASE_LAYOUT.yaxis, ...overrides.yaxis },
  };
}

export default function Plot({ data, layout, height = 320, label }) {
  return (
    <div className="plot" role="img" aria-label={label}>
      <Suspense fallback={<div className="plot-loading" style={{ height }}>Loading chart</div>}>
        <LazyPlot
          data={data}
          layout={mergeLayout(layout)}
          config={{ displayModeBar: false, responsive: true }}
          style={{ width: "100%", height }}
          useResizeHandler
        />
      </Suspense>
    </div>
  );
}
