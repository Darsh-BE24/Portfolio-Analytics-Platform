import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
  build: {
    // Plotly is a large library; it is code-split and loaded on demand
    // (see components/Plot.jsx), so the size warning is expected and safe to raise.
    chunkSizeWarningLimit: 5000,
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.js",
  },
});
