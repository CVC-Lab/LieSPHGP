import { defineConfig } from "vite";

export default defineConfig({
  // Serves physics/export.py's output (../data/*.json) at the app root, e.g.
  // /dzhanibekov_free.json -- no copying/duplicating the generated data.
  publicDir: "../data",
  test: {
    environment: "jsdom",
  },
});
