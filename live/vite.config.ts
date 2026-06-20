import { defineConfig } from "vite";
import { VitePWA } from "vite-plugin-pwa";
import basicSsl from "@vitejs/plugin-basic-ssl";

// The live coach reads the shared stroke targets from ../shared, so allow Vite
// to serve files from the repo root (one level above this package).
export default defineConfig(({ command }) => ({
  // GitHub Pages serves this project at /rowCoach/; dev stays at root.
  base: command === "build" ? "/rowCoach/" : "/",
  server: {
    host: true, // expose on LAN so the iPhone can reach the dev server
    https: true as any, // iOS getUserMedia requires HTTPS (basicSsl provides a cert)
    fs: { allow: [".."] },
  },
  plugins: [
    basicSsl(),
    VitePWA({
      registerType: "autoUpdate",
      includeAssets: ["favicon.svg"],
      manifest: {
        name: "RowCoach Live",
        short_name: "RowCoach",
        description: "Live rowing-technique cues while you erg",
        theme_color: "#0b1020",
        background_color: "#0b1020",
        display: "standalone",
        orientation: "landscape",
        icons: [
          { src: "favicon.svg", sizes: "any", type: "image/svg+xml", purpose: "any" },
        ],
      },
    }),
  ],
}));
