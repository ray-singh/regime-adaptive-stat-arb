import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const BACKEND = process.env.BACKEND_URL || "https://regime-pairs.fly.dev";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    host: true,
    proxy: {
      "/api": {
        target: BACKEND,
        changeOrigin: true,
        secure: false,
      },
    },
  },
});
