import { defineConfig } from 'vite';

// Dev: `npm run dev` proxies /api to the FastAPI server (`uvicorn aerogrid.api.main:app --port 8000` from backend/).
export default defineConfig({
  server: { port: 5173, proxy: { '/api': 'http://127.0.0.1:8000' } },
  build: { chunkSizeWarningLimit: 2500 },
});
