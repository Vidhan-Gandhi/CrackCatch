import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The dashboard talks to the FastAPI backend on :8000. In dev we proxy so the
// browser sees one origin (no CORS preflight, and the WebSocket upgrade works
// through the same host). In Docker, VITE_API_BASE points at the backend
// service instead.
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',
    port: 5173,
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
      '/media': { target: 'http://localhost:8000', changeOrigin: true },
      '/ws': { target: 'ws://localhost:8000', ws: true },
    },
  },
  preview: { host: '0.0.0.0', port: 4173 },
  build: { outDir: 'dist', sourcemap: false, chunkSizeWarningLimit: 1200 },
})
