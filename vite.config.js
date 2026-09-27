import { defineConfig } from 'vite';

export default defineConfig({
  base: './',
  server: { port: 5173, strictPort: true, host: '127.0.0.1' },
  preview: { port: 4173, strictPort: true, host: '127.0.0.1' },
  assetsInclude: ['**/*.hdr', '**/*.glb', '**/*.ktx2'],
  build: { target: 'es2022', chunkSizeWarningLimit: 6000 }, // the main chunk is ~5.2 MB: Rapier-compat inlines its wasm
  optimizeDeps: { exclude: ['@dimforge/rapier3d-compat'] },
});
