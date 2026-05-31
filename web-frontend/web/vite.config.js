import { defineConfig } from 'vite';
import path from 'path';

export default defineConfig({
  resolve: {
    alias: {
      // Points to symlinked Mill ScalaJS output (created by `make link-fast`)
      'scalajs': path.resolve(__dirname, 'lib/scala')
    }
  },
  server: {
    proxy: {
      '/api': 'http://localhost:8080'
    },
    fs: {
      // Allow Vite to serve files through the symlink
      allow: ['.', '../lib']
    }
  },
  build: {
    outDir: 'dist',
    sourcemap: true
  }
});
