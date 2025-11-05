import path from "path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import Pages from "vite-plugin-pages";
import getRepoName from "git-repo-name";

export default defineConfig(({ command, mode }) => {
    const isGitHubPages = mode === 'github-pages'

  // Set the base path manually
    const base = isGitHubPages ? `/LatentSpaceExplorer/` : '/';

    return {
      plugins: [react(), Pages()],
      base: base,
      build: {
        outDir: isGitHubPages ? 'dist-github' : 'dist',
        sourcemap: true,
        rollupOptions: {
          output: {
            assetFileNames: (assetInfo) => {
              // Keep WASM files in a predictable location
              if (assetInfo.name?.endsWith('.wasm')) {
                return 'assets/[name]-[hash][extname]';
              }
              return 'assets/[name]-[hash][extname]';
            },
          },
        },
      },
      resolve: {
        alias: {
          "@": path.resolve(__dirname, "./src"),
        },
      },
      server: {
        headers: {
          'Cross-Origin-Embedder-Policy': 'credentialless',
          'Cross-Origin-Opener-Policy': 'same-origin',
        },
      },
      optimizeDeps: {
        exclude: ['onnxruntime-web'],
      },
      assetsInclude: ['**/*.wasm'],
    }
  })