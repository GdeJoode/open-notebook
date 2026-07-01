import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import path from 'node:path'

/**
 * Vitest config for frontend unit tests.
 *
 * Two flavours share this config:
 *   - `*.test.ts` — framework-free logic (Zustand stores, pure helpers) run in
 *     the fast `node` environment (the default here).
 *   - `*.test.tsx` — React component tests (Track UX PipelineStatus onward) that
 *     opt into jsdom via a per-file `// @vitest-environment jsdom` docblock and
 *     render with @testing-library/react.
 *
 * Broader end-to-end / DOM-integration behavior remains covered by Playwright.
 */
export default defineConfig({
  // The app's tsconfig sets `jsx: preserve` (Next handles the transform), so the
  // React plugin drives the JSX/TSX transform for component tests here.
  plugins: [react()],
  test: {
    environment: 'node',
    setupFiles: ['./vitest.setup.ts'],
    include: ['src/**/*.test.ts', 'src/**/*.test.tsx'],
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})
