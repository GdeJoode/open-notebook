import { defineConfig } from 'vitest/config'
import path from 'node:path'

/**
 * Vitest config for frontend unit tests (store reducers, pure helpers).
 *
 * Scope is deliberately narrow: component / DOM behavior is covered by the
 * Playwright E2E suite. Unit tests here target framework-free logic (Zustand
 * stores, utils) that benefits from fast, isolated assertions.
 */
export default defineConfig({
  test: {
    environment: 'node',
    setupFiles: ['./vitest.setup.ts'],
    include: ['src/**/*.test.ts'],
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
})
