import { defineConfig, devices } from '@playwright/test'
import os from 'node:os'

/**
 * Playwright configuration for Open Notebook E2E tests.
 *
 * Reads the target URL from PLAYWRIGHT_BASE_URL (default http://localhost:8502 — the
 * port the Next.js frontend is exposed on by docker-compose / `next start`).
 *
 * Run locally:
 *   npm install
 *   npm run e2e:install   # one-time chromium download
 *   npm run e2e           # headless, parallel
 *   npm run e2e:headed    # see the browser
 *   npm run e2e:debug     # step through with the inspector
 */
const baseURL = process.env.PLAYWRIGHT_BASE_URL ?? 'http://localhost:8502'

// Use ~half the available CPU cores locally; CI gets a fixed conservative value
// so the runner doesn't oversubscribe and flake. Floor at 1.
const localWorkers = Math.max(1, Math.floor(os.cpus().length / 2))

export default defineConfig({
  testDir: './e2e',
  // Helpers that aren't standalone specs are prefixed with `_` (e.g. `_helpers.ts`).
  // Match `*.spec.ts` only so we don't try to execute the helper module.
  testMatch: /.*\.spec\.ts$/,
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  workers: process.env.CI ? 2 : localWorkers,
  timeout: 30_000,
  expect: { timeout: 5_000 },
  reporter: process.env.CI
    ? [['github'], ['html', { open: 'never' }]]
    : [['list'], ['html', { open: 'never' }]],
  use: {
    baseURL,
    // Capture diagnostics only when a test fails — keeps runs fast and reports lean.
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
    video: 'retain-on-failure',
    actionTimeout: 10_000,
    navigationTimeout: 15_000,
  },
  projects: [
    {
      name: 'chromium',
      use: { ...devices['Desktop Chrome'] },
    },
  ],
})
