import type { Page, Response } from '@playwright/test'

/**
 * Shared utilities for Playwright E2E specs.
 *
 * Keep this module small: only export helpers that are actually used by a spec.
 * Track-specific helpers (e.g. seedNotebook, waitForJobComplete) should be added
 * here on demand when the first spec that needs them lands — premature helpers
 * become dead code fast.
 */

/**
 * Navigate to a path and wait for the network to settle.
 *
 * Returned `Response` is the navigation response (or `null` if the page used a
 * client-side redirect — common in this app because `/` redirects to `/notebooks`).
 * Callers should not assert on the response when redirects are expected.
 */
export async function gotoAndWait(page: Page, path: string): Promise<Response | null> {
  return page.goto(path, { waitUntil: 'domcontentloaded' })
}

/**
 * Returns true when the current page is showing the login form
 * (i.e. the backend has `OPEN_NOTEBOOK_PASSWORD` set and the user isn't
 * authenticated yet). Used by smoke tests so they pass against both an
 * auth-enabled and auth-disabled stack.
 */
export async function isOnLoginPage(page: Page): Promise<boolean> {
  return page.url().includes('/login')
}
