import { expect, test } from '@playwright/test'
import { gotoAndWait, isOnLoginPage } from './_helpers'

/**
 * Smoke test: validates Playwright is wired correctly against the running stack.
 *
 * Hits `/` which redirects to `/notebooks` (see `frontend/src/app/page.tsx`).
 * From there the redirect chain settles at one of three legitimate states,
 * depending on backend config:
 *
 *   1. `/login` — backend has `OPEN_NOTEBOOK_PASSWORD` set and the user is
 *      not authenticated (auth-on path).
 *   2. `/notebooks` — auth is off (or user is authenticated) AND a default
 *      chat + embedding model are configured.
 *   3. `/models?setup_required=true` — auth is off (or user is authenticated)
 *      but no default models are configured. The dashboard layout
 *      (`frontend/src/app/(dashboard)/layout.tsx:43-50`) pushes here whenever
 *      `useModelStatus().valid === false`. This is the **default CI state**:
 *      the e2e workflow boots the stack with an empty `docker.env`, so no LLM
 *      API keys are present and no defaults can be set on first boot.
 *
 * Assertion strategy: rather than coupling to feature UI on each route, we
 * assert on the string "Noesis" — the app name. It appears in:
 *   - the LoginForm card title (login state)
 *   - the AppSidebar header (all dashboard routes wrapped in AppShell, which
 *     includes `/notebooks`, `/models`, `/sources`, etc.)
 *
 * That makes the assertion a true smoke check: if the app mounts on any of
 * the three legitimate landed states, "Noesis" is visible. If the page renders
 * only a Next.js runtime-error overlay, a 500 page, or a blank screen, the
 * assertion fails — which is what we want from a smoke test.
 */
test.describe('smoke', () => {
  test('app root mounts on a legitimate landed state', async ({ page }) => {
    await gotoAndWait(page, '/')

    // Wait for the redirect chain to settle on one of three valid terminal
    // states. Anything else is a regression. The trailing `(\?|$)` anchors
    // the path so we don't accidentally match a deeper nested route.
    const validLanded = /\/(notebooks|login|models)(\?|$)/
    await page.waitForURL(validLanded, { timeout: 15_000 })
    await expect(page).toHaveURL(validLanded)

    // The real smoke assertion: the app actually mounted and is rendering
    // the chrome (sidebar or login card), not a Next.js error overlay or a
    // blank screen. "Noesis" is present in all three legitimate states.
    await expect(page.getByText('Noesis').first()).toBeVisible()

    // Branch-specific sanity: the login form has an associated description
    // text that is unique to the auth state; assert it when present to make
    // failure modes easier to diagnose. (Optional secondary check; the
    // primary assertion above is what gates pass/fail.)
    if (await isOnLoginPage(page)) {
      await expect(
        page.getByText('Enter your password to access the application')
      ).toBeVisible()
    }
  })
})
