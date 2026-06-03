import { expect, test } from '@playwright/test'
import { gotoAndWait, isOnLoginPage } from './_helpers'

/**
 * Smoke test: validates Playwright is wired correctly against the running stack.
 *
 * Hits `/` which the app redirects to `/notebooks` (see `frontend/src/app/page.tsx`).
 * Depending on the backend's auth config the user lands either on:
 *   - `/notebooks` directly (no auth required), or
 *   - `/login` (auth required — dashboard layout pushes unauthenticated users away).
 *
 * Both outcomes prove the frontend booted and the dashboard route exists.
 * If you change the home redirect target, update the assertions below.
 */
test.describe('smoke', () => {
  test('dashboard root loads (auth-on or auth-off)', async ({ page }) => {
    await gotoAndWait(page, '/')

    // After the redirect chain settles, URL is either /notebooks (no auth)
    // or /login (auth gate). Anything else is a regression.
    await page.waitForURL(/\/(notebooks|login)(\?|$)/, { timeout: 15_000 })

    if (await isOnLoginPage(page)) {
      // Auth required path: the LoginForm card title is "Noesis".
      await expect(page.getByText('Noesis')).toBeVisible()
    } else {
      // No-auth path: the notebooks dashboard renders. We don't pin to a
      // specific element to avoid coupling to UI churn — proving the route
      // mounted (no Next.js error overlay) is enough for a smoke check.
      await expect(page).toHaveURL(/\/notebooks/)
      await expect(page.locator('body')).toBeVisible()
    }
  })
})
