import { expect, test } from '@playwright/test'
import { gotoAndWait, isOnLoginPage } from '../_helpers'

/**
 * Track G.6 — file-watcher settings panel.
 *
 * Opens Settings → Watcher tab and asserts the status panel renders (enabled or
 * disabled badge + watched-paths section). Skips gracefully when the stack
 * requires login, matching the smoke-test convention.
 */
test.describe('track-g: file-watcher settings panel', () => {
  test('watcher tab renders the status panel', async ({ page }) => {
    await gotoAndWait(page, '/settings')
    if (await isOnLoginPage(page)) {
      test.skip(true, 'stack requires login; no seeded session for this spec')
      return
    }

    await page.getByRole('tab', { name: 'Watcher' }).click()

    // The panel title always renders; enabled/disabled badge depends on config.
    await expect(page.getByText('Inbox File-Watcher')).toBeVisible()
    const badge = page
      .getByTestId('watcher-enabled')
      .or(page.getByTestId('watcher-disabled'))
    await expect(badge).toBeVisible()
    await expect(page.getByText('Watched paths')).toBeVisible()
  })
})
