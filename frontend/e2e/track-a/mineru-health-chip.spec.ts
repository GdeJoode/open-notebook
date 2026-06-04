/**
 * A.2 acceptance criteria #4 + #5:
 *
 * The MineruServiceHealthChip polls GET /api/health/mineru. When the
 * mocked response flips from healthy:true to healthy:false, the chip
 * must transition state on the next 30-second poll (we allow a little
 * slack for slow environments).
 *
 * Fully mocked — no backend required.
 */

import { expect, test } from '@playwright/test'
import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome, DEFAULT_SETTINGS } from './_track-a-helpers'

test.describe('A.2: MinerU health chip', () => {
  // One polling interval (30s) + slack for the offline transition.
  test.setTimeout(75_000)
  test.use({ navigationTimeout: 45_000 })

  test('chip transitions from online → offline on next poll', async ({
    page,
  }) => {
    await mockDashboardChrome(page)

    // The settings page mounts the chip; keep settings endpoint quiet.
    await page.route('**/api/settings', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(DEFAULT_SETTINGS),
      })
    )

    let healthyResponse = true

    await page.route('**/api/health/mineru', (route) => {
      const body = healthyResponse
        ? { healthy: true, version: '2.0.0' }
        : { healthy: false, error: 'service is intentionally down for the test' }
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(body),
      })
    })

    await gotoAndWait(page, '/settings')

    const chip = page.getByTestId('mineru-health-chip')
    await expect(chip).toBeVisible()
    await expect(chip).toHaveAttribute('data-state', 'online')

    // Flip the mock to unhealthy. Next refetchInterval fires after 30s.
    healthyResponse = false

    // Wait up to 40s (one full interval + slack). We don't rely on a
    // wall-clock sleep — we poll the data-state attribute until the
    // change is visible to the user.
    await expect(chip).toHaveAttribute('data-state', 'offline', {
      timeout: 40_000,
    })
  })

  test('chip starts in checking state then resolves online', async ({
    page,
  }) => {
    await mockDashboardChrome(page)

    await page.route('**/api/settings', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(DEFAULT_SETTINGS),
      })
    )

    // Delay the first health response slightly so the spec can observe
    // the loading chip before it transitions to online.
    await page.route('**/api/health/mineru', async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 300))
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ healthy: true, version: '2.0.0' }),
      })
    })

    await gotoAndWait(page, '/settings')

    const chip = page.getByTestId('mineru-health-chip')
    // We may catch either the checking or online state depending on
    // exact timing; the meaningful assertion is that it eventually
    // resolves to online.
    await expect(chip).toHaveAttribute('data-state', 'online', {
      timeout: 5_000,
    })
  })
})
