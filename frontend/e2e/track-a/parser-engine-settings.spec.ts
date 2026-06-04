/**
 * A.2 acceptance criterion #1 + #2:
 *
 * Settings dropdown renders four engines, the confidence slider
 * appears only when 'auto' is chosen, and the value round-trips on
 * Save + reload.
 *
 * Fully mocked — does not require any backend service to be running
 * beyond the Next.js frontend dev server.
 */

import { expect, test } from '@playwright/test'
import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome, DEFAULT_SETTINGS } from './_track-a-helpers'

test.describe('A.2: parser-engine settings round-trip', () => {
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  test('all four engines visible; slider appears only in auto mode', async ({
    page,
  }) => {
    await mockDashboardChrome(page)

    let currentSettings = { ...DEFAULT_SETTINGS }

    await page.route('**/api/settings', async (route) => {
      const request = route.request()
      if (request.method() === 'GET') {
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(currentSettings),
        })
      }
      if (request.method() === 'PUT') {
        const body = request.postDataJSON()
        currentSettings = { ...currentSettings, ...body }
        return route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(currentSettings),
        })
      }
      return route.fallback()
    })

    // Health chip endpoint — irrelevant for this spec but the chip
    // mounts on the settings page, so respond with something stable.
    await page.route('**/api/health/mineru', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ healthy: true, version: 'e2e' }),
      })
    )

    await gotoAndWait(page, '/settings')

    // The label proves we hit the right page.
    await expect(
      page.getByText('Document Processing Engine', { exact: true })
    ).toBeVisible()

    // Open the dropdown and verify the four options. Use role queries
    // so we don't couple to Radix internals.
    const trigger = page.getByRole('combobox', {
      name: 'Document processing engine',
    })
    await trigger.click()
    await expect(
      page.getByRole('option', { name: 'Simple', exact: true })
    ).toBeVisible()
    await expect(
      page.getByRole('option', { name: 'Docling (default)' })
    ).toBeVisible()
    await expect(
      page.getByRole('option', { name: 'MinerU', exact: true })
    ).toBeVisible()
    await expect(
      page.getByRole('option', { name: /^Auto/ })
    ).toBeVisible()

    // Slider is hidden when current engine is docling.
    await expect(
      page.getByLabel('Docling confidence threshold')
    ).toHaveCount(0)

    // Switch to Auto → slider appears.
    await page.getByRole('option', { name: /^Auto/ }).click()
    const slider = page.getByLabel('Docling confidence threshold')
    await expect(slider).toBeVisible()

    // Adjust the slider via keyboard. Radix Slider exposes a focusable
    // thumb (role="slider") that responds to ArrowLeft with the step
    // value (0.01). Focus the thumb explicitly to avoid keystrokes
    // landing on the SliderRoot which is non-interactive.
    const sliderThumb = page.locator(
      '[aria-label="Docling confidence threshold"] [role="slider"]'
    )
    await sliderThumb.focus()
    for (let i = 0; i < 10; i++) {
      await page.keyboard.press('ArrowLeft')
    }

    // Save. Set up the response waiter BEFORE the click so we don't
    // race the network round-trip.
    const putResponse = page.waitForResponse(
      (response) =>
        response.url().includes('/api/settings') &&
        response.request().method() === 'PUT',
      { timeout: 20_000 }
    )
    await page.getByRole('button', { name: /Save Settings/i }).click()
    await putResponse

    expect(currentSettings.docling_min_confidence).toBeLessThan(0.95)
    expect(currentSettings.parser_engine).toBe('auto')

    // Reload to verify the persisted state re-hydrates the form.
    await page.reload()
    const sliderAfterReload = page.getByLabel('Docling confidence threshold')
    await expect(sliderAfterReload).toBeVisible()

    // Switching back to Docling hides the slider again.
    const triggerAfter = page.getByRole('combobox', {
      name: 'Document processing engine',
    })
    await triggerAfter.click()
    await page.getByRole('option', { name: 'Docling (default)' }).click()
    await expect(
      page.getByLabel('Docling confidence threshold')
    ).toHaveCount(0)
  })
})
