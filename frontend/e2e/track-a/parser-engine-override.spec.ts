/**
 * A.2 acceptance criterion #6:
 *
 * The reparse modal exposes a "Parser engine for this run" Select
 * defaulting to "Use default". Choosing 'mineru' and clicking Reprocess
 * Document sends a POST to /api/sources/{id}/reprocess whose body
 * includes `parser_engine: 'mineru'`. Backend already supports this
 * (A.1b dispatcher).
 *
 * Fully mocked — no backend required.
 */

import { expect, test } from '@playwright/test'
import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from './_track-a-helpers'

test.describe('A.2: per-source parser-engine override', () => {
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  test('Reprocess POST body includes parser_engine=mineru', async ({ page }) => {
    await mockDashboardChrome(page)

    const sourceId = 'fix-reparse'
    const sourceBody = {
      id: sourceId,
      title: 'Reparse fixture',
      topics: [],
      asset: { file_path: '/tmp/fixture.pdf' },
      embedded: false,
      embedded_chunks: 0,
      insights_count: 0,
      entity_count: 0,
      relation_count: 0,
      created: '2026-06-01T00:00:00Z',
      updated: '2026-06-01T00:00:00Z',
      file_available: true,
      full_text: 'Fixture content',
      notebooks: [],
    }

    await page.route(
      `**/api/sources/${encodeURIComponent(sourceId)}`,
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify(sourceBody),
        })
    )

    await page.route(
      `**/api/sources/${encodeURIComponent(sourceId)}/chunks`,
      (route) =>
        route.fulfill({
          status: 200,
          contentType: 'application/json',
          body: JSON.stringify({
            chunks: [],
            total_chunks: 0,
            has_spatial_data: false,
          }),
        })
    )

    await page.route(`**/api/insights*`, (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      })
    )
    await page.route(`**/api/transformations*`, (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      })
    )
    await page.route(`**/api/zotero/status`, (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ connected: false }),
      })
    )

    // Capture the reparse POST so we can assert on its body.
    let capturedReprocessBody: Record<string, unknown> | null = null
    await page.route(
      `**/api/sources/${encodeURIComponent(sourceId)}/reprocess`,
      (route) => {
        if (route.request().method() === 'POST') {
          capturedReprocessBody = route.request().postDataJSON()
          return route.fulfill({
            status: 200,
            contentType: 'application/json',
            body: JSON.stringify({
              command_id: 'cmd:e2e',
              status: 'queued',
              overrides_applied: ['parser_engine'],
            }),
          })
        }
        return route.fallback()
      }
    )

    await gotoAndWait(page, `/sources/${sourceId}`)

    // Page chrome rendered.
    await expect(page.getByText('Reparse fixture').first()).toBeVisible()

    // Open the dropdown menu via the MoreVertical button and pick
    // "Reprocess Document…".
    await page.getByRole('button', { name: '' }).nth(1).click().catch(() => {})
    // More reliable: locate by the visible menu-item label after opening.
    // Use the lucide MoreVertical button via aria-haspopup.
    const moreButtons = page.locator('button[aria-haspopup="menu"]')
    await moreButtons.first().click()
    await page.getByRole('menuitem', { name: /Reprocess Document/i }).click()

    // Modal is open — the Parser engine Select should be present
    // and defaulted to "Use default".
    const parserSelect = page.getByRole('combobox', {
      name: 'Parser engine for this run',
    })
    await expect(parserSelect).toBeVisible()
    await expect(parserSelect).toContainText(/Use default/i)

    // Switch to MinerU. `exact:true` disambiguates from the
    // "Auto (docling with MinerU fallback)" option which also
    // contains 'MinerU'.
    await parserSelect.click()
    await page.getByRole('option', { name: 'MinerU', exact: true }).click()

    // Click the modal's "Reprocess Document" button. By the time the
    // modal is open, the menuitem has unmounted so the role=button
    // lookup is unambiguous.
    await page
      .getByRole('button', { name: /Reprocess Document/i })
      .click()

    // Wait for the POST to land and assert its body.
    await expect.poll(() => capturedReprocessBody, { timeout: 5000 }).not.toBeNull()
    expect(capturedReprocessBody).toMatchObject({ parser_engine: 'mineru' })
  })
})
