/**
 * A.2 acceptance criterion #3:
 *
 * The ParserEngineBadge renders four visual states driven by the
 * metadata persisted in A.1c:
 *   - docling, no fallback → no badge (legacy/default case)
 *   - mineru (manual) → blue badge "MinerU"
 *   - mineru (auto-fallback) → amber badge "MinerU (auto-fallback, …)"
 *   - extraction failed (no metadata + status='error') → red badge
 *
 * Fully mocked — no backend required.
 */

import { expect, test, type Page } from '@playwright/test'
import { mockDashboardChrome } from './_track-a-helpers'

interface SourceMetadataFixture {
  parser_engine_used?: 'docling' | 'mineru'
  extraction_confidence?: number
  extraction_confidence_signals?: Record<string, number>
  extraction_fallback_triggered?: boolean
}

async function mockSource(
  page: Page,
  sourceId: string,
  metadata: SourceMetadataFixture | null,
  status?: string,
) {
  const sourceBody = {
    id: sourceId,
    title: 'A.2 fixture source',
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
    status,
    metadata: metadata ?? undefined,
  }

  await page.route(`**/api/sources/${encodeURIComponent(sourceId)}`, (route) => {
    if (route.request().method() === 'GET') {
      return route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(sourceBody),
      })
    }
    return route.fallback()
  })

  // Side-loads from SourceDetailContent — return empty payloads so
  // the page can settle and render the badge bar.
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
  await page.route(
    `**/api/sources/${encodeURIComponent(sourceId)}/insights`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
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
}

test.describe('A.2: ParserEngineBadge visual states', () => {
  // Dev-server first-route compile can blow past the default 15s
  // navigationTimeout on cold runs. CI uses a built/prod bundle so
  // this only relaxes the local-dev path.
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  test('docling with no fallback renders no badge', async ({ page }) => {
    await mockDashboardChrome(page)
    await mockSource(page, 'fix-docling', {
      parser_engine_used: 'docling',
      extraction_fallback_triggered: false,
      extraction_confidence: 0.98,
    })

    await page.goto('/sources/fix-docling')

    // Wait for the page chrome to mount.
    await expect(page.getByText('A.2 fixture source').first()).toBeVisible()

    // The badge must be absent — default docling success is silent.
    await expect(page.getByTestId('parser-engine-badge')).toHaveCount(0)
  })

  test('mineru manual selection renders blue badge', async ({ page }) => {
    await mockDashboardChrome(page)
    await mockSource(page, 'fix-mineru', {
      parser_engine_used: 'mineru',
      extraction_fallback_triggered: false,
    })

    await page.goto('/sources/fix-mineru')

    const badge = page.getByTestId('parser-engine-badge')
    await expect(badge).toBeVisible()
    await expect(badge).toHaveAttribute('data-state', 'mineru-manual')
    await expect(badge).toContainText(/MinerU/)
  })

  test('auto-fallback renders amber badge with confidence label', async ({
    page,
  }) => {
    await mockDashboardChrome(page)
    await mockSource(page, 'fix-fallback', {
      parser_engine_used: 'mineru',
      extraction_fallback_triggered: true,
      extraction_confidence: 0.73,
      extraction_confidence_signals: { ocr: 0.81, layout: 0.65 },
    })

    await page.goto('/sources/fix-fallback')

    const badge = page.getByTestId('parser-engine-badge')
    await expect(badge).toBeVisible()
    await expect(badge).toHaveAttribute('data-state', 'mineru-auto-fallback')
    await expect(badge).toContainText(/auto-fallback/)
    await expect(badge).toContainText(/0\.73/)
  })

  test('extraction failed renders red badge', async ({ page }) => {
    await mockDashboardChrome(page)
    // metadata: null + status: error → the conservative "extraction failed" branch.
    await mockSource(page, 'fix-failed', null, 'error')

    await page.goto('/sources/fix-failed')

    const badge = page.getByTestId('parser-engine-badge')
    await expect(badge).toBeVisible()
    await expect(badge).toHaveAttribute('data-state', 'extraction-failed')
    await expect(badge).toContainText(/extraction failed/i)
  })
})
