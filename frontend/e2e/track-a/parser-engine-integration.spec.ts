/**
 * A.3 integration spec — full auto-fallback flow + graceful degradation.
 *
 * This is the single end-to-end spec the Track A acceptance criteria
 * mandate. It is fully route-mocked: no MinerU container, no real
 * Docling, no real SurrealDB — every backend interaction is faked so
 * the spec runs deterministically on CI.
 *
 * Test 1 — Auto-fallback happy path
 *   1. Settings round-trip with parser_engine='auto' + threshold 0.95.
 *   2. /api/health/mineru → healthy.
 *   3. Source detail loads with metadata simulating a low-confidence
 *      docling result that triggered MinerU fallback.
 *   4. Assert the amber "MinerU (auto-fallback, conf 0.62)" badge with
 *      the per-signal tooltip is rendered.
 *   5. Open the Reprocess modal, switch parser engine to 'docling',
 *      submit — assert the POST body carries parser_engine: 'docling'.
 *
 * Test 2 — Graceful degradation when MinerU is down
 *   Same Settings flow but /api/health/mineru returns healthy:false.
 *   The settings chip transitions to offline; the badge for an
 *   already-extracted docling-only source still renders silently
 *   (no error UI), confirming the system degrades to docling-only
 *   without a user-visible failure.
 */

import { expect, test, type Page } from '@playwright/test'
import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome, DEFAULT_SETTINGS } from './_track-a-helpers'

const SOURCE_ID = 'src_test_e2e'

interface FallbackMetadata {
  parser_engine_used: 'docling' | 'mineru'
  extraction_confidence?: number
  extraction_confidence_signals?: Record<string, number>
  extraction_fallback_triggered?: boolean
}

const FALLBACK_METADATA: FallbackMetadata = {
  parser_engine_used: 'mineru',
  extraction_confidence: 0.62,
  extraction_confidence_signals: {
    ocr_confidence: 0.4,
    text_density: 0.5,
    heading_rate: 0.3,
    table_success: 1.0,
    image_text_ratio: 0.8,
    unknown_element_ratio: 0.7,
  },
  extraction_fallback_triggered: true,
}

const DOCLING_ONLY_METADATA: FallbackMetadata = {
  parser_engine_used: 'docling',
  extraction_confidence: 0.97,
  extraction_fallback_triggered: false,
}

async function mockSourceEndpoints(
  page: Page,
  sourceId: string,
  metadata: FallbackMetadata,
): Promise<{ capturedReprocess: () => Record<string, unknown> | null }> {
  let capturedReprocess: Record<string, unknown> | null = null

  const sourceBody = {
    id: sourceId,
    title: 'Integration fixture',
    topics: [],
    asset: { file_path: '/tmp/synthetic_clean_text.pdf' },
    embedded: false,
    embedded_chunks: 0,
    insights_count: 0,
    entity_count: 0,
    relation_count: 0,
    created: '2026-06-04T00:00:00Z',
    updated: '2026-06-04T00:00:00Z',
    file_available: true,
    full_text: 'Fixture content',
    notebooks: [],
    metadata,
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
      }),
  )

  await page.route(
    `**/api/sources/${encodeURIComponent(sourceId)}/insights`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      }),
  )

  await page.route(`**/api/insights*`, (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([]),
    }),
  )
  await page.route(`**/api/transformations*`, (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify([]),
    }),
  )
  await page.route(`**/api/zotero/status`, (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ connected: false }),
    }),
  )

  await page.route(
    `**/api/sources/${encodeURIComponent(sourceId)}/reprocess`,
    (route) => {
      if (route.request().method() === 'POST') {
        capturedReprocess = route.request().postDataJSON()
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
    },
  )

  return { capturedReprocess: () => capturedReprocess }
}

test.describe('A.3: parser-engine integration', () => {
  // The integration spec stitches several pages together, so we relax
  // the default per-test timeout to absorb dev-server first-route
  // compile latency on cold runs.
  test.setTimeout(90_000)
  test.use({ navigationTimeout: 45_000 })

  test('auto-mode happy path: settings → badge → reprocess override', async ({
    page,
  }) => {
    await mockDashboardChrome(page)

    // ---- Settings: parser_engine='auto', threshold 0.95 ----
    let currentSettings = {
      ...DEFAULT_SETTINGS,
      parser_engine: 'auto',
      docling_min_confidence: 0.95,
    }

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

    // ---- MinerU health: online ----
    await page.route('**/api/health/mineru', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ healthy: true, version: 'e2e' }),
      }),
    )

    // ---- Settings page: confirm auto + slider visible (persistence check) ----
    await gotoAndWait(page, '/settings')

    await expect(
      page.getByText('Document Processing Engine', { exact: true }),
    ).toBeVisible()

    const engineTrigger = page.getByRole('combobox', {
      name: 'Document processing engine',
    })
    await expect(engineTrigger).toContainText(/Auto/i)
    await expect(page.getByLabel('Docling confidence threshold')).toBeVisible()

    // Health chip wired correctly.
    await expect(page.getByTestId('mineru-health-chip')).toHaveAttribute(
      'data-state',
      'online',
    )

    // ---- Source detail: MinerU auto-fallback badge ----
    const { capturedReprocess } = await mockSourceEndpoints(
      page,
      SOURCE_ID,
      FALLBACK_METADATA,
    )

    await gotoAndWait(page, `/sources/${SOURCE_ID}`)
    await expect(page.getByText('Integration fixture').first()).toBeVisible()

    const badge = page.getByTestId('parser-engine-badge')
    await expect(badge).toBeVisible()
    await expect(badge).toHaveAttribute('data-state', 'mineru-auto-fallback')
    await expect(badge).toContainText(/auto-fallback/)
    await expect(badge).toContainText(/0\.62/)

    // Hover the badge → tooltip with per-signal breakdown.
    await badge.hover()
    const tooltip = page.getByRole('tooltip')
    await expect(tooltip).toBeVisible()
    await expect(tooltip).toContainText(/ocr_confidence/)
    await expect(tooltip).toContainText(/heading_rate/)

    // ---- Reprocess modal: per-source override flow ----
    const moreButtons = page.locator('button[aria-haspopup="menu"]')
    await moreButtons.first().click()
    await page
      .getByRole('menuitem', { name: /Reprocess Document/i })
      .click()

    const parserSelect = page.getByRole('combobox', {
      name: 'Parser engine for this run',
    })
    await expect(parserSelect).toBeVisible()
    await parserSelect.click()
    // Pick Docling explicitly (override the global 'auto'). The Reprocess
    // modal uses bare engine names (no "(default)" suffix) — that
    // qualifier only appears on the global settings dropdown.
    await page
      .getByRole('option', { name: 'Docling', exact: true })
      .click()

    await page.getByRole('button', { name: /Reprocess Document/i }).click()

    await expect
      .poll(() => capturedReprocess(), { timeout: 5_000 })
      .not.toBeNull()
    expect(capturedReprocess()).toMatchObject({ parser_engine: 'docling' })
  })

  test('graceful degradation: MinerU offline still renders source detail', async ({
    page,
  }) => {
    await mockDashboardChrome(page)

    // Settings: stays on auto but MinerU is down.
    await page.route('**/api/settings', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          ...DEFAULT_SETTINGS,
          parser_engine: 'auto',
          docling_min_confidence: 0.95,
        }),
      }),
    )

    await page.route('**/api/health/mineru', (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          healthy: false,
          error: 'MinerU container is intentionally down for the test',
        }),
      }),
    )

    // Settings page: chip reports offline; UI still renders.
    await gotoAndWait(page, '/settings')
    const chip = page.getByTestId('mineru-health-chip')
    await expect(chip).toBeVisible()
    await expect(chip).toHaveAttribute('data-state', 'offline', {
      timeout: 5_000,
    })

    // The form is still functional — the slider stays visible in auto mode.
    await expect(page.getByLabel('Docling confidence threshold')).toBeVisible()

    // Source detail with docling-only metadata still renders normally;
    // badge correctly stays silent for the clean docling case (no
    // misleading "extraction failed" surfacing just because MinerU is down).
    await mockSourceEndpoints(page, SOURCE_ID, DOCLING_ONLY_METADATA)

    await gotoAndWait(page, `/sources/${SOURCE_ID}`)
    await expect(page.getByText('Integration fixture').first()).toBeVisible()

    // Silent for the docling-success case — degraded mode does not
    // synthesise an error badge.
    await expect(page.getByTestId('parser-engine-badge')).toHaveCount(0)
  })
})
