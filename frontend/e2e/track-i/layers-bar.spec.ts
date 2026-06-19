/**
 * Track I — Phase I.D-1 — LayersBar (element-type visibility toggles) E2E spec.
 *
 * AC mapping → docs/tracks/I-docling-studio/plan.md §I.D-1:
 *
 *   AC1  Each element type has a toggle chip
 *        → assert the "Element layer visibility" group renders a chip per
 *          element type (one of the ELEMENT_COLORS keys, e.g. "table").
 *   AC2  Toggling a chip hides the matching bbox overlays in the middle pane
 *        → click a chip, assert aria-pressed flips to "false" (hidden). The
 *          canvas overlay itself is pixel-drawn, so visibility is asserted via
 *          the chip's pressed state, which directly drives the viewer's
 *          controlled hiddenTypes prop. Pixel assertion is left to manual smoke.
 *   AC3  Hidden-types state lives in document-workspace-store
 *        → after toggling, the store reflects the hidden type (read via the
 *          chip's pressed state across both LayersBar + viewer, which share the
 *          store). Not persisted (navigation-scoped), so no localStorage check.
 *   AC4  Keyboard accessible: Space toggles the focused chip
 *        → focus a chip, press Space, assert aria-pressed flips.
 *
 * Fully route-mocked: no live backend. Mirrors the I.B inspect-workspace spec's
 * fixtures so the workspace (and thus the LayersBar) renders.
 */

import { expect, test, type Page } from '@playwright/test'
import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const SOURCE_ID = 'src_track_i_d1_fixture'

const PNG_1x1 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='

function makeChunks(n: number) {
  return Array.from({ length: n }, (_, i) => ({
    id: `chunk:${i}`,
    text: `Chunk number ${i} body text for the LayersBar fixture.`,
    order: i,
    physical_page: 1,
    printed_page: null,
    chapter: null,
    paragraph_number: null,
    element_type: i % 3 === 0 ? 'section_header' : 'paragraph',
    positions: [[1, 0.1, 0.9, 0.1 + (i % 5) * 0.05, 0.2 + (i % 5) * 0.05]],
    metadata: {},
    is_content: true,
  }))
}

async function mockInspect(page: Page): Promise<void> {
  const source = {
    id: SOURCE_ID,
    title: 'Track I.D-1 fixture source',
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
    metadata: { parser_engine_used: 'docling' },
  }

  await page.route(`**/api/sources/${encodeURIComponent(SOURCE_ID)}`, (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify(source),
    })
  )

  await page.route(
    `**/api/sources/${encodeURIComponent(SOURCE_ID)}/chunks`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          chunks: makeChunks(40),
          total_chunks: 40,
          has_spatial_data: true,
        }),
      })
  )

  await page.route(
    `**/api/sources/${encodeURIComponent(SOURCE_ID)}/page-count`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          page_count: 3,
          pages: [
            { page_number: 1, width: 595, height: 842 },
            { page_number: 2, width: 595, height: 842 },
            { page_number: 3, width: 595, height: 842 },
          ],
        }),
      })
  )

  await page.route(/\/api\/sources\/[^/]+\/page-preview.*/, (route) =>
    route.fulfill({
      status: 200,
      contentType: 'image/png',
      body: Buffer.from(PNG_1x1, 'base64'),
    })
  )

  await page.route(`**/api/insights*`, (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '[]' })
  )
  await page.route(`**/api/transformations*`, (route) =>
    route.fulfill({ status: 200, contentType: 'application/json', body: '[]' })
  )
  await page.route(`**/api/zotero/status`, (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({ connected: false }),
    })
  )
}

const INSPECT_PATH = `/sources/${encodeURIComponent(SOURCE_ID)}/inspect`

// The LayersBar labels element types with spaces (underscores stripped). "table"
// is a stable single-word key present in ELEMENT_COLORS.
const TABLE_CHIP = /^table layer (visible|hidden)$/

test.describe('I.D-1: LayersBar element-visibility toggles', () => {
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  test('renders a toggle chip per element type (AC1)', async ({ page }) => {
    const pageErrors: string[] = []
    page.on('pageerror', (err) => pageErrors.push(err.message))

    await mockDashboardChrome(page)
    await mockInspect(page)
    await gotoAndWait(page, INSPECT_PATH)

    const group = page.getByRole('group', { name: 'Element layer visibility' })
    await expect(group).toBeVisible({ timeout: 15_000 })

    // Every chip is a toggle button; there should be one per element-type key.
    const chips = group.getByRole('button', { name: /layer (visible|hidden)$/ })
    await expect(chips.first()).toBeVisible()
    expect(await chips.count()).toBeGreaterThanOrEqual(10)

    expect(pageErrors).toEqual([])
  })

  test('clicking a chip toggles its pressed (visibility) state (AC2/AC3)', async ({
    page,
  }) => {
    await mockDashboardChrome(page)
    await mockInspect(page)
    await gotoAndWait(page, INSPECT_PATH)

    const chip = page.getByRole('button', { name: TABLE_CHIP })
    await expect(chip).toBeVisible({ timeout: 15_000 })
    await expect(chip).toHaveAttribute('aria-pressed', 'true') // visible by default

    await chip.click()
    await expect(chip).toHaveAttribute('aria-pressed', 'false') // now hidden

    await chip.click()
    await expect(chip).toHaveAttribute('aria-pressed', 'true') // visible again
  })

  test('Space toggles the focused chip (AC4 keyboard)', async ({ page }) => {
    await mockDashboardChrome(page)
    await mockInspect(page)
    await gotoAndWait(page, INSPECT_PATH)

    const chip = page.getByRole('button', { name: TABLE_CHIP })
    await expect(chip).toBeVisible({ timeout: 15_000 })
    await expect(chip).toHaveAttribute('aria-pressed', 'true')

    await chip.focus()
    await expect(chip).toBeFocused()
    await page.keyboard.press('Space')
    await expect(chip).toHaveAttribute('aria-pressed', 'false')
  })

  test('"Show all" clears every hidden type', async ({ page }) => {
    await mockDashboardChrome(page)
    await mockInspect(page)
    await gotoAndWait(page, INSPECT_PATH)

    const chip = page.getByRole('button', { name: TABLE_CHIP })
    await expect(chip).toBeVisible({ timeout: 15_000 })

    await chip.click()
    await expect(chip).toHaveAttribute('aria-pressed', 'false')

    const showAll = page.getByRole('button', { name: 'Show all' })
    await expect(showAll).toBeVisible()
    await showAll.click()

    await expect(chip).toHaveAttribute('aria-pressed', 'true')
    await expect(showAll).toHaveCount(0) // hides once nothing is hidden
  })
})
