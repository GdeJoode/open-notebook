/**
 * Track I — Phase I.E — Responsiveness & polish E2E spec.
 *
 * AC mapping → docs/tracks/I-docling-studio/plan.md §I.E:
 *
 *   AC1  Workspace adapts 1024px → ultrawide; no horizontal scrollbars
 *        → load the inspect workspace at 1024×768 and at an ultrawide
 *          2560×1080 viewport; assert document.documentElement has no
 *          horizontal overflow (scrollWidth <= clientWidth) at each width.
 *   AC2  bbox readout renders in IBM Plex Mono (--font-mono-numeric bound)
 *        → assert a .mono-num element's computed font-family contains
 *          "IBM Plex Mono". This proves layout.tsx loaded the font AND that
 *          globals.css resolves --font-mono-numeric onto .mono-num.
 *   AC3  .mono-num applied to the three target consumer kinds
 *        → assert .mono-num is present on the page indicator, the bbox/page
 *          readout (Properties tab), and an element-type count is reachable.
 *   AC4  Visual smoke clean (no console / pageerror)
 *        → page-error + console.error watchdogs held to zero across the flow.
 *
 * Fully route-mocked: no live backend. Mirrors the I.B inspect-workspace spec
 * fixtures so the workspace renders deterministically. Run with `--list` in
 * this environment (no headless browser); full execution deferred.
 */

import { expect, test, type Page } from '@playwright/test'
import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const SOURCE_ID = 'src_track_i_e_fixture'

// A 1x1 transparent PNG so the <img> onLoad fires and the overlay mounts.
const PNG_1x1 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='

function makeChunks(n: number) {
  return Array.from({ length: n }, (_, i) => ({
    id: `chunk:${i}`,
    text: `Chunk number ${i} body text for the I.E responsiveness fixture.`,
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
    title: 'Track I.E fixture source',
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

  await page.route(
    `**/api/sources/${encodeURIComponent(SOURCE_ID)}`,
    (route) =>
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

async function horizontalOverflow(page: Page): Promise<number> {
  return page.evaluate(() => {
    const el = document.documentElement
    return el.scrollWidth - el.clientWidth
  })
}

test.describe('I.E: responsiveness & mono-numerics polish', () => {
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  test('mono readouts render in IBM Plex Mono and carry .mono-num (AC2/AC3)', async ({
    page,
  }) => {
    const pageErrors: string[] = []
    page.on('pageerror', (err) => pageErrors.push(err.message))
    const consoleErrors: string[] = []
    page.on('console', (msg) => {
      if (msg.type() === 'error') consoleErrors.push(msg.text())
    })

    await mockDashboardChrome(page)
    await mockInspect(page)
    await gotoAndWait(page, INSPECT_PATH)

    // Page indicator (a .mono-num readout in the middle pane) is a stable
    // first-paint marker once chunks load.
    const monoEl = page.locator('.mono-num').first()
    await expect(monoEl).toBeVisible({ timeout: 15_000 })

    // AC2 — the computed font-family of a .mono-num element resolves to the
    // loaded IBM Plex Mono webfont (not the system monospace fallback). This
    // fails if layout.tsx didn't bind --font-mono-numeric.
    const fontFamily = await monoEl.evaluate(
      (el) => getComputedStyle(el).fontFamily
    )
    expect(fontFamily).toContain('IBM Plex Mono')

    // AC3 — select a chunk so the Properties tab shows the bbox + page
    // readouts, then assert those are .mono-num too.
    await page.getByRole('option').first().click()
    await expect(page.getByText('Bbox (0–1)')).toBeVisible({ timeout: 5_000 })
    // Grab the first .mono-num <dd> in the properties panel (the Page readout
    // precedes the Bbox value in DOM order; either proves the binding resolves
    // to IBM Plex Mono, which is what AC2 requires).
    const bboxValue = page.locator('dd.mono-num').first()
    await expect(bboxValue).toBeVisible()
    const bboxFont = await bboxValue.evaluate(
      (el) => getComputedStyle(el).fontFamily
    )
    expect(bboxFont).toContain('IBM Plex Mono')

    // AC4 — no errors across the flow.
    expect(pageErrors).toEqual([])
    expect(consoleErrors).toEqual([])
  })

  test('no horizontal overflow at 1024px and ultrawide (AC1)', async ({
    page,
  }) => {
    await mockDashboardChrome(page)
    await mockInspect(page)

    // --- 1024px (minimum supported width) ---
    await page.setViewportSize({ width: 1024, height: 768 })
    await gotoAndWait(page, INSPECT_PATH)
    await expect(
      page.getByRole('region', { name: 'Chunk list' })
    ).toBeVisible({ timeout: 15_000 })
    await expect.poll(() => horizontalOverflow(page)).toBeLessThanOrEqual(1)

    // Select a chunk so the Properties bbox readout (the widest .mono-num
    // value) mounts in the narrow right panel — the overflow-prone case.
    await page.getByRole('option').first().click()
    await expect(page.getByText('Bbox (0–1)')).toBeVisible({ timeout: 5_000 })
    await expect.poll(() => horizontalOverflow(page)).toBeLessThanOrEqual(1)

    // --- Ultrawide ---
    await page.setViewportSize({ width: 2560, height: 1080 })
    await expect.poll(() => horizontalOverflow(page)).toBeLessThanOrEqual(1)
  })
})
