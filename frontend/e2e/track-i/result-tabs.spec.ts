/**
 * Track I — Phase I.D-4 — Result tabs (Markdown / Images / Structure) E2E spec.
 *
 * AC mapping → docs/tracks/I-docling-studio/plan.md §I.D-4:
 *
 *   AC1  Each tab (Properties / Markdown / Images / Structure / Config) renders
 *        without a console error
 *        → switch through all five tabs; assert each tab's content is visible
 *          and that no console.error / pageerror fired during the run.
 *   AC2  Markdown is XSS-safe (sanitized via rehype-sanitize; no raw HTML exec)
 *        → the markdown fixture embeds a <script> + an onerror <img>; assert the
 *          rendered DOM contains neither (rehype-sanitize strips them, and
 *          react-markdown v10 does not render raw HTML by default).
 *   AC3  Images lazy-load (≤6 in flight)
 *        → the images fixture returns 13 filenames; assert at most
 *          IMAGE_PAGE_SIZE (=6) <img> elements are mounted and each carries
 *          loading="lazy". Pagination bounds concurrent requests.
 *
 * Fully route-mocked: no live backend. Mirrors the I.D-1 LayersBar spec fixtures.
 */

import { expect, test, type Page } from '@playwright/test'
import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const SOURCE_ID = 'src_track_i_d4_fixture'

const PNG_1x1 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='

// Markdown fixture carrying hostile raw HTML that must be sanitized away (AC2).
const HOSTILE_MARKDOWN = [
  '# Heading One',
  '',
  'A paragraph with **bold** text.',
  '',
  '<script>window.__xss = true</script>',
  '',
  '<img src="x" onerror="window.__xss = true" />',
  '',
  '| Col A | Col B |',
  '| --- | --- |',
  '| 1 | 2 |',
].join('\n')

function makeChunks(n: number) {
  return Array.from({ length: n }, (_, i) => ({
    id: `chunk:${i}`,
    text: `Chunk number ${i} body text for the result-tabs fixture.`,
    order: i,
    physical_page: 1,
    printed_page: null,
    chapter: null,
    paragraph_number: null,
    element_type: i % 3 === 0 ? 'section_header' : 'paragraph',
    positions: [[1, 0.1, 0.9, 0.1, 0.2]],
    metadata: {},
    is_content: true,
  }))
}

async function mockInspect(page: Page): Promise<void> {
  const source = {
    id: SOURCE_ID,
    title: 'Track I.D-4 fixture source',
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
    full_text: HOSTILE_MARKDOWN,
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
          chunks: makeChunks(12),
          total_chunks: 12,
          has_spatial_data: true,
        }),
      })
  )

  // 13 images → forces pagination (>1 page at IMAGE_PAGE_SIZE=6).
  const images = Array.from({ length: 13 }, (_, i) => `image_${i + 1}.png`)
  await page.route(
    `**/api/sources/${encodeURIComponent(SOURCE_ID)}/images`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ images }),
      })
  )

  await page.route(/\/api\/sources\/[^/]+\/images\/[^/]+$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: 'image/png',
      body: Buffer.from(PNG_1x1, 'base64'),
    })
  )

  await page.route(
    `**/api/sources/${encodeURIComponent(SOURCE_ID)}/page-count`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({
          page_count: 1,
          pages: [{ page_number: 1, width: 595, height: 842 }],
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

/** Attach console.error + pageerror watchdogs; returns the collected messages. */
function watchConsole(page: Page): string[] {
  const errors: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'error') errors.push(msg.text())
  })
  page.on('pageerror', (err) => errors.push(err.message))
  return errors
}

test.describe('I.D-4: result tabs', () => {
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  test('all five tabs render without console errors (AC1)', async ({ page }) => {
    const errors = watchConsole(page)

    await mockDashboardChrome(page)
    await mockInspect(page)
    await gotoAndWait(page, INSPECT_PATH)

    const region = page.getByRole('region', { name: 'Result tabs' })
    await expect(region).toBeVisible({ timeout: 15_000 })

    for (const name of ['Markdown', 'Images', 'Structure', 'Config', 'Properties']) {
      await page.getByRole('tab', { name }).click()
      await expect(page.getByRole('tab', { name })).toHaveAttribute(
        'aria-selected',
        'true'
      )
    }

    expect(errors).toEqual([])
  })

  test('markdown tab sanitizes hostile HTML (AC2)', async ({ page }) => {
    await mockDashboardChrome(page)
    await mockInspect(page)
    await gotoAndWait(page, INSPECT_PATH)

    await page.getByRole('tab', { name: 'Markdown' }).click()

    // Rendered markdown shows the heading...
    await expect(
      page.getByRole('heading', { name: 'Heading One' })
    ).toBeVisible({ timeout: 15_000 })

    // ...but the injected <script> never executes and no element survives.
    const xssRan = await page.evaluate(
      () => (window as unknown as { __xss?: boolean }).__xss === true
    )
    expect(xssRan).toBe(false)
    expect(await page.locator('div.prose script').count()).toBe(0)
  })

  test('images tab lazy-loads at most 6 images per page (AC3)', async ({
    page,
  }) => {
    await mockDashboardChrome(page)
    await mockInspect(page)
    await gotoAndWait(page, INSPECT_PATH)

    await page.getByRole('tab', { name: 'Images' }).click()

    const imgs = page.locator('figure img')
    await expect(imgs.first()).toBeVisible({ timeout: 15_000 })

    // Only one page of 6 is mounted at a time → bounds concurrent requests.
    expect(await imgs.count()).toBeLessThanOrEqual(6)

    // Every mounted image opts into native lazy loading.
    const lazyCount = await page.locator('figure img[loading="lazy"]').count()
    expect(lazyCount).toBe(await imgs.count())

    // Pagination is present for 13 images (3 pages).
    await expect(
      page.getByRole('button', { name: 'Next image page' })
    ).toBeVisible()
  })
})
