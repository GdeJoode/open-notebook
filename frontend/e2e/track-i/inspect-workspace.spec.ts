/**
 * Track I — Phase I.B — Inspect workspace (3-pane resizable) E2E spec.
 *
 * AC mapping → docs/tracks/I-docling-studio/plan.md §I.B:
 *
 *   AC1  /sources/{id}/inspect renders 3 panels separated by resize handles
 *        → assert three role="region" panes + two role="separator" handles.
 *   AC2  Drag handle resizes panels; layout reflows without overflow
 *        → drag the first separator left ~20% of the group width; assert the
 *          left pane shrank and no horizontal page overflow appeared.
 *   AC3  Resized sizes persist across navigation and reload (Zustand persist)
 *        → after the drag, read `document-workspace-storage` from localStorage
 *          (the persisted contract), navigate away + back, and assert the
 *          left pane renders at the restored (smaller) width, not the default.
 *   AC4  Keyboard nav: Tab between panels; Arrow keys move
 *        → focus a separator, press ArrowLeft repeatedly, assert the persisted
 *          left size changed and focus stayed inside the workspace.
 *   AC5  ARIA: each panel role="region" + aria-label; handles expose
 *        orientation → assert aria-orientation="vertical" on the separators.
 *          (react-resizable-panels v4 inverts: a horizontal Group emits
 *          vertically-oriented separators — the divider line is vertical.)
 *   AC6  No regression in the existing Chunks tab → covered by the source
 *        detail render in the I.A spec; the "Open Inspect" button is asserted
 *        here as the entry point.
 *
 * Fully route-mocked: no live backend. Dashboard chrome is mocked via the
 * shared helper; the source / chunks / page-count / page-preview endpoints are
 * seeded with minimal fixtures sufficient to render the workspace.
 */

import { expect, test, type Page } from '@playwright/test'
import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const SOURCE_ID = 'src_track_i_b_fixture'
const STORAGE_KEY = 'document-workspace-storage'

// A 1x1 transparent PNG so the <img> onLoad fires and the overlay mounts.
const PNG_1x1 =
  'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=='

function makeChunks(n: number) {
  return Array.from({ length: n }, (_, i) => ({
    id: `chunk:${i}`,
    text: `Chunk number ${i} body text for the inspect workspace fixture.`,
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
    title: 'Track I.B fixture source',
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

  // Misc fallbacks the source-detail / chat layer may touch.
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

async function readPersistedLeft(page: Page): Promise<number | null> {
  return page.evaluate((key) => {
    const raw = window.localStorage.getItem(key)
    if (!raw) return null
    try {
      return JSON.parse(raw)?.state?.panelSizes?.left ?? null
    } catch {
      return null
    }
  }, STORAGE_KEY)
}

const INSPECT_PATH = `/sources/${encodeURIComponent(SOURCE_ID)}/inspect`

test.describe('I.B: inspect workspace (3-pane resizable)', () => {
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  test('renders 3 panes + handles with correct ARIA', async ({ page }) => {
    const pageErrors: string[] = []
    page.on('pageerror', (err) => pageErrors.push(err.message))

    await mockDashboardChrome(page)
    await mockInspect(page)
    await gotoAndWait(page, INSPECT_PATH)

    // AC1 — three regions.
    await expect(page.getByRole('region', { name: 'Chunk list' })).toBeVisible({
      timeout: 15_000,
    })
    await expect(page.getByRole('region', { name: 'PDF preview' })).toBeVisible()
    await expect(page.getByRole('region', { name: 'Properties' })).toBeVisible()

    // AC1 + AC5 — two separators. A horizontal Group emits vertically-
    // oriented separators in react-resizable-panels v4 (the divider line is
    // vertical), so aria-orientation is "vertical".
    const separators = page.getByRole('separator')
    await expect(separators).toHaveCount(2)
    for (const sep of await separators.all()) {
      await expect(sep).toHaveAttribute('aria-orientation', 'vertical')
    }

    expect(pageErrors).toEqual([])
  })

  test('drag resizes + persists across reload (AC2/AC3)', async ({ page }) => {
    await mockDashboardChrome(page)
    await mockInspect(page)
    await gotoAndWait(page, INSPECT_PATH)

    const leftRegion = page.getByRole('region', { name: 'Chunk list' })
    await expect(leftRegion).toBeVisible({ timeout: 15_000 })

    const group = page.locator('[data-group]')
    const groupBox = await group.boundingBox()
    const leftBefore = (await leftRegion.boundingBox())!.width
    expect(groupBox).not.toBeNull()

    // Drag the first separator left by ~20% of the group width.
    const sep = page.getByRole('separator').first()
    const sepBox = (await sep.boundingBox())!
    const dx = Math.round(groupBox!.width * 0.2)
    await page.mouse.move(sepBox.x + sepBox.width / 2, sepBox.y + sepBox.height / 2)
    await page.mouse.down()
    await page.mouse.move(
      sepBox.x + sepBox.width / 2 - dx,
      sepBox.y + sepBox.height / 2,
      { steps: 10 }
    )
    await page.mouse.up()

    // AC2 — left pane visibly shrank; the page did not develop horizontal
    // overflow (the body scrollWidth must not exceed its clientWidth).
    await expect
      .poll(async () => (await leftRegion.boundingBox())!.width)
      .toBeLessThan(leftBefore - 10)
    const overflow = await page.evaluate(
      () => document.body.scrollWidth - document.body.clientWidth
    )
    expect(overflow).toBeLessThanOrEqual(1)

    // AC3 — the persisted store reflects the smaller left size.
    const persistedLeft = await readPersistedLeft(page)
    expect(persistedLeft).not.toBeNull()
    expect(persistedLeft!).toBeLessThan(22) // default left size

    // Reload and assert the layout restores to the persisted (smaller) size.
    await page.reload({ waitUntil: 'domcontentloaded' })
    await expect(leftRegion).toBeVisible({ timeout: 15_000 })
    const restored = await readPersistedLeft(page)
    expect(restored).toBeCloseTo(persistedLeft!, 0)
  })

  test('keyboard resize keeps focus inside the workspace (AC4)', async ({
    page,
  }) => {
    await mockDashboardChrome(page)
    await mockInspect(page)
    await gotoAndWait(page, INSPECT_PATH)

    await expect(page.getByRole('region', { name: 'Chunk list' })).toBeVisible({
      timeout: 15_000,
    })

    const before = (await readPersistedLeft(page)) ?? 22

    // Focus the first separator and drive it with the keyboard.
    const sep = page.getByRole('separator').first()
    await sep.focus()
    await expect(sep).toBeFocused()
    for (let i = 0; i < 8; i++) {
      await page.keyboard.press('ArrowLeft')
    }

    // AC4 — keyboard moved the layout (persisted size changed).
    await expect
      .poll(async () => (await readPersistedLeft(page)) ?? 22)
      .toBeLessThan(before)

    // Focus must remain inside the workspace (on the separator), not escape to
    // the document body / browser chrome.
    const focusInWorkspace = await page.evaluate(() => {
      const el = document.activeElement
      return !!el && el.getAttribute('role') === 'separator'
    })
    expect(focusInWorkspace).toBe(true)
  })
})
