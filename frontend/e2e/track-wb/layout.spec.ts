/**
 * Track WB — Phase WB.4 — layout system and typography.
 *
 * AC (docs/tracks/WB-pipeline-workbench/plan.md §WB.4):
 * - every dashboard page renders one PageHeader (with the page's only h1)
 *   inside one PageBody. The pages are read from
 *   `src/app/(dashboard)/**\/page.tsx`, not listed;
 * - a screenshot of each page at 1440 and 1024 px (mocked API). The repo keeps
 *   no screenshot baselines, so they are written as artifacts: to
 *   `$WB4_SCREENSHOT_DIR` when set, else the test's output folder;
 * - at 1366x768 the sidebar fits: its nav scrolls on its own, the document
 *   does not, and Theme and Sign Out are on screen;
 * - the notebook header's rare actions sit behind "⋯"; Delete still confirms.
 *
 * Reads a page's data mostly does not matter to its frame: anything not
 * served below answers 404, as a server would for a missing route, and each
 * page shows its own error state inside its frame.
 */

import { readdirSync } from 'node:fs'
import path from 'node:path'

import { expect, test, type Page, type Route } from '@playwright/test'

import { gotoAndWait } from '../_helpers'
import { DEFAULT_SETTINGS, mockDashboardChrome } from '../track-a/_track-a-helpers'
import { NOTEBOOK, backend } from '../track-oe/_ontology-backend'
import attention from './attention.fixture.json'
import board from './sources-board.fixture.json'
import { NOTEBOOKS, fakeCounts, fakeList } from './sources-board.fake'

const DASHBOARD = path.resolve(__dirname, '../../src/app/(dashboard)')
const SOURCE = 'source:wb4'

function findPages(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap((entry) => {
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) return findPages(full)
    return entry.name === 'page.tsx' ? [full] : []
  })
}

/** A dynamic segment's value, by the segment before it. */
const PARAMS: Record<string, string> = { notebooks: NOTEBOOK, sources: SOURCE }

function routeOf(file: string): string {
  const segments = path.relative(DASHBOARD, path.dirname(file)).split(path.sep).filter(Boolean)
  return '/' + segments.map((seg, i) => {
    if (!seg.startsWith('[')) return seg
    const value = PARAMS[segments[i - 1]]
    if (!value) throw new Error(`no fixture id for ${seg} in ${file}`)
    return encodeURIComponent(value)
  }).join('/')
}

const ROUTES = findPages(DASHBOARD).map(routeOf).sort()

function json(route: Route, body: unknown, status = 200): Promise<void> {
  return route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
}

function apiSource(id: string, title: string, stage: string) {
  return {
    id, title, topics: [], asset: null, embedded: true, embedded_chunks: 4, insights_count: 0,
    entity_count: 5, relation_count: 2, created: '2026-09-29T10:00:00Z',
    updated: '2026-09-29T10:00:00Z', full_text: 'The convenant text.', metadata: {},
    private: false, notebooks: [NOTEBOOK], processing_stage: stage, command_id: null, status: null,
  }
}

async function mockBackend(page: Page) {
  await page.route('**/api/**', (route) => json(route, { detail: 'Not Found' }, 404))
  await mockDashboardChrome(page)
  await page.route('**/api/attention**', (route) => json(route, attention))
  await page.route('**/api/settings', (route) => json(route, DEFAULT_SETTINGS))

  const api = backend()
  await page.route('**/api/ontologies**', api.handle)
  await page.route(/\/api\/notebooks(\?.*)?$/, (route) =>
    json(route, [{ id: NOTEBOOK, name: 'Regio Deals', description: 'Convenants and projects',
      archived: false, created: '2026-09-01T00:00:00Z', updated: '2026-09-28T00:00:00Z',
      source_count: 3, note_count: 0 },
    ...NOTEBOOKS.map((nb) => ({ ...nb, description: '', archived: false,
      created: '2026-09-01T00:00:00Z', updated: '2026-09-01T00:00:00Z', source_count: 0,
      note_count: 0 }))]))
  await page.route(/\/api\/notebooks\/(?:notebook:oe5|notebook%3Aoe5)(?:[/?].*)?$/, api.handleNotebook)

  await page.route(/\/api\/sources\/?(\?.*)?$/, (route) =>
    json(route, fakeList(new URL(route.request().url()).searchParams)))
  await page.route(/\/api\/sources\/counts(\?.*)?$/, (route) =>
    json(route, fakeCounts(new URL(route.request().url()).searchParams)))
  const re = SOURCE.replace(':', '(?::|%3A)')
  await page.route(new RegExp(`/api/sources/${re}/insights$`), (r) => json(r, []))
  await page.route(new RegExp(`/api/sources/${re}/chunks$`), (r) =>
    json(r, { total_chunks: 0, chunks: [] }))
  await page.route(new RegExp(`/api/sources/${re}/status$`), (r) => json(r, { status: 'completed' }))
  await page.route(new RegExp(`/api/sources/${re}/logs`), (r) =>
    r.fulfill({ status: 200, contentType: 'text/event-stream', body: '' }))
  await page.route(new RegExp(`/api/sources/${re}$`), (r) =>
    json(r, apiSource(SOURCE, '410a65bba3f8__Convenant Regio Deal Achterhoek.pdf', 'embedded')))

  await page.route('**/api/triage/queue**', (r) => json(r, { items: [], total: 0 }))
  await page.route('**/api/entity-resolution/plan**', (r) => json(r, { clusters: [], cluster_count: 0 }))
  await page.route('**/api/entity-resolution/candidates**', (r) =>
    json(r, { auto_merge: [], review: [], auto_merge_count: 0, review_count: 0 }))
  await page.route('**/api/transformations*', (r) => json(r, []))
  const entity = (id: string, name: string, type: string, degree: number, confidence: number) => ({
    id, name, entity_type: type, weight: 1, confidence, status: 'active',
    structural_degree: degree, doc_count: 2 })
  await page.route(/\/api\/knowledge-graph\/entities(\?.*)?$/, (r) => json(r, {
    items: [entity('entity:1', 'Regio Deal Achterhoek', 'programme', 14, 0.96),
      entity('entity:2', 'Provincie Gelderland', 'organization', 9, 0.93),
      entity('entity:3', 'Streekagenda', 'concept', 4, 0.81)],
    total: 3, limit: 50, offset: 0 }))
  await page.route(/\/api\/knowledge-graph\/entity-types(\?.*)?$/, (r) => json(r, [
    { entity_type: 'programme', count: 1 }, { entity_type: 'organization', count: 1 },
    { entity_type: 'concept', count: 1 }]))
}

/** Every page is framed the same way, whatever its data did. */
async function expectFramed(page: Page, route: string) {
  const header = page.locator('[data-page-body] [data-page-header]')
  await expect(header, route).toHaveCount(1, { timeout: 60_000 })
  await expect(page.locator('[data-page-body]'), route).toHaveCount(1)
  await expect(header.locator('h1'), route).toHaveCount(1)
  await expect(page.locator('main h1'), route).toHaveCount(1)
  // A wide page scrolls or clips inside its body; it never squeezes the shell.
  const shell = await page.evaluate(() => ({
    sidebar: document.querySelector('.app-sidebar')!.getBoundingClientRect().width,
    scrollWidth: document.documentElement.scrollWidth,
    innerWidth: window.innerWidth,
  }))
  expect(shell.sidebar, route).toBe(256)
  expect(shell.scrollWidth, route).toBeLessThanOrEqual(shell.innerWidth)
}

const shotDir = () => process.env.WB4_SCREENSHOT_DIR

test.describe('WB.4 — one page frame, one type scale', () => {
  test.setTimeout(240_000)
  test.use({ navigationTimeout: 60_000 })

  test('the pages are derived from the app, each with a fixture id', () => {
    expect(ROUTES).toContain('/')
    expect(ROUTES).toContain(`/sources/${encodeURIComponent(SOURCE)}/inspect`)
    expect(ROUTES.length).toBeGreaterThanOrEqual(19)
  })

  for (const width of [1440, 1024]) {
    test(`every dashboard page has its header and body at ${width} px`, async ({ page }, info) => {
      // Twenty pages, some compiled cold by next dev.
      test.setTimeout(600_000)
      await page.setViewportSize({ width, height: 900 })
      await mockBackend(page)
      for (const route of ROUTES) {
        await gotoAndWait(page, route)
        await expectFramed(page, route)
        // Let the page's reads settle before the picture.
        await page.waitForLoadState('networkidle').catch(() => undefined)
        const name = `${width}${route === '/' ? '-tray' : route.replace(/[/%:]+/g, '-')}.png`
        await page.screenshot({ path: shotDir() ? path.join(shotDir()!, name) : info.outputPath(name) })
      }
    })
  }

  test('at 1366x768 the sidebar scrolls its nav, and Sign Out and Theme stay on screen', async ({ page }) => {
    await page.setViewportSize({ width: 1366, height: 768 })
    await mockBackend(page)
    await page.addInitScript(() => window.localStorage.setItem('sidebar.moreOpen', 'true'))
    await gotoAndWait(page, '/sources')
    await expectFramed(page, '/sources')

    const signOut = page.getByRole('button', { name: 'Sign Out' })
    await expect(signOut).toBeInViewport({ ratio: 1 })
    await expect(page.getByRole('button', { name: /theme/i }).first()).toBeInViewport({ ratio: 1 })

    const doc = await page.evaluate(() => ({
      scroll: document.documentElement.scrollHeight,
      inner: window.innerHeight,
    }))
    expect(doc.scroll).toBeLessThanOrEqual(doc.inner)

    const nav = page.locator('.app-sidebar nav').first()
    const box = await nav.evaluate((el) => ({
      scroll: el.scrollHeight, client: el.clientHeight, overflow: getComputedStyle(el).overflowY,
    }))
    // With More open the entries do not fit 768 px: they scroll in the nav.
    expect(box.scroll).toBeGreaterThan(box.client)
    expect(box.overflow).toBe('auto')
    await page.getByRole('navigation').getByRole('link', { name: /^Summaries/ }).scrollIntoViewIfNeeded()
    await expect(page.getByRole('navigation').getByRole('link', { name: /^Summaries/ })).toBeInViewport()
    await expect(signOut).toBeInViewport({ ratio: 1 })
  })

  test('at 1366 px the Sources Stage and Created cells keep to one line', async ({ page }) => {
    await page.setViewportSize({ width: 1366, height: 768 })
    // The fixture's rows then read "in about 8 hours", the longest the reviewer saw.
    await page.clock.setFixedTime(Date.parse(board.generated_at) - 8 * 3600_000)
    await mockBackend(page)
    await gotoAndWait(page, '/sources')
    await expectFramed(page, '/sources')
    await expect(page.locator('[data-testid^="source-row-"]').first()).toBeVisible()
    const cells = await page.evaluate(() => {
      const lines = (el: Element) => {
        const range = document.createRange()
        range.selectNodeContents(el)
        return new Set([...range.getClientRects()].map((r) => Math.round(r.top))).size
      }
      return [...document.querySelectorAll('[data-testid^="source-row-"]')].flatMap((row) => [
        { what: 'stage', text: row.querySelector('[data-testid="pipeline-stage-label"]')!.textContent,
          lines: lines(row.querySelector('[data-testid="pipeline-stage-label"]')!) },
        { what: 'created', text: row.children[3].textContent, lines: lines(row.children[3]) },
      ])
    })
    expect(cells.map((c) => c.text)).toContain('Awaiting schema review')
    expect(cells.some((c) => /^in about \d+ hours$|^about \d+ hours ago$/.test(c.text ?? ''))).toBe(true)
    expect(cells.filter((c) => c.lines > 1)).toEqual([])
  })

  test('the old /advanced address opens Settings on its Advanced tab', async ({ page }) => {
    await mockBackend(page)
    await gotoAndWait(page, '/advanced')
    await expect(page).toHaveURL(/\/settings\?tab=advanced$/, { timeout: 45_000 })
    await expect(page.getByRole('tab', { name: 'Advanced' })).toHaveAttribute('aria-selected', 'true')
  })

  test('at 1024 px the source page tabs keep each label whole and apart', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 900 })
    await mockBackend(page)
    await gotoAndWait(page, `/sources/${encodeURIComponent(SOURCE)}`)
    await expectFramed(page, 'source')
    const tabs = page.locator('[data-page-body] [role="tablist"]').first().getByRole('tab')
    await expect(tabs).toHaveCount(6)
    const boxes = await tabs.evaluateAll((els) => els.map((el) => {
      const r = el.getBoundingClientRect()
      return { left: r.left, right: r.right, clipped: el.scrollWidth > el.clientWidth + 1 }
    }))
    for (let i = 0; i < boxes.length; i++) {
      expect(boxes[i].clipped, `tab ${i} clipped`).toBe(false)
      if (i > 0) expect(boxes[i].left, `tab ${i} overlaps`).toBeGreaterThanOrEqual(boxes[i - 1].right)
    }
  })

  test('every dialog and popover the notebook menus open closes back to a working page', async ({ page }) => {
    // Review round 1: a Dialog opened from a modal menu's onSelect left
    // `pointer-events: none` on <body> once it closed; nothing responded.
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockBackend(page)
    await gotoAndWait(page, `/notebooks/${encodeURIComponent(NOTEBOOK)}`)
    await expectFramed(page, 'notebook')
    const header = page.locator('[data-page-header]')
    const exportMenu = header.getByTestId('notebook-export-menu')
    const moreMenu = header.getByRole('button', { name: 'More notebook actions' })

    const targets = [
      { menu: exportMenu, item: 'To Obsidian…', layer: page.getByTestId('obsidian-export-dialog'),
        cancel: page.getByTestId('obsidian-export-cancel') },
      { menu: exportMenu, item: 'To Open Knowledge Format…', layer: page.getByTestId('okf-export-dialog'),
        cancel: page.getByTestId('okf-export-cancel') },
      { menu: exportMenu, item: 'As JSONL…', layer: page.getByTestId('jsonl-export-popover'),
        cancel: page.getByTestId('jsonl-export-cancel') },
      { menu: moreMenu, item: 'Import OKF…', layer: page.getByTestId('okf-import-dialog'),
        cancel: page.getByTestId('okf-import-cancel') },
      { menu: moreMenu, item: 'Delete…', layer: page.getByRole('alertdialog'),
        cancel: page.getByRole('alertdialog').getByRole('button', { name: 'Cancel' }) },
    ]
    for (const t of targets) {
      for (const open of ['mouse', 'keyboard'] as const) {
        for (const close of ['Escape', 'button'] as const) {
          const what = `${t.item} (${open}, ${close})`
          if (open === 'mouse') {
            await t.menu.click()
            await page.getByRole('menuitem', { name: t.item }).click()
          } else {
            await t.menu.focus()
            await page.keyboard.press('Enter')
            await page.getByRole('menuitem', { name: t.item }).focus()
            await page.keyboard.press('Enter')
          }
          await expect(t.layer, what).toBeVisible()
          if (close === 'Escape') await page.keyboard.press('Escape')
          else await t.cancel.click()
          await expect(t.layer, what).toBeHidden()

          await expect.poll(() => page.evaluate(() => getComputedStyle(document.body).pointerEvents),
            { message: what }).toBe('auto')
          // The page still takes a click: the description turns into its editor.
          await header.getByRole('button', { name: 'Add a description...' }).click({ timeout: 5_000 })
          await expect(header.getByRole('textbox'), what).toBeVisible()
          await page.keyboard.press('Escape')
          await expect(header.getByRole('textbox'), what).toBeHidden()
        }
      }
    }
  })

  test('the notebook header: Export ▾ and ⋯, Delete confirms before anything is deleted', async ({ page }) => {
    await page.setViewportSize({ width: 1440, height: 900 })
    await mockBackend(page)
    const deletes: string[] = []
    page.on('request', (req) => { if (req.method() === 'DELETE') deletes.push(req.url()) })
    await gotoAndWait(page, `/notebooks/${encodeURIComponent(NOTEBOOK)}`)
    await expectFramed(page, 'notebook')

    const header = page.locator('[data-page-header]')
    await expect(header.getByRole('button', { name: /^(Delete|Archive|Import OKF)/ })).toHaveCount(0)
    await header.getByTestId('notebook-export-menu').click()
    const menu = page.getByRole('menu')
    await expect(menu.getByRole('menuitem')).toHaveText(
      ['To Obsidian…', 'To Open Knowledge Format…', 'As JSONL…'])
    await menu.getByRole('menuitem', { name: 'As JSONL…' }).click()
    const popover = page.getByTestId('jsonl-export-popover')
    await expect(popover).toBeVisible()
    await page.waitForTimeout(300)
    await expect(popover).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(popover).toBeHidden()
    await expect(header.getByTestId('notebook-export-menu')).toBeFocused()

    await header.getByRole('button', { name: 'More notebook actions' }).click()
    await expect(page.getByRole('menu').getByRole('menuitem')).toHaveText(
      ['Import OKF…', 'Archive', 'Delete…'])
    await page.getByRole('menuitem', { name: 'Delete…' }).click()
    const confirm = page.getByRole('alertdialog')
    await expect(confirm).toContainText('cannot be undone')
    await confirm.getByRole('button', { name: 'Cancel' }).click()
    await expect(confirm).toBeHidden()
    expect(deletes).toEqual([])
  })
})
