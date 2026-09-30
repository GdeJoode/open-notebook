/**
 * Track WB — Phase WB.5 — responsive.
 *
 * AC (docs/tracks/WB-pipeline-workbench/plan.md §WB.5, WB-D3: desktop first,
 * usable on a tablet, read-only on a phone):
 * - no horizontal page scroll at 1024, 768 and 390 px on The Tray, Sources,
 *   a source, Triage, a notebook, the Knowledge Graph and Ontologies
 *   (`document.documentElement.scrollWidth <= innerWidth`);
 * - nothing interactive is cut off: every control is inside the viewport and
 *   its boxes (a tab strip may scroll sideways to it);
 * - nothing is drawn over a control: a tap on it reaches it;
 * - at 390 px every control is at least 40 px tall and wide (inline links in
 *   running text excepted, as WCAG 2.5.8 excepts them);
 * - a screenshot of each page at each width, written to
 *   `$WB5_SCREENSHOT_DIR` when set, else the test's output folder.
 * And the pieces behind it: the sidebar's icons with their badges below xl,
 * the phone drawer, the Sources title at 1024 (WB.4 m1), the cards, and the
 * KG details sheet.
 */

import path from 'node:path'

import { expect, test, type Page } from '@playwright/test'

import { gotoAndWait } from '../_helpers'
import { NOTEBOOK, SOURCE, apiSource, json, mockBackend } from './_wb-backend'

const enc = encodeURIComponent

const PAGES = [
  { name: 'tray', route: '/' },
  { name: 'sources', route: '/sources' },
  { name: 'source', route: `/sources/${enc(SOURCE)}` },
  { name: 'triage', route: '/knowledge-graph/triage' },
  { name: 'notebook', route: `/notebooks/${enc(NOTEBOOK)}` },
  { name: 'knowledge-graph', route: '/knowledge-graph' },
  { name: 'ontologies', route: '/ontologies' },
]

const WIDTHS = [1024, 768, 390]

const TRIAGE_ROW = {
  id: 'triage_queue:t1', entity: 'entity:1', name: 'Regio Deal Achterhoek', type: 'Programme',
  entity_type: 'programme', structural_degree: 14, doc_count: 2, reason: 'isolated core actor',
  decision: 'open', batch_id: 'b1', kind: 'status', docs: [],
}

async function backend(page: Page) {
  await mockBackend(page)
  // A source in the notebook, so its column has a card with a menu.
  await page.route(/\/api\/sources\/?\?.*notebook_id=notebook(?::|%3A)oe5.*$/, (r) => json(r, [{
    ...apiSource('source:nb1', 'Convenant Regio Deal Achterhoek', 'complete'),
    stalled: false, needs_you: false }]))
  await page.route('**/api/triage/queue**', (r) => json(r, { items: [TRIAGE_ROW], total: 1 }))
  await page.route(/\/api\/knowledge-graph\/entities\/entity(?::|%3A)1(\?.*)?$/, (r) => json(r, {
    id: 'entity:1', name: 'Regio Deal Achterhoek', entity_type: 'programme', weight: 1,
    confidence: 0.96, status: 'active', structural_degree: 14, doc_count: 2,
    relations: [{ id: 'relation:1', source: 'entity:1', target: 'entity:2', relation_type: 'part_of',
      confidence: 0.9 }],
    source_ids: [SOURCE] }))
}

interface Control { what: string; width: number; height: number; inlineText: boolean
  cutOff: boolean; coveredBy: string | null; invisible: boolean }

/**
 * The page's controls as the owner meets them. For each one on screen:
 * - its target box: the control, with its <label>s (a label click is a click
 *   on the control), and for a slider thumb the whole slider (a press on the
 *   track moves it);
 * - `cutOff`: part of it is clipped by a box that does not scroll sideways
 *   (or by the viewport), so that part cannot be reached;
 * - `coveredBy`: what a tap at the middle of its visible part hits, when that
 *   is not the control (another control or a panel drawn over it).
 * Controls scrolled out of their scroll box are reachable by scrolling and
 * are skipped; so are visually hidden ones (sr-only, Radix's native select).
 */
function controls(page: Page): Promise<Control[]> {
  return page.evaluate(() => {
    const SELECTOR = 'a[href], button, input:not([type="hidden"]), select, textarea, [role="button"], '
      + '[role="tab"], [role="checkbox"], [role="switch"], [role="combobox"], [role="radio"], '
      + '[role="slider"], [role="menuitem"], [tabindex="0"]'
    const all = [...document.querySelectorAll(SELECTOR)]
    const describe = (el: Element) => {
      const label = el.getAttribute('aria-label') || el.textContent?.trim().slice(0, 40) || ''
      const id = el.getAttribute('data-testid')
      return `${el.tagName.toLowerCase()}${id ? `[${id}]` : ''} "${label}"`
    }
    const out: Control[] = []
    for (const el of all) {
      if (el.closest('[aria-hidden="true"], [inert]')) continue
      // A focusable box around other controls (a list, a table, a radio
      // group) is not itself a target.
      if (all.some((other) => other !== el && el.contains(other))) continue
      const r = el.getBoundingClientRect()
      const style = getComputedStyle(el)
      if (r.width < 2 || r.height < 2 || style.visibility === 'hidden') continue
      // Shown only on hover: a touch screen never shows it.
      if (Number(style.opacity) === 0) {
        out.push({ what: describe(el), width: r.width, height: r.height, inlineText: false,
          cutOff: false, coveredBy: null, invisible: true })
        continue
      }

      // What of it is drawn: clipped by every box that clips, and the viewport.
      let vis = { left: r.left, right: r.right, top: r.top, bottom: r.bottom }
      let clippedX = false
      let scrollsX = false
      for (let p = el.parentElement; p; p = p.parentElement) {
        const ps = getComputedStyle(p)
        const pr = p.getBoundingClientRect()
        if (ps.overflowX !== 'visible') {
          if (vis.left < pr.left - 1 || vis.right > pr.right + 1) {
            // A tab strip may scroll sideways; any other box that does hides
            // part of the page (a table's last columns) behind a scroll.
            const sideways = (ps.overflowX === 'auto' || ps.overflowX === 'scroll')
              && p.scrollWidth > p.clientWidth && p.getAttribute('role') === 'tablist'
            if (sideways) scrollsX = true
            else clippedX = true
          }
          vis = { ...vis, left: Math.max(vis.left, pr.left), right: Math.min(vis.right, pr.right) }
        }
        if (ps.overflowY !== 'visible') {
          vis = { ...vis, top: Math.max(vis.top, pr.top), bottom: Math.min(vis.bottom, pr.bottom) }
        }
      }
      if (r.left < -1 || r.right > window.innerWidth + 1) clippedX = true
      vis = { left: Math.max(vis.left, 0), right: Math.min(vis.right, window.innerWidth),
        top: Math.max(vis.top, 0), bottom: Math.min(vis.bottom, window.innerHeight) }
      if (vis.right - vis.left < 2 || vis.bottom - vis.top < 2) continue

      const cx = (vis.left + vis.right) / 2
      const cy = (vis.top + vis.bottom) / 2
      const hit = document.elementFromPoint(cx, cy)
      const labels = 'labels' in el ? [...((el as HTMLInputElement).labels ?? [])] : []
      const own = !hit || el.contains(hit) || hit.contains(el) || labels.some((l) => l.contains(hit))
        // next dev's own badge floats over the page in development only.
        || hit.tagName.toLowerCase() === 'nextjs-portal'

      // The target: the control with its labels; a slider thumb, its slider.
      const boxes = [r, ...labels.map((l) => l.getBoundingClientRect())]
      if (el.getAttribute('role') === 'slider') {
        const root = el.parentElement?.closest('[data-orientation]')
        if (root) boxes.push(root.getBoundingClientRect())
      }
      const width = Math.max(...boxes.map((b) => b.right)) - Math.min(...boxes.map((b) => b.left))
      // The whole target is on screen (a slider's track too).
      if (Math.max(...boxes.map((b) => b.right)) > window.innerWidth + 1
        || Math.min(...boxes.map((b) => b.left)) < -1) clippedX = true
      const height = Math.max(...boxes.map((b) => b.bottom)) - Math.min(...boxes.map((b) => b.top))
      // A link inside running text: the sentence around it sizes it.
      const parent = el.parentElement
      const inlineText = style.display === 'inline' && !!parent
        && [...parent.childNodes].some((n) => n.nodeType === 3 && n.textContent!.trim().length > 0)
      out.push({ what: describe(el), width, height, inlineText, cutOff: clippedX && !scrollsX,
        coveredBy: own ? null : describe(hit!), invisible: false })
    }
    return out
  })
}

const shotDir = () => process.env.WB5_SCREENSHOT_DIR

async function settle(page: Page) {
  await page.waitForLoadState('networkidle').catch(() => undefined)
  // Sheets and menus animate; let layout come to rest.
  await page.waitForTimeout(300)
}

test.describe('WB.5 — responsive', () => {
  test.setTimeout(600_000)
  test.use({ navigationTimeout: 60_000 })

  for (const width of WIDTHS) {
    test(`at ${width} px: no page scroll, nothing cut off or overlapping${width === 390 ? ', 40 px targets' : ''}`,
      async ({ page }, info) => {
        await page.setViewportSize({ width, height: width === 390 ? 844 : 900 })
        await backend(page)
        const problems: string[] = []
        for (const p of PAGES) {
          await gotoAndWait(page, p.route)
          await expect(page.locator('main h1').first(), p.route).toBeVisible({ timeout: 60_000 })
          await settle(page)
          const doc = await page.evaluate(() => ({
            scroll: document.documentElement.scrollWidth, inner: window.innerWidth }))
          if (doc.scroll > doc.inner) problems.push(`${p.name}: page scrolls sideways (${doc.scroll} > ${doc.inner})`)

          for (const c of await controls(page)) {
            if (c.invisible) {
              if (width < 768) problems.push(`${p.name}: ${c.what} shows only on hover`)
              continue
            }
            if (c.cutOff) problems.push(`${p.name}: cut off ${c.what}`)
            if (c.coveredBy) problems.push(`${p.name}: ${c.what} covered by ${c.coveredBy}`)
            if (width === 390 && !c.inlineText && (c.height < 40 || c.width < 40)) {
              problems.push(`${p.name}: small ${c.what} ${Math.round(c.width)}x${Math.round(c.height)}`)
            }
          }

          const name = `${width}-${p.name}.png`
          await page.screenshot({ path: shotDir() ? path.join(shotDir()!, name) : info.outputPath(name) })
        }
        expect(problems).toEqual([])
      })
  }

  test('at 390 px the source chat sheet and the KG details sheet have 40 px targets', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await backend(page)
    const problems: string[] = []
    // With a sheet open the page behind is aria-hidden: controls() reads the sheet.
    const check = async (what: string, sheet: ReturnType<typeof page.getByTestId>) => {
      // Measure once it has slid in.
      await expect.poll(async () => {
        const box = (await sheet.boundingBox())!
        return box.x >= 0 && box.x + box.width <= 390
      }).toBe(true)
      await page.waitForTimeout(600)
      const found = await controls(page)
      expect(found.length, what).toBeGreaterThan(2)
      for (const c of found) {
        if (c.invisible) problems.push(`${what}: ${c.what} shows only on hover`)
        else if (c.cutOff) problems.push(`${what}: cut off ${c.what}`)
        else if (c.coveredBy) problems.push(`${what}: ${c.what} covered by ${c.coveredBy}`)
        else if (!c.inlineText && (c.height < 40 || c.width < 40)) {
          problems.push(`${what}: small ${c.what} ${Math.round(c.width)}x${Math.round(c.height)}`)
        }
      }
    }

    await gotoAndWait(page, `/sources/${enc(SOURCE)}`)
    await page.getByRole('button', { name: 'Chat with source' }).click({ timeout: 60_000 })
    const chat = page.getByTestId('source-chat-sheet')
    await expect(chat).toBeVisible()
    await settle(page)
    await check('chat sheet', chat)
    await page.keyboard.press('Escape')
    await expect(chat).toBeHidden()

    await gotoAndWait(page, '/knowledge-graph')
    await page.getByTestId('entity-card-entity:1').click({ timeout: 60_000 })
    const sheet = page.getByTestId('entity-detail-sheet')
    await expect(sheet.getByTestId('relation-open-relation:1')).toBeVisible()
    await settle(page)
    await check('KG sheet', sheet)
    const shotDir = process.env.WB5_SCREENSHOT_DIR
    if (shotDir) await page.screenshot({ path: path.join(shotDir, '390-kg-sheet.png') })
    expect(problems).toEqual([])
  })

  test('below xl the sidebar is icons, and the badges stay on them', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 900 })
    await backend(page)
    await gotoAndWait(page, '/sources')
    const sidebar = page.locator('.app-sidebar')
    await expect(sidebar).toBeVisible({ timeout: 60_000 })
    expect((await sidebar.boundingBox())!.width).toBe(64)
    const badge = page.getByTestId('nav-badge-Triage')
    await expect(badge).toBeVisible()
    // No room for a number: a dot on the icon, the count in its tooltip.
    await expect(badge).toHaveAttribute('data-badge', 'dot')
    // Drawn inside the sidebar, not clipped by its scrolling nav.
    const [b, s] = [await badge.boundingBox(), await sidebar.boundingBox()]
    expect(b!.x).toBeGreaterThanOrEqual(s!.x)
    expect(b!.x + b!.width).toBeLessThanOrEqual(s!.x + s!.width)
    const nav = await page.locator('.app-sidebar nav').boundingBox()
    expect(b!.x + b!.width).toBeLessThanOrEqual(nav!.x + nav!.width)
    await expect(page.getByRole('button', { name: /(Collapse|Expand) sidebar/ })).toHaveCount(0)
  })

  test('at 390 px the drawer opens from the top bar, traps focus and closes on navigation', async ({ page }) => {
    await page.setViewportSize({ width: 390, height: 844 })
    await backend(page)
    await gotoAndWait(page, '/sources')
    const trigger = page.getByRole('button', { name: 'Open navigation' })
    await expect(trigger).toBeVisible({ timeout: 60_000 })
    await expect(page.getByTestId('mobile-top-bar-title')).toHaveText('Sources')
    await expect(page.locator('.app-sidebar')).toHaveCount(0)

    await trigger.focus()
    await page.keyboard.press('Enter')
    const drawer = page.getByRole('dialog', { name: 'Navigation' })
    await expect(drawer).toBeVisible()
    // Focus stays inside: 40 Tabs never leave the drawer.
    for (let i = 0; i < 40; i++) {
      await page.keyboard.press('Tab')
      expect(await drawer.evaluate((d) => d.contains(document.activeElement))).toBe(true)
    }
    await drawer.getByRole('link', { name: /^Triage/ }).click()
    await expect(page).toHaveURL(/\/knowledge-graph\/triage$/)
    await expect(drawer).toBeHidden()
    await expect(page.getByTestId('mobile-top-bar-title')).toHaveText('Triage')
    await expect(trigger).toBeFocused()
    await expect.poll(() => page.evaluate(() => getComputedStyle(document.body).pointerEvents)).toBe('auto')
  })

  test('at 1024 px the Sources title keeps most of the row (WB.4 m1)', async ({ page }) => {
    await page.setViewportSize({ width: 1024, height: 900 })
    await backend(page)
    await gotoAndWait(page, '/sources')
    const title = page.locator('[data-testid^="source-row-"] [data-testid="source-title"]').first()
    await expect(title).toBeVisible({ timeout: 60_000 })
    const cell = await title.evaluate((el) => el.closest('td')!.getBoundingClientRect().width)
    expect(cell).toBeGreaterThanOrEqual(360)
    // Stage and Created stay; Type, Insights and Embedded wait for xl.
    await expect(page.getByRole('columnheader')).toHaveText(['Title', 'Stage', 'Created', 'Actions'])
  })

  test('below lg the Sources board is cards; below lg the KG details are a sheet', async ({ page }) => {
    await page.setViewportSize({ width: 768, height: 900 })
    await backend(page)
    await gotoAndWait(page, '/sources')
    await expect(page.locator('[data-testid^="source-card-"]').first()).toBeVisible({ timeout: 60_000 })
    await expect(page.getByRole('table')).toHaveCount(0)

    await gotoAndWait(page, '/knowledge-graph')
    const row = page.getByTestId('entity-row-entity:1')
    await expect(row).toBeVisible({ timeout: 60_000 })
    await row.click()
    const sheet = page.getByTestId('entity-detail-sheet')
    await expect(sheet).toBeVisible()
    await expect(sheet.getByRole('heading', { name: 'Regio Deal Achterhoek' })).toBeVisible()
    // Once it has slid in, it sits inside the viewport.
    await expect.poll(async () => {
      const box = (await sheet.boundingBox())!
      return box.x + box.width
    }).toBeLessThanOrEqual(768)
    await page.keyboard.press('Escape')
    await expect(sheet).toBeHidden()
  })
})
