/**
 * Track I — Phase I.A — Design tokens visual smoke spec.
 *
 * What we verify (AC mapping → docs/tracks/I-docling-studio/plan.md §I.A):
 *
 *   AC1  No console / pageerror on light theme        → assert pageErrors[] empty
 *                                                       after dashboard hydration.
 *   AC2  Theme toggle still flips themes              → click the toggle, assert
 *                                                       <html> class changes from
 *                                                       light↔dark and no errors fire.
 *   AC4  No layout regression on 3 representative
 *        pages: dashboard, source detail, notebook    → navigate to each, assert no
 *        detail                                          pageerror after hydration.
 *
 *   AC3  (mono-numerics applied) is deferred to I.E
 *        — out of scope here.
 *   AC5  (< 30KB bundle delta) is measured in the PR
 *        description, not in this spec.
 *
 * The spec is fully route-mocked: no live backend. The dashboard chrome
 * (auth/status, config, models/status) is mocked via the shared
 * `mockDashboardChrome` helper; the source detail and notebook detail
 * routes are seeded with minimal fixtures sufficient to render their
 * skeletons without hitting the network.
 *
 * Page-error watchdog (registered at the top of every test) is the
 * single most important assertion in this spec: a broken `next/font`
 * subset or a malformed OKLCH token typically surfaces as a hydration
 * error before any visible regression, and this watchdog catches it
 * first.
 */

import { expect, test, type Page } from '@playwright/test'
import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const SOURCE_ID = 'src_track_i_a_fixture'
const NOTEBOOK_ID = 'notebook:track-i-a-fixture'
// The notebooks client uses a raw id ("notebook:..." with a colon) but
// the URL encoder turns it into "notebook%3A...". Both spellings hit
// the same handler so we match either.
const NB_ID_RE = '(?:notebook:track-i-a-fixture|notebook%3Atrack-i-a-fixture)'

async function mockNotebookDetail(page: Page): Promise<void> {
  const notebook = {
    id: NOTEBOOK_ID,
    name: 'Track I.A fixture notebook',
    description: 'Used by design-tokens visual smoke spec',
    archived: false,
    created: '2026-06-01T00:00:00Z',
    updated: '2026-06-01T00:00:00Z',
    source_count: 0,
    note_count: 0,
  }

  await page.route(
    new RegExp(`/api/notebooks/${NB_ID_RE}(?:\\?.*)?$`),
    (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(notebook),
      }),
  )

  await page.route(
    new RegExp(`/api/sources(?:\\?.*notebook_id=${NB_ID_RE}.*)?$`),
    (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      }),
  )

  await page.route(
    new RegExp(`/api/notes(?:\\?.*notebook_id=${NB_ID_RE}.*)?$`),
    (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      }),
  )

  // The notebook detail page also fetches /api/settings (vault path,
  // schema config) — return a minimal stub so the page boots clean.
  await page.route(/\/api\/settings(?:\?.*)?$/, (route) =>
    route.fulfill({
      status: 200,
      contentType: 'application/json',
      body: JSON.stringify({
        vault_path: '',
        vault_entities_folder: 'Entities',
        vault_sync_on_startup: false,
      }),
    }),
  )
}

async function mockSourceDetail(page: Page): Promise<void> {
  const source = {
    id: SOURCE_ID,
    title: 'Track I.A fixture source',
    topics: [],
    asset: { file_path: null },
    embedded: false,
    embedded_chunks: 0,
    insights_count: 0,
    entity_count: 0,
    relation_count: 0,
    created: '2026-06-01T00:00:00Z',
    updated: '2026-06-01T00:00:00Z',
    file_available: false,
    full_text: 'Fixture content',
    notebooks: [],
    metadata: {
      parser_engine_used: 'docling',
    },
  }

  await page.route(
    `**/api/sources/${encodeURIComponent(SOURCE_ID)}`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(source),
      }),
  )

  await page.route(
    `**/api/sources/${encodeURIComponent(SOURCE_ID)}/chunks`,
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
    `**/api/sources/${encodeURIComponent(SOURCE_ID)}/insights`,
    (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      }),
  )

  // Fallbacks for the misc endpoints the source detail / chat panel
  // touches. Empty lists keep the page light and prevent 404 noise.
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
}

test.describe('I.A: design tokens visual smoke', () => {
  // First-route compile can take a while on cold dev runs; same mitigation
  // as the D-track export specs.
  test.setTimeout(60_000)
  test.use({ navigationTimeout: 45_000 })

  test('dashboard hydrates clean, theme toggle flips, multi-page nav stays clean', async ({
    page,
  }) => {
    // Page-error watchdog. Registered before any navigation so that an
    // error during the first hydration would be captured. ANY uncaught
    // page error here is a regression — a malformed OKLCH value or a
    // broken next/font subset usually surfaces as a hydration error
    // before any visual breakage is observable.
    const pageErrors: string[] = []
    page.on('pageerror', (err) => pageErrors.push(err.message))

    await mockDashboardChrome(page)
    await mockNotebookDetail(page)
    await mockSourceDetail(page)

    // Empty notebooks list keeps the dashboard skeleton minimal.
    await page.route(/\/api\/notebooks(?:\?.*)?$/, (route) =>
      route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify([]),
      }),
    )

    // -------- Step 1: dashboard (`/` redirects to `/notebooks`) --------
    await gotoAndWait(page, '/')
    await expect(page).toHaveURL(/\/notebooks\/?$/, { timeout: 15_000 })
    // Wait for a stable, deterministic element of the dashboard chrome —
    // the sidebar nav — so hydration is fully complete before we assert.
    await expect(page.getByRole('navigation').first()).toBeVisible()
    expect(pageErrors, 'no pageerror after dashboard hydration').toEqual([])

    // Snapshot the current <html> class as the theme baseline. The
    // app applies either `light` or `dark` based on the persisted
    // theme + system preference; both are valid starting points.
    const themeBefore = await page.evaluate(() =>
      document.documentElement.classList.contains('dark') ? 'dark' : 'light',
    )

    // -------- Step 2: toggle theme via existing dropdown --------
    // The ThemeToggle is in the sidebar and is the only element with
    // the sr-only "Toggle theme" label. The trigger button wraps it.
    await page
      .getByRole('button', { name: /toggle theme/i })
      .first()
      .click()

    // Pick the opposite theme item from the dropdown menu.
    const target = themeBefore === 'dark' ? 'Light' : 'Dark'
    await page.getByRole('menuitem', { name: target }).click()

    // The store applies the new class synchronously via
    // setTheme(); waitForFunction polls until <html> reflects it.
    await page.waitForFunction(
      (expected) =>
        document.documentElement.classList.contains(expected),
      target.toLowerCase(),
      { timeout: 5_000 },
    )
    const themeAfter = await page.evaluate(() =>
      document.documentElement.classList.contains('dark') ? 'dark' : 'light',
    )
    expect(themeAfter).not.toBe(themeBefore)
    expect(pageErrors, 'no pageerror after theme toggle').toEqual([])

    // -------- Step 3: navigate to source detail --------
    await gotoAndWait(page, `/sources/${encodeURIComponent(SOURCE_ID)}`)
    // The "Back" button on the source-detail page is a deterministic
    // hydration marker — its label comes from useNavigation() which
    // runs after hydration.
    await expect(
      page.getByRole('button', { name: /back/i }).first(),
    ).toBeVisible({ timeout: 15_000 })
    expect(pageErrors, 'no pageerror after source detail hydration').toEqual([])

    // -------- Step 4: navigate to notebook detail --------
    await gotoAndWait(page, `/notebooks/${NOTEBOOK_ID}`)
    // The notebook page renders an AppShell + 3-column workspace. The
    // sidebar nav is the most stable visible anchor on first paint.
    await expect(page.getByRole('navigation').first()).toBeVisible({
      timeout: 15_000,
    })
    expect(pageErrors, 'no pageerror after notebook detail hydration').toEqual(
      [],
    )

    // -------- Final assertion: zero errors across the whole flow --------
    expect(pageErrors).toEqual([])
  })
})
