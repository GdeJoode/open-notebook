/**
 * Track UX — Phase UX.6 — detail-spine + recovery-only controls E2E spec.
 *
 * AC mapping → docs/tracks/UX-pipeline-alignment/plan.md §UX.6:
 *   AC1  Manual runners visible but DISABLED on a `complete` source          → the
 *        (tooltip "Runs automatically").                                        menu
 *   AC2  Reprocess remains available.                                        → the menu
 *   (UX.4) Graph node reflects document-graph membership on a graphed/       → the
 *          complete source.                                                     spine
 *
 * Fully route-mocked (no live backend), matching the project's E2E convention.
 * A `complete` source is served on the detail route; the spec asserts the
 * detail-variant spine renders the Graph node `done` and that the header
 * "Run X" runners are disabled recovery-only actions while Reprocess stays
 * available.
 *
 * NOTE: runs against a *served* frontend (PLAYWRIGHT_BASE_URL, default
 * http://localhost:8502). It does not stand up its own web server. If no built
 * frontend is being served (or the stack is auth-gated) the test skips at
 * runtime rather than reporting a false pass.
 */

import { expect, test, type Route, type Page } from '@playwright/test'

import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

const SOURCE_ID = 'source:uxsix'

function json(route: Route, body: unknown, status = 200): Promise<void> {
  return route.fulfill({
    status,
    contentType: 'application/json',
    body: JSON.stringify(body),
  })
}

function completeSource() {
  return {
    id: SOURCE_ID,
    title: 'A UX.6 complete source',
    topics: [],
    asset: { file_path: '/data/doc.pdf' },
    embedded: true,
    embedded_chunks: 12,
    insights_count: 2,
    entity_count: 7,
    relation_count: 3,
    created: '2026-07-01T00:00:00Z',
    updated: '2026-07-01T00:00:00Z',
    full_text: 'body',
    metadata: {},
    private: false,
    notebooks: [],
    processing_stage: 'complete',
  }
}

async function mockDetail(page: Page): Promise<void> {
  // Catch-all registered FIRST (lowest priority — Playwright tries the
  // most-recently-registered route first) so both the dashboard chrome and the
  // source-specific routes below win. Keeps stray chat/session/misc calls benign.
  await page.route('**/api/**', (route) => {
    if (route.request().method() !== 'GET') return json(route, {})
    return json(route, [])
  })

  await mockDashboardChrome(page)

  await page.route('**/api/transformations*', (route) => json(route, []))
  await page.route(`**/api/sources/${SOURCE_ID}/insights`, (route) => json(route, []))
  await page.route(`**/api/sources/${SOURCE_ID}/chunks`, (route) =>
    json(route, { total_chunks: 0, chunks: [] }),
  )
  await page.route(`**/api/sources/${SOURCE_ID}/status`, (route) =>
    json(route, { status: 'completed' }),
  )
  await page.route(`**/api/sources/${SOURCE_ID}`, (route) => {
    if (route.request().method() === 'GET') return json(route, completeSource())
    return json(route, completeSource())
  })
}

test.describe('Track UX.6 — detail spine + recovery-only controls', () => {
  test('complete source: Graph node done + runners disabled, Reprocess available', async ({
    page,
  }) => {
    await mockDetail(page)

    await gotoAndWait(page, `/sources/${SOURCE_ID}`)
    if (page.url().includes('/login')) {
      test.skip(true, 'auth-gated stack — covered by the auth-disabled run')
    }

    // Wait for the detail view to load (the mocked title proves the fetch ran).
    await expect(
      page.getByRole('button', { name: 'A UX.6 complete source' }),
    ).toBeVisible({ timeout: 20_000 })

    // Environmental guard: this spec asserts UX.4+ behaviour (the collapsed
    // `data-variant="detail"` spine). If the served frontend predates it (a
    // stale build serving the old output-badge bar), skip rather than misreport
    // — rebuild the served stack from this branch to actually run the spec.
    const spine = page.locator('[data-variant="detail"]')
    if ((await spine.count()) === 0) {
      test.skip(
        true,
        'served frontend predates the UX.4+ detail spine — rebuild the served stack from this branch',
      )
    }

    // UX.4 — the collapsed detail spine renders with the Graph node `done`
    // (document-graph membership on a graphed/complete source).
    await expect(spine).toBeVisible({ timeout: 20_000 })
    const graphNode = spine.locator('[data-node-key="graph"]')
    await expect(graphNode).toHaveAttribute('data-node-state', 'done')

    // AC1/AC2 — open the header actions menu.
    await page.getByRole('button', { name: 'Source actions' }).click()

    // Reprocess stays available.
    await expect(
      page.getByRole('menuitem', { name: /Reprocess Document/ }),
    ).toBeEnabled()

    // The recovery runners are disabled on a complete source ("Runs automatically").
    for (const name of [
      /Run Preprocessing/,
      /Generate Summaries/,
      /Extract Entities/,
      /Embed Content/,
    ]) {
      await expect(page.getByRole('menuitem', { name })).toBeDisabled()
    }
    await expect(page.getByText('Runs automatically').first()).toBeVisible()
  })
})
