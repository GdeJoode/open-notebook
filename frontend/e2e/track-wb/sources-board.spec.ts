/**
 * Track WB — Phase WB.3 — Sources as the pipeline board.
 *
 * AC (docs/tracks/WB-pipeline-workbench/plan.md §WB.3):
 * - a source stopped at `embedded` looks different from a `complete` one and
 *   can be filtered;
 * - no displayed title starts with the 12-hex upload prefix;
 * - the chip counts are the server's.
 *
 * `GET /api/sources/` and `/api/sources/counts` are served by
 * `sources-board.fake.ts` from `sources-board.fixture.json`; the backend test
 * `test_sources_board_db.py::test_the_e2e_fixture_is_the_real_contract` seeds
 * the same rows into a real SurrealDB and checks the real endpoints answer what
 * the fake answers.
 */

import { expect, test, type Page, type Route } from '@playwright/test'

import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'
import attention from './attention.fixture.json'
import { EXPECTED, NOTEBOOKS, fakeCounts, fakeList, sourceId } from './sources-board.fake'

function json(route: Route, body: unknown, status = 200): Promise<void> {
  return route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
}

async function mockBackend(page: Page) {
  const calls = { list: [] as URL[], counts: [] as URL[] }
  await page.route('**/api/**', (route) =>
    json(route, route.request().method() === 'GET' ? [] : {}))
  await mockDashboardChrome(page)
  // The sidebar's badges read the attention report.
  await page.route('**/api/attention**', (route) => json(route, attention))
  await page.route(/\/api\/notebooks(\?.*)?$/, (route) =>
    json(route, NOTEBOOKS.map((nb) => ({
      ...nb, description: '', archived: false, created: '2026-09-01T00:00:00Z',
      updated: '2026-09-01T00:00:00Z', source_count: 0, note_count: 0,
    }))))
  await page.route(/\/api\/sources\/?(\?.*)?$/, (route) => {
    const url = new URL(route.request().url())
    calls.list.push(url)
    return json(route, fakeList(url.searchParams))
  })
  await page.route(/\/api\/sources\/counts(\?.*)?$/, (route) => {
    const url = new URL(route.request().url())
    calls.counts.push(url)
    return json(route, fakeCounts(url.searchParams))
  })
  return calls
}

const rows = (page: Page) => page.locator('[data-testid^="source-row-"]')
const rowIds = async (page: Page) =>
  (await rows(page).evaluateAll((els) => els.map((e) => e.getAttribute('data-testid')!)))
    .map((t) => t.slice('source-row-'.length))

test.describe('WB.3 — the Sources board', () => {
  test.setTimeout(90_000)

  test('needs-you first, stopped marked, chips with the server counts', async ({ page }) => {
    const calls = await mockBackend(page)
    await gotoAndWait(page, '/sources')
    await expect(page.getByTestId(`source-row-${sourceId('gated')}`)).toBeVisible({ timeout: 45_000 })
    expect(calls.list[0].searchParams.get('sort_by')).toBe('attention')
    expect(await rowIds(page)).toEqual(EXPECTED.attention_order.all.map(sourceId))

    const stopped = page.getByTestId(`source-row-${sourceId('stopped')}`)
    const done = page.getByTestId(`source-row-${sourceId('done1')}`)
    await expect(stopped.getByTestId('pipeline-stage-label')).toHaveText('Embed done')
    await expect(stopped.getByTestId('source-stopped-badge')).toHaveText('Stopped')
    await expect(stopped).toHaveAttribute('data-needs-you', 'true')
    await expect(done.getByTestId('pipeline-stage-label')).toHaveText('Complete')
    await expect(done.getByTestId('source-stopped-badge')).toHaveCount(0)

    const counts = EXPECTED.counts.all
    const want: Record<string, number> = { all: counts.total, stalled: counts.stalled, ...counts.stages }
    for (const [key, n] of Object.entries(want)) {
      await expect(page.getByTestId(`sources-chip-${key}`), key).toHaveAttribute('data-count', String(n))
    }
    await expect(page.getByTestId('sources-add')).toHaveAttribute('href', '/sources/new')
  })

  test('a stopped source can be filtered: the Stopped chip and the Embedded chip', async ({ page }) => {
    const calls = await mockBackend(page)
    await gotoAndWait(page, '/sources')
    await page.getByTestId('sources-chip-stalled').click({ timeout: 45_000 })
    await expect(page).toHaveURL(/\/sources\?stalled=true$/, { timeout: 45_000 })
    await expect(page.getByTestId('sources-chip-stalled')).toHaveAttribute('aria-current', 'true')
    await expect.poll(() => rowIds(page)).toEqual(['stopped', 'stopped2'].map(sourceId))
    expect(calls.list.at(-1)?.searchParams.get('stalled')).toBe('true')

    await page.getByTestId('sources-chip-embedded').click()
    await expect(page).toHaveURL(/\/sources\?stage=embedded$/)
    await expect.poll(() => rowIds(page)).toEqual(['stopped', 'fresh', 'fresh2'].map(sourceId))
    await expect(page.getByTestId(`source-row-${sourceId('fresh')}`).getByTestId('source-stopped-badge'))
      .toHaveCount(0)
  })

  test('the notebook filter scopes the rows and the counts', async ({ page }) => {
    const calls = await mockBackend(page)
    await gotoAndWait(page, '/sources')
    await page.getByTestId('sources-notebook-filter').click({ timeout: 45_000 })
    await page.getByRole('option', { name: 'Beta' }).click()
    await expect(page).toHaveURL(/\/sources\?notebook=notebook%3Abeta$/)
    await expect.poll(() => rowIds(page))
      .toEqual(EXPECTED.attention_order['notebook:beta'].map(sourceId))
    await expect(page.getByTestId('sources-chip-all')).toHaveAttribute('data-count', '4')
    expect(calls.counts.at(-1)?.searchParams.get('notebook_id')).toBe('notebook:beta')
  })

  test('arrow keys and Enter open a source; titles carry no upload prefix', async ({ page }) => {
    await mockBackend(page)
    await gotoAndWait(page, '/sources')
    await expect(page.getByTestId(`source-row-${sourceId('gated')}`)).toBeVisible({ timeout: 45_000 })
    await expect(page.locator('body')).toContainText('Convenant Regio Deal Waterwegregio')
    expect(await page.locator('body').innerText()).not.toMatch(/[0-9a-f]{12}__/)

    await page.getByRole('table', { name: 'Sources' }).focus()
    await page.keyboard.press('ArrowDown')
    await page.keyboard.press('Enter')
    await expect(page).toHaveURL(/\/sources\/source%3Afailed$|\/sources\/source:failed$/, { timeout: 45_000 })
  })
})
