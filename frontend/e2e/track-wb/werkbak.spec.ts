/**
 * Track WB — Phase WB.1 — the Werkbak.
 *
 * AC (docs/tracks/WB-pipeline-workbench/plan.md §WB.1): a source added through
 * the API and a job that failed while the app was closed both show on the
 * Werkbak on the next open, each linked to the page where it is resolved.
 *
 * `GET /api/attention` is served from `attention.fixture.json`, which the
 * backend test `test_attention_db.py::test_e2e_fixture_follows_the_contract`
 * validates against the real model and group list. The mock follows the
 * contract: without `since` there are no arrivals.
 */

import { expect, test, type Page, type Route } from '@playwright/test'

import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'
import fixture from './attention.fixture.json'

const LAST_SEEN = 'werkbak.lastSeen'

function json(route: Route, body: unknown, status = 200): Promise<void> {
  return route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
}

function apiSource(id: string, title: string, stage: string) {
  return {
    id, title, topics: [], asset: null, embedded: false, embedded_chunks: 0,
    insights_count: 0, created: '2026-09-29T01:59:00Z', updated: '2026-09-29T02:00:00Z',
    full_text: '', metadata: {}, private: false, notebooks: [], processing_stage: stage,
  }
}

async function mockBackend(page: Page, lastSeen: string | null) {
  const calls = { attention: [] as URL[], sources: [] as URL[] }
  await page.route('**/api/**', (route) =>
    json(route, route.request().method() === 'GET' ? [] : {}))
  await mockDashboardChrome(page)
  if (lastSeen) {
    await page.addInitScript(([key, value]) => {
      window.localStorage.setItem(key, value)
    }, [LAST_SEEN, lastSeen])
  }
  await page.route('**/api/attention**', (route) => {
    const url = new URL(route.request().url())
    calls.attention.push(url)
    const since = url.searchParams.get('since')
    // As the server does without a mark: the last 7 days, flagged.
    const week = new Date(Date.parse(fixture.generated_at) - 7 * 864e5).toISOString()
    return json(route, since
      ? { ...fixture, since, since_default: false }
      : { ...fixture, since: week, since_default: true })
  })
  await page.route(/\/api\/sources(\?.*)?$/, (route) => {
    const url = new URL(route.request().url())
    calls.sources.push(url)
    return json(route, [
      { ...apiSource('source:emb1', 'Convenant Regio Deal Achterhoek', 'embedded'), command_id: null, status: null },
    ])
  })
  // The source page (as track-ux/pipeline-spine.spec.ts mocks it).
  await page.route('**/api/transformations*', (route) => json(route, []))
  await page.route(/\/api\/sources\/source(:|%3A)api1\/insights$/, (route) => json(route, []))
  await page.route(/\/api\/sources\/source(:|%3A)api1\/chunks$/, (route) =>
    json(route, { total_chunks: 0, chunks: [] }))
  await page.route(/\/api\/sources\/source(:|%3A)api1\/status$/, (route) =>
    json(route, { status: 'failed' }))
  await page.route(/\/api\/sources\/source(:|%3A)api1$/, (route) =>
    json(route, apiSource('source:api1', 'From the API', 'failed')))
  return calls
}

test.describe('WB.1 — the Werkbak', () => {
  test.setTimeout(90_000)

  test('an API source and a failed job from while the app was closed, each linked', async ({ page }) => {
    const calls = await mockBackend(page, fixture.since)
    await gotoAndWait(page, '/')
    await expect(page.getByTestId('werkbak-group-new_sources')).toBeVisible({ timeout: 45_000 })
    expect(new URL(page.url()).pathname).toBe('/')
    expect(calls.attention.at(-1)?.searchParams.get('since')).toBe(fixture.since)

    const newSources = page.getByTestId('werkbak-group-new_sources')
    await expect(newSources.getByText('From the API')).toBeVisible()
    const finished = page.getByTestId('werkbak-group-finished_jobs')
    await expect(finished.getByText('failed: parse failed')).toBeVisible()
    await expect(page.getByTestId('werkbak-group-jobs_failed').getByText('parse failed'))
      .toBeVisible()

    await newSources.getByTestId('werkbak-item-source:api1').click()
    await expect(page).toHaveURL(/\/sources\/source%3Aapi1$/, { timeout: 45_000 })
    await expect(page.getByText('From the API').first()).toBeVisible({ timeout: 45_000 })

    await page.goBack()
    await page.getByTestId('werkbak-group-finished_jobs')
      .getByTestId('werkbak-item-job:f1').click()
    await expect(page).toHaveURL(/\/sources\/source%3Aapi1$/, { timeout: 45_000 })
  })

  test('the sidebar carries the counts; a group action lands on its page', async ({ page }) => {
    await mockBackend(page, null)
    await gotoAndWait(page, '/')
    await expect(page.getByTestId('werkbak-group-triage')).toBeVisible({ timeout: 45_000 })
    // The fixture's merge candidates timed out: partial.
    await expect(page.getByTestId('nav-badge-The Tray'))
      .toHaveText(`${fixture.total}+ need you, some groups did not load`)
    await expect(page.getByTestId('nav-badge-Triage')).toHaveText('1238 need you')
    await expect(page.getByTestId('nav-badge-Duplicates'))
      .toHaveText('0+ need you, some groups did not load')

    await page.getByTestId('werkbak-action-triage').click()
    await expect(page).toHaveURL(/\/knowledge-graph\/triage$/, { timeout: 45_000 })
  })

  test('stopped sources open the sources list filtered on the same set', async ({ page }) => {
    const calls = await mockBackend(page, null)
    await gotoAndWait(page, '/')
    await page.getByTestId('werkbak-action-sources_stalled').click({ timeout: 45_000 })
    await expect(page).toHaveURL(/\/sources\?stalled=true$/, { timeout: 45_000 })
    await expect(page.getByTestId('sources-stage-filter')).toContainText('stopped sources')
    await expect.poll(() => calls.sources.at(-1)?.searchParams.get('stalled')).toBe('true')
  })

  test('a stage link sends the stages as repeated params', async ({ page }) => {
    const calls = await mockBackend(page, null)
    await page.goto('/sources?stage=failed&stage=embedded')
    await expect(page.getByTestId('sources-stage-filter')).toContainText('failed, embedded',
      { timeout: 45_000 })
    await expect.poll(() => calls.sources.at(-1)?.searchParams.getAll('stage'))
      .toEqual(['failed', 'embedded'])
  })

  test('first visit: nothing is stored until Mark as seen', async ({ page }) => {
    const calls = await mockBackend(page, null)
    await gotoAndWait(page, '/')
    await expect(page.getByTestId('werkbak-since-note')).toContainText('The last 7 days',
      { timeout: 45_000 })
    await expect(page.getByTestId('werkbak-group-new_sources')).toBeVisible()
    expect(calls.attention.every((u) => !u.searchParams.has('since'))).toBe(true)
    expect(await page.evaluate((k) => window.localStorage.getItem(k), LAST_SEEN)).toBeNull()

    await page.getByTestId('werkbak-mark-seen').click()
    expect(await page.evaluate((k) => window.localStorage.getItem(k), LAST_SEEN))
      .toBe(fixture.generated_at)
    await expect.poll(() => calls.attention.at(-1)?.searchParams.get('since'))
      .toBe(fixture.generated_at)
  })
})
