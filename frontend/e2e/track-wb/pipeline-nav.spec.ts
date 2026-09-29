/**
 * Track WB — Phase WB.2 — pipeline navigation.
 *
 * AC (docs/tracks/WB-pipeline-workbench/plan.md §WB.2):
 *   1. From The Tray, each group reaches its resolution page in one click and
 *      returns in one click (the sidebar's The Tray, on every page now that
 *      the shell lives in the layout).
 *   2. Upload → gated → accept → complete without typing a URL.
 *
 * `GET /api/attention` is `attention.fixture.json`, which
 * `test_attention_db.py::test_e2e_fixture_follows_the_contract` holds to the
 * real model and groups. For (1) its timed-out group is shown loaded, so every
 * group has its action. The notebook's proposal queue and accept are the OE
 * in-memory backend (`track-oe/_ontology-backend.ts`); the source's stage and
 * the paused extraction follow the server: extraction parks the source at
 * `awaiting_schema_review` with its job PAUSED_FOR_REVIEW, accepting a
 * proposal publishes it and resumes nothing, and `POST /extraction/resume`
 * re-queues the paused job, which then runs extract → graph → complete.
 */

import { expect, test, type Page, type Route } from '@playwright/test'

import { gotoAndWait } from '../_helpers'
import { DEFAULT_SETTINGS, mockDashboardChrome } from '../track-a/_track-a-helpers'
import { NOTEBOOK, backend } from '../track-oe/_ontology-backend'
import fixture from './attention.fixture.json'

type Group = (typeof fixture.groups)[number]

function json(route: Route, body: unknown, status = 200): Promise<void> {
  return route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
}

function apiSource(id: string, title: string, stage: string, notebooks: string[] = []) {
  const past = (s: string[]) => s.includes(stage)
  return {
    id, title, topics: [], asset: null,
    embedded: !past(['ingested']), embedded_chunks: past(['ingested']) ? 0 : 4,
    insights_count: 0,
    entity_count: past(['extracted', 'graphed', 'complete']) ? 5 : 0,
    relation_count: past(['graphed', 'complete']) ? 2 : 0,
    created: '2026-09-29T10:00:00Z', updated: '2026-09-29T10:00:00Z',
    full_text: 'body', metadata: {}, private: false, notebooks, processing_stage: stage,
    command_id: null, status: null,
  }
}

/** The source page's own reads (as werkbak.spec.ts / pipeline-spine.spec.ts). */
async function mockSourcePage(page: Page, id: string, body: () => unknown, jobStatus: () => string) {
  const re = id.replace(':', '(?::|%3A)')
  await page.route(new RegExp(`/api/sources/${re}/insights$`), (r) => json(r, []))
  await page.route(new RegExp(`/api/sources/${re}/chunks$`), (r) =>
    json(r, { total_chunks: 0, chunks: [] }))
  await page.route(new RegExp(`/api/sources/${re}/status$`), (r) =>
    json(r, { status: jobStatus() }))
  await page.route(new RegExp(`/api/sources/${re}/logs`), (r) =>
    r.fulfill({ status: 200, contentType: 'text/event-stream', body: '' }))
  await page.route(new RegExp(`/api/sources/${re}$`), (r) => json(r, body()))
}

async function mockChrome(page: Page, report: unknown) {
  // Anything a page reads that this spec does not serve fails as a server
  // would for a missing route; each page shows its own error state.
  await page.route('**/api/**', (route) => json(route, { detail: 'Not Found' }, 404))
  await mockDashboardChrome(page)
  await page.route('**/api/attention**', (route) => json(route, report))
  await page.route('**/api/transformations*', (route) => json(route, []))
}

const sidebarTray = (page: Page) =>
  page.getByRole('navigation').getByRole('link', { name: /^The Tray/ })

test.describe('WB.2 — pipeline navigation', () => {
  test.setTimeout(240_000)
  test.use({ navigationTimeout: 45_000 })

  test('each Tray group reaches its page in one click and returns in one click', async ({ page }) => {
    // The timed-out group, loaded: the report the server gives once it answers.
    const groups: Group[] = fixture.groups.map((g) => g.key === 'merge_candidates'
      ? { ...g, error: null, count: 48, counted: 48, action: 'Review 48 merge candidates',
          detail: '3 clusters, 5 auto, 40 to review' }
      : g)
    const report = { ...fixture, groups, partial: false,
      total: groups.reduce((n, g) => n + g.counted, 0) }
    await mockChrome(page, report)

    // What each resolution page shows first. Their data is not this spec's:
    // enough for each page to render.
    const api = backend()
    await page.route('**/api/ontologies**', api.handle)
    await page.route(/\/api\/ontologies\/schema_core$/, (r) => json(r, {
      name: 'schema_core', version: '1.0.0', description: 'Core', authors: null, extends: null,
      entity_type_count: 0, relationship_type_count: 0, concept_count: 0,
      entity_types: [], relationship_types: [], concepts: [] }))
    await page.route(/\/api\/notebooks\/(?:notebook:live|notebook%3Alive)(?:[/?].*)?$/, api.handleNotebook)
    await page.route(/\/api\/sources(\?.*)?$/, (r) =>
      json(r, [apiSource('source:emb1', 'Convenant Regio Deal Achterhoek', 'embedded')]))
    await mockSourcePage(page, 'source:api1', () => apiSource('source:api1', 'From the API', 'failed'),
      () => 'failed')
    await mockSourcePage(page, 'source:emb3', () => apiSource('source:emb3', 'Being extracted', 'embedded'),
      () => 'queued')
    await page.route('**/api/triage/queue**', (r) => json(r, { items: [], total: 0 }))
    await page.route('**/api/entity-resolution/plan**', (r) =>
      json(r, { clusters: [], cluster_count: 0 }))
    await page.route('**/api/entity-resolution/candidates**', (r) =>
      json(r, { auto_merge: [], review: [], auto_merge_count: 0, review_count: 0 }))

    const landmark: Record<string, (p: Page) => ReturnType<Page['getByRole']>> = {
      '/notebooks/notebook%3Alive/schema': (p) => p.getByTestId('schema-tab-content'),
      '/notebooks/notebook%3Alive': (p) => p.getByTestId('notebook-section-nav'),
      '/sources?stalled=true': (p) => p.getByTestId('sources-stage-filter'),
      '/sources': (p) => p.getByRole('heading', { name: 'All Sources' }),
      '/knowledge-graph/triage': (p) => p.getByRole('heading', { name: 'Triage queue' }),
      '/knowledge-graph/resolution': (p) => p.getByRole('heading', { name: 'Entity Resolution' }),
      '/ontologies?name=schema_core&tab=edit&retype=1': (p) =>
        p.getByRole('heading', { name: 'schema_core' }),
      '/sources/source%3Aapi1': (p) => p.getByText('From the API').first(),
      '/sources/source%3Aemb3': (p) => p.getByText('Being extracted').first(),
    }

    // Derived, not sampled: every card the page gives an action.
    const withAction = [...report.groups, ...report.arrived].filter((g) => g.action && g.link)
    expect(withAction.map((g) => g.key)).toEqual([
      'sources_awaiting_review', 'sources_stalled', 'extraction_paused', 'schema_proposals',
      'triage', 'merge_candidates', 'retype', 'jobs_failed', 'jobs_running',
      'new_sources', 'finished_jobs',
    ])

    await gotoAndWait(page, '/')
    for (const group of withAction) {
      await page.getByTestId(`werkbak-action-${group.key}`).click({ timeout: 45_000 })
      await expect.poll(() => { const u = new URL(page.url()); return u.pathname + u.search },
        { timeout: 45_000, message: group.key }).toBe(group.link)
      expect(landmark[group.link], group.link).toBeTruthy()
      await expect(landmark[group.link](page), group.key).toBeVisible({ timeout: 45_000 })

      await sidebarTray(page).click()
      await expect(page).toHaveURL(/\/$/, { timeout: 45_000 })
      await expect(page.getByTestId(`werkbak-group-${group.key}`)).toBeVisible({ timeout: 45_000 })
    }

    // The sidebar's Schema proposals opens the Tray's proposals card, which
    // lists each notebook's queue.
    await page.getByRole('navigation').getByRole('link', { name: /^Triage/ }).click()
    await expect(page).toHaveURL(/\/knowledge-graph\/triage$/, { timeout: 45_000 })
    await page.getByRole('navigation').getByRole('link', { name: /^Schema proposals/ }).click()
    await expect(page).toHaveURL(/\/#schema_proposals$/, { timeout: 45_000 })
    const card = page.getByTestId('werkbak-group-schema_proposals')
    await expect(card).toBeInViewport({ timeout: 45_000 })
    await expect(card.getByTestId('werkbak-item-notebook:live'))
      .toHaveAttribute('href', '/notebooks/notebook%3Alive/schema')
  })

  test('upload → gated → accept → resume → complete, without typing a URL', async ({ page }) => {
    const SOURCE = 'source:g1'
    const notebookRow = { id: NOTEBOOK, name: 'OE.5 notebook', description: '', archived: false,
      created: '2026-09-28T00:00:00Z', updated: '2026-09-28T00:00:00Z',
      source_count: 0, note_count: 0 }

    // The server's side of the gate. Extraction parks at the gate once; the
    // resume re-queues the job, which then walks to complete.
    const server = {
      created: false,
      reads: 0,
      resumed: false,
      afterResume: 0,
      calls: [] as string[],
    }
    const before = ['ingested', 'embedded', 'awaiting_schema_review']
    const after = ['extracted', 'graphed', 'complete']
    const stage = () => server.resumed
      ? after[Math.min(server.afterResume, after.length - 1)]
      : before[Math.min(server.reads, before.length - 1)]
    const jobStatus = () => {
      const s = stage()
      if (s === 'awaiting_schema_review') return 'paused_for_review'
      if (s === 'complete') return 'completed'
      return 'running'
    }

    await mockChrome(page, { ...fixture, groups: fixture.groups, arrived: fixture.arrived })
    await page.route('**/api/settings', (r) => json(r, DEFAULT_SETTINGS))

    const api = backend()
    await page.route('**/api/ontologies**', api.handle)
    await page.route(/\/api\/notebooks(\?.*)?$/, (r) => json(r, [notebookRow]))
    await page.route(/\/api\/notebooks\/(?:notebook:oe5|notebook%3Aoe5)(?:[/?].*)?$/, async (route) => {
      const path = decodeURIComponent(new URL(route.request().url()).pathname)
      const accept = path.match(/\/schema\/extensions\/([^/]+)\/accept$/)
      if (accept && route.request().method() === 'POST') server.calls.push(`accept ${accept[1]}`)
      return api.handleNotebook(route)
    })
    await page.route(/\/api\/notebooks\/(?:notebook:oe5|notebook%3Aoe5)\/extraction\/paused$/, (r) => {
      const paused = !server.resumed && stage() === 'awaiting_schema_review'
      return json(r, { notebook_id: NOTEBOOK, paused_count: paused ? 1 : 0,
        paused_source_ids: paused ? [SOURCE] : [] })
    })
    await page.route(/\/api\/notebooks\/(?:notebook:oe5|notebook%3Aoe5)\/extraction\/resume$/, (r) => {
      server.calls.push('resume')
      const paused = !server.resumed && stage() === 'awaiting_schema_review'
      server.resumed = true
      // A proposal was accepted, so the gate is already satisfied: no sentinel.
      return json(r, { notebook_id: NOTEBOOK, resumed_count: paused ? 1 : 0,
        sentinel_added: api.state.accepted.length === 0 })
    })

    await page.route(/\/api\/sources(\?.*)?$/, (route) => {
      if (route.request().method() === 'POST') {
        server.created = true
        server.calls.push('create')
        return json(route, apiSource(SOURCE, 'regio-deal.pdf', 'ingested', [NOTEBOOK]))
      }
      return json(route, server.created
        ? [apiSource(SOURCE, 'regio-deal.pdf', stage(), [NOTEBOOK])] : [])
    })
    await mockSourcePage(page, SOURCE, () => {
      const body = apiSource(SOURCE, 'regio-deal.pdf', stage(), [NOTEBOOK])
      if (server.resumed) server.afterResume += 1
      else server.reads += 1
      return body
    }, jobStatus)

    // The only URL typed: opening the app.
    await gotoAndWait(page, '/')
    await expect(page.getByTestId('werkbak-group-triage')).toBeVisible({ timeout: 45_000 })

    // Create → Source.
    await page.getByRole('button', { name: 'Create' }).click()
    await page.getByRole('menuitem', { name: 'Source' }).click()
    await expect(page).toHaveURL(/\/sources\/new$/, { timeout: 45_000 })
    await page.getByRole('tab', { name: /Upload/i }).click()
    await page.locator('#file').setInputFiles({
      name: 'regio-deal.pdf', mimeType: 'application/pdf', buffer: Buffer.from('%PDF-1.4\n'),
    })
    await page.getByRole('button', { name: 'Next', exact: true }).click()
    await page.getByText('OE.5 notebook').click()
    await page.getByRole('button', { name: 'Submit', exact: true }).click()

    // Gated: the flow says where the proposal is accepted.
    await page.getByTestId('pipeline-review-proposal').click({ timeout: 45_000 })
    await expect(page).toHaveURL(/\/notebooks\/notebook%3Aoe5\/schema$/, { timeout: 45_000 })

    // The Schema tab: the paused extraction and the proposal queue.
    await expect(page.getByTestId('extraction-paused-banner')).toBeVisible({ timeout: 45_000 })
    await page.getByTestId('accept-pass1::streekfonds').click()
    const dialog = page.getByTestId('accept-extension-dialog')
    await dialog.getByTestId('accept-preview').click()
    await expect(dialog.getByTestId('accept-preview-clean')).toBeVisible()
    await dialog.getByTestId('accept-publish').click()
    await expect(dialog).toBeHidden()
    expect(api.state.active.content.types.Streekfonds).toBeTruthy()
    // Accepting resumed nothing: the extraction is still paused.
    await expect(page.getByTestId('extraction-paused-banner')).toBeVisible()
    await page.getByTestId('extraction-paused-resume').click()
    await expect(page.getByTestId('extraction-paused-banner')).toBeHidden({ timeout: 45_000 })
    expect(server.calls).toEqual(['create', 'accept Streekfonds', 'resume'])

    // Back to the source through the sidebar: complete.
    await page.getByRole('navigation').getByRole('link', { name: /^Sources/ }).click()
    await page.getByText('regio-deal.pdf').first().click({ timeout: 45_000 })
    await expect(page).toHaveURL(/\/sources\/source(:|%3A)g1$/, { timeout: 45_000 })
    await expect(
      page.locator('[data-variant="detail"] [data-node-key="complete"]'),
    ).toHaveAttribute('data-node-state', 'done', { timeout: 45_000 })
  })
})
