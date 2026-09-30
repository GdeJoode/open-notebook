/**
 * Track WB — the mocked backend the layout specs (WB.4, WB.5) render every
 * dashboard page against. Anything not served here answers 404, as a server
 * would for a missing route, and each page shows its own error state.
 */

import type { Page, Route } from '@playwright/test'

import { DEFAULT_SETTINGS, mockDashboardChrome } from '../track-a/_track-a-helpers'
import { NOTEBOOK, backend } from '../track-oe/_ontology-backend'
import attention from './attention.fixture.json'
import { NOTEBOOKS, fakeCounts, fakeList } from './sources-board.fake'

export { NOTEBOOK }
export const SOURCE = 'source:wb4'

export function json(route: Route, body: unknown, status = 200): Promise<void> {
  return route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })
}

export function apiSource(id: string, title: string, stage: string) {
  return {
    id, title, topics: [], asset: null, embedded: true, embedded_chunks: 4, insights_count: 0,
    entity_count: 5, relation_count: 2, created: '2026-09-29T10:00:00Z',
    updated: '2026-09-29T10:00:00Z', full_text: 'The convenant text.', metadata: {},
    private: false, notebooks: [NOTEBOOK], processing_stage: stage, command_id: null, status: null,
  }
}

export async function mockBackend(page: Page) {
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
