/**
 * Track OE — Phase OE.3 — the shared-ontology editor, end to end.
 *
 * AC (docs/tracks/OE-ontology-editor/plan.md §OE.3): rename → preview →
 * publish → rollback, plus the owner's drag-and-drop re-parenting.
 *
 * Route-mocked, like the project's other E2E specs: a small in-memory backend
 * below answers the OE.2 API (draft, validate, preview, publish, versions,
 * rollback) with its real response shapes, so the flow runs as it would against
 * the server — and nothing touches a real database.
 */

import { expect, test, type Page, type Route } from '@playwright/test'

import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

type EntityType = { name: string; parent_type?: string; aliases?: string[]; [k: string]: unknown }

function backend() {
  const seed: Record<string, EntityType> = {
    Deal: { name: 'Deal', parent_type: 'GovernmentService' },
    Project: { name: 'Project', parent_type: 'GovernmentService', graph_relevance: 'active' },
  }
  const state = {
    draft: null as null | Record<string, EntityType>,
    active: { version: '1.0.0', types: seed },
    versions: [] as { version: string; note: string | null; is_active: boolean; types: Record<string, EntityType> }[],
    calls: [] as { method: string; path: string; body: unknown }[],
  }
  const own = () => state.draft ?? state.active.types
  const draftBody = () => ({
    name: 'deals',
    exists: state.draft !== null,
    base_version: state.active.version,
    ontology: {
      metadata: { name: 'deals', version: state.active.version, extends: 'instruments' },
      entity_types: own(),
      relationship_types: {},
    },
  })
  const json = (route: Route, body: unknown, status = 200) =>
    route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })

  async function handle(route: Route) {
    const request = route.request()
    const url = new URL(request.url())
    const path = decodeURIComponent(url.pathname.replace(/^.*\/api\/ontologies/, ''))
    const method = request.method()
    const body = request.postDataJSON?.() ?? null
    state.calls.push({ method, path, body })

    if (path === '' && method === 'GET') {
      return json(route, [{ name: 'deals', version: state.active.version, description: 'Deals',
        entity_type_count: Object.keys(own()).length, relationship_type_count: 0, concept_count: 0 }])
    }
    if (path === '/deals' && method === 'GET') {
      return json(route, { name: 'deals', version: state.active.version, description: 'Deals',
        authors: null, extends: 'instruments', entity_type_count: 2, relationship_type_count: 0,
        concept_count: 0, entity_types: Object.values(state.active.types).map((t) => ({
          description: null, extraction_hints: null, examples: null, aliases: null,
          parent_type: null, properties: [], ...t })), relationship_types: [], concepts: [] })
    }
    if (path === '/instruments' && method === 'GET') {
      return json(route, { name: 'instruments', entity_types: [{ name: 'Person', parent_type: 'Thing' }] })
    }
    if (path === '/deals/draft' && method === 'GET') return json(route, draftBody())
    if (path === '/deals/draft' && method === 'DELETE') {
      state.draft = null
      return route.fulfill({ status: 204 })
    }
    const put = path.match(/^\/deals\/draft\/entity-types\/([^/]+)$/)
    if (put && method === 'PUT') {
      state.draft = { ...own(), [put[1]]: { ...(body as EntityType), name: put[1] } }
      return json(route, draftBody())
    }
    const rename = path.match(/^\/deals\/draft\/entity-types\/([^/]+)\/rename$/)
    if (rename && method === 'POST') {
      const { [rename[1]]: old, ...rest } = own()
      const newName = (body as { new_name: string }).new_name
      state.draft = { ...rest, [newName]: { ...old, name: newName, aliases: [rename[1]] } }
      return json(route, draftBody())
    }
    if (path === '/deals/draft/validate') return json(route, [])
    if (path === '/deals/draft/preview') {
      return json(route, { findings: [], blocking: false, detection: [], detection_error: null,
        notebook_edits: 0, label_changes: [{ label: 'Project', before: 'programme',
          after: 'programme', entities: 3 }],
        retype: { changes: 0, collisions: 0, flag_only: 0, caused_by_this_edit: 0,
          collisions_caused_by_this_edit: 0 } })
    }
    if (path === '/deals/draft/publish') {
      if (state.versions.length === 0) {
        state.versions.push({ version: '1.0.0', note: 'YAML seed', is_active: false, types: seed })
      }
      const version = '1.0.0+20260928T120000Z'
      state.versions.forEach((v) => (v.is_active = false))
      state.versions.push({ version, note: (body as { note: string | null }).note,
        is_active: true, types: state.draft ?? {} })
      state.active = { version, types: state.draft ?? {} }
      state.draft = null
      return json(route, { name: 'deals', version, warnings: [],
        retype: { changes: 0, collisions: 0, flag_only: 0, caused_by_this_edit: 0 } })
    }
    if (path === '/deals/versions') {
      return json(route, state.versions.map(({ types: _types, ...v }) => ({
        name: 'deals', created_at: '2026-09-28T12:00:00Z', created_by: 'ontology-editor', ...v })))
    }
    const rollback = path.match(/^\/deals\/rollback\/(.+)$/)
    if (rollback) {
      const target = state.versions.find((v) => v.version === rollback[1])
      if (!target) return json(route, { detail: 'no such version' }, 404)
      state.versions.forEach((v) => (v.is_active = v === target))
      state.active = { version: target.version, types: target.types }
      return json(route, { name: 'deals', version: target.version, warnings: [] })
    }
    return json(route, { detail: `unmocked ${method} ${path}` }, 500)
  }
  return { state, handle }
}

async function openEditor(page: Page) {
  const api = backend()
  await mockDashboardChrome(page)
  await page.route('**/api/ontologies**', api.handle)
  await gotoAndWait(page, '/ontologies')
  await page.getByText('deals', { exact: true }).first().click()
  await page.getByTestId('ontology-edit-tab').click()
  await expect(page.getByTestId('ontology-editor')).toBeVisible()
  return api
}

test.describe('OE.3 — ontology editor', () => {
  test('drag a type onto its new parent', async ({ page }) => {
    const api = await openEditor(page)
    const handle = page.getByRole('button', { name: 'Move Project' })
    const target = page.getByTestId('ontology-type-Deal')
    await handle.hover()
    await page.mouse.down()
    const box = await target.boundingBox()
    if (!box) throw new Error('no target box')
    await page.mouse.move(box.x + 20, box.y + box.height / 2, { steps: 12 })
    await page.mouse.move(box.x + 24, box.y + box.height / 2, { steps: 4 })
    await page.mouse.up()

    await expect.poll(() => api.state.calls.find((c) => c.method === 'PUT')?.body).toEqual({
      name: 'Project', parent_type: 'Deal', graph_relevance: 'active',
    })
    // Project now hangs under Deal in the tree.
    await expect(page.getByTestId('ontology-type-Deal').locator('xpath=..')
      .getByTestId('ontology-type-Project')).toBeVisible()
    await expect(page.getByTestId('ontology-draft-state')).toHaveText('Draft')
  })

  test('the keyboard can move a type too', async ({ page }) => {
    const api = await openEditor(page)
    const handle = page.getByRole('button', { name: 'Move Project' })
    await handle.focus()
    await page.keyboard.press('Space')
    // Deal is listed right above Project under the same parent.
    await page.keyboard.press('ArrowUp')
    await page.keyboard.press('Space')
    await expect.poll(() => api.state.calls.some((c) => c.method === 'PUT')).toBe(true)
  })

  test('rename → preview → publish → rollback', async ({ page }) => {
    const api = await openEditor(page)
    await page.getByTestId('ontology-type-Project').getByRole('button', { name: 'Project' }).click()
    await page.getByLabel('New name').fill('Programmaproject')
    await page.getByTestId('ontology-type-rename').click()
    await expect(page.getByTestId('ontology-type-Programmaproject')).toBeVisible()

    await page.getByTestId('ontology-preview').click()
    await expect(page.getByTestId('ontology-preview-result')).toContainText('Project')

    await page.getByLabel('Publish note').fill('rename Project')
    await page.getByTestId('ontology-publish').click()
    const versions = page.getByTestId('ontology-versions')
    await expect(versions).toContainText('1.0.0+20260928T120000Z')
    await expect(versions).toContainText('rename Project')
    await expect(page.getByTestId('ontology-draft-state')).toHaveText('No draft')

    await page.getByTestId('ontology-rollback-1.0.0').click()
    await expect.poll(() => api.state.active.version).toBe('1.0.0')
    await expect(page.getByTestId('ontology-type-Project')).toBeVisible()
  })
})
