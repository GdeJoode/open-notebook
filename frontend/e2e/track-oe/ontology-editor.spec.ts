/**
 * Track OE — Phase OE.3 — the shared-ontology editor, end to end.
 *
 * AC (docs/tracks/OE-ontology-editor/plan.md §OE.3): rename → preview →
 * publish → rollback, plus the owner's drag-and-drop re-parenting (and Ctrl+X /
 * Ctrl+V as the keyboard way).
 *
 * Route-mocked, like the project's other E2E specs: a small in-memory backend
 * below answers the OE.2 API. It reproduces the contract the review found the
 * first double hid (review OE.3 M7): relationships stored as
 * `source_types`/`target_types`, validation that blocks a parent cycle, 409 on
 * a stale draft and 422 on a blocking one, a rollback that leaves the draft
 * alone (so it goes stale), and a rename that appends the old name to the
 * aliases and moves this ontology's references. Nothing touches a database.
 */

import { expect, test, type Page, type Route } from '@playwright/test'

import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

type EntityType = { name: string; parent_type?: string; aliases?: string[]; [k: string]: unknown }
type Relationship = { name: string; [k: string]: unknown }
type Content = { types: Record<string, EntityType>; rels: Record<string, Relationship> }

const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value))

function backend() {
  const seed: Content = {
    types: {
      Deal: { name: 'Deal', parent_type: 'GovernmentService' },
      Project: { name: 'Project', parent_type: 'GovernmentService', graph_relevance: 'active' },
    },
    rels: {
      BEVAT_PROJECT: { name: 'BEVAT_PROJECT', description: 'Deal bevat project',
        source_types: ['Deal'], target_types: ['Project'] },
    },
  }
  const state = {
    draft: null as null | { content: Content; base: string },
    active: { version: '1.0.0', content: clone(seed) },
    versions: [] as { version: string; note: string | null; is_active: boolean; content: Content }[],
    calls: [] as { method: string; path: string; body: unknown }[],
  }
  const current = () => state.draft?.content ?? state.active.content
  const editable = () => {
    if (!state.draft) state.draft = { content: clone(state.active.content), base: state.active.version }
    return state.draft.content
  }
  const draftBody = () => ({
    name: 'deals',
    exists: state.draft !== null,
    base_version: state.draft?.base ?? state.active.version,
    ontology: {
      metadata: { name: 'deals', version: state.active.version, extends: 'instruments' },
      entity_types: current().types,
      relationship_types: current().rels,
    },
  })
  const findings = () => {
    const types = current().types
    const out = []
    for (const t of Object.values(types)) {
      const seen = new Set([t.name])
      let parent = t.parent_type
      while (parent && types[parent]) {
        if (seen.has(parent)) {
          out.push({ rule: 'parent_cycle', ontology: null, subject: t.name, severity: 'block',
            weight: 1, detail: `parent_type loops through ${[...seen].join(',')}` })
          break
        }
        seen.add(parent)
        parent = types[parent].parent_type
      }
    }
    return out
  }
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
        entity_type_count: 2, relationship_type_count: 1, concept_count: 0 }])
    }
    if (path === '/deals' && method === 'GET') {
      return json(route, { name: 'deals', version: state.active.version, description: 'Deals',
        authors: null, extends: 'instruments', entity_type_count: 2, relationship_type_count: 1,
        concept_count: 0, relationship_types: [], concepts: [],
        entity_types: Object.values(state.active.content.types).map((t) => ({
          description: null, extraction_hints: null, examples: null, aliases: null,
          parent_type: null, properties: [], ...t })) })
    }
    if (path === '/instruments' && method === 'GET') {
      return json(route, { name: 'instruments', version: '1.0.0',
        entity_types: [{ name: 'Person', parent_type: 'Thing' }] })
    }
    if (path === '/deals/draft' && method === 'GET') return json(route, draftBody())
    if (path === '/deals/draft' && method === 'DELETE') {
      state.draft = null
      return route.fulfill({ status: 204 })
    }
    const put = path.match(/^\/deals\/draft\/entity-types\/([^/]+)$/)
    if (put && method === 'PUT') {
      editable().types[put[1]] = { ...(body as EntityType), name: put[1] }
      return json(route, draftBody())
    }
    const rename = path.match(/^\/deals\/draft\/entity-types\/([^/]+)\/rename$/)
    if (rename && method === 'POST') {
      const content = editable()
      const [old, next] = [rename[1], (body as { new_name: string }).new_name]
      const type = content.types[old]
      delete content.types[old]
      content.types[next] = { ...type, name: next, aliases: [...(type.aliases ?? []), old] }
      for (const t of Object.values(content.types)) if (t.parent_type === old) t.parent_type = next
      for (const r of Object.values(content.rels)) {
        for (const k of ['domain', 'range', 'source_types', 'target_types']) {
          if (Array.isArray(r[k])) r[k] = (r[k] as string[]).map((n) => (n === old ? next : n))
        }
      }
      return json(route, draftBody())
    }
    const rel = path.match(/^\/deals\/draft\/relationship-types\/([^/]+)$/)
    if (rel && method === 'PUT') {
      editable().rels[rel[1]] = { ...(body as Relationship), name: rel[1] }
      return json(route, draftBody())
    }
    if (path === '/deals/draft/validate') return json(route, findings())
    if (path === '/deals/draft/preview') {
      const before = state.active.content.types
      const changed = Object.values(current().types).filter(
        (t) => (before[t.name]?.parent_type ?? null) !== (t.parent_type ?? null))
      return json(route, { findings: findings(), blocking: findings().length > 0, detection: [],
        detection_error: null, notebook_edits: 0,
        label_changes: changed.map((t) => ({ label: t.name, before: 'programme', after: 'concept', entities: 2 })),
        retype: { changes: 0, collisions: 0, flag_only: 0, caused_by_this_edit: 0,
          collisions_caused_by_this_edit: 0 } })
    }
    if (path === '/deals/draft/publish') {
      if (!state.draft) return json(route, { detail: "'deals' has no draft to publish" }, 404)
      if (state.draft.base !== state.active.version) {
        return json(route, { detail: `'deals' is at version ${state.active.version} now` }, 409)
      }
      if (findings().length) {
        return json(route, { detail: { message: 'the draft has blocking findings', findings: findings() } }, 422)
      }
      if (state.versions.length === 0) {
        state.versions.push({ version: '1.0.0', note: 'YAML seed', is_active: false, content: clone(seed) })
      }
      const version = `1.0.0+2026092812000${state.versions.length}Z`
      state.versions.forEach((v) => (v.is_active = false))
      state.versions.push({ version, note: (body as { note: string | null }).note, is_active: true,
        content: clone(state.draft.content) })
      state.active = { version, content: clone(state.draft.content) }
      state.draft = null
      return json(route, { name: 'deals', version, warnings: [],
        retype: { changes: 0, collisions: 0, flag_only: 0, caused_by_this_edit: 0,
          collisions_caused_by_this_edit: 0 } })
    }
    if (path === '/deals/versions') {
      return json(route, state.versions.map((v) => ({ name: 'deals', created_at: '2026-09-28T12:00:00Z',
        created_by: 'ontology-editor', version: v.version, note: v.note, is_active: v.is_active })))
    }
    const rollback = path.match(/^\/deals\/rollback\/(.+)$/)
    if (rollback) {
      const target = state.versions.find((v) => v.version === rollback[1])
      if (!target) return json(route, { detail: 'no such version' }, 404)
      state.versions.forEach((v) => (v.is_active = v === target))
      state.active = { version: target.version, content: clone(target.content) }
      // Like the server: the draft is left as it is, and so goes stale.
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

const typeName = (page: Page, name: string) =>
  page.getByTestId(`ontology-type-${name}`).getByRole('button', { name, exact: true })

async function confirm(page: Page) {
  await page.getByTestId('confirm-action').click()
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
    await expect(page.getByTestId('ontology-type-Deal').locator('xpath=..')
      .getByTestId('ontology-type-Project')).toBeVisible()
    await expect(page.getByTestId('ontology-draft-state')).toHaveText('Draft')
  })

  test('the keyboard can move a type too', async ({ page }) => {
    const api = await openEditor(page)
    const handle = page.getByRole('button', { name: 'Move Project' })
    await handle.focus()
    await page.keyboard.press('Space')
    await expect(handle).toHaveAttribute('aria-pressed', 'true') // picked up
    // Deal is listed right above Project under the same parent. The keyboard
    // sensor moves in fixed steps; step up until Deal is the drop target (the
    // target is resolved asynchronously, so wait on it each time).
    const deal = page.getByTestId('ontology-type-Deal')
    for (let step = 0; step < 3; step++) {
      await page.keyboard.press('ArrowUp')
      const over = await deal.evaluate((el) => new Promise<boolean>((resolve) =>
        setTimeout(() => resolve(el.className.includes('bg-primary/10')), 250)))
      if (over) break
    }
    await expect(deal).toHaveClass(/bg-primary\/10/)
    await page.keyboard.press('Space')
    await expect.poll(() => api.state.calls.find((c) => c.method === 'PUT')?.body).toEqual({
      name: 'Project', parent_type: 'Deal', graph_relevance: 'active',
    })
  })

  test('cut and paste a type with Ctrl+X / Ctrl+V', async ({ page }) => {
    const api = await openEditor(page)
    await typeName(page, 'Project').focus()
    await page.keyboard.press('Control+x')
    await expect(page.getByTestId('ontology-cut-state')).toContainText('Project cut')
    await typeName(page, 'Deal').focus()
    await page.keyboard.press('Control+v')
    await expect.poll(() => api.state.calls.find((c) => c.method === 'PUT')?.body).toEqual({
      name: 'Project', parent_type: 'Deal', graph_relevance: 'active',
    })
    await expect(page.getByTestId('ontology-cut-state')).toHaveCount(0)
  })

  test('Ctrl+V on "No parent" makes a type a root; Escape cancels a cut', async ({ page }) => {
    const api = await openEditor(page)
    await typeName(page, 'Project').focus()
    await page.keyboard.press('Control+x')
    await page.keyboard.press('Escape')
    await expect(page.getByTestId('ontology-cut-state')).toHaveCount(0)
    await typeName(page, 'Project').focus()
    await page.keyboard.press('Control+x')
    await page.getByTestId('ontology-root-zone').focus()
    await page.keyboard.press('Control+v')
    await expect.poll(() => api.state.calls.find((c) => c.method === 'PUT')?.body).toEqual({
      name: 'Project', graph_relevance: 'active',
    })
  })

  test('rename → preview → publish → rollback', async ({ page }) => {
    const api = await openEditor(page)
    await typeName(page, 'Project').click()
    await page.getByLabel('New name').fill('Programmaproject')
    await page.getByTestId('ontology-type-rename').click()
    await expect(page.getByTestId('ontology-type-Programmaproject')).toBeVisible()
    // The rename moved the relationship's reference: the row shows the new name.
    await expect(page.getByLabel('BEVAT_PROJECT range')).toHaveValue('Programmaproject')

    // Re-parent it too, so the preview has a type change to show.
    await typeName(page, 'Programmaproject').click()
    await page.getByLabel('Parent type').fill('Deal')
    await page.getByTestId('ontology-type-save').click()
    await page.getByTestId('ontology-preview').click()
    await expect(page.getByTestId('ontology-preview-labels')).toContainText('Programmaproject')

    await page.getByLabel('Publish note').fill('rename Project')
    await page.getByTestId('ontology-publish').click()
    const versions = page.getByTestId('ontology-versions')
    await expect(versions).toContainText('rename Project')
    await expect(page.getByTestId('ontology-draft-state')).toHaveText('No draft')
    expect(api.state.active.content.types.Programmaproject.aliases).toEqual(['Project'])

    await page.getByTestId('ontology-rollback-1.0.0').click()
    await confirm(page)
    await expect.poll(() => api.state.active.version).toBe('1.0.0')
    await expect(page.getByTestId('ontology-type-Project')).toBeVisible()
  })

  test('editing a relationship keeps its source_types/target_types (review B1)', async ({ page }) => {
    const api = await openEditor(page)
    await page.getByLabel('BEVAT_PROJECT description').fill('Een deal bevat projecten')
    await page.getByTestId('ontology-relationship-BEVAT_PROJECT').getByRole('button', { name: 'Save' }).click()
    await expect.poll(() => api.state.calls.find((c) => c.method === 'PUT')?.body).toEqual({
      name: 'BEVAT_PROJECT', description: 'Een deal bevat projecten',
      source_types: ['Deal'], target_types: ['Project'],
    })
  })

  test('a parent cycle stays visible and blocks the publish (review B2)', async ({ page }) => {
    await openEditor(page)
    await typeName(page, 'Deal').click()
    await page.getByLabel('Parent type').fill('Project')
    await page.getByTestId('ontology-type-save').click()
    await typeName(page, 'Project').click()
    await page.getByLabel('Parent type').fill('Deal')
    await page.getByTestId('ontology-type-save').click()
    await expect(page.getByRole('alert').first()).toContainText('parent_cycle')
    await expect(page.getByTestId('ontology-type-Deal')).toBeVisible()
    await expect(page.getByTestId('ontology-type-Project')).toBeVisible()
    await expect(page.getByTestId('ontology-publish')).toBeDisabled()
  })

  test('a rollback makes the open draft stale: warned, not publishable (review M5)', async ({ page }) => {
    const api = await openEditor(page)
    // Publish once, so there is a version to roll back to.
    await typeName(page, 'Project').click()
    await page.getByLabel('Description', { exact: true }).fill('eerste')
    await page.getByTestId('ontology-type-save').click()
    await page.getByTestId('ontology-publish').click()
    await expect(page.getByTestId('ontology-draft-state')).toHaveText('No draft')
    // Start a new draft, then roll back underneath it.
    await typeName(page, 'Project').click()
    await page.getByLabel('Description', { exact: true }).fill('tweede')
    await page.getByTestId('ontology-type-save').click()
    await expect(page.getByTestId('ontology-draft-state')).toHaveText('Draft')
    await page.getByTestId('ontology-rollback-1.0.0').click()
    await confirm(page)
    await expect(page.getByTestId('ontology-draft-stale')).toBeVisible()
    await expect(page.getByTestId('ontology-publish')).toBeDisabled()
    expect(api.state.draft).not.toBeNull() // the edits are kept until the user discards
  })
})
