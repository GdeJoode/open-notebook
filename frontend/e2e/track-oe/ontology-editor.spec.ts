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
 *
 * OE.4 adds the retype (`/api/ontologies/retype/...`), modelled on
 * `retype_planner.plan_rows` and `EntityRetypeService`: entities keep the type
 * they were written with; the plan types each row's label under the ACTIVE
 * content (name or alias, up the parent chain to a base) and is conservative
 * (every current identity stays held, so a move onto a held `(name_key, type)`
 * is a collision); the preview and publish count the same plan; an apply
 * writes only when the fingerprint it echoes is the current plan's (409
 * otherwise) and returns a rollback id to undo it.
 */

import { expect, test, type Page, type Route } from '@playwright/test'

import { gotoAndWait } from '../_helpers'
import { mockDashboardChrome } from '../track-a/_track-a-helpers'

type EntityType = { name: string; parent_type?: string; aliases?: string[]; [k: string]: unknown }
type Relationship = { name: string; [k: string]: unknown }
type Content = { types: Record<string, EntityType>; rels: Record<string, Relationship> }
type Entity = { id: string; name: string; name_key: string; label: string; entity_type: string }
type RetypeRow = { id: string; name: string; label: string; status: string; old: string; new: string;
  old_flag: null; new_flag: null; held_by?: string[] }

/** The canonical type of a label under `content`: its type by name or alias,
 *  up the parent chain, to the first base. A label defined nowhere is concept. */
const BASES: Record<string, string> = { GovernmentService: 'programme', Grant: 'grant' }
function typeOf(label: string, content: Content): string {
  const find = (n: string) => Object.values(content.types)
    .find((t) => t.name === n || (t.aliases ?? []).includes(n))
  let current: string | undefined = find(label)?.name ?? label
  const seen = new Set<string>()
  while (current && !seen.has(current)) {
    if (BASES[current]) return BASES[current]
    seen.add(current)
    current = find(current)?.parent_type
  }
  return 'concept'
}

function retypePlan(entities: Entity[], content: Content) {
  const holders = new Map<string, Set<string>>()
  const key = (k: string, t: string) => `${k}|${t}`
  const hold = (k: string, id: string) => holders.set(k, (holders.get(k) ?? new Set()).add(id))
  for (const e of entities) hold(key(e.name_key, e.entity_type), e.id)
  const changes: RetypeRow[] = []
  const collisions: RetypeRow[] = []
  for (const e of entities) {
    const next = typeOf(e.label, content)
    if (next === e.entity_type) continue
    const item: RetypeRow = { id: e.id, name: e.name, label: e.label, status: 'active',
      old: e.entity_type, new: next, old_flag: null, new_flag: null }
    const held = [...(holders.get(key(e.name_key, next)) ?? [])].filter((id) => id !== e.id)
    if (held.length) {
      collisions.push({ ...item, held_by: held.sort() })
      continue
    }
    hold(key(e.name_key, next), e.id)
    changes.push(item)
  }
  const byId = (a: RetypeRow, b: RetypeRow) => a.id.localeCompare(b.id)
  const fingerprint = JSON.stringify({ changes: [...changes].sort(byId),
    collisions: [...collisions].sort(byId), flag_only: [] })
  return { total: entities.length, changes, collisions, flag_only: [] as RetypeRow[],
    counts: { changes: changes.length, collisions: collisions.length, flag_only: 0 },
    fingerprint, notebook_edits: [] as string[] }
}

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
    // Typed as the seed types their labels: nothing to retype until a publish.
    entities: [
      { id: 'entity:1', name: 'Regio Deal Twente', name_key: 'regio deal twente', label: 'Project',
        entity_type: 'programme' },
      { id: 'entity:2', name: 'Groeiagenda', name_key: 'groeiagenda', label: 'Project',
        entity_type: 'programme' },
      { id: 'entity:3', name: 'Groeiagenda', name_key: 'groeiagenda', label: 'Onderwerp',
        entity_type: 'concept' },
      { id: 'entity:4', name: 'Deal Noord', name_key: 'deal noord', label: 'Deal',
        entity_type: 'programme' },
    ] as Entity[],
    rollbacks: {} as Record<string, RetypeRow[]>,
  }
  /** The retype counts a publish of `content` reports (the preview's `retype`). */
  const retypeCounts = (content: Content) => {
    const plan = retypePlan(state.entities, content)
    const caused = (rows: RetypeRow[]) => rows.filter((r) => {
      const e = state.entities.find((x) => x.id === r.id)!
      return typeOf(e.label, state.active.content) !== r.new
    }).length
    return { ...plan.counts, caused_by_this_edit: caused(plan.changes),
      collisions_caused_by_this_edit: caused(plan.collisions) }
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
    // The chain deals → instruments → government → schema_core, each GET
    // returning its MERGED types, as the browse API does.
    const person = { name: 'Person', parent_type: 'Thing' }
    const chain: Record<string, unknown> = {
      '/instruments': { name: 'instruments', version: '1.0.0', extends: 'government', includes: [],
        entity_types: [person, { name: 'Subsidie', parent_type: 'Grant' }] },
      '/government': { name: 'government', version: '1.0.0', extends: 'schema_core', includes: [],
        entity_types: [person] },
      '/schema_core': { name: 'schema_core', version: '1.0.0', extends: null, includes: [],
        entity_types: [person] },
    }
    if (chain[path] && method === 'GET') return json(route, chain[path])
    if (path === '/retype/dry-run' && method === 'GET') {
      return json(route, retypePlan(state.entities, state.active.content))
    }
    if (path === '/retype/apply' && method === 'POST') {
      const plan = retypePlan(state.entities, state.active.content)
      if ((body as { fingerprint: string }).fingerprint !== plan.fingerprint) {
        return json(route, { detail: 'the retype plan changed since the dry run was shown' }, 409)
      }
      const written = plan.changes.filter((c) => {
        const e = state.entities.find((x) => x.id === c.id)
        if (!e || e.entity_type !== c.old) return false // guarded on the old type
        e.entity_type = c.new
        return true
      })
      const rollbackId = written.length
        ? `pc4b_retype_2026092812000${Object.keys(state.rollbacks).length}000000Z.jsonl` : null
      if (rollbackId) state.rollbacks[rollbackId] = written
      return json(route, { written: written.length, skipped: [], failed: [], rollback_id: rollbackId })
    }
    const undo = path.match(/^\/retype\/undo\/(.+)$/)
    if (undo && method === 'POST') {
      const rows = state.rollbacks[undo[1]]
      if (!rows) return json(route, { detail: 'no rollback file' }, 404)
      const restored = rows.filter((c) => {
        const e = state.entities.find((x) => x.id === c.id)
        if (!e || e.entity_type !== c.new) return false
        e.entity_type = c.old
        return true
      })
      return json(route, { written: restored.length, skipped: [], failed: [] })
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
        // Like the server: the renamed side is rewritten, and a relationship
        // spelled source_types/target_types gains domain/range for that side.
        for (const [modern, legacy] of [['domain', 'source_types'], ['range', 'target_types']]) {
          const names = (r[modern] ?? r[legacy]) as string[] | undefined
          if (!names?.includes(old)) continue
          const renamed = names.map((n) => (n === old ? next : n))
          r[modern] = renamed
          if (legacy in r) r[legacy] = renamed
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
        retype: retypeCounts(current()) })
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
      const retype = retypeCounts(state.draft.content) // counted before it takes effect
      const version = `1.0.0+2026092812000${state.versions.length}Z`
      state.versions.forEach((v) => (v.is_active = false))
      state.versions.push({ version, note: (body as { note: string | null }).note, is_active: true,
        content: clone(state.draft.content) })
      state.active = { version, content: clone(state.draft.content) }
      state.draft = null
      return json(route, { name: 'deals', version, warnings: [], retype })
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

  test('an inherited type names the ontology that defines it (round-2 major 2)', async ({ page }) => {
    await openEditor(page)
    await expect(page.getByTestId('ontology-type-Person')).toContainText('from schema_core')
    await expect(page.getByTestId('ontology-type-Subsidie')).toContainText('from instruments')
  })

  test('a save does not undo a drag of the selected type (round-2 major 1)', async ({ page }) => {
    const api = await openEditor(page)
    await typeName(page, 'Project').click()
    await page.getByLabel('Description', { exact: true }).fill('beschrijving')
    await typeName(page, 'Project').focus()
    await page.keyboard.press('Control+x')
    await typeName(page, 'Deal').focus()
    await page.keyboard.press('Control+v')
    await expect(page.getByLabel('Parent type')).toHaveValue('Deal')
    await page.getByTestId('ontology-type-save').click()
    await expect.poll(() => api.state.calls.filter((c) => c.method === 'PUT').at(-1)?.body).toEqual({
      name: 'Project', parent_type: 'Deal', graph_relevance: 'active', description: 'beschrijving',
    })
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

test.describe('OE.4 — the retype after a publish', () => {
  async function publishProjectAsRoot(page: Page) {
    await typeName(page, 'Project').focus()
    await page.keyboard.press('Control+x')
    await page.getByTestId('ontology-root-zone').focus()
    await page.keyboard.press('Control+v')
    await expect(page.getByTestId('ontology-draft-state')).toHaveText('Draft')
    await page.getByTestId('ontology-publish').click()
  }

  test('publish → dry run → apply → the dry run is empty, the collision is left', async ({ page }) => {
    const api = await openEditor(page)
    await publishProjectAsRoot(page)

    // The publish reported entities it moves, so the dry run is offered.
    const panel = page.getByTestId('ontology-retype')
    await expect(panel.getByTestId('ontology-retype-changes')).toContainText('Regio Deal Twente')
    await expect(panel.getByTestId('ontology-retype-changes')).toContainText('programme → concept')
    const collisions = panel.getByTestId('ontology-retype-collisions')
    await expect(collisions).toContainText('held by entity:3')
    await expect(collisions.getByRole('link', { name: 'resolution page' }))
      .toHaveAttribute('href', '/knowledge-graph/resolution')

    await panel.getByTestId('ontology-retype-apply').click()
    await confirm(page)
    await expect(panel.getByTestId('ontology-retype-empty')).toBeVisible()
    await expect(panel.getByTestId('ontology-retype-result')).toContainText('Written 1')
    await expect(collisions).toContainText('Groeiagenda') // left for the curator
    expect(Object.fromEntries(api.state.entities.map((e) => [e.id, e.entity_type]))).toEqual({
      'entity:1': 'concept', 'entity:2': 'programme', 'entity:3': 'concept', 'entity:4': 'programme',
    })

    await panel.getByTestId('ontology-retype-undo').click()
    await confirm(page)
    await expect(panel.getByTestId('ontology-retype-changes')).toContainText('Regio Deal Twente')
    expect(api.state.entities.find((e) => e.id === 'entity:1')?.entity_type).toBe('programme')
  })

  test('an apply of a plan that changed since the dry run writes nothing and shows the new plan',
    async ({ page }) => {
      const api = await openEditor(page)
      await publishProjectAsRoot(page)
      const panel = page.getByTestId('ontology-retype')
      await expect(panel.getByTestId('ontology-retype-changes')).toContainText('Regio Deal Twente')
      // Extracted meanwhile, under the old typing.
      api.state.entities.push({ id: 'entity:5', name: 'Fonds West', name_key: 'fonds west',
        label: 'Project', entity_type: 'programme' })

      await panel.getByTestId('ontology-retype-apply').click()
      await confirm(page)
      await expect(panel.getByTestId('ontology-retype-changes')).toContainText('Fonds West')
      expect(api.state.entities.filter((e) => e.entity_type === 'concept').map((e) => e.id))
        .toEqual(['entity:3'])

      await panel.getByTestId('ontology-retype-apply').click()
      await confirm(page)
      await expect(panel.getByTestId('ontology-retype-empty')).toBeVisible()
    })

  test('the retype is there on demand, and says when there is nothing to do', async ({ page }) => {
    await openEditor(page)
    await page.getByTestId('ontology-retype-open').click()
    await expect(page.getByTestId('ontology-retype-empty')).toBeVisible()
    await expect(page.getByTestId('ontology-retype-apply')).toHaveCount(0)
  })
})
