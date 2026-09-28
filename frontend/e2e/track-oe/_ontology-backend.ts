/**
 * The OE e2e specs' in-memory backend: the OE.2 ontology API, the OE.4 retype
 * and (OE.5) one notebook's Pass-1 proposal queue, whose Accept publishes into
 * the same ontologies through the same gate — refused on a blocking finding
 * (422 `{message, findings}`, nothing stored) or an open draft (409). Shared by
 * `ontology-editor.spec.ts` and `notebook-accept.spec.ts`.
 */

import type { Route } from '@playwright/test'

export const NOTEBOOK = 'notebook:oe5'

type Proposal = { extension_id: string; type_name: string; parent_type: string | null;
  rationale: string | null; properties: unknown[] }

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
    fingerprint }
}

const clone = <T>(value: T): T => JSON.parse(JSON.stringify(value))

export function backend() {
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
    // OE.5: one notebook's Pass-1 proposal queue.
    pending: [
      { extension_id: 'pass1::streekfonds', type_name: 'Streekfonds', parent_type: 'Deal',
        rationale: 'A regional fund', properties: [] },
      { extension_id: 'pass1::zwerfbegrip', type_name: 'Zwerfbegrip', parent_type: 'Nergens',
        rationale: null, properties: [] },
    ] as Proposal[],
    accepted: [] as (Proposal & { published_to: string; published_version: string })[],
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
  const findings = () => findingsOf(current())
  function findingsOf(content: Content) {
    const types = content.types
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

  /** Like `validation.orphan`: a type whose chain reaches no base, where the
   *  active set had none such. `findings()` adds the parent-cycle rule. */
  const gate = (content: Content) => {
    const out: Record<string, unknown>[] = [...findingsOf(content)]
    for (const t of Object.values(content.types)) {
      const before = state.active.content.types[t.name]
      if (typeOf(t.name, content) === 'concept' && !(before && typeOf(before.name, state.active.content) === 'concept')) {
        out.push({ rule: 'orphan', ontology: 'deals', subject: t.name, severity: 'block', weight: 1,
          detail: `'${t.name}' reaches no schema.org base through parent_type / schema_org_type` })
      }
    }
    return out
  }
  /** The candidate an accept would publish: the ACTIVE content (not the
   *  draft) with the new type, as `OntologyEditService._with_new_type`. */
  const withNewType = (proposal: Proposal, parent: string | null | undefined) => {
    const content = clone(state.active.content)
    const chosen = parent === undefined ? proposal.parent_type : parent
    content.types[proposal.type_name] = { name: proposal.type_name,
      ...(chosen ? { parent_type: chosen } : {}),
      ...(proposal.rationale ? { description: proposal.rationale } : {}) }
    return content
  }
  const schemaView = () => ({
    notebook_id: NOTEBOOK, base_ontology: 'deals',
    base_ontology_types: Object.values(state.active.content.types).map((t) => ({
      name: t.name, description: null, parent_type: t.parent_type ?? null, properties: [] })),
    pending_extensions: state.pending, coverage_pct: 0.8, review_required: false,
    soft_nudge_dismissed: false,
  })

  /** `/api/notebooks/{NOTEBOOK}/...`: the Schema tab and the proposal queue (OE.5). */
  async function handleNotebook(route: Route) {
    const request = route.request()
    const url = new URL(request.url())
    const path = decodeURIComponent(url.pathname.replace(/^.*\/api\/notebooks\/[^/]+/, ''))
    const method = request.method()
    const body = request.postDataJSON?.() ?? null
    state.calls.push({ method, path: `notebook${path}`, body })
    if (path === '' && method === 'GET') {
      return json(route, { id: NOTEBOOK, name: 'OE.5 notebook', description: '', archived: false,
        created: '2026-09-28T00:00:00Z', updated: '2026-09-28T00:00:00Z' })
    }
    if (path === '/schema' && method === 'GET') return json(route, schemaView())
    if (path === '/pass1_results') return json(route, [])
    const m = path.match(/^\/schema\/extensions\/([^/]+)\/(accept|accept\/preview|reject)$/)
    // Everything else on the page (orphans, audit, events, …) is not this
    // spec's: left to the network, as the track-b specs do.
    if (!m) return route.fallback()
    const proposal = state.pending.find((p) => p.type_name === m[1])
    if (m[2] === 'reject') {
      state.pending = state.pending.filter((p) => p !== proposal)
      return json(route, schemaView())
    }
    if (!proposal) {
      if (m[2] === 'accept' && method === 'POST' && state.accepted.some((a) => a.type_name === m[1])) {
        return json(route, { notebook_schema: schemaView(), published: null })
      }
      return json(route, { detail: `No pending extension named '${m[1]}'` }, 404)
    }
    if (m[2] === 'accept' && method === 'GET') {
      const defined = Object.values(state.active.content.types)
        .some((t) => t.name.toLowerCase() === (proposal.parent_type ?? '').toLowerCase())
      return json(route, { type_name: proposal.type_name, parent_type: proposal.parent_type,
        default_ontology: defined ? 'deals' : 'schema_core', ontologies: ['deals', 'schema_core'] })
    }
    const { ontology, parent_type } = body as { ontology: string; parent_type?: string | null }
    if (ontology !== 'deals') return json(route, { detail: `unmocked ontology ${ontology}` }, 500)
    if (state.active.content.types[proposal.type_name]) {
      return json(route, { detail: `'deals' already defines '${proposal.type_name}'` }, 409)
    }
    const candidate = withNewType(proposal, parent_type === '' ? null : parent_type)
    const found = gate(candidate)
    if (m[2] === 'accept/preview') {
      return json(route, { findings: found, blocking: found.some((f) => f.severity === 'block'),
        draft_open: state.draft !== null, detection: [], detection_error: null,
        label_changes: [], retype: retypeCounts(candidate) })
    }
    if (state.draft) {
      return json(route, { detail: "'deals' has an unpublished draft in the ontology editor; " +
        'publish or discard it first' }, 409)
    }
    if (found.some((f) => f.severity === 'block')) {
      return json(route, { detail: { message: 'the new type has blocking findings',
        findings: found } }, 422)
    }
    if (state.versions.length === 0) {
      state.versions.push({ version: '1.0.0', note: 'YAML seed', is_active: false, content: clone(seed) })
    }
    const retype = retypeCounts(candidate)
    const version = `1.0.0+2026092812000${state.versions.length}Z`
    state.versions.forEach((v) => (v.is_active = false))
    state.versions.push({ version, note: `accepted from ${NOTEBOOK}'s Pass-1 proposals`,
      is_active: true, content: clone(candidate) })
    state.active = { version, content: candidate }
    state.pending = state.pending.filter((p) => p !== proposal)
    state.accepted.push({ ...proposal, published_to: 'deals', published_version: version })
    return json(route, { notebook_schema: schemaView(),
      published: { name: 'deals', version, warnings: found, retype } })
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
        detection_error: null,
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
  return { state, handle, handleNotebook }
}
