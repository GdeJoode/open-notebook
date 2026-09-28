/**
 * OE.3: the drag-and-drop tree's rules. The server's gate validates what a new
 * parent does to typing; these decide which drops the editor offers at all,
 * and that every type stays reachable in the tree.
 */
import { describe, expect, it } from 'vitest'

import {
  ROOT_ZONE,
  buildTypeTree,
  canDrop,
  descendantsOf,
  dropEdit,
  inheritanceClosure,
  inheritedTypes,
  nameTaken,
  reparented,
  zoneId,
  type InheritedType,
} from '../ontology-tree'
import { endpointsOf, withEndpoints } from '../relationship-endpoints'
import type { StoredEntityType } from '@/lib/api/ontologies'

const own: Record<string, StoredEntityType> = {
  Deal: { name: 'Deal', parent_type: 'GovernmentService' },
  RegioDeal: { name: 'RegioDeal', parent_type: 'Deal', graph_relevance: 'active' },
  Project: { name: 'Project', parent_type: 'GovernmentService' },
  Losstaand: { name: 'Losstaand' },
}
const inherited: InheritedType[] = [
  { name: 'Person', parent_type: 'Thing', definedIn: 'instruments' },
  { name: 'Thing', parent_type: null, definedIn: 'instruments' },
  { name: 'Deal', parent_type: 'CreativeWork', definedIn: 'instruments' }, // redefined by `own`
  { name: 'BeleidsThema', parent_type: 'Concept', definedIn: 'policy_themes' },
]

describe('buildTypeTree', () => {
  const tree = buildTypeTree(own, inherited)

  it('marks own types, inherited types (with their ontology) and undefined parents', () => {
    expect(tree.nodes.get('regiodeal')).toMatchObject({ own: true, placeholder: false })
    expect(tree.nodes.get('person')).toMatchObject({ own: false, definedIn: 'instruments' })
    expect(tree.nodes.get('beleidsthema')).toMatchObject({ definedIn: 'policy_themes' })
    expect(tree.nodes.get('governmentservice')).toMatchObject({ placeholder: true })
  })

  it('an own type redefining an inherited one wins', () => {
    expect(tree.nodes.get('deal')).toMatchObject({ own: true, parent: 'GovernmentService' })
    expect(tree.nodes.has('creativework')).toBe(false)
  })

  it('hangs children under their parent and the rest at the root', () => {
    expect(tree.roots).toEqual(['Concept', 'GovernmentService', 'Losstaand', 'Thing'])
    expect(tree.children.get('governmentservice')).toEqual(['Deal', 'Project'])
    expect(tree.children.get('deal')).toEqual(['RegioDeal'])
  })

  it('keeps every type of a parent cycle reachable (review B2)', () => {
    const cyclic = buildTypeTree({
      A: { name: 'A', parent_type: 'B' }, B: { name: 'B', parent_type: 'A' }, C: { name: 'C' },
    }, [])
    const reachable = new Set<string>()
    const walk = (name: string) => {
      reachable.add(name.toLowerCase())
      for (const child of cyclic.children.get(name.toLowerCase()) ?? []) walk(child)
    }
    cyclic.roots.forEach(walk)
    expect(reachable).toEqual(new Set(['a', 'b', 'c']))
    expect([...cyclic.nodes.values()].filter((n) => n.inCycle).map((n) => n.name)).toEqual(['A'])
  })

  it('marks only a type ON the cycle; one below it keeps its parent (round-2 minor 1)', () => {
    const t = buildTypeTree({
      X: { name: 'X', parent_type: 'Y' }, Y: { name: 'Y', parent_type: 'X' },
      D: { name: 'D', parent_type: 'X' },
    }, [])
    expect([...t.nodes.values()].filter((n) => n.inCycle).map((n) => n.name)).toEqual(['X'])
    expect(t.children.get('x')).toEqual(expect.arrayContaining(['D']))
    expect(canDrop(t, 'X', 'D')).toBe(false) // D still counts as X's descendant
  })

  it('resolves a parent given by an alias, like the backend (review minor 5)', () => {
    const aliased = buildTypeTree({ Kind: { name: 'Kind', parent_type: 'OudeNaam' } },
      [{ name: 'NieuweNaam', aliases: ['OudeNaam'], definedIn: 'deals' }])
    expect(aliased.nodes.has('oudenaam')).toBe(false)
    expect(aliased.children.get('nieuwenaam')).toEqual(['Kind'])
  })
})

describe('canDrop', () => {
  const tree = buildTypeTree(own, inherited)

  it('allows an own type under another type or a base', () => {
    expect(canDrop(tree, 'Project', 'Deal')).toBe(true)
    expect(canDrop(tree, 'Losstaand', 'Person')).toBe(true)
    expect(canDrop(tree, 'RegioDeal', 'GovernmentService')).toBe(true)
  })

  it('refuses an inherited type: it belongs to another ontology', () => {
    expect(canDrop(tree, 'Person', 'Deal')).toBe(false)
  })

  it('refuses itself, its own descendants (a cycle) and its current parent', () => {
    expect(canDrop(tree, 'Deal', 'Deal')).toBe(false)
    expect(canDrop(tree, 'Deal', 'RegioDeal')).toBe(false)
    expect(canDrop(tree, 'RegioDeal', 'Deal')).toBe(false)
  })

  it('"no parent" only for a type that has one', () => {
    expect(canDrop(tree, 'Project', null)).toBe(true)
    expect(canDrop(tree, 'Losstaand', null)).toBe(false)
  })

  it('matches names case-insensitively, like the backend', () => {
    expect(canDrop(tree, 'project', 'deal')).toBe(true)
    expect(descendantsOf(tree, 'DEAL').has('regiodeal')).toBe(true)
  })
})

describe('reparented', () => {
  it('changes only the parent and keeps every other key', () => {
    expect(reparented(own.RegioDeal, 'Project')).toEqual({
      name: 'RegioDeal', parent_type: 'Project', graph_relevance: 'active',
    })
    expect(own.RegioDeal.parent_type).toBe('Deal') // not mutated
  })

  it('removes the parent for a drop on "no parent"', () => {
    expect(reparented(own.Project, null)).toEqual({ name: 'Project' })
  })
})

describe('dropEdit', () => {
  const tree = buildTypeTree(own, inherited)

  it('turns a drop on a type into that type as parent', () => {
    expect(dropEdit(tree, own, 'Project', zoneId('Deal'))).toEqual(
      { name: 'Project', parent_type: 'Deal' })
  })

  it('turns a drop on the root zone into no parent', () => {
    expect(dropEdit(tree, own, 'Project', ROOT_ZONE)).toEqual({ name: 'Project' })
  })

  it('is no edit when dropped outside, refused, or on an unknown zone', () => {
    expect(dropEdit(tree, own, 'Project', null)).toBeNull()
    expect(dropEdit(tree, own, 'Deal', zoneId('RegioDeal'))).toBeNull()
    expect(dropEdit(tree, own, 'Person', zoneId('Deal'))).toBeNull()
    expect(dropEdit(tree, own, 'Project', 'somewhere')).toBeNull()
  })
})

describe('nameTaken (review M1: "Add" must not replace an existing type)', () => {
  it('finds a name case-insensitively', () => {
    expect(nameTaken(own, 'project')).toBe(true)
    expect(nameTaken(own, ' Deal ')).toBe(true)
    expect(nameTaken(own, 'Nieuw')).toBe(false)
  })
})

describe('relationship endpoints (review B1: both stored spellings)', () => {
  const legacy = { name: 'BEVAT_PROJECT', description: 'x',
    source_types: ['Deal'], target_types: ['Project'] }
  const modern = { name: 'FUNDS', domain: ['Grant'], range: ['Project'], extra: 1 }

  it('reads either spelling', () => {
    expect(endpointsOf(legacy)).toEqual({ domain: ['Deal'], range: ['Project'] })
    expect(endpointsOf(modern)).toEqual({ domain: ['Grant'], range: ['Project'] })
  })

  it('writes back the spelling the relationship uses, every other key kept', () => {
    expect(withEndpoints(legacy, ['Deal'], ['Project', 'Voorziening'])).toEqual({
      name: 'BEVAT_PROJECT', description: 'x',
      source_types: ['Deal'], target_types: ['Project', 'Voorziening'],
    })
    expect(withEndpoints(modern, ['Grant'], [])).toEqual({
      name: 'FUNDS', domain: ['Grant'], range: [], extra: 1,
    })
  })
})


describe('inheritedTypes: which ontology DEFINES each inherited type (round-2 major 2)', () => {
  // deals extends instruments → government → schema_core; each GET returns the
  // MERGED chain, as the browse API does.
  const person = { name: 'Person', parent_type: 'Thing' }
  const loaded = {
    instruments: { extends: 'government', includes: ['policy_themes'],
      entity_types: [person, { name: 'Gemeente', parent_type: 'AdministrativeArea' },
        { name: 'Subsidie', parent_type: 'Grant' }, { name: 'Thema', parent_type: 'Concept' }] },
    government: { extends: 'schema_core',
      entity_types: [person, { name: 'Gemeente', parent_type: 'AdministrativeArea' }] },
    schema_core: { extends: null, entity_types: [{ name: 'Person', parent_type: 'Thing' }] },
    policy_themes: { extends: null, entity_types: [{ name: 'Thema', parent_type: 'Concept' }] },
  }

  it('walks the whole chain, extends and includes', () => {
    expect(inheritanceClosure(['instruments'], loaded)).toEqual({
      order: ['instruments', 'government', 'policy_themes', 'schema_core'], missing: [] })
    expect(inheritanceClosure(['instruments'], { instruments: loaded.instruments }).missing)
      .toEqual(['government', 'policy_themes'])
  })

  it('labels a type with the ontology that introduced it, however deep', () => {
    const by = Object.fromEntries(inheritedTypes(['instruments'], loaded).map((t) => [t.name, t.definedIn]))
    expect(by).toEqual({ Person: 'schema_core', Gemeente: 'government',
      Subsidie: 'instruments', Thema: 'policy_themes' })
  })

  it('a redefinition is labelled where it is redefined', () => {
    const redefined = { ...loaded,
      government: { ...loaded.government, entity_types: [{ name: 'Person', parent_type: 'Agent' },
        { name: 'Gemeente', parent_type: 'AdministrativeArea' }] },
      instruments: { ...loaded.instruments, entity_types: [{ name: 'Person', parent_type: 'Agent' }] } }
    expect(inheritedTypes(['instruments'], redefined).find((t) => t.name === 'Person'))
      .toMatchObject({ definedIn: 'government', parent_type: 'Agent' })
  })
})

describe('relationship endpoints with BOTH spellings (round-2 minor 2)', () => {
  it('updates every spelling present, so they cannot contradict each other', () => {
    const mixed = { name: 'R', range: ['Nieuw'], source_types: ['A'], target_types: ['Nieuw'] }
    expect(withEndpoints(mixed, ['A'], ['Ander'])).toEqual({
      name: 'R', domain: ['A'], range: ['Ander'], source_types: ['A'], target_types: ['Ander'] })
  })
})
