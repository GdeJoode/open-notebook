/**
 * OE.3: the drag-and-drop tree's rules. The server's gate validates what a new
 * parent does to typing; these decide which drops the editor offers at all.
 */
import { describe, expect, it } from 'vitest'

import {
  ROOT_ZONE,
  buildTypeTree,
  canDrop,
  descendantsOf,
  dropEdit,
  reparented,
  zoneId,
} from '../ontology-tree'
import type { StoredEntityType } from '@/lib/api/ontologies'

const own: Record<string, StoredEntityType> = {
  Deal: { name: 'Deal', parent_type: 'GovernmentService' },
  RegioDeal: { name: 'RegioDeal', parent_type: 'Deal', graph_relevance: 'active' },
  Project: { name: 'Project', parent_type: 'GovernmentService' },
  Losstaand: { name: 'Losstaand' },
}
const inherited = [
  { name: 'Person', parent_type: 'Thing' },
  { name: 'Thing', parent_type: null },
  { name: 'Deal', parent_type: 'CreativeWork' }, // redefined by `own`
]

describe('buildTypeTree', () => {
  const tree = buildTypeTree(own, inherited)

  it('marks own types, inherited types and undefined parents', () => {
    expect(tree.nodes.get('regiodeal')).toMatchObject({ own: true, placeholder: false })
    expect(tree.nodes.get('person')).toMatchObject({ own: false, placeholder: false })
    expect(tree.nodes.get('governmentservice')).toMatchObject({ own: false, placeholder: true })
  })

  it('an own type redefining an inherited one wins', () => {
    expect(tree.nodes.get('deal')).toMatchObject({ own: true, parent: 'GovernmentService' })
    expect(tree.nodes.has('creativework')).toBe(false)
  })

  it('hangs children under their parent and the rest at the root', () => {
    expect(tree.roots).toEqual(['GovernmentService', 'Losstaand', 'Thing'])
    expect(tree.children.get('governmentservice')).toEqual(['Deal', 'Project'])
    expect(tree.children.get('deal')).toEqual(['RegioDeal'])
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
