import { describe, expect, it } from 'vitest'

import type { Entity } from '@/lib/api/knowledge-graph'
import {
  sourceEntitiesView,
  toSourceEntityRow,
} from '@/components/source/source-entities'

function entity(overrides: Partial<Entity> = {}): Entity {
  return {
    id: 'entity:1',
    name: 'surface name',
    entity_type: 'concept',
    weight: 1,
    ...overrides,
  }
}

describe('toSourceEntityRow', () => {
  it('prefers canonical_name over the raw surface name', () => {
    const row = toSourceEntityRow(
      entity({ name: 'BZK', canonical_name: 'Ministerie van BZK' }),
    )
    expect(row.name).toBe('Ministerie van BZK')
  })

  it('falls back to surface name when canonical_name is absent', () => {
    const row = toSourceEntityRow(entity({ name: 'BZK', canonical_name: undefined }))
    expect(row.name).toBe('BZK')
  })

  it('prefers primary_type over the raw entity_type label', () => {
    const row = toSourceEntityRow(
      entity({ entity_type: 'org', primary_type: 'Organization' }),
    )
    expect(row.type).toBe('Organization')
  })

  it('falls back to entity_type when primary_type is absent', () => {
    const row = toSourceEntityRow(entity({ entity_type: 'org' }))
    expect(row.type).toBe('org')
  })

  it('carries status + manual_override through', () => {
    const row = toSourceEntityRow(
      entity({ status: 'reference', manual_override: true }),
    )
    expect(row.status).toBe('reference')
    expect(row.manualOverride).toBe(true)
  })

  it('keeps degree/docCount as numbers (including 0) and nulls when absent', () => {
    const present = toSourceEntityRow(
      entity({ structural_degree: 0, doc_count: 3 }),
    )
    expect(present.degree).toBe(0)
    expect(present.docCount).toBe(3)

    const absent = toSourceEntityRow(entity())
    expect(absent.degree).toBeNull()
    expect(absent.docCount).toBeNull()
  })
})

describe('sourceEntitiesView', () => {
  it('loading takes precedence over everything', () => {
    expect(
      sourceEntitiesView({ isLoading: true, isError: true, items: [entity()] }),
    ).toBe('loading')
  })

  it('error renders before data', () => {
    expect(
      sourceEntitiesView({ isLoading: false, isError: true, items: [entity()] }),
    ).toBe('error')
  })

  it('absent or empty items is the empty state', () => {
    expect(
      sourceEntitiesView({ isLoading: false, isError: false, items: undefined }),
    ).toBe('empty')
    expect(
      sourceEntitiesView({ isLoading: false, isError: false, items: [] }),
    ).toBe('empty')
  })

  it('non-empty items renders the list', () => {
    expect(
      sourceEntitiesView({
        isLoading: false,
        isError: false,
        items: [entity()],
      }),
    ).toBe('list')
  })
})
