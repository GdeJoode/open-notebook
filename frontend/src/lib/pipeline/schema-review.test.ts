import { describe, expect, it } from 'vitest'

import { schemaReviewHref } from './schema-review'

describe('schemaReviewHref', () => {
  it("is the Schema tab of the first notebook, the gate's anchor", () => {
    expect(schemaReviewHref(['notebook:nb1', 'notebook:nb2'])).toBe('/notebooks/notebook%3Anb1/schema')
  })

  it('is The Tray for a source in no notebook', () => {
    expect(schemaReviewHref([])).toBe('/')
    expect(schemaReviewHref(undefined)).toBe('/')
    expect(schemaReviewHref(null)).toBe('/')
  })
})
