import { describe, it, expect } from 'vitest'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'
import path from 'node:path'

/**
 * Guard: the stale "regenerate the document graph manually" guidance was removed
 * in Track UX.4 (GRAPH is now automatic once a source reaches the `graphed`
 * stage). This asserts the removed strings do not creep back into the empty
 * state of DocumentGraphView.
 */
describe('DocumentGraphView stale regenerate guidance removed (UX.4 AC4)', () => {
  const here = path.dirname(fileURLToPath(import.meta.url))
  const source = readFileSync(path.join(here, '..', 'DocumentGraphView.tsx'), 'utf8')

  it('no longer references the manual regenerate endpoint', () => {
    expect(source).not.toContain('document-graph/regenerate')
  })

  it('no longer tells users to "Regenerate the document graph"', () => {
    expect(source).not.toContain('Regenerate the document graph')
  })

  it('mentions the automatic graphed-stage behaviour instead', () => {
    expect(source).toContain('graphed')
  })
})
