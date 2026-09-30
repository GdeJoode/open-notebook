import { describe, expect, it } from 'vitest'

import {
  EXPECTED,
  STAGES,
  fakeCounts,
  fakeList,
  sourceId,
} from '../../../e2e/track-wb/sources-board.fake'
import { BOARD_CHIPS, activeChipKey, boardHref, chipCount } from './board'

const q = (init: Record<string, string> | [string, string][] = {}) => new URLSearchParams(init)

describe('the sources-board fake answers what the real endpoints answer', () => {
  // `expected` is checked against a real SurrealDB by
  // test_sources_board_db.py::test_the_e2e_fixture_is_the_real_contract.
  it('counts', () => {
    for (const [scope, want] of Object.entries(EXPECTED.counts)) {
      const params = scope === 'all' ? q() : q({ notebook_id: scope })
      expect(fakeCounts(params), scope).toEqual(want)
    }
  })

  it('lists needs-you first, across pages', () => {
    for (const [scope, order] of Object.entries(EXPECTED.attention_order)) {
      const base: [string, string][] = scope === 'all' ? [] : [['notebook_id', scope]]
      const rows = []
      for (let offset = 0; ; offset += 3) {
        const page = fakeList(q([...base, ['sort_by', 'attention'], ['limit', '3'], ['offset', String(offset)]]))
        rows.push(...page)
        if (page.length < 3) break
      }
      expect(rows.map((r) => r.id), scope).toEqual(order.map(sourceId))
    }
  })

  it('filters as the chips ask, and each chip count is its list length', () => {
    for (const scope of [null, 'notebook:alpha', 'notebook:beta']) {
      const base: [string, string][] = scope ? [['notebook_id', scope]] : []
      const counts = fakeCounts(q(base))
      for (const stage of STAGES) {
        expect(fakeList(q([...base, ['stage', stage], ['limit', '100']])).length).toBe(
          counts.stages[stage],
        )
      }
      expect(fakeList(q([...base, ['stalled', 'true'], ['limit', '100']])).length).toBe(counts.stalled)
      expect(fakeList(q([...base, ['limit', '100']])).length).toBe(counts.total)
    }
  })
})

describe('board chips', () => {
  const counts = fakeCounts(q())

  it('reads every chip count from the server counts, none from rows', () => {
    const byKey = Object.fromEntries(BOARD_CHIPS.map((c) => [c.key, chipCount(c, counts)]))
    expect(byKey).toEqual({
      all: 10, failed: 2, awaiting_schema_review: 1, stalled: 2,
      ingested: 1, embedded: 2, extracted: 1, graphed: 1, complete: 2,
    })
    // Every stage has a chip.
    expect(STAGES.every((s) => BOARD_CHIPS.some((c) => c.key === s))).toBe(true)
    // No counts read (or a body that is not the contract): no numbers.
    expect(chipCount(BOARD_CHIPS[0], undefined)).toBeUndefined()
    expect(chipCount(BOARD_CHIPS[0], [] as unknown as typeof counts)).toBeUndefined()
  })

  it('knows which chip a URL is', () => {
    expect(activeChipKey([], false)).toBe('all')
    expect(activeChipKey(['embedded'], false)).toBe('embedded')
    expect(activeChipKey([], true)).toBe('stalled')
    // A combination (a Tray link) is no single chip.
    expect(activeChipKey(['failed', 'embedded'], false)).toBeUndefined()
    expect(activeChipKey(['failed'], true)).toBeUndefined()
  })

  it('links keep the notebook filter', () => {
    expect(boardHref({ kind: 'all' })).toBe('/sources')
    expect(boardHref({ kind: 'stage', stage: 'embedded' }, 'notebook:alpha')).toBe(
      '/sources?stage=embedded&notebook=notebook%3Aalpha',
    )
    expect(boardHref({ kind: 'stalled' })).toBe('/sources?stalled=true')
  })
})
