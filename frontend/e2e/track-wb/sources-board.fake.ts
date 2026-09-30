/**
 * WB.3 — a test double for `GET /api/sources/` and `GET /api/sources/counts`,
 * serving `sources-board.fixture.json`.
 *
 * It follows the server's contract: `?stage=` (repeatable) and `?stalled=true`
 * filter (both apply when both are given), `?notebook_id=` scopes the list and
 * the counts, `?sort_by=attention` lists the "needs you" rows (failed, awaiting
 * schema review, stopped) first and each part by `updated` newest first, and
 * `limit`/`offset` page. The counts are counted over the same rows. The
 * fixture's `expected` block is what the real endpoints answer for these rows
 * (`test_sources_board_db.py`); `sources-board.fake.test.ts` checks this fake
 * gives the same. Consumers: `sources-board.spec.ts` and the Sources page's
 * vitest.
 */

import fixture from './sources-board.fixture.json'

type Stage =
  | 'ingested'
  | 'embedded'
  | 'extracted'
  | 'awaiting_schema_review'
  | 'graphed'
  | 'complete'
  | 'failed'

export const STAGES: Stage[] = [
  'ingested', 'embedded', 'extracted', 'awaiting_schema_review', 'graphed', 'complete', 'failed',
]
const NEEDS_YOU_STAGES: string[] = ['failed', 'awaiting_schema_review']

export interface FakeRow {
  id: string
  title: string
  topics: string[]
  asset: null
  embedded: boolean
  embedded_chunks: number
  insights_count: number
  created: string
  updated: string
  command_id: null
  status: null
  processing_info: null
  processing_stage: Stage
  stalled: boolean
  needs_you: boolean
}

export const sourceId = (key: string) => `source:${key}`

const NOW = Date.parse(fixture.generated_at)

function toRow(s: (typeof fixture.sources)[number]): FakeRow & { notebook: string | null } {
  const at = new Date(NOW - s.age_minutes * 60_000).toISOString()
  const stage = s.stage as Stage
  return {
    id: sourceId(s.key),
    title: s.title,
    topics: [],
    asset: null,
    embedded: false,
    embedded_chunks: 0,
    insights_count: 0,
    created: at,
    updated: at,
    command_id: null,
    status: null,
    processing_info: null,
    processing_stage: stage,
    stalled: s.stalled,
    needs_you: s.stalled || NEEDS_YOU_STAGES.includes(stage),
    notebook: s.notebook,
  }
}

const ROWS = fixture.sources.map(toRow)

function scoped(notebookId: string | null) {
  return notebookId ? ROWS.filter((r) => r.notebook === notebookId) : ROWS
}

const strip = ({ notebook: _notebook, ...row }: FakeRow & { notebook: string | null }): FakeRow =>
  row

/** `GET /api/sources/?…` for these query parameters. */
export function fakeList(params: URLSearchParams): FakeRow[] {
  const stages = params.getAll('stage')
  const stalled = params.get('stalled') === 'true'
  const sortBy = params.get('sort_by') ?? 'updated'
  const order = params.get('sort_order') === 'asc' ? 1 : -1
  const limit = Number(params.get('limit') ?? 50)
  const offset = Number(params.get('offset') ?? 0)
  const rows = scoped(params.get('notebook_id'))
    .filter((r) => stages.length === 0 || stages.includes(r.processing_stage))
    .filter((r) => !stalled || r.stalled)
  const field = sortBy === 'created' ? 'created' : 'updated'
  rows.sort((a, b) => {
    if (sortBy === 'attention' && a.needs_you !== b.needs_you) return a.needs_you ? -1 : 1
    return a[field] < b[field] ? -order : a[field] > b[field] ? order : 0
  })
  return rows.slice(offset, offset + limit).map(strip)
}

/** `GET /api/sources/counts?…`. */
export function fakeCounts(params: URLSearchParams) {
  const rows = scoped(params.get('notebook_id'))
  const stages = Object.fromEntries(
    STAGES.map((s) => [s, rows.filter((r) => r.processing_stage === s).length]),
  ) as Record<Stage, number>
  return {
    total: rows.length,
    stages,
    stalled: rows.filter((r) => r.stalled).length,
    needs_you: rows.filter((r) => r.needs_you).length,
  }
}

export const NOTEBOOKS = fixture.notebooks
export const EXPECTED = fixture.expected
