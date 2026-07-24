import { describe, it, expect } from 'vitest'

import { parseOkfExportReport } from '../use-okf-export'
import type { OkfExportReport } from '@/lib/types/exports'

/**
 * OKF.5 — the header-JSON parser is the one seam worth pinning in a node-only
 * context. The DOM download side-effect + React Query wiring are exercised by
 * the dialog component test (jsdom) + the Track-D e2e harness, not here.
 *
 * The report rides in the ``X-OKF-Export-Report`` header, so a malformed or
 * absent value must degrade to ``null`` (the zip is the SSOT, the ledger is a
 * nicety) rather than throw and swallow a successful download.
 */
describe('parseOkfExportReport', () => {
  const validReport: OkfExportReport = {
    entities_written: 12,
    relations_written: 7,
    files_written: 14,
    bytes_written: 4096,
    duration_ms: 42,
    filter_applied: {
      min_connections: 5,
      min_confidence: 0.9,
      include_orphans: false,
      include_archived: false,
    },
    metadata: {
      notes_written: 3,
      sources_written: 2,
      dropped_relations: 1,
      okf_version: '0.1',
      omitted_fields: {
        'entity.embedding': 'Vector embedding has no OKF representation.',
        verdict_contradiction_edges: 'Verdict edges have no OKF representation.',
      },
    },
  }

  it('parses a valid compact-JSON header into the report shape', () => {
    const header = JSON.stringify(validReport)
    const parsed = parseOkfExportReport(header)

    expect(parsed).not.toBeNull()
    expect(parsed?.entities_written).toBe(12)
    expect(parsed?.relations_written).toBe(7)
    // The omitted-field ledger is the load-bearing OKF-specific payload.
    expect(parsed?.metadata?.omitted_fields).toEqual(
      validReport.metadata?.omitted_fields,
    )
    expect(Object.keys(parsed?.metadata?.omitted_fields ?? {})).toHaveLength(2)
  })

  it('returns null for a missing header (undefined / null / empty)', () => {
    expect(parseOkfExportReport(undefined)).toBeNull()
    expect(parseOkfExportReport(null)).toBeNull()
    expect(parseOkfExportReport('')).toBeNull()
    expect(parseOkfExportReport('   ')).toBeNull()
  })

  it('returns null (does not throw) for a malformed JSON header', () => {
    expect(() => parseOkfExportReport('{ not: valid json')).not.toThrow()
    expect(parseOkfExportReport('{ not: valid json')).toBeNull()
  })
})
