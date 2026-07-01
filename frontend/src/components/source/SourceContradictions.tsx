'use client'

import { useEffect, useState } from 'react'
import { AlertTriangle } from 'lucide-react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'

/**
 * DEFERRED (Track UX.6) — contradiction verdicts surface.
 *
 * There is NO read API for contradiction verdicts yet. Track Z ships a trigger
 * endpoint (`POST /sources/{id}/judge-contradictions`) but no GET to read the
 * resulting verdicts back. This panel is therefore a deliberately data-absent
 * SAFE stub:
 *
 *   - it renders NOTHING by default (no fetcher wired ⇒ no network call, no
 *     404 noise, no verdicts to show);
 *   - if/when a verdicts read API exists, a `fetchVerdicts` fetcher can be
 *     injected (also the test seam) and the panel lazily loads on mount;
 *   - any fetch error (absent endpoint, network failure) is swallowed — the
 *     panel degrades to a quiet "No contradictions detected" empty state and
 *     NEVER throws on render nor leaks an unhandled promise rejection.
 *
 * When a real GET lands, replace the injected fetcher with the API call and
 * drop the `fetchVerdicts` prop default.
 */

export interface ContradictionVerdict {
  id: string
  /** The claim under review. */
  claim?: string
  /** The judged verdict, e.g. "contradicted" / "supported" / "unclear". */
  verdict?: string
  /** Optional human-readable rationale. */
  rationale?: string
}

/** Lazy loader for verdicts. Absent today; injected by a future API or a test. */
export type VerdictFetcher = (sourceId: string) => Promise<ContradictionVerdict[]>

interface SourceContradictionsProps {
  sourceId: string
  /**
   * Optional verdicts loader. When omitted (the current state — no read API),
   * the panel renders nothing and takes no backend dependency.
   */
  fetchVerdicts?: VerdictFetcher
}

export function SourceContradictions({
  sourceId,
  fetchVerdicts,
}: SourceContradictionsProps) {
  const [verdicts, setVerdicts] = useState<ContradictionVerdict[] | null>(null)
  const [attempted, setAttempted] = useState(false)

  useEffect(() => {
    if (!fetchVerdicts) return
    let cancelled = false
    // Swallow every failure: an absent/failing endpoint must degrade to the
    // empty state, never surface as an unhandled rejection or a thrown render.
    fetchVerdicts(sourceId)
      .then((data) => {
        if (!cancelled) setVerdicts(Array.isArray(data) ? data : [])
      })
      .catch(() => {
        if (!cancelled) setVerdicts([])
      })
      .finally(() => {
        if (!cancelled) setAttempted(true)
      })
    return () => {
      cancelled = true
    }
  }, [sourceId, fetchVerdicts])

  // No loader wired (today's reality) ⇒ render nothing at all.
  if (!fetchVerdicts) return null

  // Loader wired but produced no verdicts ⇒ quiet empty state once it settled.
  if (!verdicts || verdicts.length === 0) {
    if (!attempted) return null
    return (
      <Card data-testid="source-contradictions">
        <CardContent className="py-6 text-center text-sm text-muted-foreground">
          No contradictions detected
        </CardContent>
      </Card>
    )
  }

  return (
    <Card data-testid="source-contradictions">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <AlertTriangle className="h-4 w-4 text-amber-500" aria-hidden="true" />
          Contradictions
          <Badge variant="secondary">{verdicts.length}</Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {verdicts.map((v) => (
          <div key={v.id} className="rounded-lg border bg-background p-3">
            <div className="flex items-center justify-between gap-2">
              {v.claim && <p className="text-sm font-medium">{v.claim}</p>}
              {v.verdict && (
                <Badge variant="outline" className="text-xs uppercase">
                  {v.verdict}
                </Badge>
              )}
            </div>
            {v.rationale && (
              <p className="mt-1 text-xs text-muted-foreground">{v.rationale}</p>
            )}
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
