/**
 * The Werkbak report (WB.1), shared by the page and the sidebar badges: both
 * read the same query, keyed on the stored "last seen".
 *
 * The report is not cheap (~1.2 s of server time on the live graph, mostly
 * merge candidates). What arrives from elsewhere (a source gated later, a
 * failed job, an MCP upload) is picked up within REFRESH_MS: 90 s keeps the
 * badges within a minute and a half at about 1 % of one server core per open
 * tab; nothing here needs seconds. The mutations that resolve an item
 * invalidate `ATTENTION_QUERY_KEY`, and the Werkbak page (`fresh`) always
 * reads it anew.
 *
 * The sidebar lives in the dashboard layout (WB.2): it stays mounted across
 * navigations, so its interval reads the report every REFRESH_MS as the
 * owner moves between pages, paused while the tab is hidden. The report is
 * stale after the same delay, so a focus (or a remount, e.g. after signing
 * in) reads it once it is, and not before.
 *
 * A report with a group that failed or timed out (`partial`) is read again
 * sooner, backing off over consecutive partial reports (RETRY_BACKOFF_MS:
 * 5 s, 15 s, 45 s, then REFRESH_MS again) and resetting on a complete
 * report. A slow first read after a restart is corrected within seconds, and
 * a group that keeps failing costs a read every 90 s, not every 5 s. The
 * stale time follows the same delay, so a focus reads a partial report once
 * its delay has passed.
 */

import { useSyncExternalStore } from 'react'
import { useQuery, useQueryClient } from '@tanstack/react-query'

import { attentionApi, type AttentionReport } from '@/lib/api/attention'
import { getLastSeen, subscribeLastSeen } from '@/lib/werkbak/last-seen'

export const REFRESH_MS = 90 * 1000
/** Delays after the 1st, 2nd and 3rd partial report in a row; then REFRESH_MS. */
export const RETRY_BACKOFF_MS = [5 * 1000, 15 * 1000, 45 * 1000]

/** The report plus how many partial reports in a row led to it (0 when complete). */
export type AttentionData = AttentionReport & { partialStreak: number }

/** When the report should be read again (and counts as stale). */
export function nextReadMs(report?: AttentionData): number {
  if (!report?.partial) return REFRESH_MS
  return RETRY_BACKOFF_MS[report.partialStreak - 1] ?? REFRESH_MS
}

export const ATTENTION_QUERY_KEY = ['attention'] as const

export function useLastSeen(): string | null {
  return useSyncExternalStore(subscribeLastSeen, getLastSeen, () => null)
}

export function useAttention({ fresh = false }: { fresh?: boolean } = {}) {
  const since = useLastSeen()
  const client = useQueryClient()
  const queryKey = [...ATTENTION_QUERY_KEY, since]
  const query = useQuery({
    queryKey,
    queryFn: async (): Promise<AttentionData> => {
      const report = await attentionApi.get(since)
      const previous = client.getQueryData<AttentionData>(queryKey)
      // A complete report stores 0, so a partial one after it starts over.
      const streak = previous?.partialStreak ?? 0
      return { ...report, partialStreak: report.partial ? streak + 1 : 0 }
    },
    staleTime: (q) => nextReadMs(q.state.data),
    refetchOnMount: fresh ? 'always' : true,
    refetchOnWindowFocus: true,
    refetchInterval: (q) => nextReadMs(q.state.data),
    refetchIntervalInBackground: false,
  })
  return { ...query, since }
}
