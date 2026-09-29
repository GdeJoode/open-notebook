/**
 * The Werkbak report (WB.1), shared by the page and the sidebar badges: both
 * read the same query, keyed on the stored "last seen".
 *
 * The report is not cheap (~1.2 s of server time on the live graph, mostly
 * merge candidates), so the sidebar, mounted on every page, reads the cached
 * report instead of refetching per navigation. The Werkbak page (`fresh`)
 * always reads it anew, and the mutations that resolve an item invalidate
 * `ATTENTION_QUERY_KEY`.
 *
 * What arrives from elsewhere (a source gated later, a failed job, an MCP
 * upload) is picked up by a refresh every REFRESH_MS while the tab is
 * visible, paused while it is hidden. 90 s keeps the badges within a minute
 * and a half of the database at about 1 % of one server core per open tab;
 * nothing here needs seconds. Coming back to a tab whose report is older than
 * the stale time (5 min) refetches once on focus.
 */

export const REFRESH_MS = 90 * 1000

import { useSyncExternalStore } from 'react'
import { useQuery } from '@tanstack/react-query'

import { attentionApi } from '@/lib/api/attention'
import { getLastSeen, subscribeLastSeen } from '@/lib/werkbak/last-seen'

export const ATTENTION_QUERY_KEY = ['attention'] as const

export function useLastSeen(): string | null {
  return useSyncExternalStore(subscribeLastSeen, getLastSeen, () => null)
}

export function useAttention({ fresh = false }: { fresh?: boolean } = {}) {
  const since = useLastSeen()
  const query = useQuery({
    queryKey: [...ATTENTION_QUERY_KEY, since],
    queryFn: () => attentionApi.get(since),
    staleTime: 5 * 60 * 1000,
    refetchOnMount: fresh ? 'always' : false,
    refetchOnWindowFocus: true,
    refetchInterval: REFRESH_MS,
    refetchIntervalInBackground: false,
  })
  return { ...query, since }
}
