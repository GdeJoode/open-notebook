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
 * The sidebar is rendered inside each page (AppShell), so every navigation
 * remounts it and restarts the refresh interval. The report is therefore
 * stale after REFRESH_MS, and a remount or a focus reads it once it is: an
 * owner who keeps navigating still gets a read about every REFRESH_MS, while
 * navigating within that window reads nothing. The interval covers a page
 * left open; it is paused while the tab is hidden.
 */

import { useSyncExternalStore } from 'react'
import { useQuery } from '@tanstack/react-query'

import { attentionApi } from '@/lib/api/attention'
import { getLastSeen, subscribeLastSeen } from '@/lib/werkbak/last-seen'

export const REFRESH_MS = 90 * 1000

export const ATTENTION_QUERY_KEY = ['attention'] as const

export function useLastSeen(): string | null {
  return useSyncExternalStore(subscribeLastSeen, getLastSeen, () => null)
}

export function useAttention({ fresh = false }: { fresh?: boolean } = {}) {
  const since = useLastSeen()
  const query = useQuery({
    queryKey: [...ATTENTION_QUERY_KEY, since],
    queryFn: () => attentionApi.get(since),
    staleTime: REFRESH_MS,
    refetchOnMount: fresh ? 'always' : true,
    refetchOnWindowFocus: true,
    refetchInterval: REFRESH_MS,
    refetchIntervalInBackground: false,
  })
  return { ...query, since }
}
