/**
 * The Werkbak report (WB.1), shared by the page and the sidebar badges: both
 * read the same query, keyed on the stored "last seen".
 *
 * The report is not cheap (merge candidates alone ~1 s on the live graph), so
 * the sidebar, mounted on every page, reads the cached report and does not
 * refetch on mount or focus. The Werkbak page (`fresh`) always reads it anew.
 * The mutations that resolve an item invalidate `ATTENTION_QUERY_KEY`.
 */

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
    refetchOnWindowFocus: fresh,
  })
  return { ...query, since }
}
