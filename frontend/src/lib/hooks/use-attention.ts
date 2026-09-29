/**
 * The Werkbak report (WB.1), shared by the page and the sidebar badges: both
 * read the same query, keyed on the stored "last seen".
 */

import { useSyncExternalStore } from 'react'
import { useQuery } from '@tanstack/react-query'

import { attentionApi } from '@/lib/api/attention'
import { getLastSeen, subscribeLastSeen } from '@/lib/werkbak/last-seen'

export const ATTENTION_QUERY_KEY = ['attention'] as const

export function useLastSeen(): string | null {
  return useSyncExternalStore(subscribeLastSeen, getLastSeen, () => null)
}

export function useAttention() {
  const since = useLastSeen()
  const query = useQuery({
    queryKey: [...ATTENTION_QUERY_KEY, since],
    queryFn: () => attentionApi.get(since),
    staleTime: 60 * 1000,
    refetchOnWindowFocus: true,
  })
  return { ...query, since }
}
