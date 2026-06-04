/**
 * Polls GET /api/health/mineru every 30 seconds for the Settings page
 * health chip.
 *
 * Design notes:
 * - The backend endpoint never returns 5xx, so `retry: false` is
 *   appropriate: a non-healthy response is meaningful state, not a
 *   transient error to retry.
 * - `staleTime: 25_000` keeps the data "fresh" until just before the
 *   next refetch, avoiding double-fetches when the component
 *   re-mounts.
 * - React Query pauses `refetchInterval` for hidden tabs by default
 *   (refetchIntervalInBackground is false), so we don't burn battery
 *   on background tabs.
 */

import { useQuery } from '@tanstack/react-query'
import { healthApi } from '@/lib/api/health'
import { QUERY_KEYS } from '@/lib/api/query-client'

export function useMineruHealth() {
  return useQuery({
    queryKey: QUERY_KEYS.mineruHealth,
    queryFn: () => healthApi.mineru(),
    refetchInterval: 30_000,
    staleTime: 25_000,
    retry: false,
  })
}
