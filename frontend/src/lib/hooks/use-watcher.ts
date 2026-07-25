import { useQuery } from '@tanstack/react-query'
import { watcherApi } from '@/lib/api/watcher'

export function useWatcherStatus() {
  return useQuery({
    queryKey: ['watcher', 'status'],
    queryFn: () => watcherApi.getStatus(),
    staleTime: 15_000,
    retry: false,
  })
}
