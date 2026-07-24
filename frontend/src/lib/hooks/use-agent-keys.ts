import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { agentKeysApi, MintAgentKeyInput } from '@/lib/api/agent-keys'
import { toast } from 'sonner'

const AGENT_KEY_KEYS = {
  list: ['agent-keys'] as const,
  audit: (id: string) => ['agent-keys', 'audit', id] as const,
}

export function useAgentKeys() {
  return useQuery({
    queryKey: AGENT_KEY_KEYS.list,
    queryFn: () => agentKeysApi.list(),
    retry: false,
  })
}

export function useMintAgentKey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (input: MintAgentKeyInput) => agentKeysApi.mint(input),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AGENT_KEY_KEYS.list })
      toast.success('API key created')
    },
    onError: (error: Error) => {
      toast.error(`Could not create key: ${error.message}`)
    },
  })
}

export function useRevokeAgentKey() {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => agentKeysApi.revoke(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: AGENT_KEY_KEYS.list })
      toast.success('API key revoked')
    },
    onError: (error: Error) => {
      toast.error(`Could not revoke key: ${error.message}`)
    },
  })
}

/** A key's audit trail. Disabled (no fetch) while `id` is null — e.g. drawer closed. */
export function useKeyAuditLog(id: string | null) {
  return useQuery({
    queryKey: id ? AGENT_KEY_KEYS.audit(id) : ['agent-keys', 'audit', 'none'],
    queryFn: () => agentKeysApi.auditLog(id as string),
    enabled: !!id,
    retry: false,
  })
}
