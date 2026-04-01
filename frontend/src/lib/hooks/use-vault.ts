import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { vaultApi } from '@/lib/api/vault'
import { toast } from 'sonner'

const VAULT_KEYS = {
  pendingImports: ['vault', 'imports', 'pending'] as const,
  pendingExports: ['vault', 'exports', 'pending'] as const,
  dictionary: ['vault', 'dictionary'] as const,
}

export function usePendingImports() {
  return useQuery({
    queryKey: VAULT_KEYS.pendingImports,
    queryFn: () => vaultApi.getPendingImports(),
  })
}

export function usePendingExports() {
  return useQuery({
    queryKey: VAULT_KEYS.pendingExports,
    queryFn: () => vaultApi.getPendingExports(),
  })
}

export function useVaultDictionary() {
  return useQuery({
    queryKey: VAULT_KEYS.dictionary,
    queryFn: () => vaultApi.getDictionary(),
  })
}

export function useVaultSync() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: () => vaultApi.sync(),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: VAULT_KEYS.pendingImports })
      const total = result.new + result.changed
      if (total > 0) {
        toast.success(`Vault synced: ${result.new} new, ${result.changed} changed`)
      } else {
        toast.info('Vault is up to date — no new changes')
      }
    },
    onError: () => {
      toast.error('Failed to sync vault')
    },
  })
}

export function useApplyImports() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ acceptedIds, rejectedIds }: { acceptedIds: string[]; rejectedIds: string[] }) =>
      vaultApi.applyImports(acceptedIds, rejectedIds),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: VAULT_KEYS.pendingImports })
      queryClient.invalidateQueries({ queryKey: VAULT_KEYS.dictionary })
      toast.success(`${result.accepted} imported, ${result.rejected} rejected`)
    },
    onError: () => {
      toast.error('Failed to apply imports')
    },
  })
}

export function useWriteExports() {
  const queryClient = useQueryClient()

  return useMutation({
    mutationFn: ({ acceptedIds, rejectedIds }: { acceptedIds: string[]; rejectedIds: string[] }) =>
      vaultApi.writeExports(acceptedIds, rejectedIds),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: VAULT_KEYS.pendingExports })
      toast.success(`${result.written} written to vault, ${result.rejected} rejected`)
    },
    onError: () => {
      toast.error('Failed to write to vault')
    },
  })
}
