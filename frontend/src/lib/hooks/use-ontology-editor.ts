/**
 * OE.3: React Query hooks for the ontology editor (the OE.2 API).
 *
 * Every edit returns the new draft; it is written into the cache and the
 * validation findings are re-fetched, so the editor always shows what the
 * draft would do. A publish or rollback changes the ACTIVE ontologies, so the
 * browse queries are invalidated too.
 */

'use client'

import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'

import {
  ontologyEditApi,
  type OntologyDraft,
} from '@/lib/api/ontologies'
import { QUERY_KEYS } from '@/lib/api/query-client'
import { useToast } from '@/lib/hooks/use-toast'

export const ONTOLOGY_EDITOR_KEYS = {
  draft: (name: string) => ['ontology-editor', name, 'draft'] as const,
  findings: (name: string) => ['ontology-editor', name, 'findings'] as const,
  versions: (name: string) => ['ontology-editor', name, 'versions'] as const,
}

/** The server's reason, whatever shape it has (a string, or `{message, findings}`). */
export function errorDetail(err: unknown): string {
  const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (detail && typeof detail === 'object' && 'message' in detail) {
    return String((detail as { message: unknown }).message)
  }
  return err instanceof Error ? err.message : 'Request failed'
}

export function useOntologyDraft(name: string) {
  return useQuery({
    queryKey: ONTOLOGY_EDITOR_KEYS.draft(name),
    queryFn: () => ontologyEditApi.getDraft(name),
    enabled: !!name,
  })
}

export function useDraftFindings(name: string) {
  return useQuery({
    queryKey: ONTOLOGY_EDITOR_KEYS.findings(name),
    queryFn: () => ontologyEditApi.validate(name),
    enabled: !!name,
  })
}

export function useOntologyVersions(name: string) {
  return useQuery({
    queryKey: ONTOLOGY_EDITOR_KEYS.versions(name),
    queryFn: () => ontologyEditApi.versions(name),
    enabled: !!name,
  })
}

/** Any draft edit: `edit` calls the API and resolves to the new draft. */
export function useDraftEdit(name: string) {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  return useMutation({
    mutationFn: (edit: () => Promise<OntologyDraft>) => edit(),
    onSuccess: (draft) => {
      queryClient.setQueryData(ONTOLOGY_EDITOR_KEYS.draft(name), draft)
      queryClient.invalidateQueries({ queryKey: ONTOLOGY_EDITOR_KEYS.findings(name) })
    },
    onError: (err) =>
      toast({ title: 'Edit refused', description: errorDetail(err), variant: 'destructive' }),
  })
}

export function useDiscardDraft(name: string) {
  const queryClient = useQueryClient()
  return useMutation({
    mutationFn: () => ontologyEditApi.discardDraft(name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ontology-editor', name] })
    },
  })
}

export function useDraftPreview(name: string) {
  const { toast } = useToast()
  return useMutation({
    mutationFn: (includeDetection: boolean) => ontologyEditApi.preview(name, includeDetection),
    onError: (err) =>
      toast({ title: 'Preview failed', description: errorDetail(err), variant: 'destructive' }),
  })
}

function useActiveChanged(name: string) {
  const queryClient = useQueryClient()
  return () => {
    queryClient.invalidateQueries({ queryKey: ['ontology-editor', name] })
    queryClient.invalidateQueries({ queryKey: QUERY_KEYS.ontologies })
  }
}

export function usePublishDraft(name: string) {
  const changed = useActiveChanged(name)
  const { toast } = useToast()
  return useMutation({
    mutationFn: (note: string | null) => ontologyEditApi.publish(name, note),
    onSuccess: (result) => {
      changed()
      toast({ title: 'Published', description: `${name} is now version ${result.version}` })
    },
    onError: (err) =>
      toast({ title: 'Publish refused', description: errorDetail(err), variant: 'destructive' }),
  })
}

export function useRollback(name: string) {
  const changed = useActiveChanged(name)
  const { toast } = useToast()
  return useMutation({
    mutationFn: (version: string) => ontologyEditApi.rollback(name, version),
    onSuccess: (_r, version) => {
      changed()
      toast({ title: 'Rolled back', description: `${name} is now version ${version}` })
    },
    onError: (err) =>
      toast({ title: 'Rollback refused', description: errorDetail(err), variant: 'destructive' }),
  })
}
