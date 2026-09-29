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
  retypeApi,
  type Finding,
  type OntologyDraft,
} from '@/lib/api/ontologies'
import { QUERY_KEYS } from '@/lib/api/query-client'
import { useToast } from '@/lib/hooks/use-toast'
import { ATTENTION_QUERY_KEY } from '@/lib/hooks/use-attention'

export const ONTOLOGY_EDITOR_KEYS = {
  draft: (name: string) => ['ontology-editor', name, 'draft'] as const,
  findings: (name: string) => ['ontology-editor', name, 'findings'] as const,
  versions: (name: string) => ['ontology-editor', name, 'versions'] as const,
  /** Under the editor's prefix: a publish or rollback re-runs the dry run. */
  retype: ['ontology-editor', 'retype-dry-run'] as const,
}

/** The server's reason, whatever shape it has (a string, or `{message,
 *  findings}` on a 422 — the blocking findings are named, not dropped). */
export function errorDetail(err: unknown): string {
  const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (detail && typeof detail === 'object' && 'message' in detail) {
    const { message, findings } = detail as { message: unknown; findings?: Finding[] }
    const blocking = (findings ?? []).filter((f) => f.severity === 'block')
    return blocking.length
      ? `${String(message)}: ${blocking.map((f) => `${f.rule} (${f.subject ?? '—'})`).join(', ')}`
      : String(message)
  }
  return err instanceof Error ? err.message : 'Request failed'
}

/** Everything the editor holds for one ontology, or (no name) for all of them:
 *  a publish of a PARENT changes the findings of the ontologies that extend it. */
const editorKey = (name?: string) => (name ? ['ontology-editor', name] : ['ontology-editor'])

export function useOntologyDraft(name: string) {
  return useQuery({
    queryKey: ONTOLOGY_EDITOR_KEYS.draft(name),
    queryFn: () => ontologyEditApi.getDraft(name),
    enabled: !!name,
    staleTime: 0,
  })
}

export function useDraftFindings(name: string) {
  return useQuery({
    queryKey: ONTOLOGY_EDITOR_KEYS.findings(name),
    queryFn: () => ontologyEditApi.validate(name),
    enabled: !!name,
    staleTime: 0,
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
    onError: (err) => {
      queryClient.invalidateQueries({ queryKey: editorKey(name) })
      toast({ title: 'Edit refused', description: errorDetail(err), variant: 'destructive' })
    },
  })
}

export function useDiscardDraft(name: string) {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  return useMutation({
    mutationFn: () => ontologyEditApi.discardDraft(name),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: editorKey(name) })
    },
    onError: (err) =>
      toast({ title: 'Discard failed', description: errorDetail(err), variant: 'destructive' }),
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

function useActiveChanged() {
  const queryClient = useQueryClient()
  return () => {
    queryClient.invalidateQueries({ queryKey: editorKey() })
    queryClient.invalidateQueries({ queryKey: QUERY_KEYS.ontologies })
    // A publish or rollback can change what the retype would do.
    queryClient.invalidateQueries({ queryKey: ATTENTION_QUERY_KEY })
  }
}

export function usePublishDraft(name: string) {
  const changed = useActiveChanged()
  const { toast } = useToast()
  return useMutation({
    mutationFn: (note: string | null) => ontologyEditApi.publish(name, note),
    onSuccess: (result) => {
      changed()
      toast({ title: 'Published', description: `${name} is now version ${result.version}` })
    },
    onError: (err) => {
      changed() // a 409 or 422 means the draft or the active set moved: show it
      toast({ title: 'Publish refused', description: errorDetail(err), variant: 'destructive' })
    },
  })
}

export function useRollback(name: string) {
  const changed = useActiveChanged()
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

/** OE.4: the retype dry run. Not refetched on focus: the plan on screen is the
 *  plan an apply confirms, and it only changes when asked for or after a
 *  publish, rollback or apply. */
export function useRetypeDryRun() {
  return useQuery({
    queryKey: ONTOLOGY_EDITOR_KEYS.retype,
    queryFn: () => retypeApi.dryRun(),
    staleTime: 0,
    refetchOnWindowFocus: false,
  })
}

/** Applies the plan with this fingerprint; a 409 means the plan changed since
 *  (nothing was written) and the new dry run is fetched. */
export function useApplyRetype() {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  const refresh = () => queryClient.invalidateQueries({ queryKey: ONTOLOGY_EDITOR_KEYS.retype })
  return useMutation({
    mutationFn: (fingerprint: string) => retypeApi.apply(fingerprint),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ATTENTION_QUERY_KEY })
      refresh()
      toast({ title: 'Retyped', description: `${result.written} entities written` })
    },
    onError: (err) => {
      refresh()
      toast({ title: 'Retype refused', description: errorDetail(err), variant: 'destructive' })
    },
  })
}

export function useUndoRetype() {
  const queryClient = useQueryClient()
  const { toast } = useToast()
  return useMutation({
    mutationFn: (rollbackId: string) => retypeApi.undo(rollbackId),
    onSuccess: (result) => {
      queryClient.invalidateQueries({ queryKey: ATTENTION_QUERY_KEY })
      queryClient.invalidateQueries({ queryKey: ONTOLOGY_EDITOR_KEYS.retype })
      toast({ title: 'Retype undone', description: `${result.written} entities restored` })
    },
    onError: (err) =>
      toast({ title: 'Undo failed', description: errorDetail(err), variant: 'destructive' }),
  })
}
