/**
 * Typed client for the /api/entity-resolution endpoints (Track K.6).
 *
 * Surfaces the K.3 retroactive-merge plan/apply, the K.5 candidate review
 * queue, and the force-merge/force-split overlay CRUD. Mirrors
 * `model-routing.ts`: thin wrappers around `apiClient` returning the response
 * body; error handling is left to the calling TanStack Query hook.
 */

import apiClient from './client'

/** One deterministic-collision merge cluster from `POST /plan` (K.3). */
export interface MergeCluster {
  new_canonical: string
  entity_type: string
  winner_id: string
  loser_ids: string[]
  member_surface_forms: string[]
  total_source_docs: number
  size: number
}

export interface MergePlan {
  scope: string
  total_active_entities: number
  cluster_count: number
  clusters: MergeCluster[]
}

/**
 * A reviewed cluster echoed back to `POST /apply`. The winner is editable in
 * the UI, so this carries only the identity the apply path re-loads.
 */
export interface ApplyClusterInput {
  new_canonical: string
  entity_type: string
  winner_id: string
  loser_ids: string[]
  member_surface_forms?: string[]
}

export interface ApplyResult {
  new_canonical: string
  entity_type: string
  winner_id: string
  merged_loser_ids: string[]
  relations_repointed: number
  aliases_created: number
  skipped: boolean
}

export interface ApplyResponse {
  applied: number
  skipped: number
  results: ApplyResult[]
}

/** One fuzzy/embedding merge proposal from `GET /candidates` (K.5). */
export interface MergeCandidate {
  id_a: string
  id_b: string
  name_a: string
  name_b: string
  entity_type: string
  /** `id_b`'s type when it differs from `id_a`'s, else "". Non-empty only for
   *  the `fold_equal_cross_type` method, where the two names are byte-identical
   *  and the type is the only visible difference. */
  entity_type_b?: string
  score: number
  /** 'auto_merge' | 'review' — the review band is never auto-applied. */
  band: string
  method: string
  winner_id: string
  loser_id: string
}

export interface CandidatesResponse {
  scope: string
  total_active_entities: number
  auto_merge_count: number
  review_count: number
  auto_merge: MergeCandidate[]
  review: MergeCandidate[]
}

/** A force-merge/force-split overlay rule. `kind` is 'merge' or 'split'. */
export interface OverlayCreateInput {
  kind: 'merge' | 'split'
  name_a: string
  name_b: string
  entity_type?: string
  scope?: 'global' | 'notebook'
  notebook_id?: string | null
}

export interface OverlayCreateResponse {
  id: string
}

export const entityResolutionApi = {
  /** Dry-run merge plan (no writes). Optionally scoped to one notebook. */
  plan: async (notebookId?: string): Promise<MergePlan> => {
    const response = await apiClient.post<MergePlan>('/entity-resolution/plan', {
      notebook_id: notebookId ?? null,
    })
    return response.data
  },

  /** Apply an explicit, reviewed list of clusters (mutating, destructive). */
  apply: async (clusters: ApplyClusterInput[]): Promise<ApplyResponse> => {
    const response = await apiClient.post<ApplyResponse>(
      '/entity-resolution/apply',
      { clusters },
    )
    return response.data
  },

  /** Fuzzy/embedding review queue (read-only). */
  candidates: async (notebookId?: string): Promise<CandidatesResponse> => {
    const response = await apiClient.get<CandidatesResponse>(
      '/entity-resolution/candidates',
      { params: notebookId ? { notebook_id: notebookId } : undefined },
    )
    return response.data
  },

  /** Add a force-merge / force-split overlay rule. */
  addOverlay: async (
    rule: OverlayCreateInput,
  ): Promise<OverlayCreateResponse> => {
    const response = await apiClient.post<OverlayCreateResponse>(
      '/entity-resolution/overlay',
      rule,
    )
    return response.data
  },

  /** Remove an overlay rule by id. */
  removeOverlay: async (id: string): Promise<void> => {
    await apiClient.delete(`/entity-resolution/overlay/${id}`)
  },
}
