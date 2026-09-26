/**
 * Typed client for the /api/triage endpoints (Track Q.5).
 *
 * Surfaces the Q.4 review queue (`GET /queue`), the operator decision
 * (`POST /queue/{id}/decision` — pins status + sets `manual_override`), and the
 * read-time backbone warnings (`GET /backbone-warnings`). Mirrors
 * `entity-resolution.ts`: thin wrappers around `apiClient` returning the
 * response body; error handling is left to the calling TanStack Query hook.
 */

import apiClient from './client'

/** Triage status values an operator can pin / filter by. */
export type TriageStatus = 'active' | 'reference' | 'archived'

/**
 * One triage-queue row (the B5 read-model). Carries exactly the fields the
 * queue table renders plus the queue's own `decision` / `batch_id`.
 */
export interface QueueItem {
  /** The queue row's record id (the decision target). */
  id: string
  /** The entity record id this row is about. */
  entity: string
  name: string
  type: string
  structural_degree: number
  doc_count: number
  /** The queue-flag reason(s) — `"; "`-joined when an entity tripped several. */
  reason: string
  /** `'open'` while awaiting an operator; any other value closes the row. */
  decision: string
  batch_id: string
  /**
   * `status` (Q.4 — decide with `decide`) or `tooi` (D2 — decide with
   * `decideTooi`). One queue, one row per (entity, kind): a TOOI row never
   * shares a reason or a decision with the entity's status row (migration 83).
   */
  kind?: QueueKind
  /** A TOOI row's authority records, one of which a curator may link. */
  candidates?: TooiCandidate[]
}

export type QueueKind = 'status' | 'tooi'

/** One TOOI record offered for a `tooi` row (D2). */
export interface TooiCandidate {
  uri: string
  label: string
  /** The organisation's end date per TOOI; "" when it still exists. */
  valid_until?: string
  confidence?: number
  matched_on?: string
  current_names?: string[]
}

/** A curator's answer to a `tooi` row: link one candidate, or dismiss. */
export type TooiDecisionInput =
  | { action: 'link'; uri: string }
  | { action: 'dismiss' }

export interface TooiDecisionResponse {
  queue_id: string
  entity: string
  decision: string
  uri?: string | null
}

export interface QueueResponse {
  count: number
  items: QueueItem[]
}

/**
 * An operator decision on a queued entity. `status` is the status to pin
 * (`active` to promote, `reference` to keep out of the active KG); `decision`
 * is an optional free label recorded on the row (defaults to the status). The
 * decision ALWAYS sets `manual_override` server-side (override-wins).
 */
export interface DecisionInput {
  status: Extract<TriageStatus, 'active' | 'reference'>
  decision?: string
}

export interface DecisionResponse {
  queue_id: string
  entity: string
  status: string
  manual_override: boolean
  decision: string
}

/** Operator pin from the KG detail panel (manual_override toggle). */
export interface EntityPinInput {
  manual_override: boolean
  status?: Extract<TriageStatus, 'active' | 'reference'>
}

export interface EntityPinResponse {
  entity: string
  status: string | null
  manual_override: boolean
  closed_queue_rows: number
}

export interface WeakEdge {
  other: string
  relation_type: string
  doc_count: number
}

export interface BackboneWarning {
  entity_id: string
  structural_degree: number
  weak_edges: WeakEdge[]
}

export interface BackboneWarningsResponse {
  count: number
  warnings: BackboneWarning[]
}

export const triageApi = {
  /** The review queue (open rows by default; pass `all` for decided rows too). */
  queue: async (includeAll = false): Promise<QueueResponse> => {
    const response = await apiClient.get<QueueResponse>('/triage/queue', {
      params: includeAll ? { all: true } : undefined,
    })
    return response.data
  },

  /**
   * Apply an operator decision to one queue row: pin the entity status + set
   * `manual_override`, then close the row. A subsequent pipeline run leaves the
   * pinned status untouched (override-wins).
   */
  decide: async (
    queueId: string,
    body: DecisionInput,
  ): Promise<DecisionResponse> => {
    const response = await apiClient.post<DecisionResponse>(
      // The queue id is an `entity:<id>`-shaped path segment; the router uses a
      // `:path` converter so the colon is fine unencoded.
      `/triage/queue/${queueId}/decision`,
      body,
    )
    return response.data
  },

  /**
   * Answer a `tooi` row (D2): link one of its candidate URIs onto the entity, or
   * dismiss it. Never touches the entity's status or `manual_override`.
   */
  decideTooi: async (
    queueId: string,
    body: TooiDecisionInput,
  ): Promise<TooiDecisionResponse> => {
    const response = await apiClient.post<TooiDecisionResponse>(
      `/triage/queue/${queueId}/tooi-decision`,
      body,
    )
    return response.data
  },

  /**
   * Pin/release an entity's status from the KG detail panel (override toggle).
   * Keyed on the entity id, so the operator may toggle `manual_override` on ANY
   * entity, not only queued ones. Pinning closes the entity's open queue row;
   * override-wins, so a later pipeline run leaves the pinned status untouched.
   */
  setEntityOverride: async (
    entityId: string,
    body: EntityPinInput,
  ): Promise<EntityPinResponse> => {
    const response = await apiClient.post<EntityPinResponse>(
      `/triage/entities/${entityId}/override`,
      body,
    )
    return response.data
  },

  backboneWarnings: async (
    entityIds: string[],
  ): Promise<BackboneWarningsResponse> => {
    if (entityIds.length === 0) {
      return { count: 0, warnings: [] }
    }
    const response = await apiClient.get<BackboneWarningsResponse>(
      '/triage/backbone-warnings',
      // Repeated `entity_id` query params (the router's aliased list param).
      { params: { entity_id: entityIds } },
    )
    return response.data
  },
}
