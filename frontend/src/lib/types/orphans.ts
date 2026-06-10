/**
 * TypeScript mirrors of the Phase B.5b orphan-dashboard JSON contracts.
 *
 * Hand-typed against
 * `apps/app-main/src/app_main/api/routers/orphans.py` --
 * `OrphansResponse` + `ReconnectResponse`.
 */

export interface OrphanRow {
  id: string
  canonical_name: string
  entity_type?: string | null
  /** Either `"pending_reconnect"` or `"archived"`. */
  orphan_status: string
  reconnect_attempts: number
  /** ISO-8601 string from the backend; `null` for legacy rows. */
  first_orphaned_at?: string | null
  /** ISO-8601 string from the backend; `null` if not yet retried. */
  last_reconnect_attempt_at?: string | null
}

export interface OrphansResponse {
  notebook_id: string
  pending_count: number
  archived_count: number
  /** Pending and archived rows combined; client filters by tab. */
  items: OrphanRow[]
}

export interface ReconnectResponse {
  entity_id: string
  attempted: number
  reconnected: number
  /** `null` when the entity returned to "healthy" state. */
  new_status?: string | null
}
