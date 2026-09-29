/**
 * Typed client for `GET /api/attention` (Track WB.1, the Werkbak).
 *
 * Mirrors the backend models in
 * `apps/app-main/src/app_main/services/attention_service.py`
 * (`AttentionItem`, `AttentionGroup`, `AttentionReport`). The groups are data:
 * the page renders whatever the endpoint returns, in its order.
 */

import apiClient from './client'

export interface AttentionItem {
  id: string
  label: string
  /** The frontend route where this item is resolved. */
  link: string
  detail?: string | null
  at?: string | null
}

export interface AttentionGroup {
  key: string
  title: string
  count: number
  /** The primary action, e.g. "Review 12 proposals"; empty when count is 0. */
  action: string
  /** Where the group as a whole is resolved. */
  link: string
  /** The sidebar entry whose badge this group adds to. */
  nav?: string | null
  /** False for information that asks nothing (jobs in progress, arrivals). */
  needs_action: boolean
  detail?: string | null
  items: AttentionItem[]
  /** Set when the group failed or timed out; the rest still answered. */
  error?: string | null
  /**
   * The part of `count` that goes into `total` and the badges: a paused
   * extraction or failed job whose source a source group already lists is
   * not counted twice. 0 for a failed group or one that needs no action.
   */
  counted: number
  /**
   * How many items the group has; `items` holds the first few (per-notebook
   * groups: all). "N more" is `item_total - items.length`, never `count`.
   */
  item_total: number
}

export interface AttentionReport {
  /** Server time of the read; stored as "last seen" when marked as seen. */
  generated_at: string
  /** Start of the arrivals window: the client's mark, else 7 days back. */
  since: string
  /** True when no mark was sent and `since` is the default window. */
  since_default: boolean
  /** Sum of the groups' `counted`: each thing to act on once. */
  total: number
  groups: AttentionGroup[]
  /** What arrived after `since`. */
  arrived: AttentionGroup[]
}

export const attentionApi = {
  get: async (since?: string | null) => {
    const response = await apiClient.get<AttentionReport>('/attention', {
      params: since ? { since } : undefined,
    })
    return response.data
  },
}
