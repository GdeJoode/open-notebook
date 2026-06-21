/**
 * Types for the model-routing config + privacy + provider-health surface
 * (Track J.5). Mirrors `apps/app-main/src/app_main/api/routers/model_routes.py`.
 */

export type RoutingTask = 'entity_extraction' | 'summarization' | 'chat'

/** A privacy scope's setting. `inherit` is UI-only (maps to null on the wire). */
export type PrivacyValue = 'cloud' | 'private' | 'inherit'

export type CircuitState = 'closed' | 'open' | 'half_open'

export interface ProviderChainEntry {
  provider: string
  model_id: string | null
  enabled: boolean
}

export interface ModelRoute {
  task: RoutingTask
  provider_chain: ProviderChainEntry[]
  private_chain: ProviderChainEntry[]
  /** True when no row is persisted and this is the J.1 default chain. */
  is_default: boolean
}

export interface ModelRouteUpdate {
  provider_chain: ProviderChainEntry[]
  private_chain?: ProviderChainEntry[]
}

export interface ProviderHealth {
  provider: string
  is_local: boolean
  configured: boolean
  circuit_state: CircuitState
  recent_fallback_count: number
}

export interface ModelRouteHealth {
  providers: ProviderHealth[]
}
