/**
 * API client for Docker microservices (via main app proxy).
 *
 * All calls go through /api/services/* which the main app
 * proxies to the appropriate Docker service.
 */

import apiClient from './client'

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type PrivacyLevel = 'public' | 'internal' | 'confidential'

export interface ModelRoute {
  provider: string
  model: string
}

export interface RoutingSummary {
  default_privacy: string
  providers: string[]
  routing: Record<string, Record<string, string>>
}

export interface ServiceHealth {
  [service: string]: { status: string }
}

// Extraction
export interface ExtractRequest {
  file_path: string
  privacy?: PrivacyLevel
  write_to_db?: boolean
  model_override?: {
    provider?: string
    model?: string
    api_key?: string
  }
  override?: {
    ontology?: string
  }
}

export interface EntityResult {
  name: string
  type: string
  description?: string
}

export interface RelationResult {
  source: string
  target: string
  type: string
}

export interface ExtractResponse {
  input_file: string
  output_file: string | null
  ontology: string
  chunks_processed: number
  entities: EntityResult[]
  relations: RelationResult[]
  unique_entity_count: number
  unique_relation_count: number
  entities_written: number
  relations_written: number
  processing_seconds: number
}

// Entities (from extraction service)
export interface ServiceEntity {
  id: string
  canonical_name: string
  entity_type: string
  description?: string
  confidence: number
  pagerank?: number
  community_id?: number
  source_count?: number
  status: string
  external_uri?: string
  validation_status?: string
}

export interface ServiceEntityDetail {
  entity: ServiceEntity
  outgoing_relations: ServiceRelation[]
  incoming_relations: ServiceRelation[]
  aliases: { alias_text: string; confidence: number }[]
}

export interface ServiceRelation {
  id: string
  relation_type: string
  confidence: number
  target_name?: string
  target_type?: string
  source_name?: string
  source_type?: string
}

export interface EntityGraph {
  center: string
  depth: number
  nodes: ServiceEntity[]
  edges: { source: string; target: string; type: string }[]
  node_count: number
  edge_count: number
}

// Validation
export interface ValidationResult {
  entity_id: string
  entity_name: string
  original_type: string
  status: string
  canonical_name?: string
  corrected_type?: string
  external_uri?: string
  source_vocabulary?: string
  confidence: number
  details?: string
}

export interface ValidationReport {
  total_entities: number
  results_by_status: Record<string, number>
  validated: ValidationResult[]
  reclassified: ValidationResult[]
  enriched: ValidationResult[]
  flagged_generic: ValidationResult[]
  unvalidated: ValidationResult[]
  processing_seconds: number
}

// Deduplication
export interface DuplicateCandidate {
  entity_a_id: string
  entity_a_name: string
  entity_a_type: string
  entity_b_id: string
  entity_b_name: string
  entity_b_type: string
  string_similarity: number
  semantic_similarity: number
  combined_score: number
  auto_merge: boolean
}

export interface DeduplicateResponse {
  total_entities: number
  candidates: DuplicateCandidate[]
  auto_merge_count: number
  review_count: number
  processing_seconds: number
}

// Summarization
export interface SummarizeRequest {
  file_path: string
  preset?: string
  system_prompt?: string
  output_instructions?: string
}

export interface PresetInfo {
  name: string
  description: string
}

// Vocabularies
export interface VocabularyInfo {
  vocabularies: Record<string, number>
}

// ---------------------------------------------------------------------------
// API calls
// ---------------------------------------------------------------------------

// Health & routing
export const getServicesHealth = () =>
  apiClient.get<ServiceHealth>('/services/health').then(r => r.data)

export const getRouting = () =>
  apiClient.get<RoutingSummary>('/services/routing').then(r => r.data)

// Extraction
export const extractEntities = (req: ExtractRequest) =>
  apiClient.post<ExtractResponse>('/services/extract', req).then(r => r.data)

export const getOntologies = () =>
  apiClient.get<{ ontologies: string[]; default: string }>('/services/ontologies').then(r => r.data)

// Entity browsing
export const getServiceEntities = (params?: {
  type?: string
  status?: string
  limit?: number
  offset?: number
}) =>
  apiClient.get('/services/entities', { params }).then(r => r.data)

export const getServiceEntity = (entityId: string) =>
  apiClient.get<ServiceEntityDetail>(`/services/entities/${entityId}`).then(r => r.data)

export const getEntityGraph = (entityId: string, depth = 2) =>
  apiClient.get<EntityGraph>(`/services/entities/${entityId}/graph`, { params: { depth } }).then(r => r.data)

// Validation
export const validateEntities = (params?: {
  entity_type?: string
  use_api?: boolean
  apply_changes?: boolean
}) =>
  apiClient.post<ValidationReport>('/services/validate', null, { params }).then(r => r.data)

export const getVocabularies = () =>
  apiClient.get<VocabularyInfo>('/services/vocabularies').then(r => r.data)

export const refreshVocabularies = () =>
  apiClient.post('/services/vocabularies/refresh').then(r => r.data)

// Deduplication
export const findDuplicates = (params?: {
  entity_type?: string
  merge_threshold?: number
  review_threshold?: number
}) =>
  apiClient.post<DeduplicateResponse>('/services/deduplicate', null, { params }).then(r => r.data)

export const getDuplicates = (params?: {
  entity_type?: string
  min_score?: number
}) =>
  apiClient.get('/services/duplicates', { params }).then(r => r.data)

export const mergeEntities = (canonical_id: string, duplicate_id: string) =>
  apiClient.post('/services/merge', { canonical_id, duplicate_id }).then(r => r.data)

// Summarization
export const summarizeDocument = (req: SummarizeRequest) =>
  apiClient.post('/services/summarize', req).then(r => r.data)

export const getPresets = () =>
  apiClient.get<{ presets: PresetInfo[]; auto_detect: boolean }>('/services/presets').then(r => r.data)
