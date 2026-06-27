export interface NotebookResponse {
  id: string
  name: string
  description: string
  archived: boolean
  created: string
  updated: string
  source_count: number
  note_count: number
  // Track J.3/J.5: null = inherit the global default; 'cloud' | 'private' override.
  privacy_mode?: 'cloud' | 'private' | null
}

export interface NoteResponse {
  id: string
  title: string | null
  content: string | null
  note_type: string | null
  created: string
  updated: string
}

export interface SourceListResponse {
  id: string
  title: string | null
  topics?: string[]                  // Make optional to match Python API
  asset: {
    file_path?: string
    url?: string
  } | null
  embedded: boolean
  embedded_chunks: number            // ADD: From Python API
  insights_count: number
  entity_count: number
  relation_count: number
  created: string
  updated: string
  file_available?: boolean
  // ADD: Async processing fields from Python API
  command_id?: string
  status?: string
  processing_info?: ProcessingInfo
}

export interface ProcessingInfo {
  started_at?: string
  completed_at?: string
  error?: string
  retry?: boolean
  queued?: boolean
  [key: string]: unknown
}

/**
 * Provenance metadata persisted by the SourceExtractor in A.1c.
 *
 * Surfaced on the source-detail badge bar via <ParserEngineBadge>.
 * All fields optional: legacy sources processed before A.1c don't
 * have a metadata block, and the badge renders nothing in that case.
 */
export interface SourceMetadata {
  /** Which engine actually produced the extracted text. */
  parser_engine_used?: 'docling' | 'mineru'
  /** Aggregate confidence from the docling-confidence scorer (0..1). */
  extraction_confidence?: number
  /** Per-signal breakdown (e.g. {ocr: 0.94, layout: 0.88, ...}). */
  extraction_confidence_signals?: Record<string, number>
  /**
   * True when parser_engine='auto' picked docling first, the confidence
   * dropped below the configured threshold, and the orchestrator
   * re-ran the document through MinerU.
   */
  extraction_fallback_triggered?: boolean
}

export interface SourceDetailResponse extends SourceListResponse {
  full_text: string | null  // Can be null for async processing sources
  notebooks?: string[]  // List of notebook IDs this source is linked to
  metadata?: SourceMetadata
  // Track J.3/J.5: true when this document was/will be processed privately.
  private?: boolean
}

/**
 * One shared entity driving a KG-related-source score (Track R.2/R.5).
 *
 * The per-pair "why matched" lineage carried by `kg_entities` on a hybrid
 * result: `weight` is the entity's contribution (type_salience × rarity),
 * `document_frequency` how many sources it appears in, `via_relation` whether
 * it was reached via a 1-hop relation path (vs directly shared).
 */
export interface KGSharedEntity {
  entity_id: string
  name: string
  type: string
  document_frequency: number
  weight: number
  via_relation: boolean
}

/**
 * One signal's contribution to a fused hybrid result (Track R.3/R.5).
 *
 * `present` is whether the source appeared in this signal at all; `score` and
 * `rank` are null when absent (the source still ranks on its other signal —
 * never dropped). `contribution` is the signal's additive RRF contribution to
 * the fused score.
 */
export interface SignalProvenance {
  present: boolean
  score: number | null
  rank: number | null
  contribution: number
}

/**
 * One fused related source from GET /sources/{id}/related-hybrid (Track R.3).
 *
 * `fused_score` is the weighted RRF total; `dense`/`kg` carry per-signal
 * provenance; `kg_entities` is the driving-entities lineage (the "why matched").
 */
export interface RelatedSourceHybrid {
  id: string
  title: string | null
  fused_score: number
  dense: SignalProvenance
  kg: SignalProvenance
  kg_entities: KGSharedEntity[]
}

/** Response shape for GET /api/health/mineru. */
export interface MineruHealthResponse {
  healthy: boolean
  version?: string
  error?: string
}

export type SourceResponse = SourceDetailResponse

export interface SourceStatusResponse {
  status?: string
  message: string
  processing_info?: ProcessingInfo
  command_id?: string
}

export interface SettingsResponse {
  // Document Parser Engine (renamed from default_content_processing_engine_doc in A.1b, Q-A-6)
  // Values: 'simple' | 'docling' | 'mineru' | 'auto'.
  // 'auto' currently routes to docling; confidence-driven fallback ships in A.1c.
  parser_engine?: 'simple' | 'docling' | 'mineru' | 'auto'
  // Extensions that route to MinerU when parser_engine selects it.
  // Other extensions fall back to Docling (with INFO log on the backend).
  mineru_supported_extensions?: string[]
  // Auto-mode confidence threshold (Phase A.1c).
  // When parser_engine="auto" and the docling confidence drops below this,
  // the orchestrator re-runs the document through MinerU. Default 0.95.
  docling_min_confidence?: number
  default_content_processing_engine_url?: 'auto' | 'firecrawl' | 'jina' | 'simple'
  default_embedding_option?: 'ask' | 'always' | 'never'
  auto_delete_files?: 'yes' | 'no'
  youtube_preferred_languages?: string[]

  // GPU Acceleration Settings (content-core)
  docling_gpu_enabled?: boolean
  docling_gpu_device?: 'auto' | 'cuda' | 'cpu'

  // Pipeline Settings (content-core)
  docling_pipeline?: 'auto' | 'standard' | 'vlm'

  // VLM Settings (content-core) - used when pipeline=vlm
  docling_vlm_model?: 'granite-docling-258m' | 'smoldocling-256m'
  docling_vlm_framework?: 'auto' | 'transformers' | 'mlx'

  // OCR Settings (content-core) - used when pipeline=standard
  docling_ocr_engine?: 'auto' | 'easyocr' | 'rapidocr' | 'tesseract'
  docling_ocr_languages?: string[]
  docling_ocr_use_gpu?: boolean

  // Table Processing Settings (content-core)
  docling_table_mode?: 'accurate' | 'fast'

  // Image Export Settings (content-core) - not yet functional
  docling_auto_export_images?: boolean
  docling_image_scale?: number

  // Chunking Settings (content-core)
  docling_chunking_enabled?: boolean
  docling_chunking_method?: 'hybrid' | 'hierarchical'
  docling_chunking_max_tokens?: number

  // File Management Settings
  input_directory_path?: string
  markdown_directory_path?: string
  output_directory_path?: string
  file_operation?: 'copy' | 'move' | 'none'
  output_naming_scheme?: 'timestamp_prefix' | 'date_prefix' | 'datetime_suffix' | 'original'

  // Vault Integration
  vault_path?: string
  vault_entities_folder?: string
  vault_sync_on_startup?: boolean

  // Model routing (Track J): global default privacy mode for LLM routing.
  // 'cloud' = cloud-first with local fallback; 'private' = local-only.
  default_privacy_mode?: 'cloud' | 'private'
}

export interface CreateNotebookRequest {
  name: string
  description?: string
}

export interface UpdateNotebookRequest {
  name?: string
  description?: string
  archived?: boolean
  // Track J.3/J.5: null clears the override (inherit global); 'cloud' | 'private' set it.
  privacy_mode?: 'cloud' | 'private' | null
}

export interface CreateNoteRequest {
  title?: string
  content: string
  note_type?: string
  notebook_id?: string
}

export interface CreateSourceRequest {
  // Backward compatibility: support old single notebook_id
  notebook_id?: string
  // New multi-notebook support
  notebooks?: string[]
  // Required fields
  type: 'link' | 'upload' | 'text'
  url?: string
  file_path?: string
  content?: string
  title?: string
  transformations?: string[]
  embed?: boolean
  delete_source?: boolean
  // New async processing support
  async_processing?: boolean
  // Per-submission processing config overrides
  processing_overrides?: Partial<SettingsResponse>
  // Track J.3/J.5: process this document privately (LLM stages stay local).
  private?: boolean
}

export interface UpdateNoteRequest {
  title?: string
  content?: string
  note_type?: string
}

export interface UpdateSourceRequest {
  title?: string
  type?: 'link' | 'upload' | 'text'
  url?: string
  content?: string
}

// Entity extraction result types
export type EntityPropertyValue = string | number | boolean | string[] | null

export interface ExtractedEntity {
  text: string
  label: string
  properties: Record<string, EntityPropertyValue>
  confidence: number
  source_chunk_id?: string
  source_grounding?: { start_pos: number; end_pos: number } | null
  pagerank?: number
  community_id?: number
  /** Surface-form aliases collapsed onto this entity (Track K resolution). */
  aliases?: string[]
  /** Stable external URIs (TOOI / DOI) from K.4 reconciliation. */
  external_ids?: string[]
}

export interface ExtractedRelation {
  source_entity: string
  target_entity: string
  relation_type: string
  properties: Record<string, EntityPropertyValue>
  confidence: number
}

export interface ExtractionMetadata {
  ontology_name?: string
  extractor_type?: string
  chunk_count?: number
  total_entities?: number
  total_relations?: number
  [key: string]: unknown
}

export interface ExtractionResultResponse {
  entities: ExtractedEntity[]
  relations: ExtractedRelation[]
  metadata: ExtractionMetadata
  entity_count: number
  relation_count: number
}

export interface RunEntitiesOptions {
  ontology_name?: string
  extractor_type?: 'llm' | 'langextract'
  langextract_model_id?: string
  langextract_model_url?: string
  langextract_temperature?: number
  langextract_use_schema_constraints?: boolean
  langextract_fence_output?: boolean
}

export interface APIError {
  detail: string
}

// Source Chat Types
// Base session interface with common fields
export interface BaseChatSession {
  id: string
  title: string
  created: string
  updated: string
  message_count?: number
  model_override?: string | null
}

export interface SourceChatSession extends BaseChatSession {
  source_id: string
  model_override?: string
}

export interface SourceChatMessage {
  id: string
  type: 'human' | 'ai'
  content: string
  timestamp?: string
}

export interface SourceChatContextIndicator {
  sources: string[]
  insights: string[]
  notes: string[]
}

export interface SourceChatSessionWithMessages extends SourceChatSession {
  messages: SourceChatMessage[]
  context_indicators?: SourceChatContextIndicator
}

export interface CreateSourceChatSessionRequest {
  source_id: string
  title?: string
  model_override?: string
}

export interface UpdateSourceChatSessionRequest {
  title?: string
  model_override?: string
}

export interface SendMessageRequest {
  message: string
  model_override?: string
}

export interface SourceChatStreamEvent {
  type: 'user_message' | 'ai_message' | 'context_indicators' | 'complete' | 'error'
  content?: string
  data?: unknown
  message?: string
  timestamp?: string
}

// Notebook Chat Types
export interface NotebookChatSession extends BaseChatSession {
  notebook_id: string
}

export interface NotebookChatMessage {
  id: string
  type: 'human' | 'ai'
  content: string
  timestamp?: string
}

export interface NotebookChatSessionWithMessages extends NotebookChatSession {
  messages: NotebookChatMessage[]
}

export interface CreateNotebookChatSessionRequest {
  notebook_id: string
  title?: string
  model_override?: string
}

export interface UpdateNotebookChatSessionRequest {
  title?: string
  model_override?: string | null
}

export interface ContextSourceEntry {
  id: string
  title?: string
  full_text?: string
  insights?: string[]
  [key: string]: unknown
}

export interface ContextNoteEntry {
  id: string
  title?: string
  content?: string
  [key: string]: unknown
}

export interface ChatContext {
  sources: ContextSourceEntry[]
  notes: ContextNoteEntry[]
}

export interface SendNotebookChatMessageRequest {
  session_id: string
  message: string
  context: ChatContext
  model_override?: string
}

export interface BuildContextRequest {
  notebook_id: string
  context_config: {
    sources: Record<string, string>
    notes: Record<string, string>
  }
}

export interface BuildContextResponse {
  context: ChatContext
  token_count: number
  char_count: number
}
