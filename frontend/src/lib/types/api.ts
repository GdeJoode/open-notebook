export interface NotebookResponse {
  id: string
  name: string
  description: string
  archived: boolean
  created: string
  updated: string
  source_count: number
  note_count: number
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
  created: string
  updated: string
  file_available?: boolean
  // ADD: Async processing fields from Python API
  command_id?: string
  status?: string
  processing_info?: Record<string, unknown>
}

export interface SourceDetailResponse extends SourceListResponse {
  full_text: string | null  // Can be null for async processing sources
  notebooks?: string[]  // List of notebook IDs this source is linked to
}

export type SourceResponse = SourceDetailResponse

export interface SourceStatusResponse {
  status?: string
  message: string
  processing_info?: Record<string, unknown>
  command_id?: string
}

export interface SettingsResponse {
  // Document Engine (simplified - GPU/VLM controlled via advanced settings)
  default_content_processing_engine_doc?: 'auto' | 'docling' | 'simple'
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
}

export interface CreateNotebookRequest {
  name: string
  description?: string
}

export interface UpdateNotebookRequest {
  name?: string
  description?: string
  archived?: boolean
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

export interface SendNotebookChatMessageRequest {
  session_id: string
  message: string
  context: {
    sources: Array<Record<string, unknown>>
    notes: Array<Record<string, unknown>>
  }
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
  context: {
    sources: Array<Record<string, unknown>>
    notes: Array<Record<string, unknown>>
  }
  token_count: number
  char_count: number
}
