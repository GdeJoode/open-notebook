# Open Notebook - Complete Architecture Overview

**Project**: Open Notebook - Privacy-focused alternative to Google Notebook LM
**Version**: 2.0.0 (v1-latest-single Docker tag)
**Tech Stack**: Next.js 15 (Frontend), FastAPI (Backend), SurrealDB (Database), Python 3.11-3.12
**Current Date**: 2026-02-24

---

## 1. FRONTEND STRUCTURE

**Framework**: Next.js 15 (App Router), React 19, TypeScript
**Port**: 8502
**State Management**: Zustand, TanStack React Query (v5.83)
**UI Library**: Radix UI, Tailwind CSS 4
**Package Location**: `/frontend`

### Pages & Routes
```
(auth)/login/page.tsx
(dashboard)/
├── page.tsx (main dashboard)
├── notebooks/
│   ├── page.tsx (list view)
│   ├── [id]/page.tsx (detail view)
│   └── components/
│       ├── ChatColumn.tsx
│       ├── NotesColumn.tsx
│       ├── SourcesColumn.tsx
│       ├── NotebookHeader.tsx
│       ├── NotebookList.tsx
│       ├── NoteEditorDialog.tsx
│       └── NotebookCard.tsx
├── sources/
│   ├── page.tsx (list view)
│   ├── [id]/page.tsx (detail view)
│   ├── new/page.tsx
│   └── components/
│       ├── SourceCard.tsx
│       ├── AddSourceDialog.tsx
│       ├── AddExistingSourceDialog.tsx
│       ├── AddSourceButton.tsx
│       └── pipeline/
│           ├── CreateSourcePipeline.tsx
│           ├── PipelineStepper.tsx
│           ├── PipelineHeader.tsx
│           ├── PipelineFooter.tsx
│           ├── BatchModeDialog.tsx
│           └── tabs/
│               ├── PreprocessingTab.tsx
│               ├── ExtractionTab.tsx
│               ├── EmbeddingTab.tsx
│               ├── EntitiesTab.tsx
│               ├── SummariesTab.tsx
│               ├── CompletionTab.tsx
│               └── EntityGraphView.tsx
├── search/page.tsx
├── chat/page.tsx (implicit routing)
├── podcasts/page.tsx
├── summaries/page.tsx
├── ontologies/page.tsx
├── models/page.tsx
│   └── components/
│       ├── AddModelForm.tsx
│       ├── DefaultModelsSection.tsx
│       ├── ModelTypeSection.tsx
│       ├── EmbeddingModelChangeDialog.tsx
│       └── ProviderStatus.tsx
├── transformations/page.tsx
│   └── components/
│       ├── TransformationList.tsx
│       ├── TransformationCard.tsx
│       ├── TransformationEditorDialog.tsx
│       ├── TransformationPlayground.tsx
│       └── DefaultPromptEditor.tsx
├── settings/page.tsx
│   └── components/SettingsForm.tsx
├── knowledge-graph/page.tsx
│   └── components/SigmaGraphView.tsx
├── advanced/page.tsx
│   └── components/
│       ├── RebuildEmbeddings.tsx
│       └── SystemInfo.tsx
└── layout.tsx (dashboard wrapper)

_config/route.ts (API endpoint)
```

### Key Components
- **Authentication**: LoginForm.tsx
- **Layout**: AppShell.tsx, AppSidebar.tsx
- **Common**: EmptyState, LoadingSpinner, ContextToggle, ModelSelector, ThemeToggle, ErrorBoundary, ConnectionGuard, ConfirmDialog
- **Search**: AdvancedModelsDialog.tsx, SaveToNotebooksDialog.tsx, StreamingResponse.tsx
- **Source**: ChatPanel.tsx, SourceDetailContent.tsx, PdfChunkViewer.tsx, SourceDialog.tsx, SourceInsightDialog.tsx, MessageActions.tsx, SessionManager.tsx, NotebookAssociations.tsx
- **Podcasts**: GeneratePodcastDialog.tsx, EpisodeCard.tsx, SpeakerProfilesPanel.tsx, EpisodeProfilesPanel.tsx, TemplatesTab.tsx, EpisodesTab.tsx
- **Providers**: ModalProvider, QueryProvider, ThemeProvider

### API Clients & Hooks
**Location**: `src/lib/api/` and `src/lib/hooks/`

API clients:
- `chat.ts` - Chat functionality
- `client.ts` - Base HTTP client
- `embedding.ts` - Embedding operations
- `insights.ts` - Document insights
- `knowledge-graph.ts` - Knowledge graph queries
- `models.ts` - LLM model configuration
- `notebooks.ts` - Notebook CRUD
- `notes.ts` - Note operations
- `ontologies.ts` - Ontology management
- `podcasts.ts` - Podcast generation
- `preprocessing.ts` - Document preprocessing
- `search.ts` - Text/vector search
- `settings.ts` - App settings
- `source-chat.ts` - Chat within sources
- `sources.ts` - Source CRUD & processing
- `summaries.ts` - Document summaries
- `transformations.ts` - Content transformations
- `query-client.ts` - React Query configuration

Hooks:
- `use-ask.ts` - Ask (search + answer) functionality
- `use-auth.ts` - Authentication
- `use-insights.ts` - Insights retrieval
- `use-knowledge-graph.ts` - Knowledge graph
- `use-modal-manager.ts` - Modal management
- `use-models.ts` - Model management
- `use-navigation.ts` - Navigation state
- `use-notebooks.ts` - Notebook operations
- `use-notes.ts` - Note operations
- `use-ontologies.ts` - Ontology operations
- `use-podcasts.ts` - Podcast operations
- `use-processing-logs.ts` - Processing status
- `use-search.ts` - Search operations
- `use-settings.ts` - Settings management
- `use-sources.ts` - Source operations
- `use-summaries.ts` - Summary operations
- `use-toast.ts` - Toast notifications
- `use-transformations.ts` - Transformation operations
- `use-version-check.ts` - Version checking
- `useNotebookChat.ts` - Notebook chat
- `useSourceChat.ts` - Source chat

### State Management (Zustand Stores)
- `auth-store.ts` - Authentication state
- `navigation-store.ts` - Navigation state
- `sidebar-store.ts` - Sidebar state
- `theme-store.ts` - Theme state

### Type Definitions
- `api.ts` - API request/response types
- `auth.ts` - Authentication types
- `common.ts` - Common types
- `config.ts` - Configuration types
- `models.ts` - LLM model types
- `podcasts.ts` - Podcast types
- `search.ts` - Search types
- `transformations.ts` - Transformation types

### UI Component Library
Custom Shadcn/Radix components:
- Form elements: input, textarea, select, checkbox, radio-group, label, slider
- Layout: accordion, collapsible, separator, scroll-area, tabs
- Dialogs: dialog, alert-dialog, popover
- Navigation: dropdown-menu, command
- Feedback: badge, progress, skeleton, alert
- Advanced: markdown-editor, wizard-container, form-section, checkbox-list

---

## 2. BACKEND STRUCTURE

**Framework**: FastAPI with Uvicorn
**Port**: 5055
**Language**: Python 3.11+
**Database**: SurrealDB (async operations)
**Job Queue**: Background command processor

### Main App (`apps/app-main`)

#### API Routers (`api/routers/`)
- `auth.py` - Authentication & status checks
- `chat.py` - Chat with full knowledge base
- `commands.py` - Async job execution
- `config.py` - Application configuration
- `context.py` - Context management for retrieval
- `embedding.py` - Embedding generation
- `embedding_rebuild.py` - Re-embedding pipeline
- `episode_profiles.py` - Podcast episode profiles
- `insights.py` - Document insights generation
- `knowledge_graph.py` - Knowledge graph visualization
- `models.py` - LLM model management
- `notebooks.py` - Notebook CRUD
- `notes.py` - Note operations
- `ontologies.py` - Ontology management
- `podcasts.py` - Podcast generation
- `preprocessing.py` - Document preprocessing pipeline
- `search.py` - Text/vector search and ask endpoint
- `settings.py` - Application settings
- `sources.py` - Source CRUD, upload, processing
- `source_chat.py` - Chat within source context
- `speaker_profiles.py` - Podcast speaker profiles
- `summaries.py` - Document summaries
- `transformations.py` - Content transformations

#### Services (`services/`)
- `chat_service.py` - Chat workflow orchestration
- `command_service.py` - Background job processing
- `context_service.py` - Context window management
- `entity_extraction_service.py` - Entity extraction
- `insight_service.py` - Document insight generation
- `knowledge_graph_service.py` - Knowledge graph operations
- `log_stream.py` - Streaming logs for preprocessing
- `model_service.py` - LLM model configuration & defaults
- `notebook_service.py` - Notebook operations
- `note_service.py` - Note operations
- `ontology_service.py` - Ontology management
- `podcast_service.py` - Podcast generation
- `preprocessing_service.py` - Document preprocessing
- `search_service.py` - Text/vector search
- `settings_service.py` - Settings management
- `source_processing_service.py` - Source processing orchestration
- `source_service.py` - Source CRUD & file handling
- `summarization_service.py` - Document summarization
- `transformation_service.py` - Transformation execution

#### LangGraph Workflows (`graphs/`)
- `ask.py` - Multi-turn question answering with strategy
- `chat.py` - Chat workflow
- `source_chat.py` - Chat within source context
- `prompt.py` - Prompt management
- `transformation.py` - Transformation application

#### Other Components
- `app.py` - FastAPI application factory with lifespan management
- `auth.py` - Password authentication middleware
- `config.py` - Application configuration
- `dependencies.py` - Dependency injection setup
- `exceptions.py` - Custom exceptions
- `handlers.py` - Command handler registry for background jobs
- `schemas.py` - Pydantic request/response schemas

---

## 3. SHARED PACKAGE

**Package**: `packages/shared`
**Purpose**: Common models, types, utilities across all services

### Models (`models/`)
- `base.py` - Base ObjectModel with id, created_at, updated_at
- `source.py` - Document, Chunk, Insight, Embedding models
- `notebook.py` - Notebook, Note, ChatSession models
- `extraction.py` - Entity, Relation, Ontology models
- `podcast.py` - Podcast, Episode, Speaker models
- `jobs.py` - Job and CommandTask models
- `llm.py` - LLM provider configuration models
- `settings.py` - Application settings model
- `file_tracking.py` - File tracking models
- `transformation.py` - Transformation models

### Types (`types/`)
- `enums.py` - NoteType, ContentProcessingEngine, UrlProcessingEngine, GpuDevice, VlmModel, OcrEngine, etc.
- `pipeline.py` - Pipeline step definitions

### Utils (`utils/`)
- `text.py` - Text processing utilities
- `version.py` - Version utilities

---

## 4. PACKAGES

### A. `packages/surrealdb-service`
**Purpose**: Database abstraction layer with SurrealDB
**Key Features**:
- Async SurrealDB client wrapper
- Migration management system
- REST API endpoints for database operations
- MCP server for external integrations
- Document CRUD patterns

**Key Classes**:
- SurrealDBService - Main database client
- AsyncMigrationManager - Migration handling
- Repository pattern for data operations

### B. `packages/file-manager`
**Purpose**: File and knowledge base management
**Key Features**:
- File upload/download handling
- File watching for automatic ingestion
- MIME type detection
- MCP server for file operations
- Integration with job queue for processing

**Key Components**:
- File handling APIs
- Knowledge base organization
- File metadata tracking

### C. `packages/llm-manager`
**Purpose**: LLM provider management and routing
**Providers Supported**:
- OpenAI
- Anthropic (Claude)
- Google Generative AI
- Groq
- Mistral AI
- Local models via Esperanto
- Ollama

**Key Features**:
- Provider selection and routing
- Token counting (Tiktoken)
- Usage tracking and cost estimation
- Model defaults management
- MCP server for LLM operations

**Key Components**:
- ModelManager - Provider and model selection
- Esperanto integration for multi-provider support
- Token calculation

### D. `packages/job-queue`
**Purpose**: Async background job processing
**Features**:
- Command queue system
- Job persistence
- Worker management
- Progress tracking
- Error handling and retries

### E. `packages/ontology-manager`
**Purpose**: Ontology definitions and management
**Features**:
- Ontology CRUD operations
- Relationship management
- Schema validation

---

## 5. PIPELINES

### A. `pipelines/ingestion`
**Purpose**: Document ingestion and parsing
**Input**: PDF, video, audio, web pages
**Key Technologies**:
- Docling VLM (Vision-Language Model) for document understanding
- EasyOCR for optical character recognition
- WhisperX for audio transcription
- yt-dlp for YouTube extraction
- BeautifulSoup4 for web scraping

**Output**: Structured chunks with metadata
**Features**:
- Automatic document parsing with VLM
- Table detection and extraction
- Text + image extraction
- Audio/video transcription
- Web page extraction

### B. `pipelines/embeddings`
**Purpose**: Vector embedding generation
**Providers**:
- Sentence-Transformers (local)
- ONNX Runtime (optimized inference)
- Esperanto/LLM providers (cloud)

**Features**:
- Batch embedding generation
- Vector storage in SurrealDB
- Multiple model support

### C. `pipelines/ontology-extraction`
**Purpose**: Entity and relation extraction
**Extractors**:
- LLM-based extraction (default)
- LangExtract (alternative, faster)

**Features**:
- Ontology-guided extraction
- Entity relationship mapping
- Knowledge graph construction
- Configurable extraction models

### D. `pipelines/entity-filtering`
**Purpose**: Entity validation and filtering
**Features**:
- Duplicate detection
- Entity validation
- Relationship validation

### E. `pipelines/enrichment`
**Purpose**: Content enrichment
**Features**:
- Metadata enrichment
- Source annotation
- Relationship discovery

### F. `pipelines/summarization`
**Purpose**: Document and chunk summarization
**Features**:
- Document-level summaries
- Chunk-level summaries
- Hierarchical summarization

### G. `pipelines/retrieval`
**Purpose**: Intelligent retrieval from knowledge base
**Features**:
- Vector search
- Knowledge graph queries
- Hybrid retrieval
- Context window optimization
- NetworkX for graph operations

---

## 6. DATA LAYER

### SurrealDB Database
**Port**: 8000
**Default Credentials**: root/root
**Storage**: RocksDB backend
**Tables**:

Core entities:
- `source` - Document metadata
- `chunk` - Text chunks from documents
- `notebook` - User-created notebooks
- `note` - Notes within notebooks
- `chat_session` - Chat conversation sessions
- `chat_message` - Individual messages

Processing & results:
- `embedding` - Vector embeddings
- `entity` - Extracted entities
- `relation` - Relationships between entities
- `ontology` - Ontology definitions
- `insight` - Generated insights
- `summary` - Document summaries
- `transformation` - Content transformations

Config & tracking:
- `model_config` - LLM model configurations
- `settings` - Application settings
- `file_tracking` - File metadata
- `job` / `command_task` - Background job tracking
- `podcast` / `episode` / `speaker` - Podcast data

### Schema Examples
- **Chunk**: text, order, physical_page, element_type, positions (bounding boxes), metadata
- **Source**: title, description, file_path, url, content, notebooks (relations)
- **Embedding**: chunk_ref, vector, model, dimensions, created_at
- **Entity**: name, type, description, source_ref, confidence_score

---

## 7. EXTERNAL INTEGRATIONS

### LLM Providers (16+)
- OpenAI (GPT-4, etc.)
- Anthropic (Claude)
- Google Generative AI
- Groq
- Mistral
- Local via Ollama or LM Studio
- And more via Esperanto

### Vector Databases
- SurrealDB with vector support

### Web Content Extraction
- FireCrawl (URL processing)
- Jina (URL extraction)
- Simple HTTP + BeautifulSoup

### Audio Processing
- WhisperX (transcription)
- FFmpeg (audio processing)

### Document Processing
- Docling (VLM-based parsing)
- EasyOCR (fallback OCR)
- PDF extraction via Docling

### Media
- YouTube extraction (yt-dlp)
- Audio/video support

---

## 8. INFRASTRUCTURE & DEPLOYMENT

### Docker Setup
**Dockerfile**: Multi-stage build
- **Builder Stage**: Python 3.12, uv, Node.js 20, npm build
- **Runtime Stage**: Python 3.12-slim, ffmpeg, supervisor, Node.js 20

**Services** (docker-compose.yml):
1. **SurrealDB** (surrealdb/surrealdb:v2)
   - Port 8000
   - RocksDB storage
   - GraphQL enabled

2. **Open Notebook** (built from Dockerfile)
   - Port 8502 (Frontend - Next.js)
   - Port 5055 (API - FastAPI)
   - Depends on SurrealDB
   - Volume: `./notebook_data:/app/data`

### Ports
- **8000**: SurrealDB
- **8502**: Frontend (Next.js)
- **5055**: API (FastAPI)

### Process Management
**Supervisord** (`supervisord.conf`):
- Manages multiple processes (frontend, API, background workers)
- Auto-restart on failure
- Log management

### Environment Configuration
- `.env` and `docker.env` for configuration
- Database URL, API keys, model settings

---

## 9. KEY WORKFLOWS

### Source Creation & Processing
1. User uploads file/URL/content via frontend
2. Frontend sends to `POST /sources` with file
3. Backend saves file → generates unique filename
4. Creates Source record in SurrealDB
5. Triggers preprocessing pipeline via command queue
6. Ingestion: Extract text/images using Docling VLM
7. Chunking: Create Chunk records with metadata & bounding boxes
8. Embedding: Generate vectors via configured embedding model
9. Extraction: Entity/relation extraction via ontology
10. Enrichment: Add metadata and relationships
11. Storage: Save all results to SurrealDB
12. Notebook Association: Link to selected notebooks

### Search & Retrieval
1. User query via `/search` endpoint
2. TextSearch: Full-text search in chunks
3. VectorSearch: Embed query → find similar chunks
4. HybridSearch: Combine both approaches
5. Return ranked results with source references

### Chat & Ask
1. User question via `/ask` endpoint
2. LangGraph workflow: ask.py
3. Strategy phase: Determine what to search for
4. Retrieval: Execute searches
5. Answer phase: Generate answer with context
6. Final answer: Synthesize with follow-up context
7. Stream response back to frontend as SSE

### Podcast Generation
1. User selects sources/content
2. Configures speaker profiles
3. Triggers podcast generation via `/podcasts`
4. System generates script from content
5. Audio synthesis per speaker
6. Combines audio tracks
7. Stores podcast & episodes

---

## 10. STATE MANAGEMENT PATTERNS

### Frontend
- **Zustand stores**: Auth, navigation, sidebar, theme
- **React Query**: Server state management (API data)
- **Context API**: Implicit via providers (Modal, Theme)

### Backend
- **FastAPI Depends**: Dependency injection for services
- **SurrealDB**: Persistent state
- **Job Queue**: Async state via command_task table

---

## 11. AUTHENTICATION & SECURITY

### Password Protection
- PasswordAuthMiddleware on FastAPI
- Excluded paths: /health, /docs, /_config, /api/auth/status
- Frontend stores credentials in auth-store

### CORS
- Allow all origins (configured for flexibility)

---

## 12. DEVELOPMENT SETUP

**Root**: Workspace using UV package manager
- Python 3.10+
- Make targets for common tasks

**Key Commands**:
- `npm run dev` (frontend)
- `uv run app-main` (backend)
- Docker Compose for full stack

---

## 13. TESTING

### Framework
- pytest + pytest-asyncio
- Integration tests marked with `@pytest.mark.integration`
- Test patterns in each package/pipeline

---

## ARCHITECTURE PATTERNS SUMMARY

1. **Modular Monolith**: Core app + specialized packages + independent pipelines
2. **Service-Oriented**: Each package is a service with clear boundaries
3. **API-First**: All inter-service communication via APIs/schemas
4. **Async-First**: Async/await throughout Python codebase
5. **Schema-Driven**: Pydantic models define contracts
6. **Background Processing**: Command queue for long-running tasks
7. **LangGraph**: Orchestration of multi-step AI workflows
8. **Vector-First Data**: Embeddings as primary retrieval mechanism
