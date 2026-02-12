# Open Notebook - UV Workspace Refactoring Plan

> **Status**: IN PROGRESS - Core packages and initial pipelines implemented
> **Created**: 2025-01-25
> **Last Updated**: 2026-02-12

## Table of Contents
1. [Overview](#overview)
2. [Workspace Structure](#workspace-structure)
3. [Package Specifications](#package-specifications)
4. [Pipeline Specifications](#pipeline-specifications)
5. [Application Specifications](#application-specifications)
6. [Docker Architecture](#docker-architecture)
7. [Migration Checklist](#migration-checklist)

---

## Overview

### Goals
- [x] Split monolithic codebase into focused, reusable packages
- [x] Enable **independent operation** of each pipeline component
- [ ] Provide **standalone Streamlit UIs** for running pipelines without the full app *(deferred — architecture supports it but not building UIs now)*
- [ ] Maintain a production-ready main application that integrates all pipelines
- [ ] Support flexible Docker deployments (standalone pipelines, full app, or both)

### Current Status Summary

| Component | Status | Tests | Commit |
|-----------|--------|-------|--------|
| packages/shared | ✅ Done | 86 | `7db8a90` |
| packages/surrealdb-service | ✅ Done | 28 | `f97cf54` |
| packages/file-manager | ✅ Done | 234 | `e17c86b` |
| packages/llm-manager | ✅ Done | 73 | `3fcdd6b` |
| packages/ontology-manager | ✅ Done | 188 | `9867089` |
| pipelines/ingestion | ✅ Done | 62 | `b7295ad` |
| pipelines/ontology-extraction | ✅ Done (refactored) | 32 | `9867089` |
| pipelines/entity-filtering | ✅ Done (expanded) | 469 | `134a871` |
| pipelines/web-scraper | 📦 Scaffolded | 0 | — |
| pipelines/summarization | 📦 Scaffolded | 0 | — |
| pipelines/enrichment | 📦 Scaffolded | 0 | — |
| pipelines/embeddings | 📦 Scaffolded | 0 | — |
| pipelines/retrieval | 📦 Scaffolded | 0 | — |
| apps/app-main | 📦 Scaffolded | 0 | — |
| apps/chat | 📦 Scaffolded | 0 | — |
| apps/canvas | 📦 Scaffolded | 0 | — |
| **Total** | | **1162** | |

---

## Workspace Structure

```
open-notebook/
├── pyproject.toml                    # Workspace root
├── uv.lock                           # Shared lockfile
├── docker-compose.yml                # Production deployment
├── docker-compose.standalone.yml     # Standalone pipeline containers
├── docker-compose.connected.yml      # Connected pipeline containers
│
├── packages/                         # Core services (with APIs and UIs)
│   ├── shared/                       # ✅ Common utilities, schemas (library only)
│   ├── surrealdb-service/            # ✅ Database access layer
│   ├── file-manager/                 # ✅ File system management
│   ├── llm-manager/                  # ✅ LLM model management (Claude, Ollama)
│   └── ontology-manager/             # ✅ Ontology schema versioning, validation, evolution
│
├── pipelines/                        # Processing pipelines (CLI + UI)
│   ├── web-scraper/                  # Website scraping and content download
│   ├── ingestion/                    # ✅ Document/audio ingestion
│   ├── ontology-extraction/          # ✅ Pure LLM-based entity/relation extraction
│   ├── entity-filtering/             # ✅ 13-stage filtering, dedup, resolution, validation, scoring
│   ├── summarization/                # RAPTOR, TreeKG, and LLM summarization
│   ├── enrichment/                   # Metadata verification and enrichment
│   ├── embeddings/                   # Vector embeddings
│   └── retrieval/                    # Search and retrieval
│
└── apps/                             # User-facing applications
    ├── app-main/                     # Production app (FastAPI + Next.js)
    ├── chat/                         # Knowledge base chat interface
    └── canvas/                       # Interactive note creation canvas
```

### Dependency Rules

1. **shared** - No dependencies on other workspace packages
2. **surrealdb-service** - Depends only on `shared`
3. **file-manager** - Depends on `shared` and `surrealdb-service`
4. **llm-manager** - Depends on `shared`
5. **ontology-manager** - Depends on `shared` and `surrealdb-service`
6. **pipelines** - Depend on `shared`, `surrealdb-service`, and relevant packages
   - ontology-extraction: `shared` + `llm-manager` + `ontology-manager`
   - entity-filtering: `shared` + `surrealdb-service` + `ontology-manager`
   - ingestion: `shared` + `surrealdb-service` + `file-manager` + `llm-manager`
   - Pipelines use file-manager for all file write operations
   - Pipelines can read files directly (read-only access)
7. **apps** - Can depend on anything

### Dependency Graph

```
shared ←───────────────────────────────────────────────────┐
   ↑                                                       │
surrealdb-service ←────────────────────────────────────────┤
   ↑                                                       │
file-manager ←─────────────────────────────────────────────┤
   ↑                                                       │
llm-manager ←──────────────────────────────────────────────┤
   ↑ (provides model access to all LLM-using pipelines)    │
   │                                                       │
   ├── ontology-manager ←──────────────────────────────────┤
   │   (schema versioning, validation, prompts)            │
   │                                                       │
   ├── web-scraper                                         │
   │       ↓                                               │
   ├── ingestion ←─────────────────────────────────────────┤
   │       ↓                                               │
   ├── ontology-extraction (uses llm-manager +             │
   │       ontology-manager for LLM-guided extraction)     │
   │       ↓                                               │
   ├── entity-filtering (13-stage pipeline: noise, dedup,   │
   │       normalization, fuzzy/embedding dedup, resolution,│
   │       KG matching, ontology validation, graph analysis,│
   │       composite edge prediction)                      │
   │       ↓                                               │
   ├── summarization (uses llm-manager)                    │
   │       ↓                                               │
   ├── enrichment (external APIs: Scholar, CrossRef, etc.) │
   │       ↓                                               │
   ├── embeddings (uses llm-manager for embed models)      │
   │       ↓                                               │
   └── retrieval (uses llm-manager for reranking)          │
              ↑                                            │
   ┌──────────┴──────────┐                                 │
   │                     │                                 │
 chat ←──────────────  canvas ←────────────────────────────┤
 (RAG + LLM)          (note editor + KB)                   │
   │                     │                                 │
   └──────────┬──────────┘                                 │
              ↓                                            │
         app-main ←────────────────────────────────────────┘
```

### Pipeline Processing Order

```
web-scraper → ingestion → ontology-extraction → entity-filtering → summarization → enrichment → embeddings → retrieval
                                                                                       ↑
                                                                             (manual verification
                                                                              & metadata editing)
```

### Root `pyproject.toml` (Workspace Config)

```toml
[project]
name = "open-notebook-workspace"
version = "2.0.0"
requires-python = ">=3.11,<3.13"

[tool.uv.workspace]
members = [
    "packages/shared",
    "packages/surrealdb-service",
    "packages/file-manager",
    "packages/llm-manager",
    "packages/ontology-manager",
    "pipelines/web-scraper",
    "pipelines/ingestion",
    "pipelines/ontology-extraction",
    "pipelines/entity-filtering",
    "pipelines/summarization",
    "pipelines/enrichment",
    "pipelines/embeddings",
    "pipelines/retrieval",
    "apps/app-main",
    "apps/chat",
    "apps/canvas",
]

[tool.uv.sources]
shared = { workspace = true }
surrealdb-service = { workspace = true }
file-manager = { workspace = true }
llm-manager = { workspace = true }
ontology-manager = { workspace = true }
web-scraper = { workspace = true }
ingestion = { workspace = true }
ontology-extraction = { workspace = true }
entity-filtering = { workspace = true }
summarization = { workspace = true }
enrichment = { workspace = true }
embeddings = { workspace = true }
retrieval = { workspace = true }
chat = { workspace = true }
canvas = { workspace = true }
```

---

## Package Specifications

### packages/shared

**Purpose**: Common utilities, configuration, schemas, and constants used across all packages.

**What belongs here**:
- [ ] Pydantic base models and schemas
- [ ] Configuration management (env vars, settings)
- [ ] Logging setup (loguru configuration)
- [ ] Common utilities (text processing, file handling)
- [ ] Constants and enums
- [ ] Type definitions

**What does NOT belong here**:
- Database access code
- Business logic
- External API clients

**Source files to migrate**:
| Current Location | New Location | Notes |
|-----------------|--------------|-------|
| `open_notebook/config.py` | `packages/shared/src/shared/config.py` | |
| `open_notebook/domain/base.py` | `packages/shared/src/shared/schemas/base.py` | |
| `open_notebook/exceptions.py` | `packages/shared/src/shared/exceptions.py` | |
| | | |

**Directory structure**:
```
packages/shared/
├── pyproject.toml
├── README.md
└── src/shared/
    ├── __init__.py
    ├── config.py           # Configuration management
    ├── constants.py        # Application constants
    ├── exceptions.py       # Custom exceptions
    ├── logging.py          # Loguru setup
    ├── schemas/            # Pydantic models
    │   ├── __init__.py
    │   ├── base.py         # ObjectModel, RecordModel
    │   ├── notebook.py     # Notebook, Note, Source, Chunk
    │   ├── knowledge.py    # Entity, Relation, Claim
    │   └── settings.py     # ContentSettings, ModelConfig
    └── utils/
        ├── __init__.py
        ├── text.py         # Text processing utilities
        └── files.py        # File handling utilities
```

---

### packages/file-manager

**Purpose**: Central file system management - single point of entry for all file operations, file tracking, and storage organization.

**Design Principle**: The file manager is the **only component** that can perform write operations (copy, move, delete, rename) on files. Other components have read access but must request file operations through the file manager API.

**What belongs here**:
- [ ] File operation API (copy, move, delete, rename, create directory)
- [ ] File tracking and indexing in database
- [ ] Knowledge base management (create, list, assign files)
- [ ] Project management (create, list, assign files)
- [ ] File metadata management
- [ ] Storage location configuration
- [ ] File integrity checks (checksums, existence verification)

**What does NOT belong here**:
- File content processing (that's ingestion pipeline)
- File content analysis (that's other pipelines)
- Database queries unrelated to files

**Storage Architecture**:
```
{USER_DEFINED_ROOT}/                    # e.g., /mnt/e
├── knowledgebases/                     # Files linked to knowledge graph
│   ├── {kb_name_1}/                    # Knowledge base 1
│   │   ├── sources/                    # Original source files
│   │   ├── exports/                    # Exported content
│   │   └── media/                      # Audio/video/images
│   ├── {kb_name_2}/                    # Knowledge base 2
│   │   └── ...
│   └── _default/                       # Default KB if none specified
│
├── projects/                           # Standalone project files (NOT in KG)
│   ├── {project_name_1}/
│   │   ├── input/                      # Input files for processing
│   │   ├── output/                     # Processed output
│   │   └── exports/                    # Exported results
│   └── {project_name_2}/
│       └── ...
│
├── obsidian/                           # Obsidian vault exports
│   ├── {vault_name_1}/                 # Obsidian vault (can be opened directly)
│   │   ├── .obsidian/                  # Obsidian config (auto-generated)
│   │   ├── notes/                      # Exported notes from canvas
│   │   ├── sources/                    # Source document summaries
│   │   ├── conversations/              # Exported chat conversations
│   │   └── attachments/                # Images and media
│   └── {vault_name_2}/
│       └── ...
│
└── temp/                               # Temporary processing files
    └── {session_id}/                   # Per-session temp storage
```

**Core Rules**:
1. A file can belong to **exactly one** knowledge base OR one project (not both)
2. Files in knowledge bases are tracked in the knowledge graph
3. Files in projects are tracked but NOT part of the knowledge graph
4. All file mutations go through file manager API
5. Other components get read-only file paths from file manager
6. Obsidian vaults are standalone exports - changes sync one-way (app → Obsidian)

**Directory structure**:
```
packages/file-manager/
├── pyproject.toml
├── README.md
├── main.py                 # FastAPI server entry point
├── ui.py                   # Streamlit file browser UI
└── src/file_manager/
    ├── __init__.py
    ├── config.py           # Storage configuration
    ├── models.py           # File, KnowledgeBase, Project, ObsidianVault models
    ├── tracker.py          # File tracking in database
    ├── operations.py       # File operations (copy, move, delete)
    ├── knowledge_bases.py  # KB management
    ├── projects.py         # Project management
    ├── obsidian/
    │   ├── __init__.py
    │   ├── vault.py        # Vault creation and management
    │   ├── exporter.py     # Export notes, conversations, sources
    │   ├── formatter.py    # Format content for Obsidian markdown
    │   └── sync.py         # One-way sync from app to vault
    ├── api/
    │   ├── __init__.py
    │   ├── app.py          # FastAPI app
    │   └── routers/
    │       ├── __init__.py
    │       ├── files.py    # File operations endpoints
    │       ├── kbs.py      # Knowledge base endpoints
    │       ├── projects.py # Project endpoints
    │       └── obsidian.py # Obsidian vault endpoints
    └── mcp/
        ├── __init__.py
        └── server.py       # MCP server for Claude file access
```

**API Endpoints**:
```yaml
# Configuration
GET    /api/v1/config                    # Get storage configuration
PUT    /api/v1/config                    # Update storage root, etc.

# Knowledge Bases
GET    /api/v1/knowledge-bases           # List all KBs
POST   /api/v1/knowledge-bases           # Create new KB
GET    /api/v1/knowledge-bases/{kb}      # Get KB details
DELETE /api/v1/knowledge-bases/{kb}      # Delete KB (with confirmation)
GET    /api/v1/knowledge-bases/{kb}/files # List files in KB

# Projects
GET    /api/v1/projects                  # List all projects
POST   /api/v1/projects                  # Create new project
GET    /api/v1/projects/{project}        # Get project details
DELETE /api/v1/projects/{project}        # Delete project
GET    /api/v1/projects/{project}/files  # List files in project

# File Operations
POST   /api/v1/files/upload              # Upload file to KB or project
POST   /api/v1/files/copy                # Copy file
POST   /api/v1/files/move                # Move file (within same KB/project)
POST   /api/v1/files/delete              # Delete file
POST   /api/v1/files/assign              # Assign file to KB or project
GET    /api/v1/files/{file_id}           # Get file metadata
GET    /api/v1/files/{file_id}/path      # Get file path (read access)

# Directory Operations
POST   /api/v1/directories               # Create directory
DELETE /api/v1/directories               # Delete directory
GET    /api/v1/directories/browse        # Browse directory structure

# Obsidian Vaults
GET    /api/v1/obsidian/vaults           # List all Obsidian vaults
POST   /api/v1/obsidian/vaults           # Create new vault
GET    /api/v1/obsidian/vaults/{vault}   # Get vault details
DELETE /api/v1/obsidian/vaults/{vault}   # Delete vault
GET    /api/v1/obsidian/vaults/{vault}/path  # Get vault path (to open in Obsidian)
POST   /api/v1/obsidian/export/notes     # Export notes to vault
POST   /api/v1/obsidian/export/conversation  # Export chat conversation to vault
POST   /api/v1/obsidian/export/source    # Export source summary to vault
POST   /api/v1/obsidian/sync/{vault}     # Re-sync vault from app data

# Search & Query
GET    /api/v1/files/search              # Search files by name, type, etc.
GET    /api/v1/files/untracked           # Find files not in any KB/project
POST   /api/v1/files/sync                # Sync file system with database
```

**MCP Tools** (for Claude access):
```yaml
tools:
  - name: list_knowledge_bases
    description: List all knowledge bases

  - name: list_projects
    description: List all standalone projects

  - name: list_files
    description: List files in a knowledge base or project

  - name: get_file_path
    description: Get the file path for reading

  - name: upload_file
    description: Upload/copy a file to a knowledge base or project

  - name: move_file
    description: Move a file within the storage system

  - name: create_project
    description: Create a new standalone project

  - name: search_files
    description: Search for files by name or metadata
```

**Database Schema** (SurrealDB):
```sql
-- File tracking
DEFINE TABLE file SCHEMAFULL;
DEFINE FIELD path ON file TYPE string;
DEFINE FIELD filename ON file TYPE string;
DEFINE FIELD extension ON file TYPE string;
DEFINE FIELD size_bytes ON file TYPE int;
DEFINE FIELD checksum ON file TYPE option<string>;
DEFINE FIELD mime_type ON file TYPE option<string>;
DEFINE FIELD created ON file TYPE datetime;
DEFINE FIELD modified ON file TYPE datetime;
DEFINE FIELD storage_type ON file TYPE string;  -- 'knowledge_base' | 'project'
DEFINE FIELD knowledge_base ON file TYPE option<record<knowledge_base>>;
DEFINE FIELD project ON file TYPE option<record<project>>;

-- Knowledge bases
DEFINE TABLE knowledge_base SCHEMAFULL;
DEFINE FIELD name ON knowledge_base TYPE string;
DEFINE FIELD path ON knowledge_base TYPE string;
DEFINE FIELD description ON knowledge_base TYPE option<string>;
DEFINE FIELD created ON knowledge_base TYPE datetime;

-- Projects
DEFINE TABLE project SCHEMAFULL;
DEFINE FIELD name ON project TYPE string;
DEFINE FIELD path ON project TYPE string;
DEFINE FIELD description ON project TYPE option<string>;
DEFINE FIELD created ON project TYPE datetime;

-- Constraints
DEFINE INDEX file_path ON file FIELDS path UNIQUE;
DEFINE INDEX kb_name ON knowledge_base FIELDS name UNIQUE;
DEFINE INDEX project_name ON project FIELDS name UNIQUE;
```

#### Streamlit UI Specification: file-manager

**Purpose**: **File browser and storage management** - Browse files, manage knowledge bases and projects, perform file operations.

**Use Cases**:
- Browse and organize files across knowledge bases and projects
- Create and manage knowledge bases
- Create and manage standalone projects
- Upload files to specific locations
- Move/copy files between locations
- Find untracked files and assign them
- Configure storage locations

**Pages/Features**:
- [ ] **File Browser**: Tree view of storage, file details, preview
- [ ] **Knowledge Bases**: Create/edit/delete KBs, view contents
- [ ] **Projects**: Create/edit/delete projects, view contents
- [ ] **Upload**: Upload files to KB or project
- [ ] **Operations**: Move, copy, delete files with confirmation
- [ ] **Sync**: Find untracked files, sync with database
- [ ] **Settings**: Configure storage root, default KB

**UI Mockup**:
```
┌─────────────────────────────────────────────────────────┐
│  File Manager                         [/mnt/e configured]│
├─────────────┬───────────────────────────────────────────┤
│ Browser     │  Storage Browser                          │
│ KBs         │                                           │
│ Projects    │  ┌─ knowledgebases/                       │
│ Obsidian    │  │  ├─ research/           [KB: research] │
│ Upload      │  │  │  ├─ sources/                        │
│ Sync        │  │  │  │  ├─ paper1.pdf    [12.4 MB]     │
│ Settings    │  │  │  │  ├─ paper2.pdf    [8.2 MB]      │
│             │  │  │  │  └─ notes.md      [24 KB]       │
│ ─────────── │  │  │  └─ exports/                        │
│ Storage:    │  │  └─ personal/           [KB: personal] │
│ /mnt/e      │  │     └─ ...                             │
│             │  │                                        │
│ KBs: 3      │  ├─ projects/                             │
│ Projects: 5 │  │  ├─ docling-test/       [Project]     │
│ Vaults: 2   │  │  └─ whisperx-batch/     [Project]     │
│ Files: 234  │  │                                        │
│             │  └─ obsidian/                             │
│             │     ├─ research-vault/     [Vault]       │
│             │     └─ personal-notes/     [Vault]       │
│             │                                           │
│             │  Selected: paper1.pdf                     │
│             │  ┌─────────────────────────────────────┐  │
│             │  │ Size: 12.4 MB │ Type: application/pdf│ │
│             │  │ KB: research  │ Added: 2025-01-20   │  │
│             │  │ Checksum: a3f2...                   │  │
│             │  │ [Open] [Move] [Copy] [Delete]       │  │
│             │  └─────────────────────────────────────┘  │
└─────────────┴───────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Knowledge Bases                                        │
├─────────────────────────────────────────────────────────┤
│  [+ Create New Knowledge Base]                          │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ research                                         │   │
│  │ Path: /mnt/e/knowledgebases/research            │   │
│  │ Files: 45 │ Size: 234 MB │ Created: 2024-06-15  │   │
│  │ Description: Academic papers and research notes │   │
│  │ [Browse] [Edit] [Export] [Delete]               │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ personal                                         │   │
│  │ Path: /mnt/e/knowledgebases/personal            │   │
│  │ Files: 123 │ Size: 1.2 GB │ Created: 2024-01-10 │   │
│  │ [Browse] [Edit] [Export] [Delete]               │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Projects (Standalone)                                  │
├─────────────────────────────────────────────────────────┤
│  [+ Create New Project]                                 │
│                                                         │
│  Projects are for standalone file processing.           │
│  Files here are NOT part of the knowledge graph.        │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ docling-batch-test                               │   │
│  │ Path: /mnt/e/projects/docling-batch-test        │   │
│  │ Files: 12 │ Created: 2025-01-24                 │   │
│  │ [Browse] [Open in Ingestion] [Delete]           │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Obsidian Vaults                                        │
├─────────────────────────────────────────────────────────┤
│  [+ Create New Vault]                                   │
│                                                         │
│  Obsidian vaults for exported notes and conversations.  │
│  Open directly in Obsidian app. Sync is one-way (app→). │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 📚 research-vault                                │   │
│  │ Path: /mnt/e/obsidian/research-vault            │   │
│  │ Notes: 45 │ Conversations: 12 │ Sources: 23     │   │
│  │ Last sync: 2025-01-25 14:30                     │   │
│  │ [Open in Obsidian] [Sync Now] [Browse] [Delete] │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ 📚 personal-notes                                │   │
│  │ Path: /mnt/e/obsidian/personal-notes            │   │
│  │ Notes: 89 │ Conversations: 34 │ Sources: 0      │   │
│  │ Last sync: 2025-01-25 10:15                     │   │
│  │ [Open in Obsidian] [Sync Now] [Browse] [Delete] │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ─────────────────────────────────────────────────────  │
│  Quick Export:                                          │
│  Export to vault: [research-vault ▼]                    │
│  • [ ] Selected notes from canvas                       │
│  • [ ] Recent conversations (last 7 days)               │
│  • [ ] Source summaries                                 │
│  [Export Selected]                                      │
└─────────────────────────────────────────────────────────┘
```

**Integration with Other Components**:

| Component | File Manager Integration |
|-----------|-------------------------|
| **Ingestion** | Gets file paths for processing, stores output to project/KB |
| **Embeddings** | Reads processed files via path |
| **Ontology Extraction** | Reads processed files via path |
| **Summarization** | Reads processed files via path |
| **Chat** | Exports conversations to Obsidian vaults |
| **Canvas** | Exports notes to Obsidian vaults, links to KB sources |
| **App-Main** | Uses file manager API for all file operations |

**Ingestion Integration Example**:
```
┌─────────────────────────────────────────────────────────┐
│  Ingestion Pipeline                    [Standalone Mode]│
├─────────────┬───────────────────────────────────────────┤
│ ...         │  Output Settings                          │
│             │                                           │
│             │  Storage Type:                            │
│             │  ( ) Knowledge Base  (•) Project          │
│             │                                           │
│             │  Project: [docling-batch-test ▼]          │
│             │           [+ Create New Project]          │
│             │                                           │
│             │  Output Directory:                        │
│             │  /mnt/e/projects/docling-batch-test/output│
│             │                                           │
│             │  [Process & Save to Project]              │
└─────────────┴───────────────────────────────────────────┘
```

---

### packages/surrealdb-service

**Purpose**: Database access layer with REST API and MCP server for LLM tool access.

**What belongs here**:
- [ ] SurrealDB async client wrapper
- [ ] Repository pattern implementation
- [ ] Database migrations
- [ ] FastAPI REST API for database operations
- [ ] MCP server for Claude/LLM tool access
- [ ] Admin Streamlit UI

**What does NOT belong here**:
- Business logic beyond CRUD
- Pipeline-specific queries (those go in pipelines)

**Source files to migrate**:
| Current Location | New Location | Notes |
|-----------------|--------------|-------|
| `open_notebook/database/repository.py` | `packages/surrealdb-service/src/surrealdb_service/repository.py` | |
| `open_notebook/database/async_migrate.py` | `packages/surrealdb-service/src/surrealdb_service/migrations.py` | |
| `migrations/*.surrealql` | `packages/surrealdb-service/migrations/` | |
| | | |

**Directory structure**:
```
packages/surrealdb-service/
├── pyproject.toml
├── README.md
├── main.py                 # FastAPI server entry point
├── ui.py                   # Streamlit admin UI entry point
├── migrations/             # SurrealQL migration files
│   ├── 01.surrealql
│   └── ...
└── src/surrealdb_service/
    ├── __init__.py
    ├── client.py           # AsyncSurreal wrapper
    ├── repository.py       # CRUD operations
    ├── migrations.py       # Migration runner
    ├── api/
    │   ├── __init__.py
    │   ├── app.py          # FastAPI app
    │   └── routers/
    │       ├── __init__.py
    │       ├── notebooks.py
    │       ├── sources.py
    │       ├── entities.py
    │       └── search.py
    └── mcp/
        ├── __init__.py
        └── server.py       # MCP server implementation
```

**API Endpoints** (customize as needed):
```yaml
# Core CRUD
POST   /api/v1/{table}              # Create record
GET    /api/v1/{table}/{id}         # Get record
PUT    /api/v1/{table}/{id}         # Update record
DELETE /api/v1/{table}/{id}         # Delete record
GET    /api/v1/{table}              # List records (paginated)

# Relationships
POST   /api/v1/relate               # Create relationship
GET    /api/v1/{table}/{id}/relations  # Get related records

# Search
POST   /api/v1/search/vector        # Vector similarity search
POST   /api/v1/search/fulltext      # Full-text search
POST   /api/v1/query                # Raw SurrealQL (admin only)
```

**MCP Tools** (for Claude access):
```yaml
tools:
  - name: query_database
    description: Execute a SurrealQL query

  - name: get_record
    description: Retrieve a record by ID

  - name: search_similar
    description: Find similar records by embedding

  - name: list_sources
    description: List all sources with optional filters

  - name: get_entity_graph
    description: Get entity and its relationships
```

#### Streamlit UI Specification: surrealdb-service

**Purpose**: **Database administration interface** - Inspect data, run queries, manage migrations, monitor health.

**Use Cases**:
- Browse and inspect database tables and records
- Run ad-hoc SurrealQL queries for debugging
- Check and run database migrations
- Monitor database health and performance
- Export data for backup or analysis

**Pages/Features**:
- [ ] **Dashboard**: Database stats, table counts, recent activity, health status
- [ ] **Table Browser**: Select table, view/filter/sort/search records
- [ ] **Record Inspector**: View single record with all fields and relationships
- [ ] **Query Console**: Execute raw SurrealQL queries with syntax highlighting
- [ ] **Migration Manager**: View applied migrations, run pending, rollback
- [ ] **Backup/Export**: Export tables or full database
- [ ] **Connection Settings**: Configure database connection

**Operational Modes**:
```yaml
local_mode:
  description: "Connect to local SurrealDB instance"
  default_url: "ws://localhost:8000/rpc"

docker_mode:
  description: "Connect to containerized SurrealDB"
  default_url: "ws://surrealdb:8000/rpc"

remote_mode:
  description: "Connect to remote SurrealDB"
  requires: "SURREAL_URL environment variable"
```

**UI Mockup**:
```
┌─────────────────────────────────────────────────────────┐
│  SurrealDB Admin                    [ws://localhost:8000]│
├─────────────┬───────────────────────────────────────────┤
│ Dashboard   │  Database Overview                        │
│ Tables      │  ┌─────────────────────────────────────┐  │
│ Query       │  │ Status: 🟢 Connected                │  │
│ Migrations  │  │ Tables: 15  │ Records: 12,450       │  │
│ Backup      │  │ Relations: 8,200 │ Size: 156 MB    │  │
│ Settings    │  │ Uptime: 3d 4h 12m                   │  │
│             │  └─────────────────────────────────────┘  │
│             │                                           │
│             │  Table Overview                           │
│             │  ┌──────────────────────────────────────┐ │
│             │  │ Table       │ Records │ Last Update │ │
│             │  ├─────────────┼─────────┼─────────────┤ │
│             │  │ source      │ 1,234   │ 2 min ago   │ │
│             │  │ chunk       │ 45,678  │ 2 min ago   │ │
│             │  │ entity      │ 8,901   │ 5 min ago   │ │
│             │  │ relation    │ 12,345  │ 5 min ago   │ │
│             │  └──────────────────────────────────────┘ │
│             │                                           │
│             │  Recent Activity                          │
│             │  ├─ Source created: "paper.pdf" (2m ago) │
│             │  ├─ 45 entities extracted (5m ago)       │
│             │  └─ Embedding batch completed (10m ago)  │
└─────────────┴───────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Query Console                                          │
├─────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────┐   │
│  │ SELECT * FROM source                            │   │
│  │ WHERE status = 'processed'                      │   │
│  │ ORDER BY created DESC                           │   │
│  │ LIMIT 10;                                       │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  [Run Query]  [Format]  [Clear]  [History ▼]           │
│                                                         │
│  Results (10 rows, 23ms):                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │ id          │ title           │ status     │   │   │
│  ├─────────────┼─────────────────┼────────────┤   │   │
│  │ source:abc  │ Research Paper  │ processed  │   │   │
│  │ source:def  │ Article.pdf     │ processed  │   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  [Export: JSON ▼]  [Copy]                              │
└─────────────────────────────────────────────────────────┘
```

---

### packages/llm-manager

**Purpose**: Centralized LLM model management - configure, select, and provide models (Claude API, Ollama) to all pipeline components. Allows assigning specific models to specific tasks.

**What belongs here**:
- [ ] Model registry (available models and their capabilities)
- [ ] Provider configuration (Claude API keys, Ollama URLs)
- [ ] Pipeline-to-model mapping (which model for which task)
- [ ] Model health checking and fallback logic
- [ ] Usage tracking and cost estimation
- [ ] REST API for model access
- [ ] Streamlit UI for configuration

**What does NOT belong here**:
- Actual LLM inference (pipelines call models directly via this manager's config)
- Embedding generation (that's embeddings pipeline, though it uses llm-manager for model selection)
- Prompt engineering (that's per-pipeline responsibility)

**Directory structure**:
```
packages/llm-manager/
├── pyproject.toml
├── README.md
├── main.py                 # API entry point
├── ui.py                   # Streamlit UI entry point
└── src/llm_manager/
    ├── __init__.py
    ├── config.py           # LLMConfig, ProviderConfig
    ├── registry/
    │   ├── __init__.py
    │   ├── models.py       # ModelInfo, ModelCapabilities
    │   ├── providers.py    # ProviderRegistry (Claude, Ollama)
    │   └── discovery.py    # Auto-discover Ollama models
    ├── mapping/
    │   ├── __init__.py
    │   ├── assignments.py  # PipelineModelAssignment
    │   ├── defaults.py     # Default model assignments
    │   └── storage.py      # Persist assignments to DB/file
    ├── clients/
    │   ├── __init__.py
    │   ├── base.py         # BaseLLMClient interface
    │   ├── claude.py       # ClaudeClient (anthropic SDK)
    │   ├── ollama.py       # OllamaClient (httpx)
    │   └── factory.py      # Get client for pipeline/task
    ├── health/
    │   ├── __init__.py
    │   ├── checker.py      # Model health checks
    │   └── fallback.py     # Fallback logic when model unavailable
    ├── tracking/
    │   ├── __init__.py
    │   ├── usage.py        # Token/request tracking
    │   └── costs.py        # Cost estimation (Claude API)
    ├── api/
    │   ├── __init__.py
    │   ├── app.py          # FastAPI app
    │   └── routers/
    │       ├── models.py   # /models endpoints
    │       ├── assignments.py # /assignments endpoints
    │       └── health.py   # /health endpoints
    └── cli.py              # CLI commands
```

**Core Data Models**:
```python
from pydantic import BaseModel
from enum import Enum
from typing import Optional

class ModelProvider(str, Enum):
    CLAUDE = "claude"
    OLLAMA = "ollama"
    OPENAI = "openai"  # Future

class ModelCapability(str, Enum):
    CHAT = "chat"
    COMPLETION = "completion"
    EMBEDDING = "embedding"
    VISION = "vision"
    FUNCTION_CALLING = "function_calling"

class ModelInfo(BaseModel):
    id: str                          # e.g., "claude-sonnet-4-20250514"
    provider: ModelProvider
    name: str                        # Display name
    capabilities: list[ModelCapability]
    context_window: int              # Max tokens
    cost_per_1k_input: Optional[float]   # For Claude
    cost_per_1k_output: Optional[float]  # For Claude
    is_available: bool = True
    is_local: bool = False           # True for Ollama

class PipelineTask(str, Enum):
    # Ontology Extraction
    ONTOLOGY_ENTITY_EXTRACTION = "ontology.entity_extraction"
    ONTOLOGY_RELATION_EXTRACTION = "ontology.relation_extraction"

    # Summarization
    SUMMARIZATION_CHUNK = "summarization.chunk"
    SUMMARIZATION_RAPTOR = "summarization.raptor"
    SUMMARIZATION_DOCUMENT = "summarization.document"

    # Embeddings
    EMBEDDING_CHUNK = "embedding.chunk"
    EMBEDDING_ENTITY = "embedding.entity"
    EMBEDDING_QUERY = "embedding.query"

    # Retrieval
    RETRIEVAL_RERANK = "retrieval.rerank"
    RETRIEVAL_QUERY_EXPANSION = "retrieval.query_expansion"

    # Chat Application
    CHAT_RAG = "chat.rag"               # RAG-based conversation
    CHAT_CONVERSATIONAL = "chat.conversational"  # Multi-turn conversation

    # Canvas Application
    CANVAS_NOTE_GENERATION = "canvas.note_generation"  # Generate notes from sources
    CANVAS_NOTE_EXPANSION = "canvas.note_expansion"    # Expand/elaborate notes
    CANVAS_SUGGESTIONS = "canvas.suggestions"          # Related content suggestions

    # General
    ANALYSIS = "analysis"

class ModelAssignment(BaseModel):
    task: PipelineTask
    primary_model: str               # Model ID
    fallback_model: Optional[str]    # Fallback if primary unavailable
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    custom_params: dict = {}
```

**Default Model Assignments**:
```yaml
# Default assignments (can be overridden via UI/API)
default_assignments:
  # Ontology Extraction - needs strong reasoning
  ontology.entity_extraction:
    primary: "qwen2.5:14b-instruct"  # Local Ollama
    fallback: "claude-sonnet-4-20250514"
    temperature: 0.1

  ontology.relation_extraction:
    primary: "qwen2.5:14b-instruct"
    fallback: "claude-sonnet-4-20250514"
    temperature: 0.1

  # Summarization - benefits from large context
  summarization.chunk:
    primary: "qwen2.5:7b-instruct"   # Faster for chunks
    fallback: "claude-haiku"
    temperature: 0.2

  summarization.raptor:
    primary: "qwen2.OK, 5:14b-instruct"
    fallback: "claude-sonnet-4-20250514"
    temperature: 0.2

  summarization.document:
    primary: "claude-sonnet-4-20250514"  # Best for long docs
    fallback: "qwen2.5:32b-instruct"
    temperature: 0.2

  # Embeddings
  embedding.chunk:
    primary: "nomic-embed-text"      # Ollama embedding model
    fallback: "text-embedding-3-small"

  embedding.query:
    primary: "nomic-embed-text"
    fallback: "text-embedding-3-small"

  # Retrieval
  retrieval.rerank:
    primary: "qwen2.5:7b-instruct"
    fallback: null
    temperature: 0.0

  # Chat Application
  chat.rag:
    primary: "qwen2.5:14b-instruct"   # Good balance of speed and quality
    fallback: "claude-sonnet-4-20250514"
    temperature: 0.7

  chat.conversational:
    primary: "qwen2.5:14b-instruct"
    fallback: "claude-sonnet-4-20250514"
    temperature: 0.8                   # Slightly higher for natural conversation

  # Canvas Application
  canvas.note_generation:
    primary: "qwen2.5:14b-instruct"
    fallback: "claude-sonnet-4-20250514"
    temperature: 0.5                   # Balanced creativity for note generation

  canvas.note_expansion:
    primary: "qwen2.5:7b-instruct"     # Faster for quick expansions
    fallback: "claude-haiku"
    temperature: 0.6

  canvas.suggestions:
    primary: "qwen2.5:7b-instruct"     # Fast for real-time suggestions
    fallback: null
    temperature: 0.3
```

**CLI Commands**:
```bash
# List available models
llm-manager models list
llm-manager models list --provider ollama
llm-manager models list --capability embedding

# Discover Ollama models (refresh from Ollama API)
llm-manager models discover

# Check model health
llm-manager health check
llm-manager health check --model qwen2.5:14b-instruct

# View current assignments
llm-manager assignments list
llm-manager assignments list --pipeline ontology-extraction

# Set model assignment
llm-manager assignments set ontology.entity_extraction qwen2.5:14b-instruct
llm-manager assignments set ontology.entity_extraction claude-sonnet-4-20250514 --fallback qwen2.5:14b-instruct

# Reset to defaults
llm-manager assignments reset
llm-manager assignments reset --task ontology.entity_extraction

# Test a model
llm-manager test --model qwen2.5:14b-instruct --prompt "Hello, world!"

# Show usage statistics
llm-manager usage show
llm-manager usage show --last 7d --by-task
```

**REST API Endpoints**:
```yaml
# Models
GET    /api/models                    # List all models
GET    /api/models/{model_id}         # Get model details
POST   /api/models/discover           # Discover Ollama models
GET    /api/models/{model_id}/health  # Check model health

# Assignments
GET    /api/assignments               # List all assignments
GET    /api/assignments/{task}        # Get assignment for task
PUT    /api/assignments/{task}        # Update assignment
DELETE /api/assignments/{task}        # Reset to default

# For pipelines to use
POST   /api/get-client                # Get configured client for task
  body: { "task": "ontology.entity_extraction" }
  response: { "provider": "ollama", "model": "qwen2.5:14b-instruct", "config": {...} }

# Health
GET    /api/health                    # Overall health
GET    /api/health/providers          # Provider-specific health

# Usage
GET    /api/usage                     # Usage statistics
GET    /api/usage/costs               # Cost breakdown (Claude)
```

**Dependencies**:
```yaml
core:
  - anthropic        # Claude API
  - httpx            # Ollama API calls
  - pydantic         # Data models
  - fastapi          # REST API
  - uvicorn          # API server

ui:
  - streamlit        # Configuration UI

optional:
  - openai           # Future OpenAI support
```

#### Streamlit UI Specification: llm-manager

**Purpose**: **Model configuration dashboard** - View available models, configure pipeline-to-model assignments, monitor health and usage.

**Pages/Features**:
- [ ] **Models**: View all available models (Claude + Ollama)
- [ ] **Assignments**: Configure which model handles which task
- [ ] **Health**: Monitor model availability and response times
- [ ] **Usage**: Track token usage and costs
- [ ] **Settings**: Configure API keys, Ollama URL, defaults

**UI Mockup**:
```
┌─────────────────────────────────────────────────────────┐
│  LLM Model Manager                        [Connected]   │
├─────────────┬───────────────────────────────────────────┤
│ Models      │  Available Models                         │
│ Assignments │                                           │
│ Health      │  Claude API Models:                       │
│ Usage       │  ┌─────────────────────────────────────┐  │
│ Settings    │  │ ✅ claude-sonnet-4-20250514         │  │
│             │  │    Context: 200K │ Vision: ✓        │  │
│ ─────────── │  │    Cost: $3/$15 per 1M tokens      │  │
│ Claude:     │  ├─────────────────────────────────────┤  │
│ ✅ Connected│  │ ✅ claude-haiku-3-5                  │  │
│             │  │    Context: 200K │ Fast             │  │
│ Ollama:     │  │    Cost: $0.25/$1.25 per 1M tokens │  │
│ ✅ Connected│  └─────────────────────────────────────┘  │
│ 5 models    │                                           │
│             │  Ollama Models (local):                   │
│             │  ┌─────────────────────────────────────┐  │
│             │  │ ✅ qwen2.5:14b-instruct             │  │
│             │  │    Context: 32K │ 14B params        │  │
│             │  ├─────────────────────────────────────┤  │
│             │  │ ✅ qwen2.5:7b-instruct              │  │
│             │  │    Context: 32K │ 7B params │ Fast  │  │
│             │  ├─────────────────────────────────────┤  │
│             │  │ ✅ nomic-embed-text                 │  │
│             │  │    Embedding model │ 768 dims       │  │
│             │  └─────────────────────────────────────┘  │
│             │                                           │
│             │  [Refresh Ollama Models]  [Test Model]   │
└─────────────┴───────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Pipeline Model Assignments                             │
├─────────────────────────────────────────────────────────┤
│  Filter: [All Pipelines ▼]                             │
│                                                         │
│  Ontology Extraction                                   │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Task                    │ Primary    │ Fallback │   │
│  ├─────────────────────────┼────────────┼──────────┤   │
│  │ Entity Extraction       │ qwen2.5:14b│ claude-4 │   │
│  │ Relation Extraction     │ qwen2.5:14b│ claude-4 │   │
│  └─────────────────────────────────────────────────┘   │
│  [Edit Assignments]                                    │
│                                                         │
│  Summarization                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Task                    │ Primary    │ Fallback │   │
│  ├─────────────────────────┼────────────┼──────────┤   │
│  │ Chunk Summary           │ qwen2.5:7b │ haiku    │   │
│  │ RAPTOR Tree             │ qwen2.5:14b│ claude-4 │   │
│  │ Document Summary        │ claude-4   │ qwen:32b │   │
│  └─────────────────────────────────────────────────┘   │
│  [Edit Assignments]                                    │
│                                                         │
│  Embeddings                                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Task                    │ Primary    │ Fallback │   │
│  ├─────────────────────────┼────────────┼──────────┤   │
│  │ Chunk Embedding         │ nomic-embed│ ada-002  │   │
│  │ Query Embedding         │ nomic-embed│ ada-002  │   │
│  └─────────────────────────────────────────────────┘   │
│  [Edit Assignments]                                    │
│                                                         │
│  [Reset All to Defaults]  [Export Config]  [Import]   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Edit Assignment: ontology.entity_extraction           │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Primary Model:                                        │
│  [qwen2.5:14b-instruct ▼]                              │
│                                                         │
│  Fallback Model (when primary unavailable):            │
│  [claude-sonnet-4-20250514 ▼]                          │
│                                                         │
│  Parameters:                                           │
│  Temperature: [0.1____]  (0.0 = deterministic)        │
│  Max Tokens:  [4096___]  (leave empty for default)    │
│                                                         │
│  Custom Parameters (JSON):                             │
│  ┌─────────────────────────────────────────────────┐   │
│  │ {                                               │   │
│  │   "top_p": 0.9,                                 │   │
│  │   "num_ctx": 16384                              │   │
│  │ }                                               │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  [Test with Sample]  [Save]  [Cancel]                  │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Model Health                                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Provider Status:                                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Claude API      │ ✅ Connected │ Latency: 245ms │   │
│  │ Ollama (local)  │ ✅ Running   │ Latency: 12ms  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Model Health Checks (last 5 min):                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Model              │ Status │ Avg Latency │ Err │   │
│  ├────────────────────┼────────┼─────────────┼─────┤   │
│  │ qwen2.5:14b        │ ✅     │ 1.2s        │ 0%  │   │
│  │ qwen2.5:7b         │ ✅     │ 0.4s        │ 0%  │   │
│  │ nomic-embed-text   │ ✅     │ 0.05s       │ 0%  │   │
│  │ claude-sonnet-4    │ ✅     │ 0.8s        │ 0%  │   │
│  │ claude-haiku       │ ✅     │ 0.3s        │ 0%  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  [Run Health Check]  [Auto-refresh: 5min ▼]           │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Usage Statistics                       [Last 7 days]  │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Token Usage by Provider:                              │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Ollama (local):   2.4M tokens   │ Cost: $0      │   │
│  │ Claude API:       156K tokens   │ Cost: $2.34   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Usage by Task:                                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Task                    │ Tokens  │ Requests    │   │
│  ├─────────────────────────┼─────────┼─────────────┤   │
│  │ ontology.entity_ext     │ 1.2M    │ 3,456       │   │
│  │ summarization.chunk     │ 890K    │ 2,100       │   │
│  │ embedding.chunk         │ 450K    │ 12,000      │   │
│  │ summarization.raptor    │ 120K    │ 89          │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Estimated Monthly Cost: $12.50 (at current rate)     │
│                                                         │
│  [Export Report]  [Clear Statistics]                   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Settings                                              │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Claude API:                                           │
│  API Key: [sk-ant-...••••••••••••] [Show] [Test]      │
│  Status: ✅ Valid                                      │
│                                                         │
│  Ollama:                                               │
│  URL: [http://localhost:11434_______] [Test]          │
│  Status: ✅ Connected (5 models available)            │
│  Auto-discover models: [x]                             │
│                                                         │
│  Defaults:                                             │
│  Default Chat Model: [qwen2.5:14b-instruct ▼]         │
│  Default Embed Model: [nomic-embed-text ▼]            │
│                                                         │
│  Fallback Behavior:                                    │
│  (•) Use fallback model when primary unavailable      │
│  ( ) Fail immediately (no fallback)                   │
│  ( ) Queue and retry later                            │
│                                                         │
│  Persistence:                                          │
│  (•) Store assignments in database                    │
│  ( ) Store assignments in local file                  │
│                                                         │
│  [Save Settings]  [Export All Config]                  │
└─────────────────────────────────────────────────────────┘
```

**Integration Example** (how pipelines use llm-manager):
```python
# In ontology-extraction pipeline
from llm_manager import get_model_client, PipelineTask

async def extract_entities(text: str) -> dict:
    # Get the configured client for this task
    client = await get_model_client(PipelineTask.ONTOLOGY_ENTITY_EXTRACTION)

    # Client is pre-configured with the right model, temperature, etc.
    response = await client.chat([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text}
    ])

    return parse_response(response)

# Or use the REST API
async def extract_entities_via_api(text: str) -> dict:
    async with httpx.AsyncClient() as http:
        # Get model config
        config_resp = await http.post(
            "http://llm-manager:5120/api/get-client",
            json={"task": "ontology.entity_extraction"}
        )
        config = config_resp.json()

        # Use the configured model
        if config["provider"] == "ollama":
            # Call Ollama with config
            ...
        elif config["provider"] == "claude":
            # Call Claude with config
            ...
```

---

## Pipeline Specifications

### pipelines/web-scraper

**Purpose**: Website scraping and content download - crawl websites, download pages and assets, store as project files for further processing.

**What belongs here**:
- [ ] Single URL scraping
- [ ] Batch URL scraping (from file)
- [ ] Recursive crawling (follow links within domain)
- [ ] Content type selection (HTML, images, PDFs, videos, etc.)
- [ ] Asset downloading (CSS, JS, images, documents)
- [ ] Rate limiting and polite crawling
- [ ] Robots.txt compliance (optional)
- [ ] Authentication support (basic auth, cookies)
- [ ] JavaScript rendering (via Playwright)

**What does NOT belong here**:
- Content parsing/chunking (that's ingestion pipeline)
- Text extraction from HTML (that's ingestion pipeline)
- Embedding generation (that's embeddings pipeline)

**Output Structure** (stored via file-manager):
```
{STORAGE_ROOT}/projects/{project_name}/
├── manifest.json                    # Scrape metadata and URL mapping
├── {domain_1}/                      # Subdirectory per domain
│   ├── index.html                   # Homepage
│   ├── about/
│   │   └── index.html
│   ├── blog/
│   │   ├── post-1.html
│   │   └── post-2.html
│   └── assets/
│       ├── images/
│       │   ├── logo.png
│       │   └── hero.jpg
│       ├── documents/
│       │   └── whitepaper.pdf
│       └── styles/
│           └── main.css
├── {domain_2}/
│   └── ...
└── _metadata/
    ├── urls.json                    # All discovered URLs
    ├── errors.json                  # Failed URLs with errors
    └── stats.json                   # Scraping statistics
```

**Manifest Schema**:
```json
{
  "project": "research-websites",
  "created": "2025-01-25T10:30:00Z",
  "config": {
    "max_depth": 3,
    "content_types": ["html", "pdf", "images"],
    "follow_external": false
  },
  "sources": [
    {
      "url": "https://example.com",
      "domain": "example.com",
      "pages_scraped": 45,
      "assets_downloaded": 120,
      "total_size_mb": 234.5,
      "started": "2025-01-25T10:30:00Z",
      "completed": "2025-01-25T10:45:32Z"
    }
  ]
}
```

**Directory structure**:
```
pipelines/web-scraper/
├── pyproject.toml
├── README.md
├── main.py                 # CLI entry point
├── ui.py                   # Streamlit UI entry point
└── src/web_scraper/
    ├── __init__.py
    ├── config.py           # Scraper configuration
    ├── crawler.py          # Core crawling logic
    ├── downloader.py       # Asset downloading
    ├── parser.py           # URL/link extraction from HTML
    ├── storage.py          # File-manager integration
    ├── filters.py          # URL/content filtering
    ├── rate_limiter.py     # Polite crawling
    ├── auth.py             # Authentication handling
    ├── js_renderer.py      # Playwright integration for JS sites
    └── cli.py              # CLI commands
```

**CLI Commands**:
```bash
# Scrape a single URL (page only)
web-scraper scrape https://example.com --project my-project

# Scrape with depth (crawl linked pages)
web-scraper crawl https://example.com --depth 3 --project research

# Scrape from URL list file
web-scraper batch --urls ./urls.txt --project batch-scrape

# Scrape with specific content types
web-scraper crawl https://example.com \
  --content-types html,pdf,images \
  --project docs-collection

# Scrape JavaScript-rendered site
web-scraper crawl https://spa-site.com \
  --render-js \
  --wait 3000 \
  --project spa-content

# Resume interrupted scrape
web-scraper resume --project my-project

# Export scraped URLs for ingestion
web-scraper export --project my-project --format file-list
```

**Dependencies**:
```yaml
core:
  - httpx           # Async HTTP client
  - beautifulsoup4  # HTML parsing
  - lxml            # Fast HTML/XML parser

optional:
  - playwright      # JavaScript rendering
  - aiofiles        # Async file operations

rate_limiting:
  - aiolimiter      # Async rate limiting
```

#### Streamlit UI Specification: web-scraper

**Purpose**: **Standalone web scraping tool** - Scrape websites, download content, store as project files for later processing with ingestion pipeline.

**Use Cases**:
- Scrape a website for offline reading/analysis
- Download all PDFs/documents from a site
- Archive a website before it changes
- Collect training data from multiple sites
- Download images/media from galleries
- Prepare content for ingestion into knowledge base

**Pages/Features**:
- [ ] **Single URL**: Enter URL, configure options, scrape
- [ ] **Batch URLs**: Upload file or paste list of URLs
- [ ] **Crawl Settings**: Depth, filters, content types
- [ ] **Progress Monitor**: Real-time scraping progress
- [ ] **Results Browser**: Browse scraped content
- [ ] **Export to Ingestion**: Send to ingestion pipeline

**Operational Modes**:
```yaml
standalone_mode:
  description: "Scrape and store locally"
  requires_database: false
  requires_file_manager: true  # Always needs file-manager
  output: "Project directory with scraped content"

connected_mode:
  description: "Scrape and track in database"
  requires_database: true
  requires_file_manager: true
  output: "Project directory + database tracking"
```

**Content Type Options**:
```yaml
content_types:
  pages:
    - html          # Web pages
    - xml           # XML files, sitemaps

  documents:
    - pdf           # PDF documents
    - doc/docx      # Word documents
    - xls/xlsx      # Excel spreadsheets
    - ppt/pptx      # PowerPoint
    - txt           # Plain text files

  media:
    - images        # jpg, png, gif, webp, svg
    - video         # mp4, webm, etc.
    - audio         # mp3, wav, etc.

  code:
    - css           # Stylesheets
    - js            # JavaScript files
    - json          # JSON data files

  data:
    - csv           # CSV files
    - xml           # Data XML
    - rss           # RSS/Atom feeds
```

**UI Mockup**:
```
┌─────────────────────────────────────────────────────────┐
│  Web Scraper                           [Standalone Mode]│
├─────────────┬───────────────────────────────────────────┤
│ Single URL  │  ┌─────────────────────────────────────┐  │
│ Batch URLs  │  │ [Single URL]  [Batch URLs]  [Sitemap]│ │
│ Progress    │  └─────────────────────────────────────┘  │
│ Results     │                                           │
│ Settings    │  Scrape Website                           │
│             │                                           │
│ ─────────── │  URL: [https://example.com___________]   │
│ Project:    │                                           │
│ [new... ▼]  │  Crawl Options:                           │
│             │  Depth: [3▼]  (0 = this page only)       │
│ File Mgr:   │  [ ] Follow external links                │
│ 🟢 Connected│  [x] Respect robots.txt                   │
│             │  Rate limit: [2▼] requests/second         │
│             │                                           │
│             │  Content to Download:                     │
│             │  [x] HTML pages                           │
│             │  [x] PDF documents                        │
│             │  [x] Images (jpg, png, webp)             │
│             │  [ ] Videos                               │
│             │  [ ] Other documents (doc, xls, ppt)     │
│             │  [ ] Source files (css, js)              │
│             │                                           │
│             │  JavaScript Rendering:                    │
│             │  [ ] Enable (slower, for SPAs)           │
│             │      Wait time: [3000▼] ms               │
│             │                                           │
│             │  [Start Scraping]                         │
└─────────────┴───────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Web Scraper - Batch URLs              [Standalone Mode]│
├─────────────┬───────────────────────────────────────────┤
│ Single URL  │  Batch URL Scraping                       │
│ Batch URLs  │                                           │
│ Progress    │  Upload URL list:                         │
│ Results     │  ┌─────────────────────────────────────┐  │
│ Settings    │  │  [Drag & Drop .txt file here]       │  │
│             │  │      or click to browse             │  │
│             │  └─────────────────────────────────────┘  │
│             │                                           │
│             │  Or paste URLs (one per line):           │
│             │  ┌─────────────────────────────────────┐  │
│             │  │ https://site1.com                   │  │
│             │  │ https://site2.com/page              │  │
│             │  │ https://site3.com/docs              │  │
│             │  │                                     │  │
│             │  └─────────────────────────────────────┘  │
│             │                                           │
│             │  URLs loaded: 3                           │
│             │  Domains: site1.com, site2.com, site3.com│
│             │                                           │
│             │  [Validate URLs]  [Start Batch Scrape]   │
└─────────────┴───────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Scraping Progress                      [Running 2:34] │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Overall Progress                                       │
│  ████████████████░░░░░░░░░░░░░░ 54% (245/450 pages)   │
│                                                         │
│  Current: https://example.com/blog/post-42             │
│                                                         │
│  Statistics:                                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Pages scraped:    245    │ Errors:      3       │   │
│  │ PDFs downloaded:  12     │ Skipped:     28      │   │
│  │ Images:           89     │ Queue:       177     │   │
│  │ Total size:       156 MB │ Rate:        1.5/s   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Recent Activity:                                       │
│  ├─ ✅ /blog/post-41.html (234 KB)                    │
│  ├─ ✅ /images/chart.png (45 KB)                      │
│  ├─ ⚠️ /private/admin (403 Forbidden)                 │
│  └─ 🔄 /blog/post-42.html (downloading...)            │
│                                                         │
│  [Pause]  [Stop]  [View Errors]                        │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Scrape Results                         [Completed]    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  Project: research-websites                             │
│  Location: /mnt/e/projects/research-websites           │
│                                                         │
│  Summary:                                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Total pages:      450   │ Total size:  523 MB   │   │
│  │ HTML files:       398   │ Duration:    12m 34s  │   │
│  │ PDF documents:    23    │ Errors:      8        │   │
│  │ Images:           156   │                       │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Scraped Domains:                                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │ example.com          │ 312 pages │ 234 MB      │   │
│  │ docs.example.com     │ 138 pages │ 289 MB      │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Browse Content:                                        │
│  ┌─ example.com/                                       │
│  │  ├─ index.html                                      │
│  │  ├─ about/                                          │
│  │  ├─ blog/ (45 files)                               │
│  │  └─ assets/ (89 files)                             │
│  └─ docs.example.com/                                  │
│     └─ ...                                             │
│                                                         │
│  [Open in File Manager]  [Send to Ingestion Pipeline] │
│  [Export URL List]       [Re-scrape Failed]           │
└─────────────────────────────────────────────────────────┘
```

**Integration with Other Components**:

| Component | Integration |
|-----------|-------------|
| **File Manager** | All scraped content stored via file-manager API |
| **Ingestion** | "Send to Ingestion" button processes scraped HTML/PDFs |
| **Embeddings** | After ingestion, content can be embedded |
| **Entity Extraction** | After ingestion, extract entities from scraped content |

**Workflow Example**:
```
1. Web Scraper: Scrape https://research-site.com → Store in project "research"
2. File Manager: Browse scraped content, verify completeness
3. Ingestion: Process scraped PDFs and HTML pages → Create chunks
4. Embeddings: Generate vectors for chunks
5. Entity Extraction: Extract entities and relationships
6. (Optional) Move from project to knowledge base via file-manager
```

---

### pipelines/ingestion

**Purpose**: Document ingestion, content extraction, chunking, and audio/video transcription.

**What belongs here**:
- [ ] Document loading (PDF, URL, text, etc.)
- [ ] Content extraction via Docling
- [ ] **Audio/Video transcription via WhisperX**
  - Transcribe audio/video files to text
  - Speaker diarization support
  - Word-level timestamps for precise chunking
  - Store transcription only (not original audio/video files)
- [ ] Chunk generation with metadata
- [ ] File management (uploads, exports)

**WhisperX Integration Notes**:
```yaml
source_code: /mnt/e/repos/public/whisperx
known_issues:
  - cuDNN not found error (embedded in PyTorch but not visible)
  - Need to make CUDA/cuDNN packages explicitly visible in pyproject.toml
requirements:
  - GPU support (CUDA)
  - Explicit cuDNN dependency declaration
  - PyTorch with CUDA support
output:
  - Transcription text (stored)
  - Speaker segments (if diarization enabled)
  - Timestamps per segment
  - Original audio/video NOT stored
```

**What does NOT belong here**:
- Embedding generation (use embeddings pipeline)
- Entity extraction (use ontology-extraction pipeline)

**Source files to migrate**:
| Current Location | New Location | Notes |
|-----------------|--------------|-------|
| `open_notebook/processors/chunk_extractor.py` | `pipelines/ingestion/src/ingestion/chunk_extractor.py` | |
| `open_notebook/graphs/source.py` (partial) | `pipelines/ingestion/src/ingestion/workflow.py` | Extract content processing only |
| | | |

**Directory structure**:
```
pipelines/ingestion/
├── pyproject.toml
├── README.md
├── main.py                 # CLI entry point
├── ui.py                   # Streamlit UI entry point
└── src/ingestion/
    ├── __init__.py
    ├── loaders/
    │   ├── __init__.py
    │   ├── pdf.py          # PDF loading via Docling
    │   ├── url.py          # URL content extraction
    │   └── text.py         # Plain text handling
    ├── chunk_extractor.py  # Chunking logic
    ├── workflow.py         # LangGraph workflow (optional)
    └── cli.py              # CLI commands
```

**CLI Commands**:
```bash
# Ingest a single document
ingestion ingest --source-type pdf --file ./document.pdf --output-dir ./output

# Batch ingest from directory
ingestion batch --input-dir ./documents --pattern "*.pdf"

# Re-chunk existing source
ingestion rechunk --source-id source:abc123 --chunk-size 500
```

#### Streamlit UI Specification: ingestion

**Purpose**: **Standalone document processing** - Extract and chunk documents independently without going through the full application pipeline.

**Use Cases**:
- Process PDFs with Docling to inspect chunk quality before importing
- Extract content from URLs for review
- **Transcribe audio/video files** (podcasts, interviews, lectures)
- Batch process a folder of documents
- Export chunks to JSON/CSV for use elsewhere
- Test different chunking parameters

**Pages/Features**:
- [ ] **Process Document**: Upload/URL/text input with full processing options
- [ ] **Transcribe Audio/Video**: Upload audio/video, transcribe with WhisperX
  - Speaker diarization toggle
  - Language selection
  - Model size selection (tiny/base/small/medium/large)
  - View transcription with timestamps
- [ ] **Chunk Viewer**: Inspect generated chunks with metadata, spatial positions, hierarchy
- [ ] **Batch Processing**: Process multiple files, monitor progress
- [ ] **Export**: Download chunks as JSON, CSV, or Markdown
- [ ] **Settings**: Configure Docling options, chunk sizes, overlap, WhisperX model
- [ ] **Database Integration** (optional): Push results to SurrealDB if connected

**Operational Modes**:
```yaml
standalone_mode:
  description: "Process files locally, export results"
  requires_database: false
  output: "JSON/CSV/Markdown files"

connected_mode:
  description: "Process files and save to database"
  requires_database: true
  output: "Database records + local export"
```

**UI Mockup**:
```
┌─────────────────────────────────────────────────────────┐
│  Ingestion Pipeline                    [Standalone Mode]│
├─────────────┬───────────────────────────────────────────┤
│ Documents   │  ┌─────────────────────────────────────┐  │
│ Audio/Video │  │ [Documents]  [Audio/Video]  [URL]   │  │
│ Chunks      │  └─────────────────────────────────────┘  │
│ Batch       │                                           │
│ Export      │  Process Document                         │
│ Settings    │  ┌─────────────────────────────────────┐  │
│             │  │  [Drag & Drop PDF/DOCX/TXT here]    │  │
│ ─────────── │  │         or click to browse          │  │
│ DB Status:  │  └─────────────────────────────────────┘  │
│ ⚪ Offline  │                                           │
│             │  Processing Options:                      │
│             │  [x] Extract tables  [x] Extract figures  │
│             │  [x] OCR fallback    [ ] GPU acceleration │
│             │  Chunk size: [500▼]  Overlap: [50▼]      │
│             │                                           │
│             │  Output:                                  │
│             │  (•) Local only  ( ) Save to database    │
│             │                                           │
│             │  [Process Document]  [Process & Export]   │
└─────────────┴───────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Ingestion Pipeline                    [Standalone Mode]│
├─────────────┬───────────────────────────────────────────┤
│ Documents   │  ┌─────────────────────────────────────┐  │
│ Audio/Video │  │ [Documents]  [Audio/Video]  [URL]   │  │
│ Chunks      │  └─────────────────────────────────────┘  │
│ Batch       │                                           │
│ Export      │  Transcribe Audio/Video (WhisperX)       │
│ Settings    │  ┌─────────────────────────────────────┐  │
│             │  │  [Drag & Drop MP3/MP4/WAV here]     │  │
│ ─────────── │  │         or click to browse          │  │
│ DB Status:  │  └─────────────────────────────────────┘  │
│ ⚪ Offline  │                                           │
│             │  WhisperX Options:                        │
│ GPU: 🟢     │  Model: [large-v3 ▼]  Language: [auto ▼] │
│             │  [x] Speaker diarization                  │
│             │  [ ] Word-level timestamps                │
│             │                                           │
│             │  Output:                                  │
│             │  (•) Local only  ( ) Save to database    │
│             │                                           │
│             │  [Transcribe]  [Transcribe & Export]     │
└─────────────┴───────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Transcription Result                   [00:45:23 total]│
├─────────────────────────────────────────────────────────┤
│  Speakers detected: 2 (Speaker 1, Speaker 2)           │
│                                                         │
│  ┌─ Segment 1 ───────────────────────────────────────┐  │
│  │ [00:00:00 - 00:02:34] Speaker 1                   │  │
│  │ "Welcome to today's discussion about machine      │  │
│  │  learning applications in healthcare..."          │  │
│  ├───────────────────────────────────────────────────┤  │
│  │ [00:02:35 - 00:05:12] Speaker 2                   │  │
│  │ "Thank you for having me. I'd like to start      │  │
│  │  by explaining the fundamentals..."               │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  [Chunk by Speaker] [Chunk by Time] [Export: SRT ▼]    │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Chunk Viewer                              [12 chunks]  │
├─────────────────────────────────────────────────────────┤
│  ┌─ Chunk 1 ─────────────────────────────────────────┐  │
│  │ Page: 1  │  Section: Introduction  │  Order: 1    │  │
│  │ Type: paragraph  │  Tokens: 234                   │  │
│  ├───────────────────────────────────────────────────┤  │
│  │ "This paper presents a novel approach to..."      │  │
│  │                                                   │  │
│  │ [View Spatial] [View Metadata] [Copy]            │  │
│  └───────────────────────────────────────────────────┘  │
│                                                         │
│  [Export All: JSON ▼]  [Copy All to Clipboard]         │
└─────────────────────────────────────────────────────────┘
```

---

### pipelines/ontology-extraction

**Purpose**: Ontology-guided knowledge extraction using TTL/OWL ontologies with multi-pass processing, namespace partitioning, and token budget management.

**Key Differentiator**: Unlike generic NER or OpenIE approaches, this pipeline uses **domain-specific ontologies** to extract structured knowledge that fits a predefined schema. Designed for batch processing of document collections with page-level extraction.

**What belongs here**:
- [ ] TTL/OWL ontology parsing (classes, properties, namespaces)
- [ ] Automatic namespace partitioning and clustering
- [ ] Token budget-aware pass generation
- [ ] Multi-pass extraction (parallel independent passes, then dependent)
- [ ] Page-by-page extraction with deduplication
- [ ] Relationship extraction with known entity context
- [ ] Multi-LLM support (Claude, Ollama, OpenAI)
- [ ] Batch document processing with progress tracking
- [ ] Structured output matching ontology schema

**What does NOT belong here**:
- Document parsing (that's ingestion pipeline)
- Embedding generation (that's embeddings pipeline)

**Source files to migrate**:
| Current Location | New Location | Notes |
|-----------------|--------------|-------|
| `scripts/ontologie_extract_multipass.py` | `pipelines/ontology-extraction/src/ontology_extraction/multipass.py` | Core multi-pass extraction |
| `scripts/ontologie_extract_v3.py` | `pipelines/ontology-extraction/src/ontology_extraction/page_extractor.py` | Page-by-page extraction |
| `scripts/batch_parse_with_pages.py` | Reference for input format | Already in ingestion |
| | | |

**Directory structure**:
```
pipelines/ontology-extraction/
├── pyproject.toml
├── README.md
├── main.py                 # CLI entry point
├── ui.py                   # Streamlit UI entry point
├── ontologies/             # Bundled ontology files
│   ├── default.ttl         # Default general ontology
│   └── examples/           # Example domain ontologies
└── src/ontology_extraction/
    ├── __init__.py
    ├── config.py           # ExtractionConfig, API settings
    ├── ontology/
    │   ├── __init__.py
    │   ├── parser.py       # TTL/OWL parser (OntologyParser)
    │   ├── models.py       # OntologyClass, OntologyProperty, NamespaceCluster
    │   └── loader.py       # Load ontology from file/URL
    ├── passes/
    │   ├── __init__.py
    │   ├── generator.py    # PassGenerator - token budget aware
    │   ├── executor.py     # Run passes (parallel/sequential)
    │   └── models.py       # ExtractionPass dataclass
    ├── extractors/
    │   ├── __init__.py
    │   ├── base.py         # BaseExtractor interface
    │   ├── claude.py       # Claude API extractor
    │   ├── ollama.py       # Ollama extractor
    │   └── openai.py       # OpenAI extractor
    ├── processing/
    │   ├── __init__.py
    │   ├── page_processor.py   # Page-by-page extraction
    │   ├── document_processor.py # Full document processing
    │   ├── batch_processor.py  # Batch document processing
    │   └── merger.py       # Merge and deduplicate extractions
    ├── output/
    │   ├── __init__.py
    │   ├── schemas.py      # Output JSON schemas
    │   └── exporters.py    # Export to various formats
    └── cli.py              # CLI commands
```

**Ontology Parser Features**:
```python
# Namespace mappings (example from Dutch policy domain)
NAMESPACE_INFO = {
    "bw:": {
        "name": "Brede Welvaart",
        "description": "Welvaart dimensies, kapitaalvormen, welzijnsthema's",
        "priority": 1
    },
    "geo:": {
        "name": "Geografisch",
        "description": "Gemeenten, provincies, regio's, wijken",
        "priority": 2
    },
    "org:": {
        "name": "Organisaties",
        "description": "Overheden, bedrijven, kennisinstellingen",
        "priority": 2
    },
    "fin:": {
        "name": "Financiering",
        "description": "Geldstromen, subsidies, Europese programma's",
        "priority": 3
    },
    # ... more namespaces
}
```

**CLI Commands**:
```bash
# Extract from single document using ontology
ontology-extraction extract \
  --input ./document_elements.json \
  --ontology ./my_ontology.ttl \
  --output ./extracted.json

# Batch extract from directory
ontology-extraction batch \
  --input-dir ./parsed_documents \
  --ontology ./domain_ontology.ttl \
  --output-dir ./extractions \
  --parallel 3

# Extract with specific LLM
ontology-extraction extract \
  --input ./elements.json \
  --ontology ./ontology.ttl \
  --provider claude \
  --model claude-sonnet-4-20250514

# Analyze ontology structure
ontology-extraction analyze-ontology \
  --ontology ./ontology.ttl \
  --show-namespaces \
  --show-passes

# Merge multiple extraction results
ontology-extraction merge \
  --inputs ./extractions/*.json \
  --output ./merged.json \
  --deduplicate

# Resume interrupted batch processing
ontology-extraction resume \
  --progress-file ./progress.json
```

**Dependencies**:
```yaml
core:
  - rdflib           # TTL/OWL parsing
  - httpx            # Async HTTP for LLM APIs
  - pydantic         # Schema validation

llm_providers:
  - anthropic        # Claude API
  - openai           # OpenAI API (optional)
  # Ollama uses httpx directly

processing:
  - aiofiles         # Async file operations
  - tqdm             # Progress bars
```

**Configuration Options**:
```yaml
extraction_config:
  # Token budget settings
  max_tokens_per_pass: 3000     # Max tokens for schema in prompt
  chars_per_token: 4            # Estimate for text
  max_tokens_response: 4096     # Max response length

  # Parallel processing
  page_batch_size: 3            # Pages to process concurrently
  pass_parallelism: true        # Run independent passes in parallel

  # LLM settings
  provider: "ollama"            # claude, ollama, openai
  model: "qwen2.5:14b-instruct" # Model name
  temperature: 0.1              # Low for consistent extraction
  timeout: 180                  # Seconds per request

  # Output settings
  deduplicate: true             # Merge duplicate entities
  include_confidence: true      # Include confidence scores
  include_evidence: true        # Include source text evidence
```

#### Streamlit UI Specification: ontology-extraction

**Purpose**: **Standalone ontology-guided extraction tool** - Extract structured knowledge from documents using domain ontologies, batch process document collections, review and export results.

**Use Cases**:
- Extract domain-specific entities from policy documents
- Process batch of PDFs with consistent ontology schema
- Test and refine ontology configurations
- Review extraction quality per page/document
- Export structured data for knowledge graphs
- Compare extraction results across different ontologies

**Pages/Features**:
- [ ] **Extract**: Single document extraction with ontology selection
- [ ] **Batch Process**: Batch extraction with progress tracking
- [ ] **Ontology Manager**: Load, view, edit ontology files
- [ ] **Pass Analyzer**: View generated passes and token budgets
- [ ] **Results Browser**: Browse extracted entities/relations by document/page
- [ ] **Quality Review**: Review and correct extractions
- [ ] **Export**: Export to JSON, CSV, RDF, Neo4j format
- [ ] **Settings**: Configure LLM, token budgets, parallelism

**Operational Modes**:
```yaml
standalone_mode:
  description: "Extract from local files, export results"
  requires_database: false
  input: "JSON elements files from ingestion pipeline"
  output: "JSON extraction files, CSV, RDF"

connected_mode:
  description: "Extract and store in knowledge graph"
  requires_database: true
  input: "Source records from database"
  output: "Entity/relation records in database"
```

**UI Mockup**:
```
┌─────────────────────────────────────────────────────────┐
│  Ontology Extraction                   [Standalone Mode]│
├─────────────┬───────────────────────────────────────────┤
│ Extract     │  Single Document Extraction               │
│ Batch       │                                           │
│ Ontology    │  Input (from ingestion pipeline):         │
│ Passes      │  ┌─────────────────────────────────────┐  │
│ Results     │  │ [Select elements.json file...]      │  │
│ Export      │  └─────────────────────────────────────┘  │
│ Settings    │                                           │
│             │  Ontology:                                │
│ ─────────── │  [brede_welvaart_v32.ttl ▼] [Upload...]  │
│ Provider:   │                                           │
│ Ollama ▼    │  Ontology Summary:                        │
│             │  ┌─────────────────────────────────────┐  │
│ Model:      │  │ Namespaces: 9                       │  │
│ qwen2.5 ▼   │  │ Classes: 87  Properties: 124       │  │
│             │  │ Passes: 4 (~2,800 tokens/pass)     │  │
│ DB Status:  │  └─────────────────────────────────────┘  │
│ ⚪ Offline  │                                           │
│             │  Processing:                              │
│             │  Page batch size: [3▼]                    │
│             │  [x] Parallel passes  [x] Deduplicate    │
│             │                                           │
│             │  [Analyze Passes]  [Extract]              │
└─────────────┴───────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Batch Processing                        [Running 4:23] │
├─────────────────────────────────────────────────────────┤
│  Input Directory: /mnt/e/Regiodeals/parse/output       │
│  Ontology: brede_welvaart_v32.ttl                      │
│                                                         │
│  Overall Progress                                       │
│  ████████████████░░░░░░░░░░░░░░ 54% (27/50 documents)  │
│                                                         │
│  Current: NPVR/Document_23_Economische_Impact          │
│  ├─ Pass 1/4: Brede Welvaart ✅                        │
│  ├─ Pass 2/4: Geografisch + Organisaties ✅            │
│  ├─ Pass 3/4: Financiering + Indicatoren 🔄           │
│  └─ Pass 4/4: Relaties ⏳                              │
│                                                         │
│  Statistics:                                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Documents:  27/50    │ Pages processed:  834   │   │
│  │ Entities:   1,234    │ Relations:        567   │   │
│  │ Geo:        156      │ Org:              289   │   │
│  │ Concepts:   445      │ Financieel:       89    │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  [Pause]  [Stop]  [View Results]                       │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Pass Analyzer                          [4 passes]     │
├─────────────────────────────────────────────────────────┤
│  Ontology: brede_welvaart_v32.ttl                      │
│                                                         │
│  Generated Extraction Passes:                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Pass 1: Brede Welvaart                          │   │
│  │ ├─ Namespaces: bw:                              │   │
│  │ ├─ Classes: 23  Properties: 18                  │   │
│  │ ├─ Tokens: ~850                                 │   │
│  │ └─ Depends on: (none - runs parallel)          │   │
│  ├─────────────────────────────────────────────────┤   │
│  │ Pass 2: Geografisch + Organisaties              │   │
│  │ ├─ Namespaces: geo:, org:                       │   │
│  │ ├─ Classes: 34  Properties: 42                  │   │
│  │ ├─ Tokens: ~1,200                               │   │
│  │ └─ Depends on: (none - runs parallel)          │   │
│  ├─────────────────────────────────────────────────┤   │
│  │ Pass 3: Financiering + Indicatoren              │   │
│  │ ├─ Namespaces: fin:, ind:                       │   │
│  │ ├─ Tokens: ~950                                 │   │
│  │ └─ Depends on: (none - runs parallel)          │   │
│  ├─────────────────────────────────────────────────┤   │
│  │ Pass 4: Relaties (relationship pass)            │   │
│  │ ├─ Focus: Cross-entity relationships            │   │
│  │ ├─ Tokens: ~600                                 │   │
│  │ └─ Depends on: Pass 1, 2, 3 (known entities)   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  [Regenerate Passes]  [Export Pass Config]             │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Extraction Results                      [Document 23] │
├─────────────────────────────────────────────────────────┤
│  Document: NPVR/Economische_Impact_Regio_Noord         │
│  Pages: 45  │  Entities: 89  │  Relations: 34          │
│                                                         │
│  Filter: [All types ▼] Search: [_______________]       │
│                                                         │
│  Entities by Type:                                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Type          │ Count │ Examples                │   │
│  ├───────────────┼───────┼─────────────────────────┤   │
│  │ Geografisch   │ 23    │ Groningen, Drenthe...   │   │
│  │ Organisatie   │ 31    │ Provincie, CBS, TNO...  │   │
│  │ Indicator     │ 18    │ BBP, Werkloosheid...    │   │
│  │ Financieel    │ 12    │ €50M, subsidie...       │   │
│  │ Concept       │ 5     │ Brede welvaart...       │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  Relations:                                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Provincie Groningen ──[investeert_in]──> MKB   │   │
│  │ CBS ──[meet]──> Werkloosheidspercentage         │   │
│  │ Regiodeal Noord ──[richt_zich_op]──> Innovatie │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  [View by Page] [Export: JSON ▼] [Save to Database]   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Ontology Manager                                      │
├─────────────────────────────────────────────────────────┤
│  Loaded Ontologies:                                    │
│  ┌─────────────────────────────────────────────────┐   │
│  │ ● brede_welvaart_v32.ttl (active)              │   │
│  │   87 classes │ 124 properties │ 9 namespaces   │   │
│  │   [View] [Edit] [Validate]                     │   │
│  ├─────────────────────────────────────────────────┤   │
│  │ ○ schema.org.ttl                               │   │
│  │   156 classes │ 234 properties                 │   │
│  │   [Activate] [View]                            │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  [Upload New Ontology]  [Create from Template]         │
│                                                         │
│  Ontology Viewer: brede_welvaart_v32.ttl              │
│  ┌─────────────────────────────────────────────────┐   │
│  │ Namespace: geo:                                 │   │
│  │ ├─ Classes:                                     │   │
│  │ │   ├─ Gemeente (label: "Gemeente")            │   │
│  │ │   ├─ Provincie (label: "Provincie")          │   │
│  │ │   └─ Regio (label: "Regio")                  │   │
│  │ └─ Properties:                                  │   │
│  │     ├─ ligtIn (Gemeente → Provincie)           │   │
│  │     └─ grenstAan (Gemeente → Gemeente)         │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Integration with Other Pipelines**:

| Pipeline | Integration |
|----------|-------------|
| **Ingestion** | Ontology-extraction reads `02_elements_by_page.json` from ingestion output |
| **Summarization** | Can use extracted entities as context for better summaries |
| **Embeddings** | Extracted entities can be embedded separately |
| **Retrieval** | Entity-based retrieval using extracted knowledge |

**Workflow Example**:
```
1. Ingestion: Parse PDF → 02_elements_by_page.json (chunks with page info)
2. Ontology-Extraction:
   - Load domain ontology (TTL/OWL)
   - Generate extraction passes based on namespaces
   - Extract page-by-page with multi-pass strategy
   - Merge and deduplicate results
   - Output: ontologie_extract.json (entities, relations, concepts)
3. Summarization: Use extracted entities as context for RAPTOR
4. Embeddings: Embed both chunks and extracted entities
```

---

### pipelines/enrichment

**Purpose**: Metadata verification, enrichment, and external database lookup before embedding. Provides a human-in-the-loop checkpoint to review, edit, and enhance extracted metadata.

**Why this exists**: After extraction and summarization, sources have auto-generated metadata (titles, authors, dates, entities, summaries). Before embedding, users need to:
1. **Verify** auto-extracted metadata is correct
2. **Enrich** with external data (Google Scholar, CrossRef, DOI resolution)
3. **Edit** incorrect or missing fields manually
4. **Approve** content before it enters the vector database

**What belongs here**:
- [ ] External API integrations (Google Scholar, CrossRef, Semantic Scholar, OpenAlex)
- [ ] DOI resolution and metadata enrichment
- [ ] Metadata verification queue and workflows
- [ ] Manual metadata editor interface
- [ ] Approval/rejection workflow before embedding
- [ ] Duplicate detection and merge suggestions

**External API Integrations**:
| API | Purpose | Data Retrieved |
|-----|---------|----------------|
| Google Scholar | Academic paper lookup | Citations, related papers, author profiles |
| CrossRef | DOI resolution | Full bibliographic metadata, funding, license |
| Semantic Scholar | Paper metadata + citations | Abstract, citations, influential citations, TLDR |
| OpenAlex | Open scholarly data | Concepts, institutions, authors, works graph |
| arXiv | Preprint lookup | Categories, versions, related papers |
| PubMed | Medical/life sciences | MeSH terms, abstracts, related articles |
| ORCID | Author disambiguation | Author identifiers, affiliations, works |

**Source files to migrate**:
| Current Location | New Location | Notes |
|-----------------|--------------|-------|
| (new) | `pipelines/enrichment/src/enrichment/lookups/scholar.py` | Google Scholar integration |
| (new) | `pipelines/enrichment/src/enrichment/lookups/crossref.py` | CrossRef DOI resolution |
| (new) | `pipelines/enrichment/src/enrichment/lookups/semantic.py` | Semantic Scholar API |
| (new) | `pipelines/enrichment/src/enrichment/verification/queue.py` | Verification queue |
| (new) | `pipelines/enrichment/src/enrichment/verification/editor.py` | Metadata editor |

**Directory structure**:
```
pipelines/enrichment/
├── pyproject.toml
├── README.md
├── main.py                 # CLI entry point
├── ui.py                   # Streamlit UI entry point
└── src/enrichment/
    ├── __init__.py
    ├── service.py          # EnrichmentService class
    ├── lookups/
    │   ├── __init__.py
    │   ├── base.py         # BaseLookupProvider
    │   ├── scholar.py      # Google Scholar (scholarly/serpapi)
    │   ├── crossref.py     # CrossRef API
    │   ├── semantic.py     # Semantic Scholar API
    │   ├── openalex.py     # OpenAlex API
    │   ├── arxiv.py        # arXiv API
    │   ├── pubmed.py       # PubMed/NCBI API
    │   └── orcid.py        # ORCID author lookup
    ├── verification/
    │   ├── __init__.py
    │   ├── queue.py        # Verification queue management
    │   ├── workflow.py     # Approval/rejection workflow
    │   ├── editor.py       # Metadata editor logic
    │   └── diff.py         # Show changes before/after enrichment
    ├── models/
    │   ├── __init__.py
    │   ├── metadata.py     # EnrichedMetadata, VerificationStatus
    │   └── lookup.py       # LookupResult, LookupSource
    ├── api/
    │   ├── __init__.py
    │   ├── router.py       # FastAPI router
    │   └── endpoints.py    # REST endpoints
    └── cli.py              # CLI commands
```

**CLI Commands**:
```bash
# Lookup paper metadata from DOI
enrichment lookup --doi "10.1000/xyz123"

# Search Google Scholar for paper
enrichment search --query "attention is all you need" --provider scholar

# Enrich source with external metadata
enrichment enrich --source-id source:abc123 --providers scholar,crossref

# Show verification queue
enrichment queue --status pending

# Approve source for embedding
enrichment approve --source-id source:abc123

# Reject source (needs manual review)
enrichment reject --source-id source:abc123 --reason "Incorrect author attribution"

# Batch enrich all pending sources
enrichment batch-enrich --providers crossref,semantic --limit 50

# Export enrichment report
enrichment report --output enrichment_report.json
```

**API Endpoints**:
```yaml
# Lookup endpoints
POST /api/enrichment/lookup/doi         # Resolve DOI via CrossRef
POST /api/enrichment/lookup/scholar     # Search Google Scholar
POST /api/enrichment/lookup/semantic    # Search Semantic Scholar
POST /api/enrichment/lookup/batch       # Batch lookup multiple sources

# Verification endpoints
GET  /api/enrichment/queue              # Get verification queue
GET  /api/enrichment/queue/{source_id}  # Get source verification status
POST /api/enrichment/verify/{source_id} # Submit verification decision
PUT  /api/enrichment/metadata/{source_id} # Update source metadata

# Workflow endpoints
POST /api/enrichment/approve/{source_id}  # Approve for embedding
POST /api/enrichment/reject/{source_id}   # Reject with reason
POST /api/enrichment/enrich/{source_id}   # Trigger enrichment

# Batch operations
POST /api/enrichment/batch/enrich       # Batch enrich sources
POST /api/enrichment/batch/approve      # Batch approve verified sources
```

#### Streamlit UI Specification: enrichment

**Purpose**: **Metadata verification and enrichment hub** - Review auto-extracted metadata, lookup external databases, edit fields, and approve sources before embedding.

**Use Cases**:
- Verify and correct auto-extracted paper metadata (title, authors, date)
- Lookup papers on Google Scholar to get citation counts, related works
- Resolve DOIs to get complete bibliographic information
- Edit metadata fields manually before embedding
- Track verification status across all sources
- Batch approve verified sources for embedding

**Pages/Features**:
- [ ] **Verification Queue**: List of sources pending verification with status indicators
- [ ] **Metadata Editor**: Full metadata editing interface for a source
- [ ] **External Lookup**: Search and import metadata from Scholar, CrossRef, etc.
- [ ] **Diff View**: Compare original vs enriched metadata
- [ ] **Batch Operations**: Approve/reject multiple sources
- [ ] **Enrichment History**: Audit trail of all metadata changes

**Operational Modes**:
```yaml
standalone_mode:
  description: "Lookup external APIs, export enriched metadata"
  requires_database: false
  output: "JSON metadata files, enrichment reports"

connected_mode:
  description: "Full verification workflow with database"
  requires_database: true
  output: "Verified sources ready for embedding"
```

**UI Mockup - Verification Queue**:
```
┌─────────────────────────────────────────────────────────────────────────┐
│  Enrichment Pipeline                              [Connected Mode] 🔗   │
├─────────────┬───────────────────────────────────────────────────────────┤
│ 📋 Queue    │  Verification Queue                    [Batch Actions ▾] │
│ ✏️ Editor   │                                                          │
│ 🔍 Lookup   │  Filter: [All ▾] [Pending ▾] [Papers ▾]    🔎 Search... │
│ 📊 History  │                                                          │
│             │  ┌─────────────────────────────────────────────────────┐ │
│             │  │ ☐ │ 📄 Title                    │ Status  │ Actions │ │
│             │  ├───┼───────────────────────────────┼─────────┼─────────┤ │
│             │  │ ☐ │ Attention Is All You Need   │ ⏳ Pend │ [Edit]  │ │
│             │  │   │ Authors: Vaswani et al.     │ Scholar │ [Lookup]│ │
│             │  │   │ DOI: 10.48550/arXiv.1706... │ matched │ [Approve│ │
│             │  ├───┼───────────────────────────────┼─────────┼─────────┤ │
│             │  │ ☐ │ BERT: Pre-training Deep...  │ ⚠️ Review│ [Edit]  │ │
│             │  │   │ Authors: [Missing]          │ No DOI  │ [Lookup]│ │
│             │  │   │ DOI: —                      │ found   │ [Reject]│ │
│             │  ├───┼───────────────────────────────┼─────────┼─────────┤ │
│             │  │ ☑ │ GPT-4 Technical Report      │ ✅ Ready │ [View]  │ │
│             │  │   │ Authors: OpenAI             │ Verified│ [Embed] │ │
│             │  │   │ DOI: —                      │         │         │ │
│             │  └───┴───────────────────────────────┴─────────┴─────────┘ │
│             │                                                          │
│             │  Selected: 2    [✅ Approve Selected] [❌ Reject Selected]│
└─────────────┴──────────────────────────────────────────────────────────┘
```

**UI Mockup - Metadata Editor**:
```
┌─────────────────────────────────────────────────────────────────────────┐
│  Metadata Editor                                  [← Back to Queue]    │
├─────────────────────────────────────────────────────────────────────────┤
│  Source: Attention Is All You Need                                     │
│  Status: ⏳ Pending Verification                                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─ Bibliographic Information ─────────────────────────────────────┐   │
│  │ Title:    [Attention Is All You Need                        ]   │   │
│  │ Authors:  [Vaswani, Shazeer, Parmar, Uszkoreit, Jones, ...  ]   │   │
│  │           [+ Add Author]                                        │   │
│  │ Year:     [2017    ]  Journal: [NeurIPS                     ]   │   │
│  │ DOI:      [10.48550/arXiv.1706.03762                        ]   │   │
│  │ arXiv:    [1706.03762                                       ]   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─ External Lookups ──────────────────────────────────────────────┐   │
│  │ [🔍 Google Scholar] [🔍 CrossRef] [🔍 Semantic Scholar]         │   │
│  │                                                                  │   │
│  │ Scholar Results:                              [Import Selected] │   │
│  │ ┌──────────────────────────────────────────────────────────┐   │   │
│  │ │ ☑ Attention Is All You Need                              │   │   │
│  │ │   Citations: 98,432 | Venue: NeurIPS 2017                │   │   │
│  │ │   Authors: A Vaswani, N Shazeer, N Parmar, J Uszkoreit...│   │   │
│  │ └──────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─ Extracted Entities (from Ontology Extraction) ─────────────────┐   │
│  │ Concepts: [transformer] [self-attention] [encoder-decoder]      │   │
│  │ Methods:  [scaled dot-product attention] [multi-head attention] │   │
│  │ [✏️ Edit Entities]                                               │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─ Summaries (from Summarization Pipeline) ───────────────────────┐   │
│  │ RAPTOR Summary: "The Transformer model architecture..."         │   │
│  │ TreeKG Summary: "Chapter 1: Introduction to attention..."       │   │
│  │ [✏️ Edit Summaries]                                              │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  [💾 Save Changes]  [✅ Approve for Embedding]  [❌ Reject]  [🔄 Reset] │
└─────────────────────────────────────────────────────────────────────────┘
```

**UI Mockup - External Lookup Panel**:
```
┌─────────────────────────────────────────────────────────────────────────┐
│  External Lookup                                                        │
├─────────────────────────────────────────────────────────────────────────┤
│  Search: [attention is all you need                           ] [🔍]   │
│                                                                         │
│  Providers: [☑ Google Scholar] [☑ CrossRef] [☐ Semantic Scholar]       │
│             [☐ OpenAlex] [☐ arXiv] [☐ PubMed]                          │
├─────────────────────────────────────────────────────────────────────────┤
│  Results:                                                     (3 found) │
│                                                                         │
│  ┌─ Google Scholar ────────────────────────────────────────────────┐   │
│  │ 📄 Attention Is All You Need                                    │   │
│  │    A Vaswani, N Shazeer, N Parmar... - NeurIPS 2017            │   │
│  │    Citations: 98,432 | ⭐ Highly Influential                    │   │
│  │    [View on Scholar] [Import Metadata]                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─ CrossRef (DOI: 10.48550/arXiv.1706.03762) ─────────────────────┐   │
│  │ 📄 Attention Is All You Need                                    │   │
│  │    Type: Conference Paper | Publisher: Curran Associates        │   │
│  │    License: CC-BY-4.0 | Funder: Google                         │   │
│  │    [View DOI] [Import Full Metadata]                            │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─ Semantic Scholar ──────────────────────────────────────────────┐   │
│  │ 📄 Attention Is All You Need                                    │   │
│  │    Paper ID: 204e3073870fae3d05bcbc2f6a8e263d9b72e776          │   │
│  │    Influential Citations: 4,521 | Fields: Computer Science     │   │
│  │    TLDR: "The Transformer architecture relies entirely on..."  │   │
│  │    [View Paper] [Import + TLDR]                                 │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  [Import All Selected] [Cancel]                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

**Verification Workflow**:
```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────┐
│  Ingested   │────▶│  Extracted   │────▶│ Summarized  │────▶│ PENDING  │
│  (raw file) │     │  (entities)  │     │  (RAPTOR)   │     │ VERIFY   │
└─────────────┘     └──────────────┘     └─────────────┘     └────┬─────┘
                                                                  │
                    ┌─────────────────────────────────────────────┼─────┐
                    │                                             ▼     │
                    │  ┌─────────────┐     ┌─────────────┐     ┌──────┐│
                    │  │   ENRICH    │◀────│   REVIEW    │◀────│LOOKUP││
                    │  │  (external) │     │   (human)   │     │(APIs)││
                    │  └──────┬──────┘     └──────┬──────┘     └──────┘│
                    │         │                   │                    │
                    │         ▼                   ▼                    │
                    │  ┌─────────────┐     ┌─────────────┐             │
                    │  │  APPROVED   │     │  REJECTED   │             │
                    │  │ (→ embed)   │     │ (→ archive) │             │
                    │  └─────────────┘     └─────────────┘             │
                    │          Enrichment Pipeline                     │
                    └──────────────────────────────────────────────────┘
```

**PipelineTask Enum Entries**:
```python
# Enrichment Tasks
ENRICHMENT_LOOKUP = "enrichment.lookup"         # External API lookup
ENRICHMENT_VERIFY = "enrichment.verify"         # Verification decision
ENRICHMENT_MERGE = "enrichment.merge"           # Merge external metadata
```

**Default Model Assignments** (for AI-assisted verification):
```python
"enrichment.verify": "qwen2.5:14b",    # Suggest verification decisions
"enrichment.merge": "qwen2.5:14b",     # Intelligent metadata merging
```

**Docker Service**:
```yaml
enrichment:
  build:
    context: ./pipelines/enrichment
    dockerfile: Dockerfile
  ports:
    - "8527:8501"  # Streamlit UI
    - "5127:8000"  # FastAPI
  environment:
    - SCHOLAR_API_KEY=${SCHOLAR_API_KEY}
    - CROSSREF_EMAIL=${CROSSREF_EMAIL}
    - SEMANTIC_SCHOLAR_API_KEY=${SEMANTIC_SCHOLAR_API_KEY}
  depends_on:
    - surrealdb
    - summarization
```

**Port Assignment**: Streamlit: 8527, API: 5127

**Integration with Other Pipelines**:
```yaml
receives_from:
  - ontology-extraction: Extracted entities, concepts, relations
  - summarization: RAPTOR summaries, TreeKG summaries

sends_to:
  - embeddings: Verified, enriched metadata ready for embedding

external_apis:
  - Google Scholar (via scholarly or SerpAPI)
  - CrossRef REST API
  - Semantic Scholar API
  - OpenAlex API
  - arXiv API
  - PubMed/NCBI E-utilities
```

---

### pipelines/embeddings

**Purpose**: Generate and manage vector embeddings for chunks and entities.

**What belongs here**:
- [ ] Embedding model management (Ollama, OpenAI, etc.)
- [ ] Batch embedding generation
- [ ] Embedding updates and rebuilds
- [ ] Similarity computation

**Source files to migrate**:
| Current Location | New Location | Notes |
|-----------------|--------------|-------|
| `open_notebook/processors/embeddings.py` | `pipelines/embeddings/src/embeddings/service.py` | |
| `commands/embedding_commands.py` | `pipelines/embeddings/src/embeddings/commands.py` | |
| | | |

**Directory structure**:
```
pipelines/embeddings/
├── pyproject.toml
├── README.md
├── main.py                 # CLI entry point
├── ui.py                   # Streamlit UI entry point
└── src/embeddings/
    ├── __init__.py
    ├── service.py          # EmbeddingService class
    ├── models/
    │   ├── __init__.py
    │   ├── ollama.py       # Ollama embeddings
    │   ├── openai.py       # OpenAI embeddings
    │   └── local.py        # ONNX/local models
    ├── batch.py            # Batch processing
    └── cli.py              # CLI commands
```

**CLI Commands**:
```bash
# Embed all chunks for a source
embeddings generate --source-id source:abc123

# Rebuild all embeddings with new model
embeddings rebuild --model nomic-embed-text --batch-size 100

# Test similarity between texts
embeddings similarity --text1 "query" --text2 "document"
```

#### Streamlit UI Specification: embeddings

**Purpose**: **Standalone embedding generation** - Generate embeddings for text/files without the full pipeline, test models, export vectors.

**Use Cases**:
- Generate embeddings for a set of texts/chunks
- Compare different embedding models on same content
- Export embeddings for use in other tools (numpy, FAISS, etc.)
- Visualize embedding space with UMAP
- Test similarity before committing to a model

**Pages/Features**:
- [ ] **Generate Embeddings**: Input text/file, generate and export embeddings
- [ ] **Similarity Tester**: Compute similarity between texts
- [ ] **Model Comparison**: Same content, different models side-by-side
- [ ] **Vector Visualizer**: UMAP/t-SNE projection
- [ ] **Batch Processing**: Embed multiple files, export as numpy/JSON
- [ ] **Database Sync**: Push/pull embeddings to/from SurrealDB

**Operational Modes**:
```yaml
standalone_mode:
  description: "Generate embeddings locally, export to files"
  requires_database: false
  output: "numpy arrays, JSON, CSV"

connected_mode:
  description: "Generate and store in database"
  requires_database: true
  output: "Database vectors + local export"
```

**UI Mockup**:
```
┌─────────────────────────────────────────────────────────┐
│  Embeddings Pipeline                   [Standalone Mode]│
├─────────────┬───────────────────────────────────────────┤
│ Generate    │  Generate Embeddings                      │
│ Similarity  │                                           │
│ Compare     │  Input:                                   │
│ Visualize   │  (•) Text  ( ) File  ( ) JSON chunks     │
│ Batch       │                                           │
│ ─────────── │  ┌─────────────────────────────────────┐  │
│ Model:      │  │ Enter text to embed...              │  │
│ nomic ▼     │  │                                     │  │
│             │  │                                     │  │
│ Dimensions: │  └─────────────────────────────────────┘  │
│ 768         │                                           │
│             │  [Generate]  [Generate & Export numpy]    │
│ ─────────── │                                           │
│ DB Status:  │  Result:                                  │
│ ⚪ Offline  │  Dimensions: 768  │  Tokens: 45           │
│             │  [0.023, -0.891, 0.445, ...]             │
│             │                                           │
│             │  [Copy Vector] [Download .npy] [To DB]   │
└─────────────┴───────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  Model Comparison                                       │
├─────────────────────────────────────────────────────────┤
│  Query: "machine learning applications"                 │
│                                                         │
│  ┌─ nomic-embed-text ───┬─ all-MiniLM-L6 ───┐         │
│  │ Dims: 768            │ Dims: 384          │         │
│  │ Time: 23ms           │ Time: 12ms         │         │
│  │                      │                    │         │
│  │ Top matches:         │ Top matches:       │         │
│  │ 1. doc_a (0.92)      │ 1. doc_a (0.89)   │         │
│  │ 2. doc_c (0.87)      │ 2. doc_b (0.86)   │         │
│  │ 3. doc_b (0.85)      │ 3. doc_c (0.84)   │         │
│  └──────────────────────┴────────────────────┘         │
└─────────────────────────────────────────────────────────┘
```

---

### pipelines/retrieval

**Purpose**: Search and retrieve relevant content from the knowledge base.

**What belongs here**:
- [ ] Dense (vector) retrieval
- [ ] Sparse (BM25/keyword) retrieval
- [ ] Hybrid retrieval strategies
- [ ] PPR-based graph retrieval
- [ ] LLM reranking

**Source files to migrate**:
| Current Location | New Location | Notes |
|-----------------|--------------|-------|
| `open_notebook/graphs/kg_retrieval.py` | `pipelines/retrieval/src/retrieval/` | Split into modules |
| `api/search_service.py` | `pipelines/retrieval/src/retrieval/search.py` | |
| | | |

**Directory structure**:
```
pipelines/retrieval/
├── pyproject.toml
├── README.md
├── main.py                 # CLI entry point
├── ui.py                   # Streamlit UI entry point
└── src/retrieval/
    ├── __init__.py
    ├── config.py           # RetrievalConfig
    ├── dense.py            # DenseRetriever
    ├── sparse.py           # BM25/keyword retrieval
    ├── hybrid.py           # Hybrid strategies
    ├── ppr.py              # PPRRetriever (graph-based)
    ├── reranker.py         # LLMReranker
    └── cli.py              # CLI commands
```

**CLI Commands**:
```bash
# Search with default settings
retrieval search "What is machine learning?" --limit 10

# Search with specific strategy
retrieval search "query" --strategy hybrid --rerank

# Batch evaluation on test queries
retrieval evaluate --queries ./test_queries.json --output ./results.json
```

#### Streamlit UI Specification: retrieval

**Purpose**: **Standalone search interface** - Query your knowledge base, test retrieval strategies, export results for analysis.

**Use Cases**:
- Search across indexed documents without the full app
- Compare retrieval strategies (dense, sparse, hybrid, PPR)
- Export search results for reports or further processing
- Build and test evaluation datasets
- Debug retrieval quality issues

**Pages/Features**:
- [ ] **Search**: Query interface with strategy selection
- [ ] **Results Viewer**: Results with scores, snippets, source links
- [ ] **Strategy Comparison**: Side-by-side comparison of strategies
- [ ] **Evaluation**: Run test queries, compute metrics
- [ ] **Export**: Download results as JSON/CSV for analysis
- [ ] **Query Builder**: Build complex queries with filters

**Operational Modes**:
```yaml
connected_mode:
  description: "Search against SurrealDB index"
  requires_database: true

local_index_mode:
  description: "Search against local FAISS/numpy index"
  requires_database: false
  note: "Requires pre-built local index file"
```

**UI Mockup**:
```
┌─────────────────────────────────────────────────────────┐
│  Retrieval Pipeline                     [Connected Mode]│
├─────────────┬───────────────────────────────────────────┤
│ Search      │  Search Query                             │
│ Compare     │  [What is the main argument?__________]   │
│ Evaluate    │                                           │
│ Export      │  Strategy: [Hybrid ▼]  Limit: [10▼]      │
│ Settings    │  [x] Rerank  [x] Include KG context       │
│             │                                           │
│ ─────────── │  Filters:                                 │
│ DB Status:  │  Source type: [All ▼]  Date: [Any ▼]     │
│ 🟢 Online   │                                           │
│             │  [Search]  [Search & Export]              │
│ Sources:    │                                           │
│ 1,234       │  Results (10 found, 0.23s)                │
│             │  ┌─────────────────────────────────────┐  │
│ Chunks:     │  │ 1. [0.92] Chapter 3: Main Arguments │  │
│ 45,678      │  │    Source: research_paper.pdf       │  │
│             │  │    "The central thesis posits..."   │  │
│             │  │    [View Full] [View Source] [Copy] │  │
│             │  ├─────────────────────────────────────┤  │
│             │  │ 2. [0.87] Section 2.1: Overview     │  │
│             │  │    Source: methodology.pdf          │  │
│             │  │    "This paper argues that..."      │  │
│             │  └─────────────────────────────────────┘  │
│             │                                           │
│             │  [Export Results: JSON ▼]                 │
└─────────────┴───────────────────────────────────────────┘
```

---

### pipelines/summarization

**Purpose**: Generate summaries at various levels of abstraction.

**What belongs here**:
- [ ] RAPTOR hierarchical summarization
- [ ] Simple LLM summarization
- [ ] Multi-document summarization
- [ ] Summary caching

**Source files to migrate**:
| Current Location | New Location | Notes |
|-----------------|--------------|-------|
| `open_notebook/processors/raptor/` | `pipelines/summarization/src/summarization/raptor/` | |
| `open_notebook/graphs/treekg_summarizer.py` | `pipelines/summarization/src/summarization/treekg.py` | |
| | | |

**Directory structure**:
```
pipelines/summarization/
├── pyproject.toml
├── README.md
├── main.py                 # CLI entry point
├── ui.py                   # Streamlit UI entry point
└── src/summarization/
    ├── __init__.py
    ├── config.py           # SummarizationConfig
    ├── simple.py           # Simple LLM summarization
    ├── raptor/
    │   ├── __init__.py
    │   ├── tree_builder.py
    │   ├── summarizer.py
    │   └── clustering.py
    ├── multi_doc.py        # Multi-document summarization
    └── cli.py              # CLI commands
```

**CLI Commands**:
```bash
# Summarize a source
summarization summarize --source-id source:abc123 --method raptor

# Summarize with simple LLM
summarization summarize --text "..." --method simple --model gpt-4

# Build RAPTOR tree
summarization raptor-build --source-id source:abc123 --levels 3
```

#### Streamlit UI Specification: summarization

**Purpose**: **Standalone summarization tool** - Generate summaries of any text/document, build RAPTOR trees, compare methods and models.

**Use Cases**:
- Summarize a document without importing to the main app
- Build and visualize RAPTOR hierarchical summaries
- Compare different summarization approaches
- Test different LLMs for summarization quality
- Export summaries for use elsewhere

**Pages/Features**:
- [ ] **Summarize**: Input text/file, generate summary with chosen method
- [ ] **RAPTOR Builder**: Build hierarchical summary tree, configure levels
- [ ] **RAPTOR Viewer**: Interactive tree visualization, drill down into clusters
- [ ] **Method Comparison**: Same text, different methods side-by-side
- [ ] **Model Comparison**: Same method, different LLMs
- [ ] **Export**: Download summaries as Markdown/JSON

**Operational Modes**:
```yaml
standalone_mode:
  description: "Summarize text/files locally"
  requires_database: false
  output: "Markdown, JSON, RAPTOR tree files"

connected_mode:
  description: "Summarize and store in database"
  requires_database: true
  output: "Database records + local export"
```

**UI Mockup**:
```
┌─────────────────────────────────────────────────────────┐
│  Summarization Pipeline                [Standalone Mode]│
├─────────────┬───────────────────────────────────────────┤
│ Summarize   │  Summarize Document                       │
│ RAPTOR      │                                           │
│ Compare     │  Input:                                   │
│ Models      │  (•) Text  ( ) File  ( ) URL             │
│ Export      │                                           │
│             │  ┌─────────────────────────────────────┐  │
│ ─────────── │  │ Paste or type text to summarize... │  │
│ Method:     │  │                                     │  │
│ RAPTOR ▼    │  │                                     │  │
│             │  └─────────────────────────────────────┘  │
│ Model:      │                                           │
│ GPT-4 ▼     │  RAPTOR Settings:                         │
│             │  Levels: [3▼]  Cluster size: [5▼]        │
│ DB Status:  │                                           │
│ ⚪ Offline  │  [Summarize]  [Build RAPTOR Tree]         │
└─────────────┴───────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│  RAPTOR Tree Viewer                    [3 levels built] │
├─────────────────────────────────────────────────────────┤
│                                                         │
│                    [Root Summary]                       │
│                    "The document presents..."           │
│                         /      \                        │
│              [Cluster 1]        [Cluster 2]             │
│              "Methods..."       "Results..."            │
│               /    \              /    \                │
│           [C1.1] [C1.2]       [C2.1] [C2.2]            │
│             |      |            |      |               │
│           chunks  chunks      chunks chunks            │
│                                                         │
│  ─────────────────────────────────────────────────────  │
│  Selected: Cluster 1 (Level 2)                         │
│  Chunks: 5  │  Tokens: 2,340                           │
│                                                         │
│  Summary:                                               │
│  "This section discusses the theoretical framework     │
│   underlying the proposed methodology..."              │
│                                                         │
│  [View Chunks] [Export Tree: JSON ▼] [Copy Summary]    │
└─────────────────────────────────────────────────────────┘
```

---

## Application Specifications

### apps/app-main

**Purpose**: Production application with full-featured UI and API.

**What belongs here**:
- [ ] FastAPI backend integrating all pipelines
- [ ] Next.js frontend (current frontend/)
- [ ] Authentication and authorization
- [ ] Async command/job management
- [ ] WebSocket support for real-time updates

**Source files to migrate**:
| Current Location | New Location | Notes |
|-----------------|--------------|-------|
| `api/` | `apps/app-main/backend/api/` | Most routers, services |
| `frontend/` | `apps/app-main/frontend/` | As-is |
| `commands/` | `apps/app-main/backend/commands/` | Async command handlers |
| | | |

**Directory structure**:
```
apps/app-main/
├── pyproject.toml          # Python dependencies
├── README.md
├── docker-compose.yml      # App-specific compose
├── backend/
│   ├── main.py             # FastAPI entry point
│   ├── api/
│   │   ├── __init__.py
│   │   ├── app.py          # FastAPI app setup
│   │   ├── auth.py         # Authentication
│   │   ├── routers/        # API routes
│   │   └── services/       # Business logic services
│   └── commands/           # Async command handlers
│       ├── __init__.py
│       ├── source_commands.py
│       ├── embedding_commands.py
│       └── podcast_commands.py
└── frontend/               # Next.js application
    ├── package.json
    ├── next.config.ts
    └── src/
        ├── app/
        ├── components/
        └── lib/
```

---

### apps/chat

**Purpose**: Conversational interface to chat with your knowledge base using RAG (Retrieval-Augmented Generation).

**What belongs here**:
- [ ] RAG chat interface with streaming responses
- [ ] Conversation history and context management
- [ ] Source citation and reference linking
- [ ] Multi-turn conversation handling
- [ ] Knowledge base selection
- [ ] Chat export and sharing

**Key Features**:
| Feature | Description |
|---------|-------------|
| **RAG Chat** | Retrieve relevant context from KB, generate grounded responses |
| **Source Citations** | Show which documents/chunks informed each response |
| **Conversation Memory** | Maintain context across multiple turns |
| **KB Selection** | Choose which knowledge base(s) to query |
| **Streaming** | Real-time token streaming for better UX |
| **Export** | Export conversations as Markdown/PDF |

**Dependencies**:
- `retrieval` - For searching the knowledge base
- `llm-manager` - For chat model access
- `surrealdb-service` - For conversation persistence
- `shared` - Common schemas

**Directory structure**:
```
apps/chat/
├── pyproject.toml
├── README.md
├── main.py                 # CLI entry point
├── ui.py                   # Streamlit chat UI
└── src/chat/
    ├── __init__.py
    ├── config.py           # ChatConfig
    ├── models.py           # Conversation, Message, Citation
    ├── rag/
    │   ├── __init__.py
    │   ├── retriever.py    # KB retrieval integration
    │   ├── context.py      # Context window management
    │   └── prompts.py      # System prompts, templates
    ├── conversation/
    │   ├── __init__.py
    │   ├── manager.py      # Conversation state management
    │   ├── memory.py       # Multi-turn memory handling
    │   └── history.py      # Conversation persistence
    ├── streaming/
    │   ├── __init__.py
    │   └── handler.py      # Streaming response handler
    ├── api/
    │   ├── __init__.py
    │   ├── app.py          # FastAPI app
    │   └── routers/
    │       ├── chat.py     # Chat endpoints
    │       └── history.py  # Conversation history endpoints
    └── cli.py              # CLI commands
```

**CLI Commands**:
```bash
# Start chat session
chat start --kb "my-knowledge-base"

# Continue existing conversation
chat continue --conversation-id conv:abc123

# Export conversation
chat export --conversation-id conv:abc123 --format markdown

# List conversations
chat list --kb "my-knowledge-base"
```

**API Endpoints**:
```yaml
# Chat
POST   /api/v1/chat                    # Send message, get response (streaming)
POST   /api/v1/chat/complete           # Non-streaming completion
GET    /api/v1/chat/sources            # Get sources for last response

# Conversations
GET    /api/v1/conversations           # List conversations
POST   /api/v1/conversations           # Create new conversation
GET    /api/v1/conversations/{id}      # Get conversation with messages
DELETE /api/v1/conversations/{id}      # Delete conversation
POST   /api/v1/conversations/{id}/export  # Export conversation

# Knowledge Bases (for selection)
GET    /api/v1/knowledge-bases         # List available KBs
```

#### Streamlit UI Specification: chat

**Purpose**: **Standalone chat interface** - Conversational access to your knowledge base with source citations and conversation management.

**Use Cases**:
- Ask questions about your documents and get grounded answers
- Research topics across multiple sources with citations
- Explore knowledge base content through natural conversation
- Save and revisit important conversations
- Export Q&A sessions for documentation

**Pages/Features**:
- [ ] **Chat**: Main conversation interface with streaming
- [ ] **Sources**: View and navigate cited sources for responses
- [ ] **History**: Browse and search past conversations
- [ ] **Settings**: Configure model, retrieval settings, prompts
- [ ] **Export**: Export conversations in various formats

**Operational Modes**:
```yaml
standalone_mode:
  description: "Chat with local files (no KB persistence)"
  requires_database: false
  features: ["basic_chat", "local_file_context"]

connected_mode:
  description: "Chat with knowledge base, save conversations"
  requires_database: true
  features: ["full_rag", "conversation_history", "citations"]
```

**UI Mockup**:
```
┌─────────────────────────────────────────────────────────────────────┐
│  Knowledge Base Chat                              [Connected Mode]  │
├───────────────┬─────────────────────────────────────────────────────┤
│               │                                                     │
│ Conversations │  ┌─────────────────────────────────────────────┐   │
│               │  │ 🤖 Based on your documents, the main        │   │
│ ▶ Current     │  │    findings from the 2024 research are:    │   │
│   "Research   │  │                                             │   │
│    Q&A"       │  │    1. Improved efficiency by 34% [1]       │   │
│               │  │    2. Cost reduction of €2.1M [2]          │   │
│ ─────────────│  │    3. User satisfaction increased [3]       │   │
│               │  │                                             │   │
│ Yesterday     │  │    Sources: [1] report_2024.pdf p.12       │   │
│ • Project     │  │             [2] financials.xlsx            │   │
│   planning    │  │             [3] survey_results.pdf p.45    │   │
│               │  └─────────────────────────────────────────────┘   │
│ Last week     │                                                     │
│ • Budget      │  ┌─────────────────────────────────────────────┐   │
│   review      │  │ 👤 What were the main challenges faced?     │   │
│               │  └─────────────────────────────────────────────┘   │
│ ─────────────│                                                     │
│               │  ┌─────────────────────────────────────────────┐   │
│ Knowledge     │  │ 🤖 The documents highlight several key      │   │
│ Base:         │  │    challenges encountered during the        │   │
│ ┌──────────┐  │  │    implementation phase...                  │   │
│ │Research ▼│  │  │    ▌ (streaming...)                         │   │
│ └──────────┘  │  └─────────────────────────────────────────────┘   │
│               │                                                     │
│ [+ New Chat]  │  ┌─────────────────────────────────────────────┐   │
│               │  │ Ask a question...                      [↵] │   │
│ [Export]      │  └─────────────────────────────────────────────┘   │
│               │                                                     │
│ Retrieval:    │  Retrieved: 8 chunks │ Confidence: High │ 1.2s    │
│ Chunks: [5▼]  │                                                     │
└───────────────┴─────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  Source Viewer                                    [Source [1]]      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  📄 report_2024.pdf - Page 12                                       │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  "The implementation of the new system resulted in a 34%            │
│   improvement in processing efficiency, measured across all         │
│   departments over a 6-month period. Key contributing factors       │
│   included automated workflows and reduced manual data entry."      │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│  Chunk ID: chunk:abc123  │  Relevance: 0.92  │  Section: Results    │
│                                                                     │
│  [Open Source] [Copy Citation] [View in Context] [◀ Prev] [Next ▶] │
└─────────────────────────────────────────────────────────────────────┘
```

---

### apps/canvas

**Purpose**: Interactive visual canvas for creating, organizing, and connecting notes with knowledge base integration.

**What belongs here**:
- [ ] Visual note editor with rich text and markdown
- [ ] Spatial canvas for organizing notes
- [ ] Connections/links between notes
- [ ] Knowledge base integration (link notes to sources)
- [ ] AI-assisted note generation
- [ ] Export to various formats

**Key Features**:
| Feature | Description |
|---------|-------------|
| **Visual Canvas** | Spatial arrangement of notes, zoom/pan navigation |
| **Rich Editor** | Markdown, code blocks, images, tables |
| **Connections** | Visual links between related notes |
| **KB Integration** | Link notes to sources, auto-suggest related content |
| **AI Assist** | Generate notes from sources, expand/summarize |
| **Templates** | Pre-built templates for common note types |
| **Export** | Export as Markdown, PDF, Obsidian vault |

**Dependencies**:
- `retrieval` - For finding related content
- `summarization` - For AI-assisted note generation
- `llm-manager` - For AI features
- `surrealdb-service` - For note persistence
- `shared` - Common schemas

**Directory structure**:
```
apps/canvas/
├── pyproject.toml
├── README.md
├── main.py                 # CLI entry point
├── ui.py                   # Streamlit canvas UI
└── src/canvas/
    ├── __init__.py
    ├── config.py           # CanvasConfig
    ├── models.py           # Note, Canvas, Connection, Block
    ├── editor/
    │   ├── __init__.py
    │   ├── blocks.py       # Block types (text, code, image, etc.)
    │   ├── markdown.py     # Markdown parsing/rendering
    │   └── templates.py    # Note templates
    ├── spatial/
    │   ├── __init__.py
    │   ├── canvas.py       # Canvas state management
    │   ├── layout.py       # Auto-layout algorithms
    │   └── connections.py  # Connection/link management
    ├── ai/
    │   ├── __init__.py
    │   ├── generator.py    # Note generation from sources
    │   ├── expander.py     # Expand/elaborate on notes
    │   └── suggestions.py  # Related content suggestions
    ├── integration/
    │   ├── __init__.py
    │   ├── kb_linker.py    # Link notes to KB sources
    │   └── search.py       # Search notes and KB
    ├── export/
    │   ├── __init__.py
    │   ├── markdown.py     # Export to Markdown
    │   ├── obsidian.py     # Export to Obsidian vault
    │   └── pdf.py          # Export to PDF
    ├── api/
    │   ├── __init__.py
    │   ├── app.py          # FastAPI app
    │   └── routers/
    │       ├── notes.py    # Note CRUD endpoints
    │       ├── canvas.py   # Canvas state endpoints
    │       └── ai.py       # AI feature endpoints
    └── cli.py              # CLI commands
```

**CLI Commands**:
```bash
# Create new canvas
canvas create --name "Research Notes" --notebook notebook:abc123

# List canvases
canvas list --notebook notebook:abc123

# Export canvas
canvas export --canvas-id canvas:xyz --format obsidian --output ./export/

# Generate note from source
canvas generate-note --source-id source:abc123 --type summary

# Open canvas UI
canvas ui --canvas-id canvas:xyz
```

**API Endpoints**:
```yaml
# Canvases
GET    /api/v1/canvases                # List canvases
POST   /api/v1/canvases                # Create canvas
GET    /api/v1/canvases/{id}           # Get canvas with notes
DELETE /api/v1/canvases/{id}           # Delete canvas
PUT    /api/v1/canvases/{id}/layout    # Update canvas layout

# Notes
GET    /api/v1/notes                   # List notes (with filters)
POST   /api/v1/notes                   # Create note
GET    /api/v1/notes/{id}              # Get note
PUT    /api/v1/notes/{id}              # Update note
DELETE /api/v1/notes/{id}              # Delete note

# Connections
POST   /api/v1/connections             # Create connection between notes
DELETE /api/v1/connections/{id}        # Delete connection

# AI Features
POST   /api/v1/ai/generate             # Generate note from source
POST   /api/v1/ai/expand               # Expand/elaborate note
POST   /api/v1/ai/suggest              # Get related content suggestions

# Export
POST   /api/v1/export                  # Export canvas/notes
```

#### Streamlit UI Specification: canvas

**Purpose**: **Standalone note canvas** - Visual workspace for creating and organizing notes with knowledge base integration.

**Use Cases**:
- Create research notes linked to source documents
- Visually organize ideas and connections
- Generate notes from sources using AI
- Build knowledge maps from your documents
- Export notes to Obsidian or other tools

**Pages/Features**:
- [ ] **Canvas**: Main visual workspace with notes and connections
- [ ] **Note Editor**: Rich markdown editor with blocks
- [ ] **AI Assistant**: Generate, expand, summarize notes
- [ ] **KB Browser**: Browse and link to knowledge base sources
- [ ] **Templates**: Apply note templates
- [ ] **Export**: Export canvas/notes in various formats

**Operational Modes**:
```yaml
standalone_mode:
  description: "Create notes locally without KB"
  requires_database: false
  features: ["basic_editing", "local_export", "basic_ai"]

connected_mode:
  description: "Full canvas with KB integration"
  requires_database: true
  features: ["full_editing", "kb_integration", "ai_features", "sync"]
```

**UI Mockup**:
```
┌────────────────────────────────────────────────────────────────────────────┐
│  Note Canvas: Research Project                           [Connected Mode]  │
├─────────┬──────────────────────────────────────────────────────────────────┤
│         │  ┌─ Toolbar ──────────────────────────────────────────────────┐  │
│ Notes   │  │ [+ Note] [+ Connection] [Auto-Layout] [Zoom: 100%▼] [Grid] │  │
│         │  └────────────────────────────────────────────────────────────┘  │
│ ▼ All   │                                                                  │
│   • Main│  ╔═══════════════════╗          ╔═══════════════════╗           │
│     thesis│ ║ 📝 Main Thesis    ║─────────▶║ 📝 Key Finding 1  ║           │
│   • Key │  ║                   ║          ║                   ║           │
│     findings║ The research shows ║          ║ 34% improvement   ║           │
│   • Method│  ║ that AI-assisted  ║          ║ in efficiency     ║           │
│         │  ║ workflows...      ║          ║ 📎 report.pdf     ║           │
│ ────────│  ╚═══════════════════╝          ╚═══════════════════╝           │
│         │          │                              │                        │
│ Sources │          │                              │                        │
│         │          ▼                              ▼                        │
│ 📄 report│  ╔═══════════════════╗          ╔═══════════════════╗           │
│ 📄 survey│  ║ 📝 Methodology    ║          ║ 📝 Key Finding 2  ║           │
│ 📊 data │  ║                   ║          ║                   ║           │
│         │  ║ Mixed methods     ║          ║ Cost savings of   ║           │
│ ────────│  ║ approach using... ║          ║ €2.1M achieved    ║           │
│         │  ║ 📎 methodology.pdf║          ║ 📎 financials.xlsx║           │
│ AI Tools│  ╚═══════════════════╝          ╚═══════════════════╝           │
│         │                                                                  │
│ [Generate│  ─────────────────────────────────────────────────────────────  │
│  Note]  │                                                                  │
│         │  Mini-map: [▪]          Pan: Click+Drag    Zoom: Scroll         │
│ [Expand]│                                                                  │
│         │                                                                  │
│ [Suggest│                                                                  │
│  Links] │                                                                  │
└─────────┴──────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  Note Editor                                            [Editing: note:123]│
├────────────────────────────────────────────────────────────────────────────┤
│  Title: Main Thesis                                                        │
│  ─────────────────────────────────────────────────────────────────────────│
│  [B] [I] [H1▼] [Code] [List] [Link] [Image] [Table] │ [AI: Expand▼]       │
│  ─────────────────────────────────────────────────────────────────────────│
│                                                                            │
│  # Main Thesis                                                             │
│                                                                            │
│  The research demonstrates that **AI-assisted workflows** significantly    │
│  improve organizational efficiency while reducing operational costs.       │
│                                                                            │
│  ## Key Points                                                             │
│                                                                            │
│  - 34% efficiency improvement across departments                          │
│  - €2.1M cost reduction in first year                                     │
│  - 89% user satisfaction rating                                           │
│                                                                            │
│  ## Linked Sources                                                         │
│  📎 [report_2024.pdf](source:abc123) - Page 12-15                         │
│  📎 [financials.xlsx](source:def456) - Sheet: Summary                     │
│                                                                            │
│  ─────────────────────────────────────────────────────────────────────────│
│  Tags: [research] [findings] [2024]  │  Created: 2025-01-25  │  Modified: │
│                                                                            │
│  [Save] [Cancel] [Delete] [Link to Source▼] [Export: MD▼]                 │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Docker Architecture

### Design Principle: Standalone-First

Each pipeline is designed to run **independently** with its own Streamlit UI. The Docker architecture should support:
1. Running individual pipelines standalone (no database required for basic operations)
2. Running pipelines connected to the database
3. Running the full application stack

---

### Option 1: Standalone Pipeline Containers

Each pipeline has its own container that can run independently.

```yaml
# docker-compose.standalone.yml
# Run individual pipelines without the full stack

services:
  # Run just the ingestion pipeline with Streamlit UI
  ingestion:
    build:
      context: .
      dockerfile: pipelines/ingestion/Dockerfile
    ports:
      - "8501:8501"  # Streamlit UI
    volumes:
      - ./data/input:/app/input    # Input documents
      - ./data/output:/app/output  # Processed output
    environment:
      - MODE=standalone  # No database connection

  # Run just embeddings pipeline
  embeddings:
    build:
      context: .
      dockerfile: pipelines/embeddings/Dockerfile
    ports:
      - "8502:8501"
    volumes:
      - ./data:/app/data
    environment:
      - MODE=standalone
      - OLLAMA_HOST=http://host.docker.internal:11434  # Use host Ollama

  # Run just summarization
  summarization:
    build:
      context: .
      dockerfile: pipelines/summarization/Dockerfile
    ports:
      - "8504:8501"
    volumes:
      - ./data:/app/data
    environment:
      - MODE=standalone
```

**Usage**:
```bash
# Run just ingestion to process some PDFs
docker compose -f docker-compose.standalone.yml up ingestion

# Run just ontology extraction
docker compose -f docker-compose.standalone.yml up ontology-extraction

# Run multiple pipelines (they're independent)
docker compose -f docker-compose.standalone.yml up ingestion embeddings
```

**Pros**:
- True independence - each pipeline works alone
- Minimal resource usage for specific tasks
- Easy to understand and debug
- No database setup required for basic operations

**Cons**:
- Can't save to database without additional setup
- No integration between pipelines

**Best for**: Quick processing tasks, testing, development

---

### Option 2: Connected Pipeline Containers

Pipelines can connect to database when available.

```yaml
# docker-compose.connected.yml
# Pipelines with optional database connection

services:
  # Infrastructure (optional)
  surrealdb:
    image: surrealdb/surrealdb:latest
    command: start --user root --pass root file:/data/database.db
    volumes:
      - surreal_data:/data
    ports:
      - "8000:8000"

  # Database API (optional, for connected mode)
  surrealdb-api:
    build:
      context: .
      dockerfile: packages/surrealdb-service/Dockerfile
    depends_on: [surrealdb]
    ports:
      - "5100:5100"
    environment:
      - SURREAL_URL=ws://surrealdb:8000/rpc

  # Pipelines with database connection
  ingestion:
    build:
      context: .
      dockerfile: pipelines/ingestion/Dockerfile
    ports:
      - "8501:8501"
    volumes:
      - ./data:/app/data
    environment:
      - MODE=connected
      - SURREAL_URL=ws://surrealdb:8000/rpc
      - SURREALDB_API=http://surrealdb-api:5100
    depends_on:
      - surrealdb-api

  embeddings:
    build:
      context: .
      dockerfile: pipelines/embeddings/Dockerfile
    ports:
      - "8502:8501"
    environment:
      - MODE=connected
      - SURREAL_URL=ws://surrealdb:8000/rpc
    depends_on:
      - surrealdb-api

  # ... other pipelines
```

**Usage**:
```bash
# Start database + specific pipeline
docker compose -f docker-compose.connected.yml up surrealdb surrealdb-api ingestion

# Start all connected
docker compose -f docker-compose.connected.yml up
```

---

### Option 3: Full Application Stack

Complete production deployment with all components.

```yaml
# docker-compose.yml (Production)
services:
  # === Infrastructure ===
  surrealdb:
    image: surrealdb/surrealdb:latest
    command: start --user root --pass root file:/data/database.db
    volumes:
      - surreal_data:/data
    restart: unless-stopped

  redis:
    image: redis:alpine
    restart: unless-stopped

  # === Core Services ===
  surrealdb-api:
    build:
      context: .
      dockerfile: packages/surrealdb-service/Dockerfile
    depends_on: [surrealdb]
    ports:
      - "5100:5100"
    restart: unless-stopped

  # === Main Application ===
  app-backend:
    build:
      context: .
      dockerfile: apps/app-main/Dockerfile.backend
    depends_on: [surrealdb-api, redis]
    ports:
      - "5055:5055"
    restart: unless-stopped

  app-frontend:
    build:
      context: ./apps/app-main/frontend
      dockerfile: Dockerfile
    ports:
      - "3000:3000"
    depends_on: [app-backend]
    restart: unless-stopped

  # === Pipeline Workers (background processing) ===
  worker:
    build:
      context: .
      dockerfile: Dockerfile.worker
    depends_on: [surrealdb-api, redis]
    environment:
      - WORKER_MODE=background
    deploy:
      replicas: 2
    restart: unless-stopped

  # === Optional: Pipeline UIs for debugging ===
  # Uncomment to expose pipeline UIs in production
  # ingestion-ui:
  #   build:
  #     context: .
  #     dockerfile: pipelines/ingestion/Dockerfile
  #   ports:
  #     - "8501:8501"
  #   environment:
  #     - MODE=connected
  #   profiles: ["debug"]

volumes:
  surreal_data:
```

---

### Option 4: Hybrid with On-Demand Pipelines (Recommended)

Core services always running, pipelines started on-demand.

```yaml
# docker-compose.yml
services:
  # === Always Running ===
  surrealdb:
    image: surrealdb/surrealdb:latest
    command: start --user root --pass root file:/data/database.db
    volumes:
      - surreal_data:/data
    ports:
      - "8000:8000"

  surrealdb-api:
    build:
      context: .
      dockerfile: packages/surrealdb-service/Dockerfile
    depends_on: [surrealdb]
    ports:
      - "5100:5100"

  file-manager:
    build:
      context: .
      dockerfile: packages/file-manager/Dockerfile
    depends_on: [surrealdb-api]
    ports:
      - "5110:5110"   # API
      - "8510:8501"   # Streamlit UI
    volumes:
      - ${STORAGE_ROOT:-./data}:/storage  # User-defined storage root
    environment:
      - STORAGE_ROOT=/storage

  app-main:
    build:
      context: .
      dockerfile: apps/app-main/Dockerfile
    depends_on: [surrealdb-api, file-manager]
    ports:
      - "5055:5055"
      - "3000:3000"

  # === On-Demand Pipelines (use profiles) ===
  # All pipelines connect to file-manager for file operations
  ingestion-ui:
    build:
      context: .
      dockerfile: pipelines/ingestion/Dockerfile
    ports:
      - "8501:8501"
    environment:
      - MODE=connected
      - SURREAL_URL=ws://surrealdb:8000/rpc
      - FILE_MANAGER_URL=http://file-manager:5110
    depends_on: [file-manager]
    profiles: ["ingestion", "all-pipelines"]

  embeddings-ui:
    build:
      context: .
      dockerfile: pipelines/embeddings/Dockerfile
    ports:
      - "8502:8501"
    environment:
      - MODE=connected
    profiles: ["embeddings", "all-pipelines"]

  retrieval-ui:
    build:
      context: .
      dockerfile: pipelines/retrieval/Dockerfile
    ports:
      - "8503:8501"
    environment:
      - MODE=connected
    profiles: ["retrieval", "all-pipelines"]

  summarization-ui:
    build:
      context: .
      dockerfile: pipelines/summarization/Dockerfile
    ports:
      - "8504:8501"
    environment:
      - MODE=connected
    profiles: ["summarization", "all-pipelines"]

  # Applications
  chat:
    build:
      context: .
      dockerfile: apps/chat/Dockerfile
    ports:
      - "8530:8501"   # Streamlit UI
      - "5130:5130"   # API
    environment:
      - MODE=connected
      - SURREALDB_URL=ws://surrealdb:8000
    depends_on: [surrealdb, retrieval]
    profiles: ["chat", "apps", "all"]

  canvas:
    build:
      context: .
      dockerfile: apps/canvas/Dockerfile
    ports:
      - "8531:8501"   # Streamlit UI
      - "5131:5131"   # API
    environment:
      - MODE=connected
      - SURREALDB_URL=ws://surrealdb:8000
    depends_on: [surrealdb]
    profiles: ["canvas", "apps", "all"]

  surrealdb-admin:
    build:
      context: .
      dockerfile: packages/surrealdb-service/Dockerfile.ui
    ports:
      - "8500:8501"
    profiles: ["admin", "all-pipelines"]

volumes:
  surreal_data:
```

**Usage**:
```bash
# Start core services only
docker compose up -d

# Start core + ingestion UI when needed
docker compose --profile ingestion up -d

# Start core + all pipeline UIs
docker compose --profile all-pipelines up -d

# Run standalone (no database) - use separate file
docker compose -f docker-compose.standalone.yml up ingestion
```

---

### Docker Decision Matrix

| Factor | Standalone | Connected | Full Stack | Hybrid (Rec.) |
|--------|-----------|-----------|------------|---------------|
| Complexity | Low | Medium | High | Medium |
| Independence | Full | Partial | None | Flexible |
| Resource Usage | Minimal | Medium | High | Flexible |
| Use Case | Quick tasks | Dev/Test | Production | All |

### Your Choice

<!-- Mark your preferred option -->
- [ ] Option 1: Standalone Pipeline Containers
- [ ] Option 2: Connected Pipeline Containers
- [ ] Option 3: Full Application Stack
- [ ] Option 4: Hybrid with On-Demand Pipelines (Recommended)
- [ ] Custom: _____________________

### Port Assignments

| Component | Streamlit Port | API Port |
|-----------|---------------|----------|
| **Core Services** | | |
| surrealdb-admin | 8500 | 5100 |
| file-manager | 8510 | 5110 |
| llm-manager | 8515 | 5120 |
| **Pipelines** | | |
| web-scraper | 8520 | - |
| ingestion | 8521 | - |
| ontology-extraction | 8522 | - |
| summarization | 8523 | - |
| enrichment | 8527 | 5127 |
| embeddings | 8524 | - |
| retrieval | 8525 | - |
| **Applications** | | |
| app-main (frontend) | 3000 | 5055 |
| chat | 8530 | 5130 |
| canvas | 8531 | 5131 |

### Additional Docker Considerations

**Streamlit UIs Deployment**:
- [ ] Always available (include in base docker-compose)
- [ ] On-demand via profiles (recommended)
- [ ] Standalone only (separate compose file)
- [ ] Not in Docker (run locally with `uv run`)

**GPU Support**:
- [ ] Need GPU for embeddings (CUDA container)
- [ ] Need GPU for LLM inference
- [ ] CPU-only is fine
- [ ] Mixed (embeddings on GPU, rest on CPU)

**Message Queue Choice**:
- [ ] Redis (simple, in-memory)
- [ ] RabbitMQ (robust, persistent)
- [ ] surreal-commands (current, database-backed)
- [ ] None (direct API calls only)

**Volume Strategy**:
- [ ] Shared volume for all pipelines (`./data:/app/data`)
- [ ] Separate volumes per pipeline
- [ ] Host mounts for development
- [ ] Named volumes for production

---

## Detailed Implementation Plan

This plan focuses on implementing the workspace structure with ingestion and extraction pipelines first, including all their dependencies.

### Dependency Graph for Initial Implementation

```
                    ┌─────────────────────────────────────────────┐
                    │           packages/shared                    │  ✅
                    │  (base models, utilities, config)            │
                    └─────────────────┬───────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              │                       │                       │
              ▼                       ▼                       ▼
    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
    │ surrealdb-service│    │  file-manager   │    │   llm-manager   │  ✅
    │  (database ops)  │    │ (file storage)  │    │   (LLM calls)   │
    └────────┬────────┘    └────────┬────────┘    └────────┬────────┘
             │                      │                      │
             ├──────────────────────┼──────────────────────┘
             │                      │
             ▼                      ▼
    ┌─────────────────┐    ┌─────────────────────────────────────────────┐
    │ontology-manager │    │         pipelines/ingestion                  │  ✅
    │ (schema, valid.) │    │  (PDF parsing, chunking, source creation)    │
    └────────┬────────┘    └─────────────────┬───────────────────────────┘
             │                               │
             ├───────────────────────────────┤
             ▼                               ▼
    ┌─────────────────────────────────────────────┐
    │     pipelines/ontology-extraction            │  ✅
    │  (pure LLM-based extraction via llm-manager) │
    └─────────────────┬───────────────────────────┘
                      │
                      ▼
    ┌─────────────────────────────────────────────┐
    │     pipelines/entity-filtering               │  ✅
    │  (13-stage: noise → dedup → resolution →     │
    │   validation → composite edge prediction)     │
    └─────────────────────────────────────────────┘
```

### Implementation Order

1. **Step 1**: Base Workspace Structure — ✅ Done
2. **Step 2**: packages/shared — ✅ Done (84 tests)
3. **Step 3**: packages/surrealdb-service — ✅ Done (70 tests)
4. **Step 4**: packages/file-manager — ✅ Done (54 tests)
5. **Step 5**: packages/llm-manager — ✅ Done (64 tests)
6. **Step 6**: pipelines/ingestion — ✅ Done (60 tests)
7. **Step 7a**: packages/ontology-manager — ✅ Done (188 tests)
8. **Step 7b**: pipelines/ontology-extraction (refactored to pure LLM) — ✅ Done (32 tests)
9. **Step 7c**: pipelines/entity-filtering (expanded) — ✅ Done (469 tests)
10. **Step 8**: Source-Centric File Management — ✅ Done (68 tests)
    - Added `SourceFolder` and `PipelineCacheEntry` models to packages/shared
    - Added `SourceFolderRepository` and `PipelineCacheRepository` to packages/surrealdb-service
    - Added `SourceFolderService`, `PipelineCacheService`, `DuplicateDetector` services to packages/file-manager
    - Added source-folders API router (14 endpoints) to packages/file-manager
    - Added migration 26 (source_folder + pipeline_cache tables)
    - Integrated source-folder-aware exports into ingestion pipeline exporters
    - Per-source directories with `{source_id}_` file prefixing for global uniqueness
    - Versioned pipeline cache with automatic timestamped archiving (never deletes)
    - Metadata-based duplicate detection (no content hashing)
11. **Step 9**: Integration Testing — Pending
12. **Step 10**: Remaining pipelines (web-scraper, summarization, enrichment, embeddings, retrieval)
13. **Step 11**: Applications (app-main, chat, canvas)

---

## Step 1: Base Workspace Structure ✅

**Goal**: Create the UV workspace skeleton with all directories and root configuration.

### 1.1 Create Root pyproject.toml

```toml
# /pyproject.toml
[project]
name = "open-notebook"
version = "2.0.0"
description = "Knowledge management system with RAG pipelines"
requires-python = ">=3.11"
readme = "README.md"

[tool.uv.workspace]
members = [
    "packages/*",
    "pipelines/*",
    "apps/*",
]

[tool.uv.sources]
shared = { workspace = true }
surrealdb-service = { workspace = true }
file-manager = { workspace = true }
llm-manager = { workspace = true }
ingestion = { workspace = true }
ontology-extraction = { workspace = true }
# Add more as implemented

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
```

### 1.2 Create Directory Structure

```bash
# Run from project root
mkdir -p packages/{shared,surrealdb-service,file-manager,llm-manager}/src
mkdir -p pipelines/{ingestion,ontology-extraction,summarization,enrichment,embeddings,retrieval,web-scraper}/src
mkdir -p apps/{app-main,chat,canvas}/src
```

### 1.3 Verification Checklist

- [x] Root pyproject.toml created with workspace config
- [x] All directories created
- [x] `uv sync` runs without errors (even with empty packages)
- [x] Git: Create feature branch `feature/workspace-refactor`

---

## Step 2: packages/shared ✅

**Goal**: Create the foundation package with base models, utilities, and configuration that all other packages depend on.

### 2.1 Package Structure

```
packages/shared/
├── pyproject.toml
├── README.md
└── src/shared/
    ├── __init__.py
    ├── config.py              # Environment config, settings
    ├── models/
    │   ├── __init__.py
    │   ├── base.py            # ObjectModel base class
    │   ├── source.py          # Source model
    │   ├── chunk.py           # Chunk model
    │   ├── notebook.py        # Notebook, Note models
    │   ├── knowledge_graph.py # Entity, Relation models
    │   └── content_settings.py # Pipeline settings
    ├── utils/
    │   ├── __init__.py
    │   ├── text.py            # Text processing utilities
    │   ├── hashing.py         # Content hashing
    │   └── logging.py         # Logging configuration
    └── types/
        ├── __init__.py
        └── pipeline.py        # PipelineTask enum, shared types
```

### 2.2 pyproject.toml

```toml
# packages/shared/pyproject.toml
[project]
name = "shared"
version = "0.1.0"
description = "Shared models and utilities for open-notebook"
requires-python = ">=3.11"
dependencies = [
    "pydantic>=2.0",
    "loguru",
    "python-dotenv",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/shared"]
```

### 2.3 Files to Migrate

| Current Location | New Location | Action |
|-----------------|--------------|--------|
| `open_notebook/domain/base.py` | `packages/shared/src/shared/models/base.py` | Copy + refactor |
| `open_notebook/domain/notebook.py` | `packages/shared/src/shared/models/notebook.py` | Copy + refactor |
| `open_notebook/domain/content_settings.py` | `packages/shared/src/shared/models/content_settings.py` | Copy + refactor |
| `open_notebook/domain/knowledge_graph.py` | `packages/shared/src/shared/models/knowledge_graph.py` | Copy + refactor |
| `open_notebook/config.py` (parts) | `packages/shared/src/shared/config.py` | Extract shared config |

### 2.4 Key Implementation Tasks

- [x] 2.4.1 Create package structure and pyproject.toml
- [x] 2.4.2 Migrate `ObjectModel` base class (remove database-specific code)
- [x] 2.4.3 Create pure Pydantic models (no database operations)
- [x] 2.4.4 Create `PipelineTask` enum with all task types
- [x] 2.4.5 Create shared configuration class
- [x] 2.4.6 Create text utilities (chunking helpers, text cleaning)
- [x] 2.4.7 Write unit tests for models (84 tests)
- [x] 2.4.8 Verify: `uv sync` and `uv run pytest packages/shared`
- [x] 2.4.9 Added extraction models: `ExtractedEntity`, `ExtractedRelation`, `ExtractionResult`, `FilteredResult`

### 2.5 Critical Design Decision: Pure Models

**Important**: Models in `shared` must be **pure Pydantic models** without database operations.

```python
# WRONG - Don't include database operations in shared models
class Source(ObjectModel):
    def save(self):  # NO - this couples to database
        db.query(...)

# CORRECT - Pure data models only
class Source(BaseModel):
    id: str
    title: str
    content: str
    # ... fields only, no methods that touch DB
```

Database operations go in `surrealdb-service`.

---

## Step 3: packages/surrealdb-service ✅

**Goal**: Centralize all SurrealDB operations with REST API, MCP server, and admin UI.

### 3.1 Package Structure

```
packages/surrealdb-service/
├── pyproject.toml
├── README.md
├── main.py                    # CLI entry point
├── ui.py                      # Streamlit admin UI
└── src/surrealdb_service/
    ├── __init__.py
    ├── client.py              # SurrealDB async client wrapper
    ├── repositories/
    │   ├── __init__.py
    │   ├── base.py            # BaseRepository[T]
    │   ├── source.py          # SourceRepository
    │   ├── chunk.py           # ChunkRepository
    │   ├── notebook.py        # NotebookRepository
    │   ├── entity.py          # EntityRepository
    │   └── settings.py        # SettingsRepository
    ├── api/
    │   ├── __init__.py
    │   ├── app.py             # FastAPI app
    │   └── routers/
    │       ├── sources.py
    │       ├── chunks.py
    │       ├── notebooks.py
    │       └── entities.py
    ├── mcp/
    │   ├── __init__.py
    │   └── server.py          # MCP server implementation
    └── migrations/
        └── manager.py         # Migration management
```

### 3.2 pyproject.toml

```toml
[project]
name = "surrealdb-service"
version = "0.1.0"
description = "SurrealDB service layer for open-notebook"
requires-python = ">=3.11"
dependencies = [
    "shared",                  # Workspace dependency
    "surrealdb>=0.3.0",
    "fastapi>=0.100.0",
    "uvicorn[standard]",
    "httpx",
    "mcp",
]

[project.optional-dependencies]
ui = ["streamlit>=1.30.0"]

[project.scripts]
surrealdb-service = "surrealdb_service.api.app:main"
surrealdb-mcp = "surrealdb_service.mcp.server:main"

[tool.uv.sources]
shared = { workspace = true }
```

### 3.3 Files to Migrate

| Current Location | New Location | Action |
|-----------------|--------------|--------|
| `open_notebook/database/repository.py` | `src/surrealdb_service/repositories/base.py` | Refactor to generic |
| `open_notebook/database/client.py` | `src/surrealdb_service/client.py` | Copy + improve |
| `api/routers/sources.py` | `src/surrealdb_service/api/routers/sources.py` | Move CRUD parts |
| `api/routers/notebooks.py` | `src/surrealdb_service/api/routers/notebooks.py` | Move CRUD parts |

### 3.4 Key Implementation Tasks

- [x] 3.4.1 Create package structure and pyproject.toml
- [x] 3.4.2 Implement `SurrealDBClient` async wrapper
- [x] 3.4.3 Create `BaseRepository[T]` generic class
- [x] 3.4.4 Implement `SourceRepository` with CRUD operations
- [x] 3.4.5 Implement `ChunkRepository` with vector search
- [x] 3.4.6 Implement `EntityRepository` for knowledge graph
- [x] 3.4.7 Create FastAPI REST API with all routers
- [ ] 3.4.8 Create MCP server for Claude access
- [ ] 3.4.9 Create Streamlit admin UI (database browser, query runner)
- [x] 3.4.10 Write integration tests with test database (70 tests)
- [ ] 3.4.11 Verify: API runs on port 5100, UI on 8500

### 3.5 Repository Pattern Implementation

```python
# src/surrealdb_service/repositories/base.py
from typing import TypeVar, Generic, List, Optional
from shared.models.base import BaseModel

T = TypeVar('T', bound=BaseModel)

class BaseRepository(Generic[T]):
    def __init__(self, client: SurrealDBClient, table: str, model_class: type[T]):
        self.client = client
        self.table = table
        self.model_class = model_class

    async def get(self, id: str) -> Optional[T]:
        result = await self.client.query(f"SELECT * FROM {self.table} WHERE id = $id", {"id": id})
        return self.model_class(**result[0]) if result else None

    async def create(self, item: T) -> T:
        result = await self.client.query(f"CREATE {self.table} CONTENT $data", {"data": item.model_dump()})
        return self.model_class(**result[0])

    async def update(self, id: str, item: T) -> T:
        ...

    async def delete(self, id: str) -> bool:
        ...

    async def list(self, limit: int = 100, offset: int = 0) -> List[T]:
        ...
```

---

## Step 4: packages/file-manager ✅

**Goal**: Centralize all file operations, storage management, and knowledge base organization.

### 4.1 Package Structure

```
packages/file-manager/
├── pyproject.toml
├── README.md
├── main.py
├── ui.py
└── src/file_manager/
    ├── __init__.py
    ├── config.py              # Storage paths configuration
    ├── storage/
    │   ├── __init__.py
    │   ├── base.py            # StorageBackend ABC
    │   ├── local.py           # LocalStorageBackend
    │   └── paths.py           # Path resolution utilities
    ├── knowledge_base/
    │   ├── __init__.py
    │   ├── manager.py         # KnowledgeBaseManager
    │   └── models.py          # KnowledgeBase, Project models
    ├── files/
    │   ├── __init__.py
    │   ├── tracker.py         # FileTracker (DB integration)
    │   ├── operations.py      # File CRUD operations
    │   └── watcher.py         # File system watcher
    ├── obsidian/
    │   ├── __init__.py
    │   ├── vault.py           # Obsidian vault management
    │   └── exporter.py        # Export to Obsidian format
    ├── api/
    │   ├── __init__.py
    │   ├── app.py
    │   └── routers/
    │       ├── files.py
    │       ├── knowledge_bases.py
    │       └── obsidian.py
    └── mcp/
        ├── __init__.py
        └── server.py
```

### 4.2 pyproject.toml

```toml
[project]
name = "file-manager"
version = "0.1.0"
description = "File and knowledge base management for open-notebook"
requires-python = ">=3.11"
dependencies = [
    "shared",
    "surrealdb-service",
    "aiofiles",
    "watchfiles",
    "python-magic",
    "fastapi>=0.100.0",
    "mcp",
]

[tool.uv.sources]
shared = { workspace = true }
surrealdb-service = { workspace = true }
```

### 4.3 Key Implementation Tasks

- [x] 4.3.1 Create package structure and pyproject.toml
- [x] 4.3.2 Implement `StorageConfig` with environment-based paths
- [x] 4.3.3 Implement `LocalStorageBackend` for file operations
- [x] 4.3.4 Implement `KnowledgeBaseManager` for KB CRUD
- [x] 4.3.5 Implement `FileTracker` for database sync
- [ ] 4.3.6 Create Obsidian vault management and export
- [x] 4.3.7 Create FastAPI REST API
- [ ] 4.3.8 Create MCP server for Claude file access
- [ ] 4.3.9 Create Streamlit file browser UI
- [x] 4.3.10 Write tests for file operations (54 tests)
- [ ] 4.3.11 Verify: API runs on port 5110, UI on 8510

### 4.4 Storage Configuration

```python
# src/file_manager/config.py
from pydantic_settings import BaseSettings
from pathlib import Path

class StorageConfig(BaseSettings):
    storage_root: Path = Path.home() / ".open-notebook"

    @property
    def knowledgebases_path(self) -> Path:
        return self.storage_root / "knowledgebases"

    @property
    def projects_path(self) -> Path:
        return self.storage_root / "projects"

    @property
    def obsidian_path(self) -> Path:
        return self.storage_root / "obsidian"

    @property
    def temp_path(self) -> Path:
        return self.storage_root / "temp"

    def ensure_directories(self):
        for path in [self.knowledgebases_path, self.projects_path,
                     self.obsidian_path, self.temp_path]:
            path.mkdir(parents=True, exist_ok=True)
```

---

## Step 5: packages/llm-manager ✅

**Goal**: Centralize LLM provider management, model routing, and token tracking.

### 5.1 Package Structure

```
packages/llm-manager/
├── pyproject.toml
├── README.md
├── main.py
├── ui.py
└── src/llm_manager/
    ├── __init__.py
    ├── config.py              # Model configurations
    ├── providers/
    │   ├── __init__.py
    │   ├── base.py            # BaseLLMProvider ABC
    │   ├── ollama.py          # OllamaProvider
    │   ├── openai.py          # OpenAIProvider
    │   ├── anthropic.py       # AnthropicProvider
    │   └── openrouter.py      # OpenRouterProvider
    ├── router.py              # Model router (task → provider)
    ├── usage/
    │   ├── __init__.py
    │   ├── tracker.py         # Token usage tracking
    │   └── cost.py            # Cost calculation
    ├── api/
    │   ├── __init__.py
    │   ├── app.py
    │   └── routers/
    │       ├── chat.py
    │       ├── models.py
    │       └── usage.py
    └── mcp/
        ├── __init__.py
        └── server.py
```

### 5.2 pyproject.toml

```toml
[project]
name = "llm-manager"
version = "0.1.0"
description = "LLM provider management for open-notebook"
requires-python = ">=3.11"
dependencies = [
    "shared",
    "surrealdb-service",
    "esperanto>=0.5.0",        # Multi-provider LLM library
    "httpx",
    "tiktoken",
    "fastapi>=0.100.0",
    "mcp",
]

[tool.uv.sources]
shared = { workspace = true }
surrealdb-service = { workspace = true }
```

### 5.3 Key Implementation Tasks

- [x] 5.3.1 Create package structure and pyproject.toml
- [x] 5.3.2 Implement `BaseLLMProvider` abstract class
- [x] 5.3.3 Implement `OllamaProvider` (primary local provider)
- [x] 5.3.4 Implement `OpenAIProvider` for GPT models
- [x] 5.3.5 Implement `ModelRouter` for task → model mapping
- [x] 5.3.6 Implement token tracking and cost calculation
- [x] 5.3.7 Create FastAPI REST API
- [ ] 5.3.8 Create Streamlit model management UI
- [x] 5.3.9 Write tests for providers (64 tests)
- [ ] 5.3.10 Verify: API runs on port 5120, UI on 8515

### 5.4 Model Router Implementation

```python
# src/llm_manager/router.py
from shared.types.pipeline import PipelineTask

class ModelRouter:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.default_models = {
            PipelineTask.INGESTION_PARSE: "qwen2.5:14b",
            PipelineTask.EXTRACTION_OPENIE: "qwen2.5:14b",
            PipelineTask.SUMMARIZATION_RAPTOR: "qwen2.5:14b",
            PipelineTask.ENRICHMENT_LOOKUP: "qwen2.5:14b",
            # ... etc
        }

    def get_model_for_task(self, task: PipelineTask) -> str:
        # Check user overrides first, then defaults
        return self.config.overrides.get(task) or self.default_models.get(task)

    def get_provider_for_model(self, model: str) -> BaseLLMProvider:
        # Route model name to appropriate provider
        if ":" in model:  # Ollama format
            return self.ollama_provider
        elif model.startswith("gpt"):
            return self.openai_provider
        # ... etc
```

---

## Step 6: pipelines/ingestion ✅

**Goal**: Create the ingestion pipeline for processing documents into the knowledge base.

### 6.1 Package Structure

```
pipelines/ingestion/
├── pyproject.toml
├── README.md
├── main.py                    # CLI entry point
├── ui.py                      # Streamlit UI
└── src/ingestion/
    ├── __init__.py
    ├── service.py             # IngestionService orchestrator
    ├── parsers/
    │   ├── __init__.py
    │   ├── base.py            # BaseParser ABC
    │   ├── pdf.py             # Docling PDF parser
    │   ├── html.py            # HTML/web parser
    │   ├── audio.py           # WhisperX transcription
    │   ├── video.py           # YouTube/video handler
    │   └── text.py            # Plain text/markdown
    ├── chunking/
    │   ├── __init__.py
    │   ├── strategies.py      # ChunkingStrategy enum
    │   ├── semantic.py        # Semantic chunking
    │   ├── fixed.py           # Fixed-size chunking
    │   └── hybrid.py          # Hybrid approach
    ├── models/
    │   ├── __init__.py
    │   ├── document.py        # ParsedDocument, Element
    │   └── chunk.py           # Chunk with metadata
    ├── api/
    │   ├── __init__.py
    │   ├── router.py
    │   └── endpoints.py
    └── cli.py                 # CLI commands
```

### 6.2 pyproject.toml

```toml
[project]
name = "ingestion"
version = "0.1.0"
description = "Document ingestion pipeline for open-notebook"
requires-python = ">=3.11"
dependencies = [
    "shared",
    "surrealdb-service",
    "file-manager",
    "llm-manager",
    "docling>=2.0.0",          # PDF parsing
    "whisperx",                # Audio transcription
    "yt-dlp",                  # YouTube downloads
    "beautifulsoup4",          # HTML parsing
    "httpx",
    "aiofiles",
]

[project.optional-dependencies]
ui = ["streamlit>=1.30.0"]
gpu = ["torch", "torchaudio"]  # For WhisperX GPU

[project.scripts]
ingestion = "ingestion.cli:main"

[tool.uv.sources]
shared = { workspace = true }
surrealdb-service = { workspace = true }
file-manager = { workspace = true }
llm-manager = { workspace = true }
```

### 6.3 Files to Migrate

| Current Location | New Location | Action |
|-----------------|--------------|--------|
| `open_notebook/graphs/source.py` | `src/ingestion/service.py` | Refactor to service |
| `open_notebook/processors/chunk_extractor.py` | `src/ingestion/chunking/` | Split by strategy |
| `scripts/batch_parse_with_pages.py` | `src/ingestion/parsers/pdf.py` | Integrate Docling |
| `open_notebook/plugins/audio.py` | `src/ingestion/parsers/audio.py` | WhisperX integration |

### 6.4 Key Implementation Tasks

- [x] 6.4.1 Create package structure and pyproject.toml
- [x] 6.4.2 Implement `BaseParser` abstract class
- [x] 6.4.3 Implement `PDFParser` using Docling with GPU support
- [x] 6.4.4 Implement `AudioParser` using WhisperX
- [x] 6.4.5 Implement `HTMLParser` for web content
- [x] 6.4.6 Implement chunking strategies (semantic, fixed, hybrid)
- [x] 6.4.7 Create `IngestionService` orchestrator
- [x] 6.4.8 Integrate with file-manager for storage
- [x] 6.4.9 Integrate with surrealdb-service for persistence
- [x] 6.4.10 Create CLI commands
- [ ] 6.4.11 Create Streamlit UI (file upload, progress, results) *(deferred)*
- [x] 6.4.12 Write tests for each parser (60 tests)
- [ ] 6.4.13 Verify: CLI and UI work independently

### 6.5 IngestionService Design

```python
# src/ingestion/service.py
from dataclasses import dataclass
from typing import List, Optional
from file_manager import FileManager
from surrealdb_service import SourceRepository, ChunkRepository
from llm_manager import LLMManager

@dataclass
class IngestionResult:
    source_id: str
    chunks_created: int
    parse_time: float
    chunk_time: float
    errors: List[str]

class IngestionService:
    def __init__(
        self,
        file_manager: FileManager,
        source_repo: SourceRepository,
        chunk_repo: ChunkRepository,
        llm_manager: Optional[LLMManager] = None,  # For semantic chunking
    ):
        self.file_manager = file_manager
        self.source_repo = source_repo
        self.chunk_repo = chunk_repo
        self.llm_manager = llm_manager
        self.parsers = self._init_parsers()

    async def ingest(
        self,
        file_path: str,
        knowledge_base_id: str,
        chunking_strategy: str = "semantic",
        **options
    ) -> IngestionResult:
        # 1. Detect file type and select parser
        parser = self._select_parser(file_path)

        # 2. Parse document
        parsed = await parser.parse(file_path)

        # 3. Create source record
        source = await self.source_repo.create(Source(
            title=parsed.title,
            content=parsed.full_text,
            knowledge_base_id=knowledge_base_id,
            file_path=file_path,
            metadata=parsed.metadata,
        ))

        # 4. Chunk content
        chunks = await self._chunk(parsed, chunking_strategy)

        # 5. Store chunks
        for chunk in chunks:
            chunk.source_id = source.id
            await self.chunk_repo.create(chunk)

        # 6. Copy file to storage
        await self.file_manager.store(file_path, knowledge_base_id, source.id)

        return IngestionResult(
            source_id=source.id,
            chunks_created=len(chunks),
            ...
        )
```

### 6.6 Streamlit UI Mockup

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Ingestion Pipeline                               [Standalone Mode] ⚡  │
├─────────────┬───────────────────────────────────────────────────────────┤
│ 📄 Upload   │  Document Ingestion                                      │
│ ⚙️ Settings │                                                          │
│ 📊 Results  │  ┌─ Upload ────────────────────────────────────────────┐ │
│ 📜 History  │  │                                                      │ │
│             │  │  📁 Drop files here or click to browse              │ │
│             │  │                                                      │ │
│             │  │  Supported: PDF, HTML, TXT, MD, MP3, MP4, YouTube   │ │
│             │  └──────────────────────────────────────────────────────┘ │
│             │                                                          │
│             │  ┌─ Options ───────────────────────────────────────────┐ │
│             │  │ Knowledge Base: [Select or Create ▾]                │ │
│             │  │ Chunking: (•) Semantic  ( ) Fixed  ( ) Hybrid       │ │
│             │  │ Chunk Size: [500] tokens    Overlap: [50] tokens    │ │
│             │  │ ☑ Extract page numbers   ☑ GPU acceleration        │ │
│             │  └──────────────────────────────────────────────────────┘ │
│             │                                                          │
│             │  [🚀 Start Ingestion]                                    │
└─────────────┴──────────────────────────────────────────────────────────┘
```

---

## Step 7a: packages/ontology-manager ✅

**Goal**: Ontology schema management — loading, versioning, validation, evolution tracking, prompt generation.

Migrated from `open_notebook/ontology/` with import refactoring:
- `schema.py` (424 lines) — Zero changes, pure Pydantic models
- `registry.py` (341 lines) — Replaced `open_notebook.database.repository` → `surrealdb_service.repositories`
- `validator.py` (530 lines) — SHACL-like validation for entities, properties, relationships
- `evolution.py` (619 lines) — Gap tracking and schema proposal logic
- `prompts.py` (515 lines) — LLM prompt generation for extraction
- `document_mapper.py` (125 lines) — Document type → ontology mapping
- `config.py` — `OntologyManagerConfig(BaseSettings)` with `ONTOLOGY_` env prefix
- `manager.py` — Singleton facade coordinating all submodules

**Tests**: 188 passing (schema, config, validator, prompts, document_mapper, manager)

---

## Step 7b: pipelines/ontology-extraction ✅ (refactored)

**Goal**: Create the entity and relation extraction pipeline using OpenIE with ontology support.

### 7.1 Package Structure

```
pipelines/ontology-extraction/
├── pyproject.toml
├── README.md
├── main.py
├── ui.py
└── src/ontology_extraction/
    ├── __init__.py
    ├── service.py             # ExtractionService orchestrator
    ├── extractors/
    │   ├── __init__.py
    │   ├── base.py            # BaseExtractor ABC
    │   ├── openie.py          # LLM-based OpenIE
    │   ├── ner.py             # Named Entity Recognition
    │   └── relation.py        # Relation extraction
    ├── ontology/
    │   ├── __init__.py
    │   ├── loader.py          # Load TTL/OWL ontologies
    │   ├── matcher.py         # Match entities to ontology
    │   └── schema.py          # Ontology schema models
    ├── postprocessing/
    │   ├── __init__.py
    │   ├── deduplication.py   # Entity deduplication
    │   ├── linking.py         # Entity linking
    │   └── validation.py      # Schema validation
    ├── models/
    │   ├── __init__.py
    │   ├── entity.py          # ExtractedEntity
    │   ├── relation.py        # ExtractedRelation
    │   └── result.py          # ExtractionResult
    ├── api/
    │   ├── __init__.py
    │   └── router.py
    └── cli.py
```

### 7.2 pyproject.toml

```toml
[project]
name = "ontology-extraction"
version = "0.1.0"
description = "Entity and relation extraction pipeline for open-notebook"
requires-python = ">=3.11"
dependencies = [
    "shared",
    "surrealdb-service",
    "llm-manager",
    "rdflib",                  # Ontology parsing (TTL/OWL)
    "networkx",                # Graph operations
    "spacy>=3.7.0",            # NER support
    "httpx",
]

[project.optional-dependencies]
ui = ["streamlit>=1.30.0"]

[project.scripts]
ontology-extraction = "ontology_extraction.cli:main"

[tool.uv.sources]
shared = { workspace = true }
surrealdb-service = { workspace = true }
llm-manager = { workspace = true }
```

### 7.3 Files to Migrate

| Current Location | New Location | Action |
|-----------------|--------------|--------|
| `open_notebook/processors/openie.py` | `src/ontology_extraction/extractors/openie.py` | Refactor |
| `open_notebook/processors/entity_linking.py` | `src/ontology_extraction/postprocessing/linking.py` | Move |
| `open_notebook/ontology/` | `src/ontology_extraction/ontology/` | Move directory |
| `scripts/ontologie_extract_v3.py` | `src/ontology_extraction/extractors/openie.py` | Integrate |

### 7.4 Key Implementation Tasks

- [x] 7b.1 Refactored to pure LLM-based extraction (removed spacy, rdflib, networkx deps)
- [x] 7b.2 Implement `ExtractorBase` abstract class
- [x] 7b.3 Implement `LLMExtractor` using llm-manager + ontology-manager prompts
- [x] 7b.4 Create `ExtractionWorkflow` orchestrator (batch processing with chunk_id tagging)
- [x] 7b.5 Create `ExtractionConfig` dataclass
- [x] 7b.6 Create CLI entry point
- [x] 7b.7 Write tests (32 tests — config, extractors, workflow)
- [ ] 7b.8 Entity deduplication → moved to entity-filtering pipeline
- [ ] 7b.9 Entity linking across sources → moved to entity-filtering pipeline

---

## Step 7c: pipelines/entity-filtering ✅ (expanded)

**Goal**: Generic, pluggable entity/relation filtering, deduplication, resolution, validation, and scoring pipeline.

New pipeline (not migrated — fresh implementation inspired by monolith patterns). Has grown
significantly beyond the original 5-stage design into a comprehensive 13-stage pipeline.

**Source modules** (25 files across 6 subpackages):

| Subpackage | Module | Description |
|------------|--------|-------------|
| `filters/` | `base.py` | Base filter interface |
| | `noise_filter.py` | Citation, number, URL, punctuation removal + custom patterns |
| | `normalizer.py` | Article stripping, NFKC, whitespace, diacritics, OCR cleanup, HTML strip |
| | `reclassifier.py` | Generic rules (hyphenated→PERSON, all-caps→ABBREVIATION) + custom rules |
| `deduplication/` | `entity_deduplicator.py` | Case-insensitive string dedup with merge group tracking |
| | `fuzzy_resolver.py` | Levenshtein / Jaro-Winkler fuzzy matching with phonetic support |
| | `embedding_deduplicator.py` | Semantic dedup via embedding similarity (FAISS optional) |
| | `union_find.py` | Union-Find data structure for merge group tracking |
| `resolution/` | `embedding_resolver.py` | Semantic match enrichment via embedding similarity |
| | `entity_linker.py` | External KB linking (DBpedia Spotlight) |
| | `contextual_clusterer.py` | Co-occurrence-based entity clustering |
| | `kg_resolver.py` | Match against existing KG entities (cascade/fuzzy/semantic) |
| `validation/` | `ontology_constraint_filter.py` | Validate entities/relations against an ontology schema |
| | `graph_analyzer.py` | PageRank/betweenness centrality filtering + outlier classification |
| `scoring/` | `edge_predictor.py` | Composite edge prediction: cosine similarity + Adamic-Adar + hierarchy/source-proximity |
| `summarization/` | *(empty)* | Reserved for TreeKG/RAPTOR integration |
| *(root)* | `workflow.py` | 13-stage orchestrator |
| | `config.py` | `FilteringConfig` + 8 sub-config dataclasses |
| | `cli.py` | CLI entry point |

**13-stage workflow** (all stages after stage 4 are optional, default disabled):

1. Noise filtering
2. Normalization (with syntactic pre-processing: diacritics, OCR, HTML, page numbers)
3. Reclassification
4. String deduplication
5. Fuzzy resolution (Levenshtein/Jaro-Winkler + phonetic)
6. Embedding deduplication (semantic, FAISS-accelerated)
7. Embedding resolution (semantic match enrichment)
8. Entity linking (DBpedia Spotlight)
9. Contextual clustering (co-occurrence)
10. KG resolution (cascade against existing knowledge graph)
11. Ontology constraint validation
12. Graph centrality analysis (PageRank/betweenness, outlier detection)
13. Edge prediction (cosine + Adamic-Adar + hierarchy/source-proximity composite scoring)

**Config sub-dataclasses**: `SyntacticConfig`, `FuzzyDedupConfig`, `EmbeddingDedupConfig`,
`SemanticConfig`, `KGResolutionConfig`, `OntologyValidationConfig`, `LLMVerificationConfig`,
`EdgePredictionConfig` — all default to disabled for backward compatibility.

**Edge predictor** (Phase 5 complete): Ported from monolith with 3 scoring algorithms.
Accepts an optional `hierarchy_graph: nx.DiGraph` for TreeKG common-ancestor scoring;
falls back to `source_chunk_id` proximity when no hierarchy is available.

**Tests**: 469 passing across 19 test files (config, config_extended, noise_filter, normalizer,
normalizer_extended, reclassifier, deduplicator, fuzzy_resolver, embedding_deduplicator,
union_find, embedding_resolver, entity_linker, contextual_clusterer, kg_resolver,
ontology_constraint_filter, graph_analyzer, edge_predictor, workflow, workflow_all_options)

### 7.5 OpenIE Extractor Design

```python
# src/ontology_extraction/extractors/openie.py
from llm_manager import LLMManager
from shared.types.pipeline import PipelineTask

EXTRACTION_PROMPT = """Extract all entities, concepts, and relations from the following text.

Ontology Schema:
{ontology_schema}

Text:
{text}

Output as JSON following this schema:
{output_schema}
"""

class OpenIEExtractor:
    def __init__(self, llm_manager: LLMManager, ontology: Optional[Ontology] = None):
        self.llm_manager = llm_manager
        self.ontology = ontology

    async def extract(self, text: str, source_id: str) -> ExtractionResult:
        # Get model for this task
        model = self.llm_manager.get_model_for_task(PipelineTask.EXTRACTION_OPENIE)

        # Build prompt with ontology schema if available
        prompt = EXTRACTION_PROMPT.format(
            ontology_schema=self.ontology.to_prompt() if self.ontology else "None",
            text=text,
            output_schema=ExtractionResult.schema_json(),
        )

        # Call LLM
        response = await self.llm_manager.chat(model, prompt, json_mode=True)

        # Parse and validate result
        result = ExtractionResult.model_validate_json(response)

        # Post-process: deduplicate, link, validate
        result = await self._postprocess(result, source_id)

        return result
```

### 7.6 Streamlit UI Mockup

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Ontology Extraction Pipeline                     [Connected Mode] 🔗   │
├─────────────┬───────────────────────────────────────────────────────────┤
│ 🎯 Extract  │  Entity Extraction                                       │
│ 📚 Ontology │                                                          │
│ 🔗 Graph    │  Source: [Select Source ▾]        [🔄 Refresh]           │
│ ⚙️ Settings │                                                          │
│             │  ┌─ Extraction Options ────────────────────────────────┐ │
│             │  │ Ontology: [Dutch Policy Ontology ▾] [📤 Upload New] │ │
│             │  │ Extractor: (•) OpenIE (LLM)  ( ) NER only           │ │
│             │  │ ☑ Multi-pass extraction  ☑ Entity linking          │ │
│             │  │ ☑ Deduplication          ☑ Schema validation       │ │
│             │  └──────────────────────────────────────────────────────┘ │
│             │                                                          │
│             │  ┌─ Extraction Results ────────────────────────────────┐ │
│             │  │ Entities: 47  │  Relations: 23  │  Concepts: 15    │ │
│             │  ├────────────────┼─────────────────┼──────────────────┤ │
│             │  │ 🏛️ Organizations: 12                                │ │
│             │  │ 📍 Locations: 8                                      │ │
│             │  │ 👤 Persons: 5                                        │ │
│             │  │ 📊 Indicators: 15                                    │ │
│             │  │ 💰 Financial: 7                                      │ │
│             │  └──────────────────────────────────────────────────────┘ │
│             │                                                          │
│             │  [🚀 Run Extraction] [📥 Export JSON] [💾 Save to DB]   │
└─────────────┴──────────────────────────────────────────────────────────┘
```

---

## Step 8: Integration Testing

**Goal**: Verify all components work together in the complete pipeline flow.

### 8.1 Integration Test Scenarios

| Test | Description | Components |
|------|-------------|------------|
| E2E-1 | Ingest PDF → Extract entities → Verify in DB | ingestion, ontology-extraction, surrealdb-service |
| E2E-2 | File manager stores files in correct KB | file-manager, surrealdb-service |
| E2E-3 | LLM manager routes tasks correctly | llm-manager, all pipelines |
| E2E-4 | Standalone mode works without database | ingestion, ontology-extraction |
| E2E-5 | Connected mode persists all data | All components |

### 8.2 Test Implementation Tasks

- [ ] 8.2.1 Create test fixtures (sample PDF, ontology)
- [ ] 8.2.2 Write E2E test: ingest → extract → verify
- [ ] 8.2.3 Write standalone mode tests for each pipeline
- [ ] 8.2.4 Write connected mode integration tests
- [ ] 8.2.5 Write API endpoint tests
- [ ] 8.2.6 Verify Docker compose brings up all services
- [ ] 8.2.7 Performance benchmarks for ingestion

### 8.3 Integration Test Example

```python
# tests/integration/test_pipeline_flow.py
import pytest
from ingestion import IngestionService
from ontology_extraction import ExtractionService
from surrealdb_service import SurrealDBClient, SourceRepository, EntityRepository

@pytest.mark.asyncio
async def test_ingest_and_extract_pdf():
    # Setup
    client = SurrealDBClient("ws://localhost:8000/rpc")
    await client.connect()

    source_repo = SourceRepository(client)
    entity_repo = EntityRepository(client)

    ingestion = IngestionService(source_repo=source_repo, ...)
    extraction = ExtractionService(entity_repo=entity_repo, ...)

    # Ingest
    result = await ingestion.ingest(
        file_path="tests/fixtures/sample.pdf",
        knowledge_base_id="test-kb",
    )
    assert result.chunks_created > 0

    # Extract
    extract_result = await extraction.extract(source_id=result.source_id)
    assert len(extract_result.entities) > 0

    # Verify in database
    source = await source_repo.get(result.source_id)
    assert source is not None
    assert source.status == "extracted"

    entities = await entity_repo.list_for_source(result.source_id)
    assert len(entities) == len(extract_result.entities)
```

---

## Implementation Timeline

```
✅ COMPLETED:
├── Steps 1-2: Workspace structure + shared package (86 tests)
├── Step 3: surrealdb-service (28 tests)
├── Steps 4-5: file-manager (83 tests) + llm-manager (73 tests)
├── Step 6: ingestion pipeline (62 tests)
├── Step 7a: ontology-manager package (188 tests)
├── Step 7b: ontology-extraction refactored to pure LLM (32 tests)
└── Step 7c: entity-filtering pipeline (469 tests, 13-stage pipeline)

REMAINING:
├── Step 8: Integration testing across pipelines
├── Step 9: Remaining pipelines
│   ├── web-scraper
│   ├── summarization (TreeKG/RAPTOR — edge predictor already accepts hierarchy_graph)
│   ├── enrichment
│   ├── embeddings
│   └── retrieval
├── Step 10: Applications (app-main, chat, canvas)
└── Step 11: Docker configuration + deployment
```

---

## Quick Reference: Commands

```bash
# Build entire workspace
uv sync --all-packages --all-extras

# Run all tests (1021 total)
uv run pytest packages/ pipelines/

# Run specific package/pipeline tests
uv run pytest packages/shared/tests/
uv run pytest packages/surrealdb-service/tests/
uv run pytest packages/file-manager/tests/
uv run pytest packages/llm-manager/tests/
uv run pytest packages/ontology-manager/tests/
uv run pytest pipelines/ingestion/tests/
uv run pytest pipelines/ontology-extraction/tests/
uv run pytest pipelines/entity-filtering/tests/

# Import verification
uv run python -c "from shared.models.extraction import ExtractionResult; print('shared OK')"
uv run python -c "from ontology_manager import OntologyManager; print('ontology-manager OK')"
uv run python -c "from ontology_extraction.workflow import ExtractionWorkflow; print('extraction OK')"
uv run python -c "from entity_filtering.workflow import FilteringWorkflow; print('filtering OK')"
```

---

## Notes & Questions

### Open Questions
1. ~~Should WhisperX be a separate pipeline or part of ingestion?~~ → Part of ingestion (resolved)
2. How to handle large file uploads in Streamlit (>200MB)?
3. GPU memory management when running multiple pipelines?
4. ~~When to build TreeKG/RAPTOR summarization models in entity-filtering?~~ → Edge predictor accepts optional `hierarchy_graph` from TreeKG; `summarization/` subpackage reserved but empty
5. Integration testing strategy across pipelines

### Decisions Made
1. Pure Pydantic models in shared, database operations in surrealdb-service
2. Each pipeline has standalone + connected modes
3. File-manager is the single source of truth for file locations
4. LLM-manager handles all model routing via PipelineTask enum
5. **Filtering pipeline is generic + pluggable** (not domain-specific/Dutch-policy)
6. **Entity deduplication belongs in entity-filtering** (not ontology-extraction)
7. **ontology-manager is a package** (service pattern like llm-manager, not a pipeline)
8. **Ontology-extraction uses pure LLM calls** via llm-manager (no spacy, no rdflib)
9. **TreeKG/RAPTOR are summarization models** for entity-filtering, not extraction. Edge predictor accepts optional hierarchy_graph from TreeKG
10. **No hardcoded localhost URLs** — use env vars via BaseSettings or in-process workspace imports
11. **Streamlit UIs deferred** — architecture supports them but not building now
12. **pytest uses `--import-mode=importlib`** with NO `tests/__init__.py` to prevent import collisions

### Additional Components Needed
1. Web scraper pipeline (after extraction is working)
2. Summarization pipeline (RAPTOR + TreeKG) — edge predictor already accepts hierarchy_graph; `summarization/` subpackage in entity-filtering is reserved
3. Enrichment pipeline (Google Scholar, CrossRef)
4. Embeddings pipeline
5. Retrieval pipeline
6. Application integration (app-main, chat, canvas)
