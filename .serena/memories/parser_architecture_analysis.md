# Open Notebook Parser Architecture - Comprehensive Analysis

## Executive Summary
The open-notebook codebase uses a sophisticated multi-engine parser architecture built on top of content-core with LangGraph orchestration. The system supports multiple document processing backends (docling, docling_gpu, simple) and integrates chunk extraction with spatial information for PDF documents. The architecture is designed for extensibility to add new parsers like docling-granite.

---

## 1. CURRENT PARSER STRUCTURE

### 1.1 Parser Implementation Pattern
Parsers in this codebase follow an **async function-based pattern** rather than classes:

**Location**: `/mnt/e/Repos/Private/open-notebook/open_notebook/processors/`

Key files:
- `chunk_extractor.py`: Generic Docling chunk extraction with bounding box support
- `docling_gpu.py`: GPU-accelerated Docling processor wrapper
- `__init__.py`: Exports public parser functions

### 1.2 Document Processor Interface

The standard interface is an async function that:
- **Input**: `ProcessSourceState` from content-core (contains file_path, url, content, metadata)
- **Output**: `ProcessSourceState` updated with extracted content and metadata
- **Signature Pattern**:
```python
async def extract_with_parser_name(state: ProcessSourceState) -> ProcessSourceState:
    # Perform extraction
    # Update state.content with extracted markdown/html/json
    # Update state.metadata with engine info
    # Return state
```

### 1.3 Current Docling Implementation

**File**: `open_notebook/processors/docling_gpu.py`

```python
async def extract_with_docling_gpu(state: ProcessSourceState) -> ProcessSourceState:
    """
    Use GPU-accelerated Docling to parse files, URLs, or content into the desired format.
    """
    - Gets GPU-enabled DocumentConverter
    - Supports file_path, url, or content from state
    - Exports to markdown/html/json based on config
    - Records metadata: docling_format, docling_gpu_enabled, extraction_engine
    - Returns updated state
```

### 1.4 Chunk Extraction Utilities

**File**: `open_notebook/processors/chunk_extractor.py`

Two main functions:
1. `extract_chunks_from_docling(source_path, output_format)`:
   - Returns tuple: (content_string, chunks_list, conversion_result)
   - Handles PDF, Word, Excel, and image files
   
2. `extract_chunks_with_positions(doc, result)`:
   - Extracts text items, tables, and pictures
   - Preserves spatial information (bounding boxes)
   - Returns chunk format with metadata:
   ```python
   {
       'text': str,
       'order': int,
       'physical_page': int,
       'printed_page': int | None,
       'chapter': str | None,
       'paragraph_number': int | None,
       'element_type': str,  # 'paragraph', 'table', 'picture', etc.
       'positions': [[page_num, x1, x2, y1, y2], ...],
       'metadata': dict
   }
   ```

---

## 2. PARSER ROUTING & SELECTION

### 2.1 Engine Selection Flow

**Location**: `open_notebook/graphs/source.py` - `content_process()` function

```
1. Load ContentSettings.default_content_processing_engine_doc
2. Check content_state["document_engine"] override
3. Route to correct processor:
   - "docling_gpu" → extract_with_docling_gpu()
   - "docling", "auto", other → extract_content() from content-core
```

**Key Code** (lines 55-61):
```python
if document_engine == "docling_gpu":
    logger.info("🚀 Using GPU-accelerated Docling processor")
    processed_state = await extract_with_docling_gpu(content_state)
else:
    # Use standard content-core processing for all other engines
    processed_state = await extract_content(content_state)
```

### 2.2 Configuration System

**Location**: `open_notebook/domain/content_settings.py`

```python
class ContentSettings(RecordModel):
    default_content_processing_engine_doc: Literal["auto", "docling", "docling_gpu", "simple"]
    default_content_processing_engine_url: Literal["auto", "firecrawl", "jina", "simple"]
    default_embedding_option: Literal["ask", "always", "never"]
    auto_delete_files: Literal["yes", "no"]
    youtube_preferred_languages: List[str]
```

**Storage**: Stored in SurrealDB with record_id "open_notebook:content_settings"

### 2.3 Engine Registry Pattern

The system doesn't use a formal registry class. Instead:
- **For documents**: String-based routing in `content_process()` 
- **For URLs**: content-core handles routing internally
- **Extension point**: Add new `if document_engine == "docling_granite":` branches in `content_process()`

---

## 3. DOCUMENT PROCESSING FLOW

### 3.1 Complete Processing Pipeline

**Entry Point**: `api/routers/sources.py` - `create_source()` endpoint

```
HTTP POST /sources
    ↓
parse_source_form_data()  [Validates & parses form data]
    ↓
save_uploaded_file()  [If file upload]
    ↓
execute_command_sync() or submit_command_job()
    ↓
commands/source_commands.py::process_source_command()
    ↓
open_notebook/graphs/source.py::source_graph.ainvoke()
    ↓
    Step 1: content_process()  [Extract content & chunks]
    ↓
    Step 2: save_source()  [Save to database]
    ↓
    Step 3: transform_content()  [Apply transformations]
    ↓
    Step 4: vectorize()  [Create embeddings]
```

### 3.2 Processing Modes

**Async Processing** (Default, recommended):
```
POST /sources → Returns immediately with status=queued
Command queued → Processes in background
Status trackable via GET /sources/{id}/status
```

**Sync Processing** (Legacy):
```
POST /sources → Blocks until processing complete
Returns final processed source immediately
```

### 3.3 State Flow Through Graph

**SourceState TypedDict** (LangGraph state):
```python
{
    "content_state": ProcessSourceState,  # From content-core
    "apply_transformations": List[Transformation],
    "source_id": str,
    "notebook_ids": List[str],
    "source": Source,  # Database model
    "transformation": List[...],  # Results accumulator
    "embed": bool,
    "chunks": Optional[List[Dict[str, Any]]]  # Extracted chunks with positions
}
```

### 3.4 Chunk Extraction Integration

**Location**: `source.py::content_process()` (lines 63-91)

Chunks are extracted post-content-extraction:
```
1. Content is extracted via docling_gpu or content-core
2. Check if PDF file or docling was used
3. If yes, re-invoke docling to extract chunks with spatial data
4. Return both content and chunks
```

**Rationale**: Separation of concerns - content extraction for full text, chunk extraction for visualization

### 3.5 Chunk Persistence

**Location**: `source.py::save_source()` (lines 118-154)

```python
# Delete existing chunks for source (idempotency)
# Create Chunk records for each chunk_data:
chunk = Chunk(
    source=source.id,
    text=chunk_data["text"],
    order=chunk_data["order"],
    physical_page=chunk_data["physical_page"],
    printed_page=chunk_data.get("printed_page"),
    chapter=chunk_data.get("chapter"),
    paragraph_number=chunk_data.get("paragraph_number"),
    element_type=chunk_data["element_type"],
    positions=chunk_data.get("positions", []),
    metadata=chunk_data.get("metadata", {})
)
```

**Database**: SurrealDB `chunk` table (with source foreign key)

---

## 4. CONFIGURATION SYSTEM

### 4.1 Settings Retrieval

**API**: `GET /settings` → Returns ContentSettings

**Frontend**: 
- File: `frontend/src/app/(dashboard)/settings/components/SettingsForm.tsx`
- Hook: `useSettings()` - fetches current settings
- Hook: `useUpdateSettings()` - mutates settings

### 4.2 Settings Update Flow

```
Frontend SettingsForm
    ↓
PUT /settings  [JSON body with updates]
    ↓
api/routers/settings.py::update_settings()
    ↓
ContentSettings.get_instance()  [Load from SurrealDB]
    ↓
Update provided fields (only non-null)
    ↓
await settings.update()  [Save back to DB]
    ↓
Return updated SettingsResponse
```

### 4.3 Settings in Processing

**Location**: `source.py::content_process()` (lines 35-53)

Settings are read fresh before each processing:
```python
content_settings = ContentSettings(
    default_content_processing_engine_doc="auto",  # OR from DB
    ...
)
document_engine = content_settings.default_content_processing_engine_doc
content_state["document_engine"] = document_engine
```

**Note**: Currently hardcoded defaults in code. Should be loaded from DB via:
```python
content_settings = await ContentSettings.get_instance()
```

### 4.4 Available Engines

**Document Engines**:
- `auto` - content-core decides (usually docling for PDFs)
- `docling` - CPU-only Docling
- `docling_gpu` - GPU-accelerated Docling (8-14x faster)
- `simple` - Basic text extraction

**URL Engines**:
- `auto` - Try firecrawl, fallback to jina, then simple
- `firecrawl` - Paid service (free tier available)
- `jina` - Free with rate limits
- `simple` - Basic HTTP extraction

---

## 5. DEPENDENCIES & INTEGRATIONS

### 5.1 Core Dependencies

**Python Packages** (from pyproject.toml):
- `content-core>=1.0.2` - Main document processing abstraction
- `docling[cuda12]>=2.58.0` - Docling with CUDA 12 support
- `langgraph>=0.2.38` - Orchestration framework
- `langchain>=0.3.3` - LLM framework
- `surrealdb>=1.0.4` - Database
- `surreal-commands>=1.0.13` - Command/job queue system

### 5.2 Docling Integration

**GPU Setup**: `open_notebook/utils/docling_gpu.py`

```python
def create_gpu_document_converter() -> DocumentConverter:
    # Tries CUDA device first
    accelerator_options = AcceleratorOptions(
        num_threads=8,
        device=AcceleratorDevice.CUDA
    )
    # Falls back to AUTO if CUDA fails
```

**Architecture**:
- Single global GPU converter instance (cached in module)
- Lazy initialization on first use
- Auto-fallback to CPU if CUDA unavailable

### 5.3 Content-Core Architecture

**Location**: `.venv/lib/python3.12/site-packages/content_core/`

content-core provides:
- `ProcessSourceState` - Standard state object
- `extract_content()` - Main extraction function with engine routing
- Document engine implementations (docling, pdf, office, text, video, youtube, etc.)
- URL engine implementations (firecrawl, jina, simple)

**Why content-core?**
- Abstracts multiple document formats
- Handles diverse input types (files, URLs, content strings)
- Standardized output format (ProcessSourceState)
- Built-in engine selection logic

### 5.4 LangGraph Orchestration

**Location**: `open_notebook/graphs/source.py`

LangGraph is used for:
- State management (SourceState TypedDict)
- Sequential node execution (content_process → save_source → transform_content)
- Conditional edges (trigger_transformations uses runtime state)
- Checkpoint persistence (via langgraph-checkpoint-sqlite)

**Benefits**:
- Clear data flow visualization
- Fault tolerance via checkpoints
- Easy to add/remove processing steps
- Streaming support for long operations

### 5.5 Command/Job Queue

**System**: surreal-commands (built on SurrealDB)

**Pattern**:
```python
@command("process_source", app="open_notebook")
async def process_source_command(input_data: SourceProcessingInput):
    # Process asynchronously
    # Can be queued, retried, monitored
```

**Async Flow**:
```
API submits command → surreal-commands queues → Worker processes → Status tracked
```

### 5.6 Database

**System**: SurrealDB

**Key Tables** (based on codebase analysis):
- `source` - Source records
- `chunk` - Chunk records with spatial data
- `source_embedding` - Vector embeddings
- `source_insight` - Transformation results
- `reference` - Notebook-source associations
- `open_notebook:content_settings` - Application settings

---

## 6. PARSER INTERFACE SPECIFICATION

### 6.1 Standard Parser Function Signature

```python
async def extract_with_[parser_name](state: ProcessSourceState) -> ProcessSourceState:
    """
    Extract content from document/URL using [parser_name].
    
    Args:
        state: ProcessSourceState containing:
            - file_path: Optional path to file
            - url: Optional URL to fetch
            - content: Optional direct content string
            - metadata: Dict with context info
    
    Returns:
        ProcessSourceState with:
            - content: Extracted text (markdown, html, or json)
            - file_path: Path (if was file)
            - url: URL (if was URL)
            - title: Extracted title
            - metadata: Updated with extraction info
                - extraction_engine: Name of engine used
                - docling_format: Output format used
                - docling_gpu_enabled: Whether GPU was used
                - Any engine-specific metadata
    """
```

### 6.2 Chunk Extraction Interface

```python
def extract_chunks_from_[parser_name](
    source_path: str,
    output_format: str = "markdown"
) -> tuple[str, List[Dict[str, Any]], Any]:
    """
    Extract content and chunks with spatial information.
    
    Returns:
        Tuple of:
        - content_string: Full extracted content
        - chunks_list: List of chunk dicts with:
            {
                'text': str,
                'order': int,
                'element_type': str,
                'positions': [[page, x1, x2, y1, y2], ...],
                'metadata': dict
            }
        - conversion_result: Engine-specific result object
    """
```

### 6.3 Integration Points

**Point 1**: Engine routing in `source.py::content_process()`
```python
if document_engine == "docling_granite":
    processed_state = await extract_with_docling_granite(content_state)
else:
    processed_state = await extract_content(content_state)
```

**Point 2**: Chunk extraction in `source.py::content_process()`
```python
if should_extract and extraction_engine == "docling_granite":
    _, chunks, _ = extract_chunks_from_docling_granite(file_path, "markdown")
```

**Point 3**: Settings configuration
```python
class ContentSettings:
    default_content_processing_engine_doc: Literal[
        "auto", "docling", "docling_gpu", "docling_granite", "simple"
    ]
```

**Point 4**: Frontend option in SettingsForm.tsx
```tsx
<SelectItem value="docling_granite">Docling Granite (Fast & Accurate)</SelectItem>
```

---

## 7. KEY ARCHITECTURAL PATTERNS

### 7.1 Async-First Architecture
- All processing is async/await based
- Uses LangGraph for state management
- Supports both sync and async API modes

### 7.2 Separation of Concerns
- **Content extraction**: Full text via docling/content-core
- **Chunk extraction**: Spatial data via re-processing with docling
- **Transformations**: Applied post-extraction
- **Embeddings**: Optional vectorization

### 7.3 Configuration-Driven Selection
- Single source of truth: ContentSettings in DB
- Can change behavior without code changes
- Frontend controls all settings
- Settings loaded at processing time (not cached)

### 7.4 Error Handling & Resilience
- GPU processing falls back to AUTO device
- Chunk extraction failures don't fail whole process
- Source records created immediately for UI responsiveness
- Command/status tracking for monitoring

### 7.5 Database-Centric Design
- All persistent state in SurrealDB
- Settings stored as records
- Chunk data persisted with source
- Command tracking for async operations

---

## 8. EXTENSION POINTS FOR DOCLING-GRANITE

### 8.1 Primary Integration Point
**File**: `open_notebook/graphs/source.py`, lines 55-61

Add new conditional:
```python
elif document_engine == "docling_granite":
    logger.info("🚀 Using Docling Granite processor")
    processed_state = await extract_with_docling_granite(content_state)
```

### 8.2 New Processor Function
**File**: `open_notebook/processors/docling_granite.py` (new file)

```python
from content_core.common.state import ProcessSourceState

async def extract_with_docling_granite(state: ProcessSourceState) -> ProcessSourceState:
    # Implementation
    return state
```

### 8.3 Chunk Extraction (Optional)
**File**: `open_notebook/processors/chunk_extractor.py`

Add function:
```python
def extract_chunks_from_docling_granite(...):
    # Implementation
    return content, chunks, result
```

### 8.4 Configuration Update
**File**: `open_notebook/domain/content_settings.py`, line 10-12

Update Literal:
```python
default_content_processing_engine_doc: Literal[
    "auto", "docling", "docling_gpu", "docling_granite", "simple"
]
```

### 8.5 Settings Route Update
**File**: `api/routers/settings.py`, lines 36-42

Update allowed values in cast:
```python
settings.default_content_processing_engine_doc = cast(
    Literal["auto", "docling", "docling_gpu", "docling_granite", "simple"],
    settings_update.default_content_processing_engine_doc
)
```

### 8.6 Frontend Update
**File**: `frontend/src/app/(dashboard)/settings/components/SettingsForm.tsx`, line 18

Update schema:
```tsx
default_content_processing_engine_doc: z.enum([
    'auto', 'docling', 'docling_gpu', 'docling_granite', 'simple'
]).optional(),
```

Update dropdown:
```tsx
<SelectItem value="docling_granite">Docling Granite (Fast & Accurate)</SelectItem>
```

### 8.7 Exports Update
**File**: `open_notebook/processors/__init__.py`

```python
from open_notebook.processors.docling_granite import extract_with_docling_granite, extract_chunks_from_docling_granite

__all__ = ["extract_chunks_from_docling", "extract_with_docling_gpu", "extract_with_docling_granite", "extract_chunks_from_docling_granite"]
```

---

## 9. TESTING CONSIDERATIONS

### 9.1 Integration Test Pattern
```python
async def test_extract_with_docling_granite():
    state = ProcessSourceState(file_path="/path/to/test.pdf")
    result = await extract_with_docling_granite(state)
    
    assert result.content is not None
    assert "extraction_engine" in result.metadata
    assert result.metadata["extraction_engine"] == "docling_granite"
```

### 9.2 End-to-End Test Pattern
```python
async def test_source_processing_with_docling_granite():
    # Create source via API
    response = await POST /sources with docling_granite in settings
    
    # Check processing
    assert response.status == 200
    assert response.command_id is not None
    
    # Monitor completion
    status = await GET /sources/{id}/status
    assert status.status == "completed"
```

---

## 10. DEPENDENCIES FOR DOCLING-GRANITE

### Assumptions
- Docling Granite will have similar API to standard Docling:
  - DocumentConverter class
  - convert(source_path) → ConversionResult
  - result.document.export_to_markdown/html/json()
  
- Installation via pip or extras in pyproject.toml

### Integration Steps
1. Add `docling-granite` dependency to pyproject.toml
2. Implement `extract_with_docling_granite()` function
3. Implement chunk extraction if spatial data needed
4. Update configuration enums
5. Update routing logic
6. Update frontend options
7. Test end-to-end

---

## 11. PERFORMANCE NOTES

### Current Benchmarks
- **Docling (CPU)**: Baseline
- **Docling GPU**: 8-14x faster (as documented in settings)

### Considerations for Docling Granite
- If faster: Can target "fast" or "faster" description in UI
- If requires GPU: Similar pattern to docling_gpu with fallback
- If uses quantized models: May have accuracy trade-off worth documenting

---

## 12. METADATA PROPAGATION

All processors should populate `state.metadata` with:
- `extraction_engine`: Name of the engine ("docling_granite")
- `docling_format`: Output format used ("markdown", "html", "json")
- Any engine-specific fields (e.g., "processing_time", "model_used", etc.)

These metadata are:
1. Stored with chunks
2. Queryable in frontend
3. Useful for debugging and monitoring
