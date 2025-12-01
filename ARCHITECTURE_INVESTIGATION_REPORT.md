# Open Notebook Architecture Investigation Report
**Date**: November 2025  
**Purpose**: Foundation analysis for docling-granite parser integration and file management improvements

---

## Executive Summary

The open-notebook application has a well-designed, extensible architecture for document processing. The system is built on:

1. **Multi-layered File Processing Pipeline**: Upload → Parse → Chunk → Store → Embed
2. **Plugin-based Parser Architecture**: Easy to add new parsers via async function interface
3. **Configuration-Driven Engine Selection**: Runtime selection of processing engines
4. **Spatial-Aware Chunking**: Document chunks include bounding box positions for visualization
5. **Command Queue System**: Asynchronous job processing with status tracking

### Key Finding
The codebase is **production-ready** for docling-granite integration with minimal changes required. The architecture already supports the necessary abstraction level.

---

## 1. FILE PROCESSING PIPELINE

### 1.1 Complete Flow

```
User Upload (Frontend)
    ↓
POST /sources → save_uploaded_file()
    ↓
./data/uploads/ (unique filename)
    ↓
async_processing=true?
    ├─ true  → Submit to command queue → Return immediately with status
    └─ false → Execute synchronously → Wait for result
    ↓
process_source_command()
    ↓
source_graph.ainvoke(SourceState)
    ↓
    [Step 1] content_process()
        ├─ Select engine (docling_gpu / docling / content-core)
        ├─ Extract content → markdown/html/json
        ├─ Extract chunks with spatial data
        └─ Return content + chunks
    ↓
    [Step 2] save_source()
        ├─ Update source record
        ├─ Save chunks with positions
        └─ Store full_text
    ↓
    [Step 3] transform_content() (optional)
        └─ Apply transformations
    ↓
    [Step 4] vectorize() (optional)
        └─ Create embeddings
    ↓
Database (SurrealDB)
    ├─ source table (content, metadata)
    ├─ chunk table (text, positions, spatial info)
    └─ source_embedding table (vectors)
```

### 1.2 File Upload Details

**Location**: `api/routers/sources.py:62-83`

```python
async def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded file to uploads folder and return file path."""
    file_path = generate_unique_filename(upload_file.filename, UPLOADS_FOLDER)
    
    try:
        with open(file_path, "wb") as f:
            content = await upload_file.read()
            f.write(content)
        return file_path
    except Exception as e:
        if os.path.exists(file_path):
            os.unlink(file_path)  # Cleanup on failure
        raise
```

**Key Features**:
- Unique naming: Prevents overwrites with auto-incrementing counter
- Cleanup on failure: Removes partial files
- Error handling: Proper exception propagation

### 1.3 Directory Structure

```
./data/
├── uploads/                    # All user uploaded files
│   ├── document.pdf
│   ├── document (1).pdf       # Auto-renamed duplicates
│   ├── image.png
│   └── ...
├── sqlite-db/
│   └── checkpoints.sqlite     # LangGraph checkpoint state
└── tiktoken-cache/
    └── ...                     # Token counting cache
```

### 1.4 Processing Modes

**Async (Default)** - Recommended for production
```
POST /sources → Returns immediately
Response: { id, status: "queued", command_id }
        ↓
Client polls GET /sources/{id}/status
        ↓
Processing happens in background via surreal-commands
```

**Sync (Legacy)** - For tests/scripts
```
POST /sources → Blocks until complete (300s timeout)
Response: { id, status: "completed", chunks, embedded_chunks }
```

---

## 2. PARSER ARCHITECTURE

### 2.1 Current Parsers

| Engine | Location | Input | Output | Performance |
|--------|----------|-------|--------|-------------|
| `docling_gpu` | `open_notebook/processors/docling_gpu.py` | Files, URLs, content | Markdown/HTML/JSON | 8-14x faster (GPU) |
| `docling` | content-core | Files, URLs, content | Markdown/HTML/JSON | Baseline (CPU) |
| `simple` | content-core | Text only | Plain text | Fast (minimal processing) |
| `auto` | content-core | Any | Auto-format | Auto-selects best |

### 2.2 Parser Interface Specification

All parsers follow this async function pattern:

```python
async def extract_with_[name](state: ProcessSourceState) -> ProcessSourceState:
    """
    Extract and process document content.
    
    Args:
        state: ProcessSourceState with:
            - file_path: Optional path to document
            - url: Optional URL to fetch
            - content: Optional direct text content
            - metadata: Dict with context
    
    Returns:
        ProcessSourceState with:
            - content: Extracted text (markdown/html/json)
            - metadata: Updated with engine info
                - extraction_engine: "engine_name"
                - docling_format: Output format used
                - docling_gpu_enabled: Whether GPU used
                - [Additional engine-specific fields]
    """
```

### 2.3 Example: Docling GPU Implementation

**File**: `open_notebook/processors/docling_gpu.py`

```python
async def extract_with_docling_gpu(state: ProcessSourceState) -> ProcessSourceState:
    # Get GPU-enabled converter (singleton pattern)
    converter = get_gpu_converter()
    
    # Determine source
    source = state.file_path or state.url or state.content
    if not source:
        raise ValueError("No input provided")
    
    # Process document
    result = converter.convert(source)
    doc = result.document
    
    # Export in desired format
    output = doc.export_to_markdown()
    
    # Update metadata
    state.metadata["docling_format"] = "markdown"
    state.metadata["docling_gpu_enabled"] = True
    state.metadata["extraction_engine"] = "docling_gpu"
    
    # Update state
    state.content = output
    return state
```

**GPU Initialization** (`open_notebook/utils/docling_gpu.py`):
- CUDA GPU first (lines 26-29)
- Falls back to AUTO device if CUDA fails (lines 51-54)
- Singleton pattern avoids re-initialization

### 2.4 Engine Selection Logic

**Location**: `open_notebook/graphs/source.py:49-61`

```python
# Load settings (currently hardcoded, should use get_instance())
content_settings = ContentSettings(
    default_content_processing_engine_doc="auto",
    ...
)

document_engine = content_settings.default_content_processing_engine_doc

# Route to correct processor
if document_engine == "docling_gpu":
    logger.info("🚀 Using GPU-accelerated Docling processor")
    processed_state = await extract_with_docling_gpu(content_state)
else:
    # Use standard content-core processing for all other engines
    processed_state = await extract_content(content_state)
```

### 2.5 Configuration System

**Location**: `open_notebook/domain/content_settings.py`

```python
class ContentSettings(RecordModel):
    record_id: ClassVar[str] = "open_notebook:content_settings"
    
    default_content_processing_engine_doc: Optional[
        Literal["auto", "docling", "docling_gpu", "simple"]
    ] = Field("auto", description="...")
    
    default_content_processing_engine_url: Optional[
        Literal["auto", "firecrawl", "jina", "simple"]
    ] = Field("auto", description="...")
    
    default_embedding_option: Optional[Literal["ask", "always", "never"]]
    auto_delete_files: Optional[Literal["yes", "no"]]
    youtube_preferred_languages: Optional[List[str]]
```

**Storage**: Single record in SurrealDB with ID "open_notebook:content_settings"

**Retrieval**: `ContentSettings.get_instance()` loads from database

---

## 3. CHUNK EXTRACTION WITH SPATIAL DATA

### 3.1 Purpose

Documents are processed twice:
1. **Content Extraction**: Full text for search/embedding
2. **Chunk Extraction**: Structural elements with bounding boxes for visualization

This separation allows:
- Full-text search on extracted content
- PDF annotation with chunk visualization
- Accurate spatial positioning in UI

### 3.2 Chunk Data Structure

**Location**: `open_notebook/domain/notebook.py:143-195`

```python
class Chunk(ObjectModel):
    table_name: ClassVar[str] = "chunk"
    
    source: Union[str, RecordID]           # Link to source
    text: str                               # Chunk content
    order: int                              # Position in document
    
    # Page information
    physical_page: int                      # PDF page number (0-indexed)
    printed_page: Optional[int]             # Print edition page
    
    # Structural metadata
    chapter: Optional[str]                  # Section heading
    paragraph_number: Optional[int]         # Index within section
    element_type: str                       # 'paragraph', 'title', 'table', 'picture'
    
    # Bounding boxes: [[page, x1, x2, y1, y2], ...]
    positions: List[List[float]]            # Normalized coordinates (0-1)
    
    # Additional metadata
    metadata: Optional[Dict[str, Any]]      # Engine-specific info
```

### 3.3 Extraction Process

**Location**: `open_notebook/processors/chunk_extractor.py`

```python
def extract_chunks_from_docling(
    source_path: str,
    output_format: str = "markdown"
) -> tuple[str, List[Dict[str, Any]], Optional[ConversionResult]]:
    """
    Extract content and chunks with spatial information.
    
    Returns:
        (content_string, chunks_list, conversion_result)
    """
    # Initialize converter
    converter = DocumentConverter()
    result = converter.convert(source_path)
    doc = result.document
    
    # Extract content
    content = doc.export_to_markdown()
    
    # Extract chunks with positions
    chunks = extract_chunks_with_positions(doc, result)
    
    return content, chunks, result
```

**Chunk Elements Extracted**:
- Text items (paragraphs)
- Tables
- Pictures/Images
- Structural hierarchy (chapters, sections)

### 3.4 Chunk Persistence

**Location**: `open_notebook/graphs/source.py:118-154`

```python
# Delete existing chunks (idempotent)
# For each extracted chunk:
chunk = Chunk(
    source=source.id,
    text=chunk_data["text"],
    order=chunk_data["order"],
    physical_page=chunk_data["physical_page"],
    printed_page=chunk_data.get("printed_page"),
    chapter=chunk_data.get("chapter"),
    paragraph_number=chunk_data.get("paragraph_number"),
    element_type=chunk_data["element_type"],
    positions=chunk_data.get("positions", []),  # [[page, x1, x2, y1, y2], ...]
    metadata=chunk_data.get("metadata", {})
)
await chunk.save()
```

**Auto-Trigger Logic** (lines 72-86):
```python
# Check if we should extract chunks
extraction_engine = (processed_state.metadata or {}).get("extraction_engine", "")
is_pdf = file_path and file_path.lower().endswith('.pdf')
should_extract = ("docling" in extraction_engine.lower() or is_pdf) and file_path

if should_extract:
    try:
        _, chunks, _ = extract_chunks_from_docling(file_path, output_format="markdown")
    except Exception as e:
        logger.warning(f"Failed to extract chunks: {e}")
        chunks = None
```

**Key**: If extraction_engine metadata contains "docling", chunk extraction auto-triggers.

---

## 4. SETTINGS & CONFIGURATION

### 4.1 Settings Flow

```
Frontend (Settings UI)
    ↓
PUT /settings { default_content_processing_engine_doc: "docling_gpu" }
    ↓
api/routers/settings.py:update_settings()
    ├─ Load ContentSettings from DB
    ├─ Update provided fields
    └─ Save back to DB
    ↓
SurrealDB (record: "open_notebook:content_settings")
    ↓
Next Processing Run
    ↓
source.py:content_process()
    ├─ ContentSettings.get_instance()
    ├─ Read default_content_processing_engine_doc
    └─ Route to selected engine
```

### 4.2 Frontend Settings UI

**File**: `frontend/src/app/(dashboard)/settings/components/SettingsForm.tsx`

```tsx
const settingsSchema = z.object({
  default_content_processing_engine_doc: z.enum(['auto', 'docling', 'docling_gpu', 'simple']).optional(),
  // ...
})

// In form render:
<SelectItem value="auto">Auto (Recommended)</SelectItem>
<SelectItem value="docling">Docling (CPU)</SelectItem>
<SelectItem value="docling_gpu">Docling GPU (Fastest)</SelectItem>
<SelectItem value="simple">Simple</SelectItem>
```

### 4.3 API Route

**File**: `api/routers/settings.py`

```python
@router.put("/settings", response_model=SettingsResponse)
async def update_settings(settings_update: SettingsUpdate):
    settings: ContentSettings = await ContentSettings.get_instance()
    
    if settings_update.default_content_processing_engine_doc is not None:
        settings.default_content_processing_engine_doc = cast(
            Literal["auto", "docling", "simple"],  # ⚠️ MISSING "docling_gpu"
            settings_update.default_content_processing_engine_doc
        )
    
    await settings.update()
    return SettingsResponse(...)
```

### 4.4 Critical Bug Found

**Issue**: Settings route doesn't support "docling_gpu" in type cast (line 40)

**Impact**: 
- Cannot save "docling_gpu" preference via API
- Frontend allows selection, but backend rejects
- Bug prevents docling_gpu from persisting in database

**Fix**: Add "docling_gpu" to Literal type:
```python
settings.default_content_processing_engine_doc = cast(
    Literal["auto", "docling", "docling_gpu", "simple"],  # ✅ FIXED
    settings_update.default_content_processing_engine_doc
)
```

---

## 5. INTEGRATION OPPORTUNITIES

### 5.1 Docling-Granite Parser Integration

**Optimal Integration Point**: `open_notebook/graphs/source.py:56-61`

```python
# Current code
if document_engine == "docling_gpu":
    processed_state = await extract_with_docling_gpu(content_state)
else:
    processed_state = await extract_content(content_state)

# After integration:
if document_engine == "docling_gpu":
    processed_state = await extract_with_docling_gpu(content_state)
elif document_engine == "docling_granite":
    processed_state = await extract_with_docling_granite(content_state)
else:
    processed_state = await extract_content(content_state)
```

**Required Changes**:
1. Add "docling_granite" to Literal in content_settings.py:10-11
2. Create `open_notebook/processors/docling_granite.py`
3. Add routing conditional in source.py:56-61
4. Update chunk extraction condition if needed (likely works auto)
5. Fix settings route bug (add missing docling_gpu and docling_granite)
6. Update frontend schema and UI options
7. Export new functions in processors/__init__.py

### 5.2 File Management System Enhancement

#### Directory Organization
```
# Current flat structure
./data/uploads/
├── document.pdf
├── document (1).pdf
└── ...

# Proposed hierarchical structure
./data/uploads/
├── by_date/
│   └── 2025-11/
│       ├── document_20251105.pdf
│       └── ...
└── by_notebook/
    └── {notebook_id}/
        ├── document.pdf
        └── ...
```

#### Implementation Points
1. **File naming** (`api/routers/sources.py:39-59`):
   - Add optional subdirectory parameter
   - Preserve backward compatibility

2. **Storage quota system**:
   - Add to ContentSettings: `max_storage_per_notebook: Optional[int]`
   - Check before upload in create_source()

3. **File lifecycle**:
   - Soft delete (move to trash folder)
   - Archive after processing
   - Scheduled cleanup job

#### File Access Control
**Security**: `api/routers/sources.py:707-721`

```python
def download_source_file(file_path: str) -> FileResponse:
    # Prevent directory traversal
    resolved_path = Path(file_path).resolve()
    uploads_path = Path(UPLOADS_FOLDER).resolve()
    
    if not str(resolved_path).startswith(str(uploads_path)):
        raise HTTPException(status_code=403, detail="Access denied")
    
    return FileResponse(resolved_path)
```

### 5.3 Ollama Integration Options

#### Option 1: Document Parsing (Direct)
- Use docling-granite with local granite4 models
- No Ollama needed if docling-granite is PyPI package
- Simplest path forward

#### Option 2: Ollama-Based Parsing
- Ollama serves granite4 model (localhost:11434)
- Custom processor calls Ollama API
- Requires docling-granite fork or custom implementation
- More flexible for model switching

#### Option 3: Content Enhancement (Future)
- After document extraction, use granite4 for:
  - Summarization
  - Classification
  - Key entity extraction
- Implement as transformation feature
- Uses Ollama as optional LLM backend

**Recommended**: Start with Option 1 (direct docling-granite integration)

---

## 6. DATABASE SCHEMA

### 6.1 Source Table

**Location**: `open_notebook/domain/notebook.py:197-250+`

```sql
source {
  id: string,
  asset: {
    file_path?: string,     -- Path in ./data/uploads/
    url?: string            -- For URL sources
  },
  title?: string,
  topics?: [string],
  full_text?: string,       -- Extracted content
  command?: RecordID,       -- Link to processing job
  created: string,
  updated: string
}
```

### 6.2 Chunk Table

```sql
chunk {
  id: string,
  source: RecordID,         -- Foreign key to source
  text: string,
  order: int,
  physical_page: int,
  printed_page?: int,
  chapter?: string,
  paragraph_number?: int,
  element_type: string,     -- 'paragraph', 'title', 'table', 'picture'
  positions: [[float]],     -- [[page, x1, x2, y1, y2], ...]
  metadata?: object,
  created: string,
  updated: string
}
```

### 6.3 Related Tables

```
source_embedding {
  id, source, content, embedding_vector
}

source_insight {
  id, source, insight_type, content
}

reference {
  from: notebook,
  to: source,
  (notebook-source associations)
}

command {
  id, status, result, starttime, endtime, error, input, output
  (surreal-commands job tracking)
}
```

---

## 7. CRITICAL CODE SECTIONS FOR MODIFICATION

### 7.1 Parser Invocation (source.py:56-61)

```python
if document_engine == "docling_gpu":
    logger.info("🚀 Using GPU-accelerated Docling processor")
    processed_state = await extract_with_docling_gpu(content_state)
else:
    # Use standard content-core processing for all other engines
    processed_state = await extract_content(content_state)
```

**Add After Line 58**:
```python
elif document_engine == "docling_granite":
    logger.info("⚡ Using Docling Granite processor")
    processed_state = await extract_with_docling_granite(content_state)
```

### 7.2 Settings Configuration (content_settings.py:10-11)

```python
default_content_processing_engine_doc: Optional[
    Literal["auto", "docling", "docling_gpu", "simple"]
]
```

**Update To**:
```python
default_content_processing_engine_doc: Optional[
    Literal["auto", "docling", "docling_gpu", "docling_granite", "simple"]
]
```

### 7.3 Settings Route Fix (settings.py:39-41)

```python
settings.default_content_processing_engine_doc = cast(
    Literal["auto", "docling", "simple"],  # WRONG
    settings_update.default_content_processing_engine_doc
)
```

**Update To**:
```python
settings.default_content_processing_engine_doc = cast(
    Literal["auto", "docling", "docling_gpu", "docling_granite", "simple"],  # CORRECT
    settings_update.default_content_processing_engine_doc
)
```

### 7.4 Frontend Schema (SettingsForm.tsx:18)

```tsx
default_content_processing_engine_doc: z.enum(['auto', 'docling', 'docling_gpu', 'simple']).optional(),
```

**Update To**:
```tsx
default_content_processing_engine_doc: z.enum(['auto', 'docling', 'docling_gpu', 'docling_granite', 'simple']).optional(),
```

### 7.5 Frontend UI (SettingsForm.tsx:114-119)

```tsx
<SelectItem value="auto">Auto (Recommended)</SelectItem>
<SelectItem value="docling">Docling (CPU)</SelectItem>
<SelectItem value="docling_gpu">Docling GPU (Fastest)</SelectItem>
<SelectItem value="simple">Simple</SelectItem>
```

**Add After docling_gpu**:
```tsx
<SelectItem value="docling_granite">Docling Granite (Fast & Accurate)</SelectItem>
```

---

## 8. IMPLEMENTATION SEQUENCE

### Phase 1: Fix Existing Bug (1-2 hours)
1. Fix settings route type cast (add docling_gpu, docling_granite)
2. Test settings persistence
3. Verify frontend-backend alignment

### Phase 2: Docling-Granite Integration (4-6 hours)
1. Create `open_notebook/processors/docling_granite.py`
2. Update ContentSettings enum
3. Add routing in source.py
4. Update frontend schema and UI
5. Test end-to-end

### Phase 3: Chunk Extraction (2-3 hours)
1. Verify auto-trigger works for docling_granite
2. Add custom chunk extraction if needed
3. Test spatial data accuracy

### Phase 4: File Management Enhancement (8-12 hours)
1. Design directory structure
2. Implement subdirectory support
3. Add storage quota system
4. Implement soft delete/archive
5. Test file operations

### Phase 5: Optional - Ollama Integration (TBD)
1. Research docling-granite Ollama support
2. Design integration approach
3. Implement model management
4. Add to settings UI

---

## 9. TESTING STRATEGY

### Unit Tests
- Parser interface compliance
- Settings persistence
- Chunk extraction accuracy

### Integration Tests
- End-to-end file upload → parse → chunk
- Settings change → routing verification
- GPU fallback behavior

### Frontend Tests
- Settings form validation
- API integration
- UI option rendering

### Performance Tests
- Baseline vs docling_granite speed
- GPU memory usage
- Chunk extraction overhead

---

## 10. APPENDIX: KEY FILES

| File | Purpose | Lines |
|------|---------|-------|
| `open_notebook/graphs/source.py` | Main processing graph | 1-250+ |
| `open_notebook/processors/docling_gpu.py` | GPU parser | 1-81 |
| `open_notebook/processors/chunk_extractor.py` | Spatial extraction | 1-200+ |
| `open_notebook/domain/content_settings.py` | Configuration model | 1-26 |
| `open_notebook/domain/notebook.py` | Data models | 1-300+ |
| `api/routers/sources.py` | File upload API | 1-1200+ |
| `api/routers/settings.py` | Settings API | 1-79 |
| `api/models.py` | API schemas | 1-400+ |
| `frontend/src/app/(dashboard)/settings/components/SettingsForm.tsx` | Settings UI | 1-250+ |
| `open_notebook/config.py` | Configuration | All |

---

## 11. RECOMMENDATIONS

### Immediate Actions
1. ✅ Fix settings route bug (docling_gpu/granite support)
2. ✅ Create docling_granite processor
3. ✅ Update configuration layers
4. ✅ Test end-to-end

### Short-Term Improvements
1. Move hardcoded defaults to database loading
2. Implement file directory organization
3. Add storage quota system
4. Comprehensive logging for debugging

### Long-Term Enhancements
1. Optimize chunk extraction (avoid re-processing)
2. Local Ollama integration for content enhancement
3. Advanced file lifecycle management
4. Multi-engine benchmarking UI

---

## 12. CONCLUSION

The open-notebook codebase provides an excellent foundation for:
- ✅ Docling-granite parser integration (minimal changes required)
- ✅ File management system improvements (clear extension points)
- ✅ Local LLM integration (architectural support exists)

The async/await architecture, configuration-driven design, and separation of concerns make it straightforward to add new processing engines while maintaining backward compatibility.

**Estimated Effort for Full Integration**:
- Fix bug + implement docling-granite: 6-8 hours
- File management system: 10-16 hours
- Testing & documentation: 4-6 hours
- **Total**: 20-30 hours for production-ready implementation

