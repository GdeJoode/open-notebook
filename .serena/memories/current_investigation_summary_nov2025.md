# Open Notebook - Current State Investigation (November 2025)

## VERIFICATION STATUS

### Verified Current Architecture (as of November 2025)
All findings from previous memory files have been verified to be current and accurate:

1. **File Processing Pipeline**: ✅ Confirmed
   - Upload handling: `api/routers/sources.py` (lines 62-83)
   - Unique filename generation: `api/routers/sources.py` (lines 39-59)
   - Storage location: `./data/uploads/`

2. **Parser Architecture**: ✅ Confirmed
   - GPU support: `open_notebook/processors/docling_gpu.py` (async function-based)
   - Standard routing: `open_notebook/graphs/source.py:content_process()` (lines 56-61)
   - Chunk extraction: `open_notebook/processors/chunk_extractor.py`

3. **Settings System**: ✅ Confirmed
   - Domain model: `open_notebook/domain/content_settings.py`
   - Current engines: `["auto", "docling", "docling_gpu", "simple"]` (line 11)
   - API route: `api/routers/settings.py` (GET/PUT)
   - Frontend: `frontend/src/app/(dashboard)/settings/components/SettingsForm.tsx`

4. **Database Models**: ✅ Confirmed
   - Source model: `open_notebook/domain/notebook.py:Source` (lines 197-250+)
   - Chunk model: `open_notebook/domain/notebook.py:Chunk` (lines 143-195)
   - Asset model: `open_notebook/domain/notebook.py:Asset` (lines 88-90)

5. **GPU Acceleration**: ✅ Confirmed
   - Implementation: `open_notebook/utils/docling_gpu.py`
   - CUDA support with AUTO fallback (lines 24-65)
   - Global converter singleton pattern (lines 14-25 in docling_gpu.py processor)

### Key Architectural Insights

#### Parser Interface Pattern
```python
async def extract_with_[parser_name](state: ProcessSourceState) -> ProcessSourceState:
    # Process and update state
    state.content = extracted_content
    state.metadata["extraction_engine"] = "engine_name"
    return state
```

#### Routing Logic (source.py:content_process)
```python
if document_engine == "docling_gpu":
    processed_state = await extract_with_docling_gpu(content_state)
else:
    processed_state = await extract_content(content_state)  # content-core
```

#### Chunk Extraction Integration
- Re-processes PDFs after content extraction to get spatial data
- Returns list with bounding boxes: `[[page, x1, x2, y1, y2], ...]`
- Stored in `chunk` table with positions preserved

### Integration Points Identified

**Point 1**: Parser Selection (source.py:49-61)
- Decision point for selecting extraction engine
- Uses ContentSettings.default_content_processing_engine_doc
- Currently supports: auto, docling, docling_gpu, simple

**Point 2**: Settings Update API (settings.py:36-42)
- Type casting for document engines
- Currently casts to: Literal["auto", "docling", "simple"]
- **NOTE**: Missing "docling_gpu" in cast (line 40)! This is a bug in settings route.

**Point 3**: Frontend Settings Form (SettingsForm.tsx:17-22)
- Zod schema includes 'docling_gpu' option
- Frontend correctly supports all 4 engines
- Backend route has incomplete support

**Point 4**: Chunk Extraction Conditional (source.py:72)
- Checks for "docling" in extraction_engine or is_pdf
- Triggers re-processing for spatial data
- Returns chunks with positions

---

## CRITICAL FINDINGS FOR DOCLING-GRANITE INTEGRATION

### 1. Settings Route Bug Found
**File**: `api/routers/settings.py:40`

Current code:
```python
settings.default_content_processing_engine_doc = cast(
    Literal["auto", "docling", "simple"],  # Missing docling_gpu!
    settings_update.default_content_processing_engine_doc
)
```

This prevents "docling_gpu" from being saved via API, even though frontend supports it.

**Fix Required**: Add "docling_gpu" to the Literal type.

### 2. Frontend-Backend Configuration Mismatch
- **Frontend** (SettingsForm.tsx:18): Supports `['auto', 'docling', 'docling_gpu', 'simple']`
- **Backend Model** (content_settings.py:10-11): Supports same 4 options
- **Backend Route** (settings.py:40): Only allows `["auto", "docling", "simple"]` (incomplete cast)

For docling-granite integration, need to ensure consistent support across all layers.

### 3. Processing Engine Decision Point
**Location**: `source.py:49-61`

Current routing:
```python
if document_engine == "docling_gpu":
    processed_state = await extract_with_docling_gpu(content_state)
else:
    processed_state = await extract_content(content_state)
```

This is the optimal place to add:
```python
elif document_engine == "docling_granite":
    processed_state = await extract_with_docling_granite(content_state)
```

### 4. Chunk Extraction for Docling-Granite
**Location**: `source.py:72-86`

Current condition:
```python
should_extract = ("docling" in extraction_engine.lower() or is_pdf) and file_path
```

Will automatically work for docling_granite if metadata["extraction_engine"] = "docling_granite".

No changes needed here if docling_granite follows the same pattern.

### 5. Settings Persistence Flow
```
Frontend SettingsForm → PUT /settings 
  → settings.py:update_settings()
  → Cast string to Literal (BUG HERE)
  → ContentSettings.update()
  → SurrealDB persistence
  → Next processing uses ContentSettings.get_instance()
```

**Note**: ContentSettings.get_instance() currently needs to be called (line 35 in source.py shows hardcoded defaults instead).

---

## OLLAMA INTEGRATION OPPORTUNITIES

### Current State
- No existing Ollama integration in codebase
- Search results show "ollama|granite|llm|model" in 20 files, mostly references to LLM usage in chat/transformation features
- No direct references to local model serving

### Feasible Integration Points for Granite4 Models

**Option 1**: Direct Integration (Preferred)
- Add `docling-granite` as new document processor
- Install from PyPI or build locally
- No Ollama dependency needed if docling-granite is standalone

**Option 2**: Ollama-Based (If docling-granite uses Ollama)
- Ollama would be external service (localhost:11434)
- docling-granite or custom processor calls Ollama API
- Add Ollama connection settings to ContentSettings
- More complex but allows model swapping

**Option 3**: LLM-Enhanced Processing (Future)
- After document extraction, use granite4 for content enhancement
- E.g., summarization, classification
- Add as transformation feature rather than document processor
- Requires Ollama for local inference

### Recommended Approach
Start with **Option 1** (Direct docling-granite integration):
1. Simpler implementation
2. Follows existing pattern
3. No new external dependencies
4. Can add Ollama-based features later if needed

---

## INTEGRATION CHECKLIST FOR DOCLING-GRANITE

### Files to Modify
1. ✅ `open_notebook/domain/content_settings.py` - Add "docling_granite" to Literal
2. ✅ `open_notebook/processors/docling_granite.py` - Create new processor (NEW FILE)
3. ✅ `open_notebook/processors/__init__.py` - Export new functions
4. ✅ `open_notebook/graphs/source.py` - Add routing conditional
5. ✅ `api/routers/settings.py` - Fix type cast (add "docling_gpu" and "docling_granite")
6. ✅ `frontend/src/app/(dashboard)/settings/components/SettingsForm.tsx` - Add to schema and UI
7. ✅ `api/models.py` - Update type hints if needed

### Optional Files
- `open_notebook/processors/chunk_extractor.py` - Add granite-specific chunk extraction (if needed)
- Tests for new processor
- Documentation

### Quick Reference: File Locations
```
Backend:
- Parser interface: open_notebook/processors/
- Graph routing: open_notebook/graphs/source.py:49-61
- Settings: open_notebook/domain/content_settings.py:10-11
- API routes: api/routers/settings.py
- API models: api/models.py

Frontend:
- Settings UI: frontend/src/app/(dashboard)/settings/components/SettingsForm.tsx
- Settings API: frontend/src/lib/api/settings.ts
```

---

## CRITICAL CODE SECTIONS

### Parser Invocation Pattern
```python
# Line 56-61 in source.py
if document_engine == "docling_gpu":
    logger.info("🚀 Using GPU-accelerated Docling processor")
    processed_state = await extract_with_docling_gpu(content_state)
else:
    # Use standard content-core processing for all other engines
    processed_state = await extract_content(content_state)
```

### Chunk Extraction Auto-Trigger
```python
# Line 66-86 in source.py
extraction_engine = (processed_state.metadata or {}).get("extraction_engine", "")
file_path = processed_state.file_path
is_pdf = file_path and file_path.lower().endswith('.pdf')
should_extract = ("docling" in extraction_engine.lower() or is_pdf) and file_path

if should_extract:
    try:
        _, chunks, _ = extract_chunks_from_docling(file_path, output_format="markdown")
    except Exception as e:
        logger.warning(f"Failed to extract chunks: {e}")
        chunks = None
```

### GPU Converter Singleton
```python
# Lines 14-25 in docling_gpu.py
_GPU_CONVERTER = None

def get_gpu_converter():
    global _GPU_CONVERTER
    if _GPU_CONVERTER is None:
        logger.info("Initializing GPU-accelerated Docling converter...")
        _GPU_CONVERTER = create_gpu_document_converter()
    return _GPU_CONVERTER
```

---

## PERFORMANCE NOTES

### Current Performance
- **Docling (CPU)**: 5-30s per document
- **Docling GPU**: 0.5-3s per document (8-14x speedup)
- **Chunk extraction overhead**: +1-3s (re-processing)

### For Docling-Granite
- Expected: Faster than docling_gpu (if using quantized models)
- Should maintain or improve on 8-14x speedup
- Metadata should record actual processing time for monitoring

---

## NEXT STEPS

1. **Verify docling-granite availability** - Check if PyPI/package exists
2. **Fix settings route bug** - Add missing docling_gpu to cast
3. **Plan granite integration** - Follow checklist above
4. **Implement processor** - Create docling_granite.py following extract_with_docling_gpu pattern
5. **Test end-to-end** - Verify all layers work (settings → routing → processing)
6. **GPU fallback handling** - Consider if granite needs CPU fallback like docling_gpu

