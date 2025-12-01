# Implementation Checklist: Docling-Granite + File Management System

**Status**: Investigation Complete  
**Recommendations**: Ready for Implementation  
**Estimated Effort**: 20-30 hours

---

## PART 1: FIX CRITICAL BUG (1-2 hours)

### 1.1 Settings Route Type Cast Bug

**File**: `api/routers/settings.py:39-41`

**Current Code**:
```python
settings.default_content_processing_engine_doc = cast(
    Literal["auto", "docling", "simple"],  # ❌ MISSING docling_gpu and granite
    settings_update.default_content_processing_engine_doc
)
```

**Impact**: Cannot persist "docling_gpu" or future "docling_granite" preferences via API

**Fix**:
```python
settings.default_content_processing_engine_doc = cast(
    Literal["auto", "docling", "docling_gpu", "docling_granite", "simple"],  # ✅ FIXED
    settings_update.default_content_processing_engine_doc
)
```

**Testing**: 
- [ ] Test API accepts "docling_gpu" value
- [ ] Verify persists to database
- [ ] Confirm next processing uses selected engine

---

## PART 2: DOCLING-GRANITE PARSER INTEGRATION (4-6 hours)

### 2.1 Create Granite Processor Module

**File**: `open_notebook/processors/docling_granite.py` (NEW)

**Template**:
```python
"""
Docling Granite parser for fast and accurate document processing.
Implements the extract_with_[parser_name] interface pattern.
"""

from content_core.common.state import ProcessSourceState
from loguru import logger

async def extract_with_docling_granite(state: ProcessSourceState) -> ProcessSourceState:
    """
    Extract content using Docling Granite.
    
    Docling Granite provides fast and accurate document processing
    using advanced language models for better understanding of document structure.
    
    Args:
        state: ProcessSourceState containing document to process
    
    Returns:
        ProcessSourceState with extracted content
    """
    # TODO: Implement docling_granite extraction
    # 1. Get converter (docling_granite API)
    # 2. Determine source (file_path, url, or content)
    # 3. Process document
    # 4. Export to desired format (markdown/html/json)
    # 5. Update metadata with extraction info
    # 6. Return state
    
    state.metadata["extraction_engine"] = "docling_granite"
    state.metadata["docling_format"] = state.output_format or "markdown"
    return state
```

**Checklist**:
- [ ] Create file with proper structure
- [ ] Implement async function following docling_gpu pattern
- [ ] Handle GPU fallback if applicable
- [ ] Implement metadata population
- [ ] Add error handling and logging
- [ ] Implement singleton converter pattern if needed

### 2.2 Update Configuration

**File**: `open_notebook/domain/content_settings.py:10-11`

**Current**:
```python
default_content_processing_engine_doc: Optional[
    Literal["auto", "docling", "docling_gpu", "simple"]
]
```

**Updated**:
```python
default_content_processing_engine_doc: Optional[
    Literal["auto", "docling", "docling_gpu", "docling_granite", "simple"]
]
```

**Checklist**:
- [ ] Add "docling_granite" to Literal type
- [ ] Test model instantiation with new value
- [ ] Verify database query still works

### 2.3 Update Parser Router

**File**: `open_notebook/graphs/source.py:56-61`

**Current**:
```python
if document_engine == "docling_gpu":
    logger.info("🚀 Using GPU-accelerated Docling processor")
    processed_state = await extract_with_docling_gpu(content_state)
else:
    processed_state = await extract_content(content_state)
```

**Updated**:
```python
if document_engine == "docling_gpu":
    logger.info("🚀 Using GPU-accelerated Docling processor")
    processed_state = await extract_with_docling_gpu(content_state)
elif document_engine == "docling_granite":
    logger.info("⚡ Using Docling Granite processor")
    processed_state = await extract_with_docling_granite(content_state)
else:
    processed_state = await extract_content(content_state)
```

**Checklist**:
- [ ] Add elif conditional for docling_granite
- [ ] Import new processor at top of file
- [ ] Test routing logic with different engine values
- [ ] Verify error handling if processor unavailable

### 2.4 Export Processor Functions

**File**: `open_notebook/processors/__init__.py`

**Current**:
```python
from open_notebook.processors.chunk_extractor import extract_chunks_from_docling
from open_notebook.processors.docling_gpu import extract_with_docling_gpu

__all__ = ["extract_chunks_from_docling", "extract_with_docling_gpu"]
```

**Updated**:
```python
from open_notebook.processors.chunk_extractor import extract_chunks_from_docling
from open_notebook.processors.docling_gpu import extract_with_docling_gpu
from open_notebook.processors.docling_granite import extract_with_docling_granite

__all__ = ["extract_chunks_from_docling", "extract_with_docling_gpu", "extract_with_docling_granite"]
```

**Checklist**:
- [ ] Add import statement
- [ ] Add to __all__ list
- [ ] Test imports work correctly

### 2.5 Update Frontend Schema

**File**: `frontend/src/app/(dashboard)/settings/components/SettingsForm.tsx:18`

**Current**:
```tsx
default_content_processing_engine_doc: z.enum(['auto', 'docling', 'docling_gpu', 'simple']).optional(),
```

**Updated**:
```tsx
default_content_processing_engine_doc: z.enum(['auto', 'docling', 'docling_gpu', 'docling_granite', 'simple']).optional(),
```

**Checklist**:
- [ ] Update Zod schema
- [ ] Test schema validation with new value

### 2.6 Update Frontend UI

**File**: `frontend/src/app/(dashboard)/settings/components/SettingsForm.tsx:114-119`

**Current**:
```tsx
<SelectItem value="auto">Auto (Recommended)</SelectItem>
<SelectItem value="docling">Docling (CPU)</SelectItem>
<SelectItem value="docling_gpu">Docling GPU (Fastest)</SelectItem>
<SelectItem value="simple">Simple</SelectItem>
```

**Updated**:
```tsx
<SelectItem value="auto">Auto (Recommended)</SelectItem>
<SelectItem value="docling">Docling (CPU)</SelectItem>
<SelectItem value="docling_gpu">Docling GPU (Fastest)</SelectItem>
<SelectItem value="docling_granite">Docling Granite (Fast & Accurate)</SelectItem>
<SelectItem value="simple">Simple</SelectItem>
```

**Add Help Text** (around line 133):
```tsx
<p>• <strong>Docling Granite</strong> uses advanced language models for faster and more accurate document understanding. Requires specific hardware setup.</p>
```

**Checklist**:
- [ ] Add SelectItem for docling_granite
- [ ] Update help text in Collapsible
- [ ] Test UI renders correctly
- [ ] Test selection works

### 2.7 Chunk Extraction (If Needed)

**File**: `open_notebook/processors/chunk_extractor.py`

Only needed if docling_granite requires custom chunk extraction.

**Auto-Trigger Verification**:
The condition at `source.py:72` already handles granite:
```python
should_extract = ("docling" in extraction_engine.lower() or is_pdf) and file_path
```

If your metadata sets `extraction_engine = "docling_granite"`, chunks will auto-trigger.

**Checklist**:
- [ ] Test chunk extraction works with docling_granite
- [ ] Verify bounding boxes are accurate
- [ ] If fails, create `extract_chunks_from_docling_granite()`

---

## PART 3: COMPREHENSIVE TESTING (4-6 hours)

### 3.1 Unit Tests

**File**: Create or update `tests/test_parsers.py`

```python
async def test_extract_with_docling_granite():
    """Test docling_granite processor."""
    state = ProcessSourceState(
        file_path="/path/to/test.pdf",
        output_format="markdown"
    )
    result = await extract_with_docling_granite(state)
    
    assert result.content is not None
    assert "extraction_engine" in result.metadata
    assert result.metadata["extraction_engine"] == "docling_granite"
```

**Checklist**:
- [ ] Test with PDF file
- [ ] Test with URL
- [ ] Test with plain text
- [ ] Test metadata population
- [ ] Test error handling (missing input)
- [ ] Test fallback behavior if GPU unavailable

### 3.2 Integration Tests

**File**: Create or update `tests/test_source_graph.py`

```python
async def test_source_processing_with_docling_granite():
    """Test end-to-end processing with docling_granite."""
    # 1. Set settings to docling_granite
    settings = await ContentSettings.get_instance()
    settings.default_content_processing_engine_doc = "docling_granite"
    await settings.update()
    
    # 2. Create source via API
    response = await client.post("/sources", data={
        "type": "upload",
        "file": test_pdf,
        "title": "Test Document",
        "notebooks": ["notebook_id"],
        "async_processing": "false"
    })
    
    # 3. Verify processing
    assert response.status_code == 200
    source = response.json()
    assert source["title"] == "Test Document"
    assert source["embedded"] or source["embedded_chunks"] > 0
```

**Checklist**:
- [ ] Test file upload with docling_granite
- [ ] Test settings persistence
- [ ] Test chunk extraction
- [ ] Test embedding (if enabled)
- [ ] Test error recovery

### 3.3 Frontend Tests

**File**: Create or update tests for SettingsForm

```typescript
test('should render docling_granite option', () => {
  render(<SettingsForm />)
  expect(screen.getByText('Docling Granite (Fast & Accurate)')).toBeInTheDocument()
})

test('should save docling_granite setting', async () => {
  render(<SettingsForm />)
  const select = screen.getByDisplayValue('Auto')
  await userEvent.click(select)
  await userEvent.click(screen.getByText('Docling Granite'))
  await userEvent.click(screen.getByText('Save'))
  
  // Verify API was called with correct value
})
```

**Checklist**:
- [ ] Test option renders
- [ ] Test selection works
- [ ] Test API integration
- [ ] Test success/error states

### 3.4 Performance Tests

**Checklist**:
- [ ] Benchmark docling_granite vs docling_gpu
- [ ] Measure memory usage
- [ ] Test with various document sizes
- [ ] Profile chunk extraction time
- [ ] Verify GPU utilization (if applicable)

### 3.5 End-to-End Testing

**Manual Checklist**:
- [ ] Upload PDF with settings set to docling_granite
- [ ] Verify processing completes successfully
- [ ] Check extracted content quality
- [ ] Verify chunks extracted with positions
- [ ] Test PDF viewer with bounding boxes
- [ ] Change settings back to docling
- [ ] Upload another document
- [ ] Verify correct engine used
- [ ] Test error scenarios (invalid file, network error, etc.)

---

## PART 4: FILE MANAGEMENT SYSTEM (8-12 hours)

### 4.1 Directory Structure Enhancement

**File**: `api/routers/sources.py:39-59` (modify `generate_unique_filename`)

**Current Structure**:
```
./data/uploads/
├── document.pdf
├── document (1).pdf
└── ...
```

**Proposed Structure Options**:

**Option A: By Date**
```
./data/uploads/
├── 2025-11/
│   ├── document.pdf
│   └── ...
└── 2025-10/
    └── ...
```

**Option B: By Notebook**
```
./data/uploads/
├── notebook_abc123/
│   ├── document.pdf
│   └── ...
└── notebook_xyz789/
    └── ...
```

**Option C: Hybrid**
```
./data/uploads/
├── notebooks/
│   └── {notebook_id}/
│       ├── {date}/
│       │   └── document.pdf
│       └── ...
└── ...
```

**Implementation Steps**:
- [ ] Design subdirectory strategy
- [ ] Update generate_unique_filename() to create subdirs
- [ ] Add subdirectory parameter to save_uploaded_file()
- [ ] Pass notebook_id to upload function
- [ ] Update file_path storage in Asset
- [ ] Test backward compatibility
- [ ] Update download endpoint path resolution

### 4.2 Storage Quota System

**File**: `open_notebook/domain/content_settings.py` (add fields)

```python
class ContentSettings(RecordModel):
    # ... existing fields ...
    max_storage_per_notebook: Optional[int] = Field(
        None, description="Maximum storage per notebook in bytes (None = unlimited)"
    )
    auto_cleanup_after_days: Optional[int] = Field(
        30, description="Auto-delete files older than N days"
    )
```

**Implementation**:
- [ ] Add storage quota fields to settings
- [ ] Create storage calculation function
- [ ] Check quota before upload
- [ ] Implement quota enforcement in create_source()
- [ ] Create cleanup scheduled task
- [ ] Add quota display to frontend
- [ ] Test quota exceeded scenario

### 4.3 Soft Delete & Archive

**File**: `open_notebook/domain/notebook.py` (add fields to Source)

```python
class Source(ObjectModel):
    # ... existing fields ...
    deleted_at: Optional[str] = None  # Soft delete timestamp
    archived_at: Optional[str] = None  # Archive timestamp
```

**Implementation**:
- [ ] Add deleted_at and archived_at to Source model
- [ ] Implement soft delete (mark, don't remove)
- [ ] Create restore endpoint
- [ ] Implement archive endpoint
- [ ] Filter deleted/archived from normal queries
- [ ] Create admin endpoint to permanently delete
- [ ] Test soft delete recovery

### 4.4 File Lifecycle Management

**New File**: `commands/file_lifecycle_commands.py`

```python
@command("cleanup_old_files", app="open_notebook")
async def cleanup_old_files():
    """Delete files older than configured days."""
    # 1. Get settings
    # 2. Find old files
    # 3. Delete or archive
    # 4. Update source records

@command("archive_large_files", app="open_notebook")
async def archive_large_files():
    """Archive files larger than threshold."""
    # Implementation
```

**Implementation**:
- [ ] Create cleanup command
- [ ] Implement scheduled execution
- [ ] Add archival logic
- [ ] Create restoration function
- [ ] Test cleanup doesn't break references
- [ ] Add logging and monitoring

### 4.5 File Usage Statistics

**New File**: `api/routers/file_stats.py`

```python
@router.get("/file-stats")
async def get_file_stats():
    """Get storage usage statistics."""
    # Total uploaded files
    # Total storage used
    # Per-notebook usage
    # Largest files
    # Usage trends
```

**Implementation**:
- [ ] Create stats calculation functions
- [ ] Add storage tracking to Source
- [ ] Implement stats endpoint
- [ ] Add frontend display widget
- [ ] Create usage warning system
- [ ] Test stats accuracy

### 4.6 Frontend UI Updates

**File**: `frontend/src/app/(dashboard)/settings/components/SettingsForm.tsx`

Add new Card section:
```tsx
<Card>
  <CardHeader>
    <CardTitle>File Management</CardTitle>
  </CardHeader>
  <CardContent className="space-y-6">
    {/* Storage quota */}
    {/* Auto-cleanup days */}
    {/* Storage stats */}
  </CardContent>
</Card>
```

**Checklist**:
- [ ] Add file management settings section
- [ ] Add storage quota input
- [ ] Add auto-cleanup days input
- [ ] Show current usage
- [ ] Show quota status
- [ ] Add clear cache button (if applicable)

---

## PART 5: OPTIONAL - OLLAMA INTEGRATION (TBD)

### 5.1 Research Phase

- [ ] Determine if docling-granite supports Ollama
- [ ] Check granite4 model availability
- [ ] Review Ollama API documentation
- [ ] Estimate performance characteristics
- [ ] Plan model management strategy

### 5.2 Architecture Decision

Choose one of:
1. **Direct Integration** (no Ollama needed)
   - Use docling-granite PyPI package directly
   - Simplest path

2. **Ollama-Based** (Ollama serves models)
   - Ollama server at localhost:11434
   - Custom processor calls Ollama API
   - More flexible but more complex

3. **Hybrid** (Ollama for enhancements)
   - Document parsing via docling-granite
   - Content enhancement via Ollama (summarization, classification)
   - Implement as transformation feature

### 5.3 Implementation (If Needed)

Depends on research findings above.

---

## SUMMARY CHECKLIST

### Critical (Must Do)
- [ ] Fix settings route bug (docling_gpu/granite support)
- [ ] Create docling_granite processor
- [ ] Update configuration layers
- [ ] Update frontend schema and UI
- [ ] Comprehensive testing

### High Priority (Should Do)
- [ ] File directory organization
- [ ] Storage quota system
- [ ] Soft delete implementation
- [ ] File cleanup automation
- [ ] Performance monitoring

### Medium Priority (Nice to Have)
- [ ] File usage statistics
- [ ] Advanced archival system
- [ ] Ollama integration
- [ ] Enhanced documentation
- [ ] Automated migration tools

---

## EFFORT ESTIMATION

| Task | Hours | Notes |
|------|-------|-------|
| Fix settings bug | 1 | Quick fix, thorough testing |
| Docling-granite integration | 6 | Create processor + routing + UI |
| Testing (units + integration) | 5 | Comprehensive coverage |
| File management system | 10 | Multiple sub-components |
| Ollama research + integration | 5-15 | Depends on findings |
| Documentation & polish | 3 | Code comments, user guide |
| **TOTAL** | **20-30** | **Production ready** |

---

## NOTES & CONSIDERATIONS

### Security
- Validate all file paths (prevent directory traversal)
- Check user permissions on files
- Implement role-based access control for settings
- Sanitize file names if needed

### Backward Compatibility
- All changes must work with existing documents
- Settings migration for old configs
- Graceful fallback if new engine unavailable

### Performance
- Chunk extraction re-processing overhead (1-3s extra per PDF)
- Consider caching chunk data
- Monitor GPU memory usage
- Implement rate limiting on processing

### Monitoring
- Log all parsing operations
- Track extraction time and quality metrics
- Monitor storage usage trends
- Alert on quota violations

