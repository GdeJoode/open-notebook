# DoclingDocument Serialization in LangGraph

## Problem

DoclingDocument objects from the Docling SDK are Pydantic models that cannot be directly passed through LangGraph's state serialization. LangGraph converts state to JSON between nodes, which loses complex Python objects.

## Solution

We serialize DoclingDocument to JSON using Pydantic's built-in serialization, store it in metadata, and reconstruct it when needed in downstream nodes.

### Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Processor Node (docling_granite, docling_gpu)              │
│                                                             │
│ 1. Convert document → DoclingDocument                       │
│ 2. Export markdown with assets (before serialization)       │
│ 3. Serialize: doc.model_dump_json()                        │
│ 4. Store in metadata["docling_document_json"]              │
└─────────────────────────────────────────────────────────────┘
                          │
                          ├─ LangGraph Serialization ─┐
                          │                            │
                          ▼                            ▼
┌─────────────────────────────────────┐  ┌────────────────────────────────┐
│ File Management Node (source.py)    │  │ Chunk Extraction Node          │
│                                      │  │                                │
│ 1. Check if markdown export done     │  │ 1. Reconstruct DoclingDocument │
│ 2. If not: reconstruct_docling_doc  │  │    from JSON                   │
│ 3. Export markdown with assets       │  │ 2. Extract chunks efficiently  │
└─────────────────────────────────────┘  └────────────────────────────────┘
```

### Key Components

#### 1. Serialization in Processors

**Files**: `docling_granite.py`, `docling_gpu.py`

```python
# Serialize DoclingDocument to JSON for LangGraph state persistence
try:
    # Use exclude_none=True to reduce JSON size (per Docling documentation)
    state.metadata["docling_document_json"] = doc.model_dump_json(exclude_none=True)
    logger.debug("✅ Serialized DoclingDocument to JSON for state persistence")
except Exception as e:
    logger.warning(f"Failed to serialize DoclingDocument: {e}")
```

**Why here?**
- Before LangGraph serialization
- DoclingDocument is still in memory
- Export markdown with assets before serialization (DoclingDocument has images/tables)

#### 2. Reconstruction Utility

**File**: `utils/docling_utils.py`

```python
from pydantic import ValidationError

def reconstruct_docling_document(metadata: dict) -> Optional[DoclingDocument]:
    """
    Reconstruct a DoclingDocument from serialized JSON in metadata.

    Returns:
        DoclingDocument instance if reconstruction succeeds, None otherwise
    """
    docling_json = metadata.get("docling_document_json")
    if not docling_json:
        return None

    try:
        # Use Pydantic's model_validate_json for proper validation
        doc = DoclingDocument.model_validate_json(docling_json)
        logger.debug("✅ Successfully reconstructed DoclingDocument from JSON")
        return doc
    except ValidationError as e:
        logger.error(f"Validation failed while reconstructing DoclingDocument: {e}")
        for error in e.errors():
            logger.error(f"  - {error['loc']}: {error['msg']}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error reconstructing DoclingDocument: {e}")
        return None
```

**Usage in any LangGraph node:**

```python
from open_notebook.utils.docling_utils import reconstruct_docling_document

# In your node function
doc = reconstruct_docling_document(state["content_state"].metadata)
if doc:
    # Use DoclingDocument for embeddings, analysis, etc.
    markdown = doc.export_to_markdown()
    chunks = extract_chunks_from_existing_document(doc)
```

#### 3. Efficient Chunk Extraction

**File**: `processors/chunk_extractor.py`

```python
def extract_chunks_from_existing_document(
    doc: DoclingDocument
) -> List[Dict[str, Any]]:
    """
    Extract chunks from an existing DoclingDocument (no re-conversion needed).

    This is more efficient when you already have a DoclingDocument from processing.
    """
    chunks = extract_chunks_with_positions(doc, result=None)
    return chunks
```

**Usage in source.py:**

```python
# Try to use reconstructed DoclingDocument first (more efficient)
doc = reconstruct_docling_document(processed_state.metadata)
if doc:
    logger.debug("Using reconstructed DoclingDocument for chunk extraction")
    chunks = extract_chunks_from_existing_document(doc)
else:
    # Fallback: Re-convert document
    _, chunks, _ = extract_chunks_from_docling(file_path)
```

### Benefits

1. **Proper Architecture**: DoclingDocument available throughout LangGraph pipeline
2. **Efficiency**: No re-conversion needed for chunk extraction or other processing
3. **Flexibility**: Any downstream node can reconstruct and use the document
4. **Fallback**: Graceful degradation if serialization fails

### Use Cases

#### Embeddings with Visual Context

```python
# In embedding node
doc = reconstruct_docling_document(state.metadata)
if doc:
    # Extract images for visual embeddings
    for item in doc.iterate_items():
        if isinstance(item, PictureItem):
            # Process image with vision model
            visual_embedding = embed_image(item.image)
```

#### Advanced Table Processing

```python
# In analysis node
doc = reconstruct_docling_document(state.metadata)
if doc:
    for item in doc.iterate_items():
        if isinstance(item, TableItem):
            df = item.export_to_dataframe()
            # Perform data analysis, validation, etc.
```

#### Document Structure Analysis

```python
# In structure analysis node
doc = reconstruct_docling_document(state.metadata)
if doc:
    structure = analyze_document_structure(doc)
    # Get hierarchy, sections, references, etc.
```

### Performance Considerations

**Serialization Cost**:
- One-time cost in processor (~100-500ms for typical documents)
- JSON size: ~5-10x content size (includes structure, metadata)
- LangGraph handles JSON efficiently

**Reconstruction Cost**:
- Fast: Pydantic validation + object creation (~50-100ms)
- Much cheaper than re-converting document (5-30 seconds)

**Memory**:
- JSON stored in metadata (cleaned up after processing)
- Reconstructed document only exists during node execution
- No memory leaks from retained objects

### Migration Guide

If you have existing code that passes `docling_document` directly in metadata:

**Before (broken with LangGraph serialization):**
```python
state.metadata["docling_document"] = doc  # Lost during serialization
```

**After (proper serialization):**
```python
state.metadata["docling_document_json"] = doc.model_dump_json()
```

**Using in downstream nodes:**
```python
from open_notebook.utils.docling_utils import reconstruct_docling_document

doc = reconstruct_docling_document(state.metadata)
if doc:
    # Use document
    pass
```

### Testing

To verify DoclingDocument survives serialization:

```python
# In your test
def test_docling_serialization():
    # Convert document
    result = converter.convert("test.pdf")
    doc = result.document

    # Serialize
    json_str = doc.model_dump_json()

    # Simulate LangGraph state passing
    metadata = {"docling_document_json": json_str}

    # Reconstruct
    reconstructed_doc = reconstruct_docling_document(metadata)

    assert reconstructed_doc is not None
    assert reconstructed_doc.export_to_markdown() == doc.export_to_markdown()
```

### Future Enhancements

1. **Compression**: Compress JSON before storing (gzip, zstd)
2. **Partial Serialization**: Only serialize needed parts of document
3. **Caching**: Cache reconstructed documents within same execution
4. **Validation**: Add schema validation for serialized documents
