# Refactoring Plan: Migrate GPU/VLM Processing to content-core

## Overview

Open-notebook currently has duplicate implementations for GPU-accelerated document processing:
- `open_notebook/processors/docling_gpu.py` - Custom GPU implementation
- `open_notebook/processors/docling_granite.py` - Custom Granite VLM implementation
- `open_notebook/utils/docling_gpu.py` - Custom GPU converter factory

These should be removed in favor of using content-core's built-in GPU and VLM support.

---

## Phase 1: Backend Refactoring

### 1.1 Files to Delete

```
open_notebook/processors/docling_gpu.py
open_notebook/processors/docling_granite.py
open_notebook/utils/docling_gpu.py
```

### 1.2 Update `open_notebook/processors/__init__.py`

**Current:**
```python
from open_notebook.processors.chunk_extractor import extract_chunks_from_docling
from open_notebook.processors.docling_gpu import extract_with_docling_gpu
from open_notebook.processors.docling_granite import extract_with_docling_granite

__all__ = ["extract_chunks_from_docling", "extract_with_docling_gpu", "extract_with_docling_granite"]
```

**New:**
```python
from open_notebook.processors.chunk_extractor import extract_chunks_from_docling

__all__ = ["extract_chunks_from_docling"]
```

### 1.3 Update `open_notebook/graphs/source.py`

**Current logic (lines 54-72):**
```python
if document_engine in ["docling_gpu", "docling_granite"]:
    content_input.document_engine = "auto"
else:
    content_input.document_engine = document_engine

if document_engine == "docling_gpu":
    logger.info("🚀 Using GPU-accelerated Docling processor")
    processed_state = await extract_with_docling_gpu(content_input)
elif document_engine == "docling_granite":
    logger.info("🚀 Using Docling-Granite (IBM Granite4 VLM) processor")
    processed_state = await extract_with_docling_granite(content_input)
else:
    processed_state = await extract_content(content_input)
```

**New logic:**
```python
from content_core.config import (
    set_docling_gpu_enabled,
    set_docling_gpu_device,
    set_docling_pipeline,
    set_docling_vlm_model,
)

# Apply content-core configuration based on settings
content_settings = await ContentSettings.get_instance()

# Configure content-core based on user settings
if content_settings.docling_gpu_enabled:
    set_docling_gpu_enabled(True)
    set_docling_gpu_device(content_settings.docling_gpu_device or "auto")

if content_settings.docling_pipeline:
    set_docling_pipeline(content_settings.docling_pipeline)

if content_settings.docling_vlm_model:
    set_docling_vlm_model(content_settings.docling_vlm_model)

# Always use content-core's extract_content
content_input.document_engine = content_settings.default_content_processing_engine_doc or "auto"
content_input.url_engine = content_settings.default_content_processing_engine_url or "auto"

processed_state = await extract_content(content_input)
```

### 1.4 Update `open_notebook/domain/content_settings.py`

**Current:**
```python
class ContentSettings(RecordModel):
    default_content_processing_engine_doc: Optional[
        Literal["auto", "docling", "docling_gpu", "docling_granite", "simple"]
    ] = Field("auto", ...)
```

**New - Add content-core configuration fields:**
```python
class ContentSettings(RecordModel):
    record_id: ClassVar[str] = "open_notebook:content_settings"

    # Document Engine (simplified - GPU/VLM controlled separately)
    default_content_processing_engine_doc: Optional[
        Literal["auto", "docling", "simple"]
    ] = Field("auto", description="Default Content Processing Engine for Documents")

    default_content_processing_engine_url: Optional[
        Literal["auto", "firecrawl", "jina", "simple"]
    ] = Field("auto", description="Default Content Processing Engine for URLs")

    # GPU Acceleration Settings (content-core)
    docling_gpu_enabled: Optional[bool] = Field(
        True, description="Enable GPU acceleration for Docling"
    )
    docling_gpu_device: Optional[Literal["auto", "cuda", "cpu"]] = Field(
        "auto", description="GPU device selection"
    )

    # Pipeline Settings (content-core)
    docling_pipeline: Optional[Literal["auto", "standard", "vlm"]] = Field(
        "standard", description="Docling processing pipeline (standard=GPU OCR, vlm=Vision-Language Model)"
    )

    # VLM Settings (content-core)
    docling_vlm_model: Optional[
        Literal["granite-docling-258m", "smoldocling-256m"]
    ] = Field("granite-docling-258m", description="VLM model for document processing")
    docling_vlm_framework: Optional[Literal["auto", "transformers", "mlx"]] = Field(
        "auto", description="VLM framework selection"
    )

    # OCR Settings (content-core)
    docling_ocr_engine: Optional[
        Literal["auto", "easyocr", "rapidocr", "tesseract"]
    ] = Field("easyocr", description="OCR engine for text recognition")
    docling_ocr_languages: Optional[List[str]] = Field(
        ["en"], description="OCR languages"
    )
    docling_ocr_use_gpu: Optional[bool] = Field(
        True, description="Use GPU for OCR acceleration"
    )

    # Table Processing Settings (content-core)
    docling_table_mode: Optional[Literal["accurate", "fast"]] = Field(
        "accurate", description="Table structure recognition mode"
    )

    # Image Export Settings (content-core)
    docling_auto_export_images: Optional[bool] = Field(
        False, description="Automatically export images during extraction"
    )
    docling_image_scale: Optional[float] = Field(
        2.0, description="Image extraction scale (1.0-4.0)"
    )

    # Chunking Settings (content-core)
    docling_chunking_enabled: Optional[bool] = Field(
        False, description="Enable automatic chunking"
    )
    docling_chunking_method: Optional[Literal["hybrid", "hierarchical"]] = Field(
        "hybrid", description="Chunking method"
    )
    docling_chunking_max_tokens: Optional[int] = Field(
        512, description="Maximum tokens per chunk"
    )

    # Existing fields...
    default_embedding_option: Optional[Literal["ask", "always", "never"]] = Field(
        "ask", description="Default Embedding Option for Vector Search"
    )
    auto_delete_files: Optional[Literal["yes", "no"]] = Field(
        "yes", description="Auto Delete Uploaded Files"
    )
    youtube_preferred_languages: Optional[List[str]] = Field(
        ["en", "pt", "es", "de", "nl", "en-GB", "fr", "de", "hi", "ja"],
        description="Preferred languages for YouTube transcripts",
    )

    # File Management Settings (existing)
    input_directory_path: Optional[str] = Field(...)
    markdown_directory_path: Optional[str] = Field(...)
    output_directory_path: Optional[str] = Field(...)
    file_operation: Optional[Literal["copy", "move", "none"]] = Field(...)
    output_naming_scheme: Optional[...] = Field(...)
```

### 1.5 Create Configuration Bridge (`open_notebook/utils/content_core_config.py`)

```python
"""
Bridge between open-notebook settings and content-core configuration.
"""
from content_core.config import (
    set_docling_gpu_enabled,
    set_docling_gpu_device,
    set_docling_pipeline,
    set_docling_vlm_model,
    set_docling_vlm_framework,
    set_docling_ocr_engine,
    set_docling_table_structure_mode,
    set_docling_image_scale,
    set_docling_auto_export_images,
    set_docling_chunking_enabled,
    set_docling_chunking_config,
)
from loguru import logger


async def apply_content_core_settings(content_settings) -> None:
    """
    Apply open-notebook ContentSettings to content-core configuration.

    This should be called before any content-core extraction operations.
    """
    logger.debug("Applying content-core configuration from settings")

    # GPU Settings
    if content_settings.docling_gpu_enabled:
        set_docling_gpu_enabled(True)
        logger.info("🚀 GPU acceleration enabled for content-core")

    if content_settings.docling_gpu_device:
        set_docling_gpu_device(content_settings.docling_gpu_device)

    # Pipeline Settings
    if content_settings.docling_pipeline:
        set_docling_pipeline(content_settings.docling_pipeline)
        logger.info(f"📄 Using {content_settings.docling_pipeline} pipeline")

    # VLM Settings
    if content_settings.docling_vlm_model:
        set_docling_vlm_model(content_settings.docling_vlm_model)

    if content_settings.docling_vlm_framework:
        set_docling_vlm_framework(content_settings.docling_vlm_framework)

    # OCR Settings
    if content_settings.docling_ocr_engine:
        set_docling_ocr_engine(content_settings.docling_ocr_engine)

    # Table Settings
    if content_settings.docling_table_mode:
        set_docling_table_structure_mode(content_settings.docling_table_mode)

    # Image Settings
    if content_settings.docling_auto_export_images is not None:
        set_docling_auto_export_images(content_settings.docling_auto_export_images)

    if content_settings.docling_image_scale:
        set_docling_image_scale(content_settings.docling_image_scale)

    # Chunking Settings
    if content_settings.docling_chunking_enabled is not None:
        set_docling_chunking_enabled(content_settings.docling_chunking_enabled)

    if content_settings.docling_chunking_method or content_settings.docling_chunking_max_tokens:
        set_docling_chunking_config(
            method=content_settings.docling_chunking_method,
            max_tokens=content_settings.docling_chunking_max_tokens,
        )

    logger.debug("✅ Content-core configuration applied")
```

---

## Phase 2: Frontend Settings UI

### 2.1 Update Settings Schema (`frontend/src/app/(dashboard)/settings/components/SettingsForm.tsx`)

**Add new fields to schema:**
```typescript
const settingsSchema = z.object({
  // Simplified document engine (no more docling_gpu, docling_granite)
  default_content_processing_engine_doc: z.enum(['auto', 'docling', 'simple']).optional(),
  default_content_processing_engine_url: z.enum(['auto', 'firecrawl', 'jina', 'simple']).optional(),

  // GPU Settings
  docling_gpu_enabled: z.boolean().optional(),
  docling_gpu_device: z.enum(['auto', 'cuda', 'cpu']).optional(),

  // Pipeline Settings
  docling_pipeline: z.enum(['auto', 'standard', 'vlm']).optional(),

  // VLM Settings
  docling_vlm_model: z.enum(['granite-docling-258m', 'smoldocling-256m']).optional(),
  docling_vlm_framework: z.enum(['auto', 'transformers', 'mlx']).optional(),

  // OCR Settings
  docling_ocr_engine: z.enum(['auto', 'easyocr', 'rapidocr', 'tesseract']).optional(),
  docling_ocr_languages: z.array(z.string()).optional(),
  docling_ocr_use_gpu: z.boolean().optional(),

  // Table Settings
  docling_table_mode: z.enum(['accurate', 'fast']).optional(),

  // Image Settings
  docling_auto_export_images: z.boolean().optional(),
  docling_image_scale: z.number().min(1).max(4).optional(),

  // Chunking Settings
  docling_chunking_enabled: z.boolean().optional(),
  docling_chunking_method: z.enum(['hybrid', 'hierarchical']).optional(),
  docling_chunking_max_tokens: z.number().min(64).max(4096).optional(),

  // Existing fields...
  default_embedding_option: z.enum(['ask', 'always', 'never']).optional(),
  auto_delete_files: z.enum(['yes', 'no']).optional(),
  // ...
})
```

### 2.2 Add New Settings Card for Content-Core Configuration

```tsx
<Card>
  <CardHeader>
    <CardTitle>Advanced Document Processing (content-core)</CardTitle>
    <CardDescription>
      Fine-tune document extraction with GPU acceleration, VLM models, and OCR settings
    </CardDescription>
  </CardHeader>
  <CardContent className="space-y-6">

    {/* GPU Acceleration Section */}
    <div className="space-y-4 p-4 border rounded-lg">
      <h4 className="font-medium">GPU Acceleration</h4>

      <div className="flex items-center justify-between">
        <Label htmlFor="gpu_enabled">Enable GPU Acceleration</Label>
        <Controller
          name="docling_gpu_enabled"
          control={control}
          render={({ field }) => (
            <Switch
              checked={field.value || false}
              onCheckedChange={field.onChange}
            />
          )}
        />
      </div>

      <div className="space-y-2">
        <Label htmlFor="gpu_device">GPU Device</Label>
        <Controller
          name="docling_gpu_device"
          control={control}
          render={({ field }) => (
            <Select value={field.value} onValueChange={field.onChange}>
              <SelectTrigger>
                <SelectValue placeholder="Select device" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">Auto-detect</SelectItem>
                <SelectItem value="cuda">CUDA (NVIDIA GPU)</SelectItem>
                <SelectItem value="cpu">CPU Only</SelectItem>
              </SelectContent>
            </Select>
          )}
        />
      </div>
    </div>

    {/* Pipeline Section */}
    <div className="space-y-4 p-4 border rounded-lg">
      <h4 className="font-medium">Processing Pipeline</h4>

      <div className="space-y-2">
        <Label>Pipeline Type</Label>
        <Controller
          name="docling_pipeline"
          control={control}
          render={({ field }) => (
            <Select value={field.value} onValueChange={field.onChange}>
              <SelectTrigger>
                <SelectValue placeholder="Select pipeline" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">Auto (Recommended)</SelectItem>
                <SelectItem value="standard">Standard (CPU/GPU OCR)</SelectItem>
                <SelectItem value="vlm">VLM (Vision-Language Model)</SelectItem>
              </SelectContent>
            </Select>
          )}
        />
        <p className="text-sm text-muted-foreground">
          VLM provides best quality for scientific documents with equations and complex tables
        </p>
      </div>

      {/* Show VLM options when VLM pipeline selected */}
      {watchedPipeline === 'vlm' && (
        <>
          <div className="space-y-2">
            <Label>VLM Model</Label>
            <Controller
              name="docling_vlm_model"
              control={control}
              render={({ field }) => (
                <Select value={field.value} onValueChange={field.onChange}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="granite-docling-258m">
                      Granite Docling (IBM, 258M params)
                    </SelectItem>
                    <SelectItem value="smoldocling-256m">
                      SmolDocling (256M params)
                    </SelectItem>
                  </SelectContent>
                </Select>
              )}
            />
          </div>

          <div className="space-y-2">
            <Label>VLM Framework</Label>
            <Controller
              name="docling_vlm_framework"
              control={control}
              render={({ field }) => (
                <Select value={field.value} onValueChange={field.onChange}>
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Auto-detect</SelectItem>
                    <SelectItem value="transformers">Transformers (CUDA)</SelectItem>
                    <SelectItem value="mlx">MLX (Apple Silicon)</SelectItem>
                  </SelectContent>
                </Select>
              )}
            />
          </div>
        </>
      )}
    </div>

    {/* OCR Section */}
    <div className="space-y-4 p-4 border rounded-lg">
      <h4 className="font-medium">OCR Settings (Standard Pipeline)</h4>

      <div className="space-y-2">
        <Label>OCR Engine</Label>
        <Controller
          name="docling_ocr_engine"
          control={control}
          render={({ field }) => (
            <Select value={field.value} onValueChange={field.onChange}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="easyocr">EasyOCR (Recommended)</SelectItem>
                <SelectItem value="rapidocr">RapidOCR</SelectItem>
                <SelectItem value="tesseract">Tesseract</SelectItem>
                <SelectItem value="auto">Auto</SelectItem>
              </SelectContent>
            </Select>
          )}
        />
        <p className="text-sm text-muted-foreground">
          EasyOCR is 9x faster than RapidOCR for European languages
        </p>
      </div>

      <div className="flex items-center justify-between">
        <Label>Use GPU for OCR</Label>
        <Controller
          name="docling_ocr_use_gpu"
          control={control}
          render={({ field }) => (
            <Switch
              checked={field.value ?? true}
              onCheckedChange={field.onChange}
            />
          )}
        />
      </div>
    </div>

    {/* Table Processing Section */}
    <div className="space-y-4 p-4 border rounded-lg">
      <h4 className="font-medium">Table Processing</h4>

      <div className="space-y-2">
        <Label>Table Recognition Mode</Label>
        <Controller
          name="docling_table_mode"
          control={control}
          render={({ field }) => (
            <Select value={field.value} onValueChange={field.onChange}>
              <SelectTrigger>
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="accurate">Accurate (95% accuracy)</SelectItem>
                <SelectItem value="fast">Fast (Lower quality)</SelectItem>
              </SelectContent>
            </Select>
          )}
        />
      </div>
    </div>

    {/* Image Export Section */}
    <div className="space-y-4 p-4 border rounded-lg">
      <h4 className="font-medium">Image Extraction</h4>

      <div className="flex items-center justify-between">
        <Label>Auto-export Images</Label>
        <Controller
          name="docling_auto_export_images"
          control={control}
          render={({ field }) => (
            <Switch
              checked={field.value || false}
              onCheckedChange={field.onChange}
            />
          )}
        />
      </div>

      <div className="space-y-2">
        <Label>Image Scale (1.0 - 4.0)</Label>
        <Controller
          name="docling_image_scale"
          control={control}
          render={({ field }) => (
            <input
              type="number"
              min="1"
              max="4"
              step="0.5"
              value={field.value || 2.0}
              onChange={(e) => field.onChange(parseFloat(e.target.value))}
              className="w-full px-3 py-2 border rounded-md"
            />
          )}
        />
        <p className="text-sm text-muted-foreground">
          Higher values = better quality, slower processing
        </p>
      </div>
    </div>

  </CardContent>
</Card>
```

### 2.3 Update API Types (`frontend/src/lib/types/api.ts`)

```typescript
export interface ContentSettings {
  // Simplified document engine
  default_content_processing_engine_doc?: 'auto' | 'docling' | 'simple';
  default_content_processing_engine_url?: 'auto' | 'firecrawl' | 'jina' | 'simple';

  // GPU Settings
  docling_gpu_enabled?: boolean;
  docling_gpu_device?: 'auto' | 'cuda' | 'cpu';

  // Pipeline Settings
  docling_pipeline?: 'auto' | 'standard' | 'vlm';

  // VLM Settings
  docling_vlm_model?: 'granite-docling-258m' | 'smoldocling-256m';
  docling_vlm_framework?: 'auto' | 'transformers' | 'mlx';

  // OCR Settings
  docling_ocr_engine?: 'auto' | 'easyocr' | 'rapidocr' | 'tesseract';
  docling_ocr_languages?: string[];
  docling_ocr_use_gpu?: boolean;

  // Table Settings
  docling_table_mode?: 'accurate' | 'fast';

  // Image Settings
  docling_auto_export_images?: boolean;
  docling_image_scale?: number;

  // Chunking Settings
  docling_chunking_enabled?: boolean;
  docling_chunking_method?: 'hybrid' | 'hierarchical';
  docling_chunking_max_tokens?: number;

  // Existing...
  default_embedding_option?: 'ask' | 'always' | 'never';
  auto_delete_files?: 'yes' | 'no';
  youtube_preferred_languages?: string[];
  input_directory_path?: string;
  markdown_directory_path?: string;
  output_directory_path?: string;
  file_operation?: 'copy' | 'move' | 'none';
  output_naming_scheme?: 'timestamp_prefix' | 'date_prefix' | 'datetime_suffix' | 'original';
}
```

---

## Phase 3: Migration Steps

### Step 1: Database Migration
Users with existing `docling_gpu` or `docling_granite` settings need migration:

```python
# Migration script to run once
async def migrate_content_settings():
    settings = await ContentSettings.get_instance()

    if settings.default_content_processing_engine_doc == "docling_gpu":
        settings.default_content_processing_engine_doc = "docling"
        settings.docling_gpu_enabled = True
        settings.docling_pipeline = "standard"

    elif settings.default_content_processing_engine_doc == "docling_granite":
        settings.default_content_processing_engine_doc = "docling"
        settings.docling_gpu_enabled = True
        settings.docling_pipeline = "vlm"
        settings.docling_vlm_model = "granite-docling-258m"

    await settings.save()
```

### Step 2: Remove Deprecated Files
```bash
rm open_notebook/processors/docling_gpu.py
rm open_notebook/processors/docling_granite.py
rm open_notebook/utils/docling_gpu.py
```

### Step 3: Update Imports
Update all files that import from deleted modules.

---

## Benefits of This Refactoring

1. **Single Source of Truth**: All GPU/VLM logic lives in content-core
2. **Easier Maintenance**: Updates to content-core automatically benefit open-notebook
3. **More Configuration Options**: Users get access to all content-core features
4. **Cleaner Codebase**: Remove ~300 lines of duplicate code
5. **Better Testing**: content-core is tested independently
6. **Consistent Behavior**: Same extraction behavior in CLI and open-notebook

---

## Estimated Effort

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1 (Backend) | 2-3 hours | None |
| Phase 2 (Frontend) | 3-4 hours | Phase 1 |
| Phase 3 (Migration) | 1 hour | Phase 1 & 2 |
| Testing | 2-3 hours | All phases |

**Total: ~8-11 hours**

---

## Resolved Questions

### 1. Markdown export with assets
**RESOLVED**: Content-core now supports export with assets via `export_to_markdown(image_mode="referenced")` and chunking with bounding boxes via `chunk_content()`. No need to duplicate in open-notebook.

### 2. DoclingDocument serialization for LangGraph
**RESOLVED**: Content-core stores the `DoclingDocument` object in `metadata["docling_document"]`. For LangGraph state persistence, we need to serialize it to JSON after extraction. Add a post-processing step in `source.py` to:
```python
# After extract_content(), serialize DoclingDocument for LangGraph
if processed_state.metadata.get("docling_document"):
    doc = processed_state.metadata["docling_document"]
    processed_state.metadata["docling_document_json"] = doc.model_dump_json(exclude_none=True)
    del processed_state.metadata["docling_document"]  # Remove non-serializable object
```

### 3. Settings UI complexity
**RESOLVED**: Use collapsible "Advanced Document Processing" section. Default settings work for most users.

### 4. Default values
**RESOLVED - Updated based on user feedback:**
- **Pipeline**: `standard` (with GPU acceleration) - NOT `auto`
- **GPU**: `enabled=True`, `device=auto` - most users have GPUs
- **VLM model**: `granite-docling-258m` (when VLM pipeline selected)
- **Image extraction**: Configurable but not yet functional (work in progress)
