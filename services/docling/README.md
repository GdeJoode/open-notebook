# Docling Processing Service

Standalone Docker service for document parsing via [Docling](https://github.com/DS4SD/docling).

## Quick Start

```bash
# From the project root
docker compose up docling

# Process a file (paths are inside the container)
curl -X POST http://localhost:8100/process \
  -H "Content-Type: application/json" \
  -d '{"input_path": "/data/input/document.pdf", "output_path": "/data/output"}'

# Process a directory
curl -X POST http://localhost:8100/process \
  -H "Content-Type: application/json" \
  -d '{"input_path": "/data/input", "output_path": "/data/output"}'

# Health check
curl http://localhost:8100/health
```

## Volumes

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./docling_input` | `/data/input` | Drop files/directories here |
| `./docling_output` | `/data/output` | Parsed output appears here |

You can mount additional volumes as needed. The `input_path` and `output_path` in API
requests refer to paths **inside the container**.

## Environment Variables

All Docling pipeline options are configurable via environment variables:

### GPU / Accelerator
| Variable | Default | Values |
|----------|---------|--------|
| `DOCLING_DEVICE` | `cuda` | `auto`, `cuda`, `mps`, `cpu` |
| `DOCLING_NUM_THREADS` | `4` | integer |

### OCR
| Variable | Default | Values |
|----------|---------|--------|
| `DOCLING_DO_OCR` | `true` | `true`, `false` |
| `DOCLING_OCR_ENGINE` | `easyocr` | `easyocr`, `rapidocr`, `tesseract`, `none` |
| `DOCLING_OCR_LANG` | `en` | comma-separated language codes (e.g. `en,nl,de`) |

### Tables
| Variable | Default | Values |
|----------|---------|--------|
| `DOCLING_DO_TABLE_STRUCTURE` | `true` | `true`, `false` |
| `DOCLING_TABLE_MODE` | `accurate` | `accurate`, `fast` |

### Images
| Variable | Default | Values |
|----------|---------|--------|
| `DOCLING_GENERATE_PAGE_IMAGES` | `false` | `true`, `false` |
| `DOCLING_GENERATE_PICTURE_IMAGES` | `true` | `true`, `false` |
| `DOCLING_IMAGES_SCALE` | `2.0` | float (1.0 - 4.0) |
| `DOCLING_DO_PICTURE_CLASSIFICATION` | `true` | `true`, `false` |
| `DOCLING_DO_PICTURE_DESCRIPTION` | `true` | `true`, `false` |
| `DOCLING_VLM_MODEL` | `HuggingFaceTB/SmolVLM-256M-Instruct` | HuggingFace model ID |
| `DOCLING_VLM_PROMPT` | *(built-in)* | custom prompt string |

### Other
| Variable | Default | Values |
|----------|---------|--------|
| `DOCLING_DO_FORMULA_EXTRACTION` | `true` | `true`, `false` |
| `DOCLING_DO_CODE_ENRICHMENT` | `true` | `true`, `false` |
| `DOCLING_ORGANIZE_BY_PAGE` | `true` | `true`, `false` |

## GPU Support

NVIDIA CUDA GPU acceleration is enabled by default. The container uses the
`nvidia/cuda` base image and reserves one GPU via Docker Compose device
reservations. Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).

To run **without** a GPU (CPU-only), override the device setting:

```yaml
# docker-compose.cpu.yml
services:
  docling:
    image: ... # or build without nvidia base
    deploy: {}  # remove GPU reservation
    environment:
      - DOCLING_DEVICE=cpu
```

```bash
docker compose -f docker-compose.yml -f docker-compose.cpu.yml up docling
```

## Output Structure

Each processed file gets its own subdirectory under the output path:

```
/data/output/
  document_name/
    original_file/        # Copy of source document
    output/
      extracted_info/
        document.md       # Full markdown
        document.json     # Structured JSON
        metadata.json     # Extraction metadata
        images/           # Extracted images
        tables/           # Tables (MD + CSV)
```

## API Reference

### `POST /process`

Process a file or directory.

**Request body:**
```json
{
  "input_path": "/data/input/document.pdf",
  "output_path": "/data/output",
  "copy_original": true
}
```

**Response:**
```json
{
  "total_files": 1,
  "succeeded": 1,
  "failed": 0,
  "results": [
    {
      "source": "/data/input/document.pdf",
      "success": true,
      "output_dir": "/data/output/document",
      "markdown_path": "/data/output/document/output/extracted_info/document.md",
      "json_path": "/data/output/document/output/extracted_info/document.json",
      "table_count": 3,
      "image_count": 5,
      "page_count": 12,
      "processing_seconds": 45.2
    }
  ],
  "total_seconds": 45.2
}
```

### `GET /health`

Returns `{"status": "ok", "service": "docling"}`.
