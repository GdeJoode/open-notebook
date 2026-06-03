# MinerU Processing Service

GPU-backed FastAPI service that parses documents through
[MinerU 2.x](https://github.com/opendatalab/MinerU). Runs as the
`mineru:8104` compose service alongside `docling:8100` and is reached
from `app-main` via `MineruHttpClient`.

The service is a thin wrapper around the `mineru` CLI; each
`POST /process` shells out one CLI invocation per input file and
returns the on-disk paths the CLI writes. Behavioural contract is
intentionally symmetric with `services/docling`.

## Endpoints

| Method | Path        | Purpose                                              |
| ------ | ----------- | ---------------------------------------------------- |
| `GET`  | `/health`   | Returns `{"status": "ok", "service": "mineru"}`      |
| `POST` | `/process`  | Process a single file or a directory of files        |

### `POST /process` request

```json
{
  "input_path": "/data/input/foo.pdf",
  "output_path": "/data/mineru_output",
  "copy_original": true
}
```

### `POST /process` response

```json
{
  "total_files": 1,
  "succeeded": 1,
  "failed": 0,
  "total_seconds": 42.18,
  "results": [{
    "source": "/data/input/foo.pdf",
    "success": true,
    "output_dir": "/data/mineru_output/foo/auto",
    "markdown_path": "/data/mineru_output/foo/auto/foo.md",
    "content_list_path": "/data/mineru_output/foo/auto/foo_content_list.json",
    "middle_json_path": "/data/mineru_output/foo/auto/foo_middle.json",
    "page_count": 12,
    "table_count": 3,
    "image_count": 8,
    "processing_seconds": 42.18,
    "backend": "pipeline",
    "method": "auto"
  }]
}
```

The MinerU CLI writes everything under
`<output_path>/<stem>/<method>/`. See `docs/tracks/A-mineru/MINERU_OUTPUT_SPIKE.md`
for the full schema characterisation.

## Configuration (env vars)

| Variable                     | Default     | Notes                                                                                          |
| ---------------------------- | ----------- | ---------------------------------------------------------------------------------------------- |
| `MINERU_DEVICE`              | `cuda`      | Forwarded to MinerU; `cpu` works but is slow                                                   |
| `MINERU_BACKEND`             | `pipeline`  | `pipeline` \| `hybrid-auto-engine` \| `vlm-auto-engine` \| `hybrid-http-client` \| `vlm-http-client` |
| `MINERU_METHOD`              | `auto`      | `auto` \| `txt` \| `ocr` (pipeline / hybrid backends only)                                     |
| `MINERU_LANG`                | `en`        | OCR language hint                                                                              |
| `MINERU_FORMULA_ENABLE`      | `true`      | Enable formula parsing                                                                         |
| `MINERU_TABLE_ENABLE`        | `true`      | Enable table parsing                                                                           |
| `MINERU_TIMEOUT_SECONDS`     | `1800`      | Per-file CLI timeout                                                                           |
| `MINERU_MODEL_SOURCE`        | `huggingface` | Set to `modelscope` if HuggingFace is unreachable                                            |
| `HF_HOME` / `TRANSFORMERS_CACHE` | `/data/models/huggingface` | Cache mount points; persist via a named volume so first-run downloads survive restarts |

## First-run notes

- The `mineru[all]` install in the Dockerfile pulls PyTorch + Ray +
  layout/OCR/formula/table models. First-time install builds the
  image to ~10–15 GB; on first request, model weights (~5–8 GB) are
  fetched from HuggingFace (or modelscope) and cached under
  `/data/models/...`. Mount a persistent volume so this doesn't
  repeat across container restarts.
- The first `POST /process` after a cold start may take several
  minutes (model download + warmup). Subsequent calls run in
  seconds-to-tens-of-seconds depending on PDF size.

## GPU memory

MinerU's `pipeline` backend needs roughly 6–8 GB of VRAM. Running
MinerU **alongside** docling on a single 24 GB card is feasible but
the dual-parser auto-fallback path (Phase A.1c) runs the two parsers
sequentially within one extraction call, so they never compete for
VRAM concurrently. If you co-locate both with other VLM workloads
(Ollama, etc.) plan for ≥ 16 GB free VRAM.

## Integration tests (manual / GPU required)

A pytest marker reserved for live MinerU runs is
`@pytest.mark.requires_gpu`. The unit-test suite (`apps/app-main/tests/
test_mineru_http_client.py`) mocks HTTP — no GPU needed.

Live smoke test:

```bash
docker compose up -d mineru
curl -sf http://localhost:8104/health
cp data/uploads/yardstickbias.pdf docling_input/
curl -X POST http://localhost:8104/process \
  -H 'Content-Type: application/json' \
  -d '{
        "input_path": "/data/input/yardstickbias.pdf",
        "output_path": "/data/mineru_output",
        "copy_original": true
      }'
ls -la mineru_output/yardstickbias/auto/
```

## Why a CLI wrapper instead of `mineru-api`?

MinerU ships its own `mineru-api` FastAPI service with an async task
queue. We deliberately wrap the CLI synchronously because:

1. The request/response shape stays symmetric with `services/docling`
   (same `MineruHttpClient` / `DoclingHttpClient` call pattern).
2. No need to poll task state or handle MinerU's 24h task eviction.
3. Deterministic on-disk output we can read back directly.

If MinerU later exposes a stable Python-API surface (current docs only
guarantee CLI + REST), we can swap the `subprocess.run` for an
in-process call without changing the HTTP contract.

## Attribution

The dual-parser pattern (docling + MinerU with confidence-driven
fallback) is inspired by the architecture discussed in
[SenolIsci/mykg](https://github.com/SenolIsci/mykg) (MIT). This
service is an independent implementation; only the architectural
shape is borrowed.
