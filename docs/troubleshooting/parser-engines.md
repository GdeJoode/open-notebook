# Parser engine troubleshooting

Open Notebook ships with four parser engines (Docling, MinerU, Auto,
Simple) selectable per-document or globally in Settings → Content
Processing. This page covers the four operational issues you are most
likely to hit.

## MinerU container fails to start

**Symptom**: `docker compose up mineru` exits with `unable to find driver capabilities` or `nvidia-smi: command not found`.

**Cause**: MinerU's container image is CUDA 12.6-based and requires the NVIDIA Container Toolkit on the host.

**Fix**:
1. Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on the host.
2. Verify with `docker run --rm --gpus all nvidia/cuda:12.6.3-runtime-ubuntu24.04 nvidia-smi`.
3. If you don't have an NVIDIA GPU at all, **do not start the MinerU service** — the main app boots fine without it (no `depends_on` link). Settings → Parser engine = `docling` will avoid every call site.

## Auto-fallback fires too often (confidence too aggressive)

**Symptom**: nearly every uploaded PDF shows the amber "MinerU (auto-fallback, conf …)" badge even when the docling output looks clean.

**Cause**: the default confidence threshold (0.95) is calibrated for academic / structured docs. Highly heterogeneous corpora (mixed Word→PDF exports, multi-language docs, image-rich marketing material) may legitimately score 0.85-0.94.

**Fix**:
1. Go to Settings → Content Processing → Parser engine.
2. Select **Auto** and lower the **Confidence threshold** slider to 0.85 (or run `apps/app-main/scripts/score_pdf_corpus.py` against your corpus and pick a percentile-based value; see [`docs/tracks/A-mineru/threshold-tuning.md`](../tracks/A-mineru/threshold-tuning.md)).
3. Reprocess the affected source(s) from the source-detail page.

## Rolling back to docling-only

**Symptom**: MinerU is causing more problems than it solves, or model weights are eating disk space.

**Fix**:
1. Settings → Parser engine → **Docling** (or **Simple** for the lightest path).
2. `docker compose stop mineru && docker compose rm mineru` to free the container.
3. (Optional) `docker volume rm open-notebook_mineru_models` to reclaim the 5-8 GB model cache. **Note**: this means re-downloading on the next start.
4. Per-source overrides are reset by reprocessing — the next `POST /sources/{id}/reprocess` without `parser_engine` in the body honours the new global setting.

## Bad confidence score on a document that looks fine

**Symptom**: a clean native PDF scores 0.78 and gets fallback'd; manually inspecting the docling output shows it's perfectly acceptable.

**Cause**: the score is signal-driven — most likely culprits: low heading count (`heading_rate` signal saturates at 2 headings/page; a doc with one heading per 5 pages scores 0.1 on this signal), or unusual element types showing up as `unknown`, or docling marking tables as detected with zero parsed rows (`table_success` → 0.0).

**Fix**:
1. Run `uv run --project apps/app-main python apps/app-main/scripts/score_pdf_corpus.py path/to/file.pdf` to see the per-signal breakdown.
2. Decide if the low-scoring signal is meaningful for your corpus. If most docs lack headings or have unparsable tables legitimately, lower the threshold (see "Auto-fallback fires too often" above).
3. As a per-document override, reprocess with `parser_engine = 'docling'` to force the docling path even if auto would fall back.
