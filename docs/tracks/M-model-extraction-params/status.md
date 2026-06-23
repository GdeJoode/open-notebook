# Track M — Status Ledger

| Phase | Title | Status | Branch | Notes |
|---|---|---|---|---|
| M.1 | Per-model config: backfill context_window/max_output_tokens + pin factory threading | **Done (review pending)** | `track/m-interim-gemini-ratelimit` | `5fdc17c` |
| M.2 | Per-provider RPM caps (google/nvidia/ollama) + pace-vs-skip | **Done (review pending)** | same | `3883cd3` |
| M.3 | Chunk packing derived from the active model's context_window | **Done (review pending)** | same | `6a884d6`, `bb53b27` |
| M.4 | Oversized-chunk guard for the small-context fallback (interim V1, Decision M-D3 (b)) | **Done (review pending)** | same | `7c71f31` |
| M.5 | Heterogeneous-chain integration test + metrics + ARCHITECTURE docs | **Not in this slice** | — | Out of the requested core scope |

## Slice delivered (2026-06-23)

**Scope:** M.1 + M.2 + M.3 + the M.4 oversized-chunk guard — the minimum that makes
extraction MODEL-AWARE so a big-context model uses a few large LLM calls instead of
~28 tiny ones. M.5 (the full heterogeneous-chain integration test + chunking_metrics
module + ARCHITECTURE/ROADMAP doc edits) and the full M.4 per-document failover
re-architecture (Decision M-D3 (a)) are deliberately deferred.

### Commit range
`3883cd3 .. bb53b27` on `track/m-interim-gemini-ratelimit`
(branch already carried the interim Gemini rate-limit fix at `dependencies.py` +
`failover_executor.py`, formalized here as M.2).

### How chunks-per-call now derive from context_window (the re-pack mechanism)

- The ingestion pipeline still chunks once at a fixed ~2000 chars and persists to the
  `chunk` table (unchanged storage unit).
- **Both** extraction loops issue **one LLM call per chunk dict** —
  `ExtractionWorkflow.extract` single-mode and `run_pass2`. So raising `batch_size`
  alone would NOT have reduced calls (it is a logging/grouping knob, not a
  combined-prompt knob). The fix is to RE-PACK the chunk dicts before the loop.
- `extraction_chunking.context_packer.pack_chunks_for_model` greedily concatenates the
  persisted 2000-char chunks into model-sized windows. Each packed window is one
  combined-text dict → **one LLM call**. Input budget =
  `(context_window − max_output_tokens − prompt_overhead) × 0.85`, token-estimated at a
  conservative `ceil(chars/4)`. Provenance is preserved as a `constituent_chunk_ids`
  list on each window (Decision M-D4); structural metadata mirrors the first chunk.
- `entity_extraction_service._pack_chunks_for_route_head` resolves the route head's
  model record and packs `chunk_dicts` before the multi/single handoff. The matching
  context-derived token budget is threaded into Pass-2 (`run_pass2(token_budget=…)` via
  `workflow.extract` → `run_multi_schema`) so a legitimately-large packed window is not
  rejected by the legacy fixed 2400-token guard — while a genuine overflow still raises.
  Null `context_window` → degrade to the un-packed chunks + the legacy 2400 cap.

### Measured call-count drop (28-chunk document, ~2000 chars/chunk)

| Model | context_window | windows (was 28) | input budget | max window | overflow |
|---|---|---|---|---|---|
| llama3.1:8b (local) | 8 192 | **4** | 4 828 tok | 4 502 tok | no |
| NIM mistral-medium | 128 000 | **1** | 103 618 tok | 14 007 tok | no |
| mistral-large-3 | 256 000 | **1** | 212 418 tok | 14 007 tok | no |
| Ministral-14B | 131 072 | **1** | 106 229 tok | 14 007 tok | no |

A 128K/256K model reads the whole document in **one** call (28 → 1, a 28× drop); the
8K local fallback re-packs into 4 bounded windows with **no overflow**. This removes the
~28-tiny-batch distortion so a cross-model speed/accuracy comparison is fair.

### M.4 oversized-chunk guard (interim V1) + the num_ctx / context_window decision

- **Decision M-D3 = (b)**: the guard, not the full per-document failover re-architecture.
- **Two layers:**
  1. The packer re-splits a single oversized ingestion chunk and re-packs a document for
     a SMALL candidate so no emitted window exceeds that candidate's input budget.
  2. `llm_call.call_candidate`: for a local Ollama candidate, request
     `num_ctx = candidate.model.context_window` so the runtime allocates the window the
     packer sized against (Ollama's `num_ctx` default ~2-4K would otherwise truncate
     silently), and log a WARNING when the estimated prompt exceeds the candidate's
     context window (a per-call-failover from a bigger primary is observable).
- **The num_ctx / context_window decision (both):** the local model row's
  `context_window` is seeded to a **conservative 8192** (the SAFE effective window, NOT
  llama's 128K theoretical max). `num_ctx = candidate.model.context_window` is passed so
  the runtime allocates that window instead of Ollama's tiny ~2-4K default.
  - **Scope of the interim — what it does and does NOT do:** the pack is sized for the
    route-HEAD (cloud) candidate, NOT for the local fallback. On a per-call failover to
    llama, `llm_call.py:172-188` sets `num_ctx` + logs a WARNING when the prompt exceeds
    the local window, but does **NOT re-split** the already-packed window. Ollama then
    **truncates** an oversized prompt (silent loss). So the interim **bounds the runtime
    `num_ctx` allocation and makes the overflow observable (WARNING), but does NOT
    re-split the fallback prompt.** The full per-document re-chunk on fallback is an
    accepted deferral — **M.4(a)**.

### M.2 per-provider caps

`resolve_per_provider_rpm()` (DI/env layer, Decision M-D2 — no registry/DB schema
change) wired into the rate-limiter singleton:

- `google` ≈ 6/min (Gemini free tier ~10; conservatively under)
- `nvidia` ≈ 30/min (NIM fair-use ~40; headroom)
- `ollama`/local — effectively unlimited (high-cap local window)
- env-overridable: `GOOGLE_RPM` / `NVIDIA_RPM` / `OLLAMA_RPM`.

PACE preserved (Decision M-D1): an over-cap acquire blocks until a slot ages out within
`max_wait`; only past `max_wait` does it raise → `SKIPPED_RATE_LIMITED` failover. The
message-based `default_is_rate_limit` fix (catches esperanto-wrapped 429 /
`RESOURCE_EXHAUSTED`) is kept so a transient cap backoff-retries on the same provider.

### Contract regression results (all green)

- **B.8 output shape:** `test_entity_extraction_service.py` + `test_entity_persistence_service.py`
  pass (73 tests) — bigger packs yield the same per-entity/relation structures; dedup
  absorbs chunk-boundary differences. Persistence unchanged.
- **Track L typing bridge:** `applicable_schemas` still threads through the packed path
  (`pass2_token_budget` is additive, schemas are document-level). `test_schema_application_regiodeal.py`
  (9) + the L-threading assertions in the entity service tests pass.
- **Track J:** `test_rate_limiter.py`, `test_provider_rate_limiter.py`,
  `test_failover_executor.py`, `test_route_resolver.py` all green — the cap is populated,
  the algorithm unchanged.
- **I.G embedding pin canary:** `len(LLMTask) == 3` asserted; no embedding routing /
  new task added.
- **Full slice:** `pipelines/ontology-extraction/tests/` + the keyworded `apps/app-main/tests/`
  selection = **682 passed, 1 skipped, 0 failed**. `from app_main.api.app import create_app`
  imports clean. Changed files lint clean (pre-existing repo-wide I001/F821 in
  `entity_extraction_service.py` / `workflow.py` are unchanged by this work).

### New / changed files
- `apps/app-main/src/app_main/services/extraction_chunking/{__init__,context_packer}.py` (new)
- `apps/app-main/src/app_main/services/model_routing/seed_chain_models.py` (new)
- `apps/app-main/src/app_main/services/model_routing/rate_limiter.py` (resolve_per_provider_rpm)
- `apps/app-main/src/app_main/services/model_routing/llm_call.py` (M.4 guard)
- `apps/app-main/src/app_main/services/entity_extraction_service.py` (pack seam + budget thread)
- `apps/app-main/src/app_main/dependencies.py` (wire per-provider caps)
- `apps/app-main/src/app_main/api/app.py` (startup backfill)
- `pipelines/ontology-extraction/src/ontology_extraction/{pass2_typed_extraction,workflow,multi_schema_orchestrator}.py` (token_budget thread)
- Tests: `test_context_packer.py`, `test_per_provider_rate_cap.py`, `test_chain_model_params.py`, `test_oversized_chunk_guard.py` (new) + two fakes updated in `test_multi_schema_orchestrator.py`, one test added in `test_pass2.py`.

### Deferred (not in this slice)
- **M.5**: `chunking_metrics.py` (overflow_count / est_calls CI gate), the
  `test_heterogeneous_chain_extraction.py` 3-candidate integration test, `ARCHITECTURE.md`
  + `FEATURE_ROADMAP.md` edits.
- **M.4 (a)**: the full per-document failover re-architecture (each candidate re-chunks
  inside the attempt). The interim guard closes the live overflow bug; the full
  re-architecture is the follow-up.
