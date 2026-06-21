# THINKING — KG extraction debug + entity-resolution live test

## Goals (measurable)
1. Root-cause why the 2 docs ingested today (`source:1k3c4…` Economics, `source:bc6xa…` Cohesion Policy) produced **0 entities**, with evidence + a reproduction.
2. Switch entity extraction to **qwen2.5:14b-instruct** (user choice) and prove the model is actually invoked (entities carry `extraction_method='llm'` + the model recorded).
3. Fix the extraction trigger so ingest → entities end-to-end (no silently-swallowed failures).
4. Ship the committed `ORDER BY name` fix by **rebuilding the Docker image**; verify the UI shows entities AND a **per-document KG**.
5. Review entity-resolution logic; ingest **all 11 Convenant/Regio Deal PDFs**; measure cross-document resolution (shared entities → single canonical entity).
6. Harden: tests, regression-green, document the schema-drift finding.

## Established facts (recon this session)
- Entity extraction is a **separate, notebook-triggered async job** (frontend polls `/api/notebooks/{id}/extraction/paused`), NOT automatic on source ingest. Likely never run / paused for the test notebook → 0 entities.
- `entity_extraction_service.py` already defaults to `extractor_type="llm"` (`ontology_extraction.extractors.llm_extractor`); `config.llm_model` selects the model.
- 147 existing entities are **legacy spaCy** data from 2025-12 (`spacy_noun_chunk`/`spacy`), noisy (`## Introduction`, ligature `/uniFB01`), `extraction_model=None`, flat confidence 1.0. Only 3 relations.
- `EXTRACTION_MODEL=llama3.1:8b-instruct-q4_0` (docker-compose.yml:130) → change to qwen2.5:14b-instruct.
- Ollama reachable on host:11434 with qwen2.5:14b-instruct + llama3.1:8b + embeddings pulled.
- `ORDER BY name` KG bug already fixed + committed on branch `fix/kg-entity-order-by-name` (WITH NOINDEX); NOT yet in the running 18GB image.
- Schema drift: `idx_entity_fulltext` (the index that broke ORDER BY) exists only on the live DB, in NO migration file. `entity` table is SCHEMALESS.
- Disk: 231 GB free — rebuild headroom OK. Live DB at migration v49.

## Constraints
- Live ops required: Ollama (local), SurrealDB, Docker rebuild. Autonomous executor must drive real containers.
- qwen2.5:14b extraction over 11 PDFs on local Ollama is **slow** — sequential, per-doc timeouts, checkpoint progress.
- `.claude/` is gitignored; settings/scripts changes are local. Code on feature branches per repo convention.

## Top 3 risks → mitigations
1. **Resolution may be weak by design** — if canonical matching is exact `canonical_name+type` only (no fuzzy/embedding dedup), a better model won't yield "high resolution." → Phase 5 reviews the algorithm FIRST; if exact-only, surface to user before claiming success (may need a resolution-logic change, not just a model swap).
2. **12× qwen14b extraction is slow / may OOM Ollama** → sequential ingest, per-doc timeout, checkpoint each doc's entity count to STATE; if Ollama OOMs, fall back to a smaller qwen or reduce concurrency and flag.
3. **18GB image rebuild slow / could fail** → verify disk first; rebuild only the `open_notebook` service; keep hot-patch as emergency verify path if rebuild stalls.

## Dependencies
P1 (diagnose) → P3 (fix trigger). P2 (model wiring) → P3, P5. P4 (rebuild+deploy) needs P3. P5 (resolution test) needs P2+P3+P4. P6 last.

## Tools/skills relied on
Docker CLI, local Ollama HTTP API, SurrealDB via app `.venv` python, ruff/pytest (uv). No web research needed. Context7 not required.

## Open questions (assumed; correct in review)
- Extraction concurrency: assume sequential (1 doc at a time) for stability.
- "High resolution" success metric: assume ≥1 shared cross-doc canonical entity per obvious shared concept (e.g. "Rijk", "Regio Deal", a ministry) AND no exact-duplicate canonical_name+type rows across docs. Refined in Phase 5.
- Notebook for the 11 Convenant docs: assume a fresh dedicated notebook.

## REVISION (post track-methodology reconciliation)
Track B is COMPLETE (2026-06-12); this is a **B.8 follow-up**, not greenfield.
- "No entities" = OPERATIONAL: extraction is a manual trigger (`POST /sources/{id}/run-entities`, schema-review pause) — never run/resumed for the 2 parsed docs. Multi-schema LLM orchestrator IS the live path; spaCy data is Dec-2025 legacy (dead path).
- Real bugs: `extraction_method` never populated (always defaults "llm"); upsert SELECT-then-UPDATE atomicity (B.1e-deferred) — note, only bites with parallel writers (run is sequential).
- UI: ORDER BY fix (done, needs deploy) + default KG filters min_conf=0.9/min_conn=5 (Q1) may hide fresh entities.
- Resolution: V1 `name_normalizer` stub by design; full TOOI/Crossref = Q9/Track M4 (deferred). Scope = ASSESS + RECOMMEND only.
- Execution: Supergoal /goal + adversarial-reviewer gate + status.md ledger per phase. Artifacts under docs/tracks/B-kg-quality/.
Revised to 4 phases (B.8a-d).
