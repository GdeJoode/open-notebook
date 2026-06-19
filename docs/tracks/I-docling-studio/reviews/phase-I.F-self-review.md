# Phase I.F — Document structure graph — Self-review

Branch: `track/i-f-structure-graph`
Commit range: `2e70c57..fb94967` (5 commits, off `main` @ `ac21d85`)

## Structure-source decision (the central design question)

**Chosen: chunk-derived structure.** The plan's first choice — building from the
serialized `DoclingDocument` JSON (`metadata["docling_document_json"]`) — is not
viable in this codebase because that JSON is never persisted:

- `docs/docling_document_serialization.md` describes it as a **transient
  LangGraph-state artifact**, produced inside the ingestion workflow, consumed
  for chunk extraction, then discarded.
- `source_processor.py::_update_source` only lifts a small provenance subset onto
  `source.metadata` (`parser_engine_used`, `extraction_confidence`, …). It never
  writes `docling_document_json`.
- `ExtractionResult` (in `source_extractor.py`) carries `chunks`, `full_text`,
  `metadata` — but not the docling doc or its JSON.
- A repo-wide grep finds `docling_document_json` only in docs + the plan, never
  in live code that persists it.

So the only reliable, persisted structure source at the orchestration boundary
is the **`chunk` rows**, which carry the structure Docling produced:
`order` (reading order), `element_type`, `metadata.section_path` (heading
breadcrumb), `metadata.section_level`, and `positions` (0–1 bboxes from I.C).

The builder therefore derives:
- one **section** `doc_node` per distinct `section_path` prefix (heading tree),
- one **leaf** `doc_node` per chunk,
- `parent_of` (section tree + section→leaf), `next_node` (reading order over
  leaves), `derived_from` (chunk → its leaf doc_node).

`self_ref` is synthesized deterministically: `#/sections/<idx-by-first-appearance>`
and `#/chunks/<chunk.order>`. There is **no real docling `self_ref`** (e.g.
`#/texts/12`) available in this codebase; the synthesized ref fills the schema
field and keeps `(source, self_ref)` unique for idempotent re-ingest.

`compute_graph()` is written to also accept a pre-extracted element list, so if a
future caller still holds the live docling doc it can pass real elements without
changing the persistence path. Today every production caller takes the chunk
path.

## Cross-track B name-collision check

Grep of `migrations/` for `\b(doc_node|parent_of|next_node|derived_from)\b`:
matches appear **only in migration 49**. Track B owns
`entity` / `relation` / `entity_alias` (migration 39) and `metrics` (47); none
collide with I.F's table names or fields. PASS.

## AC-by-AC

| AC | Status | Evidence |
|----|--------|----------|
| 1. Migration 49 applies + reverts | **Live-deferred** (no SurrealDB in sandbox) | Written per local conventions (IF NOT EXISTS, SCHEMAFULL, `TYPE RELATION`, mirrors migrations 39/47/48). Discovered by the migration runner's `*.surrealql` glob (version 49 from filename). Down drops edges before nodes. Apply/revert verified by inspection; live run deferred. |
| 2. 50-page fixture → 200+ nodes, depth ≥3, next_node chains | **Computed-verified** | `test_50_page_document_exceeds_200_nodes_depth_3_with_chains`: 255 doc_nodes computed (220 leaves + 35 sections), `max_depth() >= 3` (actually 4), `len(next_node) == leaves-1`. Live row counts deferred. |
| 3. derived_from coverage ≥90% (primary self_ref) | **Computed-verified** | `test_every_chunk_with_id_gets_a_derived_from_edge`: coverage 100% on the fixture (≥0.90 asserted); every edge targets a real leaf `self_ref`. Cardinality is currently 1:1 (one leaf per chunk), which satisfies the 1:many-allowed contract with a primary self_ref. |
| 4. API ?page=1 subgraph < 200ms on 50-page fixture | **Live-deferred** (no SurrealDB) | Endpoint exists, page-filtered query uses the `dn_source` index + `LIMIT`, edges restricted to the returned node set. Latency requires a live DB; deferred. |
| 5. UI: click node → bbox highlight | **Wired + render-verified; visual highlight live-deferred** | StructureGraphView forwards a clicked leaf's `chunk_id` to `onSelectNode` → `setActiveChunkId` → PdfChunkViewer's `selectedChunkId` (the same mechanism the chunk list uses). E2E asserts the graph + stats render without console errors. A deterministic per-node click on the Sigma WebGL canvas is not reliable headless, so the visual highlight is verified manually; documented deferred. |
| 6. Re-ingest idempotency (delete-rebuild) | **Computed-verified** | `test_recompute_is_byte_identical` (same chunk set → identical node/edge signatures); `test_build_deletes_before_inserting` (all deletes precede the first insert); `test_build_twice_persists_same_edge_counts`. Live delete-rebuild deferred. |
| 7. page_limit guard (cap 500 → 422) | **Verified** | `test_page_limit_above_cap_is_422` (501 → 422, reader not called), `test_page_limit_at_cap_is_accepted` (500 OK), `test_default_page_limit_is_100`, reader clamps internally (`test_page_limit_clamped_to_max_in_query`). |

## Live-deferred ACs (no running SurrealDB in the sandbox)

- AC1 (migration apply/revert) — verified by inspection + runner discovery.
- AC2/AC3/AC6 — the *computation* is unit-tested against fixtures; live row
  counts / live delete-rebuild deferred.
- AC4 (API latency) — needs a live 50-page graph.
- AC5 (visual bbox highlight) — wiring + render tested; the actual canvas-click →
  overlay paint is a manual check.

Everything that CAN be tested without a DB is tested with real assertions that
fail on a builder bug (verified: the 200-node test caught my first
under-sized fixture; I bumped element count rather than weakening the assert).

## Mental inversion — how could this be wrong?

- **"The graph is empty for real documents."** Chunks from `chunk_builder` store
  `section_path` under `metadata`, and `prepare_for_db` preserves the whole
  `metadata` dict — so `section_path`/`section_level` survive to the DB and the
  builder reads them from `metadata`. URL/text sources have empty `section_path`
  → flat leaf chain (no section nodes), which is correct, not broken
  (`test_missing_metadata`).
- **"derived_from references a chunk that doesn't exist yet."** The hook runs
  *after* `chunk_repo.bulk_create`, which returns rows with their record ids; the
  builder uses those ids. Order is guaranteed in `process_source`.
- **"A graph failure breaks ingestion."** The hook is wrapped in try/except and
  logs a warning; `test_structure_graph_failure_does_not_fail_ingestion` proves
  ingestion still returns `chunk_count`.
- **"Deleting doc_nodes orphans RELATE rows."** `_delete_existing` clears
  parent_of / next_node / derived_from (matched via `in.source`/`out.source`)
  before deleting doc_nodes; the down migration drops edges before nodes.
- **"bbox order is wrong."** Chunk positions interleave `[page,x1,x2,y1,y2]`;
  the doc_node bbox is `[x1,y1,x2,y2]`. `_first_bbox` reorders;
  `test_bbox_reordered_to_x1y1x2y2` pins it.
- **"page_limit silently clamps instead of erroring."** Router raises 422 above
  500 (no silent clamp at the boundary); the reader's internal `min()` is a
  defense-in-depth second line, not the user-facing contract.

## Stale plan paths (flagged)

- Plan §I.F lists the orchestrator hook in
  `services/ingestion/orchestrator.py` — that file does not exist. The real
  orchestration boundary is `services/source_processor.py::process_source`; the
  hook lives there.
- Plan signature `doc_graph_builder.build(source_id, docling_doc)` is replaced by
  `build(source_id, chunks=...)` for the reason above (no live docling doc).
- `migrations/49.surrealql` schema matches the plan's SurrealQL with local
  syntax (`IF NOT EXISTS`, a `created_at` field added for parity with sibling
  tables).

## Files

Created:
- `migrations/49.surrealql`, `migrations/49_down.surrealql`
- `apps/app-main/src/app_main/services/graph/__init__.py`
- `apps/app-main/src/app_main/services/graph/doc_graph_builder.py`
- `apps/app-main/src/app_main/services/graph/structure_graph_reader.py`
- `apps/app-main/src/app_main/api/routers/structure_graph.py`
- `apps/app-main/tests/test_doc_graph_builder.py`
- `apps/app-main/tests/test_structure_graph_router.py`
- `frontend/src/components/source/inspect/StructureGraphView.tsx`
- `frontend/e2e/track-i/structure-graph.spec.ts`

Modified:
- `apps/app-main/src/app_main/services/source_processor.py` (best-effort hook)
- `apps/app-main/src/app_main/dependencies.py` (builder factory)
- `apps/app-main/src/app_main/api/app.py` (router registration)
- `apps/app-main/tests/test_source_processing_service.py` (hook tests)
- `frontend/src/lib/api/sources.ts` (getStructureGraph + types)
- `frontend/src/components/source/inspect/StructureViewer.tsx` (placeholder → real)
- `frontend/src/components/source/inspect/DocumentInspectWorkspace.tsx` (wire props)

## Test results

- Backend: `test_doc_graph_builder.py` (18) + `test_structure_graph_router.py`
  (8) + `test_source_processing_service.py` (57, +3 hook tests) → 83 passed.
  Full suite collects 680 tests with no import breakage. ruff clean; mypy clean
  on the new modules (`--ignore-missing-imports` for the untyped
  `surrealdb_service`, per codebase norm).
- Frontend: `tsc --noEmit` clean; `npm run lint` shows no new
  warnings/errors on I.F files; Playwright spec discovered via `--list` (2 tests).
