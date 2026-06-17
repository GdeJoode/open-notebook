# Track I — Docling Studio Integration

> **Status**: 📋 PLAN — niet gestart
> **Source-of-truth**: `docs/docling-studio-integration-plan.md` (Dutch, 247 lines, dated 2026-06-16)
> **Pattern**: mirrors `docs/tracks/D-output-richness/plan.md` shape

## 1. Goal

Adopt Docling Studio's (DS) visual identity, PDF inspection UX, and a curated set of capabilities (ANN search, document structure graph, audit, optional external stores) into Open Notebook (ON) **without breaking ON's broader functionality** (audio/video/URL/podcast/KG). DS = Vue + vanilla CSS; ON = React + Tailwind 4 / shadcn. Framework-free assets (design tokens, JSON contracts, pure utility modules) are ported 1:1; everything else is rewritten.

**Core insight**: ON already has the bulk of DS's PDF inspection (`PdfChunkViewer`, bbox overlay, `PipelineConfigPanel`, `/page-preview`, `/page-count`, persistent `chunk.positions`). This track is **style adoption + polish + correctness fixes**, not a greenfield port.

## 2. User-confirmed decisions

| ID | Question | Decision |
|---|---|---|
| Q-I-B-1 | Resizable panel impl | `react-resizable-panels` (mature, ARIA, keyboard) |
| Q-I-G-1 | HNSW embedding-dimension pin | `nomic-embed-text` 768 |
| Q-I-H3-scope | Phase H3 (external stores) in/out scope | **IN scope** — final phase |

## 3. Phase decomposition

Confirmed sequence: **A → H1 → C → B → D → E → G → F → H2 → H3**. Rationale: A = quick visual win (low risk first); H1 = early safety (OOM guard); C = correctness bug (must precede B's workspace); B unlocks D/E; G/F/H2/H3 are larger.

---

### Phase I.A — Design tokens

**Goal**: Adopt DS's visual identity (near-black surfaces, orange `#f97316` accent, Inter UI / IBM Plex Mono numerics, 8px radii, thin scrollbars, active pill in `accent-muted`) by mapping ~230 lines of DS CSS variables onto ON's Tailwind 4 `@theme inline` token system.

**Files to create**: none.

**Files to modify**:
- `frontend/src/app/globals.css` — extend `@theme inline` + OKLCH tokens in `:root` and `.dark`. Token mapping:
  - `--accent #f97316` → `--primary`, `--ring`, `--sidebar-primary`
  - `--bg`, `--bg-surface`, `--bg-elevated` → `--background`, `--card`, `--popover`
  - `--accent-muted` → `--accent` (active state)
  - 8px corners → `--radius: 0.5rem`
- `frontend/src/app/layout.tsx` — load Inter + IBM Plex Mono via `next/font` (Q-I-A-1).

**Acceptance criteria**:
1. Light + dark themes render without console errors.
2. Mode toggle (existing) still flips themes correctly.
3. Mono-numerics utility class (`.mono-num`) is **defined** in `globals.css`. Application to specific consumers (`token-count`, `page-pill`, `bbox-coords`) is **scope of I.E** and explicitly NOT required in I.A. I.A may also defer the IBM Plex Mono font *load* itself to I.E (so the utility class falls back to system monospace until I.E lands), to reserve bundle headroom against AC5.
4. No layout regressions in 3 representative pages: dashboard, source detail, notebook detail.
5. Bundle-size delta < 30KB (fonts subset via `next/font`).

**Tests required**:
- Unit: none.
- E2E: 1 Playwright spec — load app, toggle theme, snapshot critical regions.

**PR boundary**: ONE PR titled `feat(frontend): adopt Docling Studio design tokens (I.A)`.

**Effort estimate**: 0.5–1 day. Reviewer cycle budget: ×1.0 (low complexity).

**Risk mitigations**: Visual-only, fully reversible by token rollback. No data-path changes.

---

### Phase I.H1 — Upload guards + per-IP rate limiting

**Goal**: Prevent backend OOM on large scanned PDFs and add proper per-IP rate limiting. Pre-validates file size and page count before processing; rejects with appropriate HTTP status. Replaces the in-process `RateLimitError` with `slowapi`-backed middleware (existing handler stays; only the limiter changes).

**Files to create**:
- `apps/app-main/tests/test_upload_guards.py` — fixture-driven 413/422 cases.
- `apps/app-main/tests/test_rate_limiter.py` — slowapi burst test.

**Files to modify**:
- `apps/app-main/src/app_main/api/app.py` — register `slowapi` middleware.
- `apps/app-main/src/app_main/api/routers/sources_files.py` — add upload-guard preflight using `pypdfium` (already used by `/page-count`).
- `apps/app-main/src/app_main/config.py` (or env loader) — `MAX_FILE_SIZE_MB` (default 500), `MAX_PAGE_COUNT` (default 500), `RATE_LIMIT_RPM` (default 120 per Q-I-H1-1).
- `pyproject.toml` (`apps/app-main/`) — add `slowapi>=0.1.9`.

**Acceptance criteria**:
1. POST with file > `MAX_FILE_SIZE_MB` → HTTP 413 with `detail` referencing the limit.
2. POST PDF with pages > `MAX_PAGE_COUNT` → HTTP 422; page count read via `pypdfium`.
3. Burst > `RATE_LIMIT_RPM` from same IP → HTTP 429 with `Retry-After` header.
4. `RateLimitError` handler still active (rendered as JSON, not stack trace).
5. Rate limit is **per-IP not per-process** (verified in multi-worker test).

**Tests required**:
- Unit (`test_upload_guards.py`): 4 scenarios (oversize file, oversize pages, valid, missing pdf).
- Unit (`test_rate_limiter.py`): burst from one IP triggers 429; second IP unaffected.

**PR boundary**: ONE PR titled `feat(api): upload guards + per-IP rate limiting (I.H1)`.

**Effort estimate**: 1 day. Reviewer cycle budget: ×1.5.

**Risk mitigations**: Defaults are generous (500MB, 500 pages, 120 RPM); won't surprise existing users. `slowapi` is well-maintained.

---

### Phase I.C — Coordinate canonicalization (correctness bug)

**Goal**: Fix the bbox-coordinate mismatch bug. Docling writes raw PDF points; MinerU writes 0–1 normalized; both land in the same `chunk.positions = [[page,x1,x2,y1,y2]]` field. Frontend currently *guesses* format via `analyzeChunkFormat` — fragile, and the Docling y-flip relies on `prov.page_height` which is often missing. **Fix at the backend**: canonicalize to 0–1 at extraction time. Drop the frontend heuristic. Backfill existing data.

**Files to modify**:
- `pipelines/ingestion/src/ingestion/models/document.py` — `BoundingBox.from_docling`: divide by `doc.pages[page_no].size` (width, height); flip y correctly.
- `apps/app-main/src/app_main/services/chunking/chunk_builder.py` — `from_document` emits 0–1 positions; remove any remaining raw-pt assumption.
- `apps/app-main/src/app_main/services/parsing/mineru_layout_parser.py` — sanity-check (should already be 0–1); add `assert 0.0 <= x <= 1.0` in dev mode.
- `frontend/src/components/source/PdfChunkViewer.tsx` — drop `analyzeChunkFormat` per Q-I-C-2; treat all positions as 0–1.

**Files to create**:
- `apps/app-main/scripts/backfill_chunk_positions.py` — paginated SurrealQL batch script (1000 rows/batch). Reads `chunk.positions` + matching `source.page_dimensions`; writes normalized values in place. Fallback: log and skip rows where dimensions are missing (operator runs full re-ingest on those).
- `apps/app-main/tests/test_bbox_canonicalization.py` — Docling fixture → assert all positions in [0, 1] AND match MinerU's emit for the same fixture page (parity test).

**Acceptance criteria**:
1. `BoundingBox.from_docling` emits 0–1 normalized coords on the fixture set.
2. `analyzeChunkFormat` is removed from `PdfChunkViewer.tsx`; no callers remain (grep clean).
3. Backfill script processes 10K chunks in < 60s on a dev machine; reports counts and skipped rows.
4. Regression test: MinerU's emit path produces identical positions to Docling's, byte-for-byte on a fixture document.
5. Visual smoke: load a pre-fix source after backfill in the UI; overlay aligns with rasterized page.

**Tests required**:
- Unit (`test_bbox_canonicalization.py`): 5 scenarios (Docling 0–1 round-trip; MinerU 0–1 round-trip; cross-parser parity; missing page_dimensions handling; y-flip correctness).
- Integration: ingest fixture document via real Docling pipeline; assert positions are 0–1.

**PR boundary**: ONE PR titled `fix(ingestion): canonicalize bbox to 0–1 in BoundingBox.from_docling (I.C)`.

**Effort estimate**: 1–2 days + backfill operator time. Reviewer cycle budget: ×1.5.

**Risk mitigations**:
- **Risk 1 (data corruption)**: backfill script is idempotent — running twice produces same result. Wrap in transaction per batch.
- **Risk 2 (visual regression for already-converted MinerU sources)**: parity test catches this at PR time.
- **Risk 3 (production rollout)**: backfill before deploying frontend change. The frontend change is the only thing that breaks if both old + new data coexist.

---

### Phase I.B — Inspect workspace (3-pane resizable)

**Goal**: Replace the cramped Chunks-tab viewer with a full-height 3-pane workspace (chunk-list left, PDF + overlay middle, properties right) at a dedicated route. Uses `react-resizable-panels` (Q-I-B-1) for ARIA-compliant drag handles. Panel sizes persist via Zustand persist middleware.

**Files to create**:
- `frontend/src/app/(dashboard)/sources/[id]/inspect/page.tsx` — new route wrapping `<AppShell>` with `<DocumentInspectWorkspace>`.
- `frontend/src/components/source/inspect/DocumentInspectWorkspace.tsx` — top-level `PanelGroup` from `react-resizable-panels`.
- `frontend/src/components/source/inspect/ChunkListPanel.tsx` — left pane (virtualized scroll for >1K chunks).
- `frontend/src/components/source/inspect/PropertiesPanel.tsx` — right pane (active chunk metadata + Pipeline config).
- `frontend/src/lib/stores/document-workspace-store.ts` — Zustand store: active page, active chunk id, panel sizes; with `persist` middleware.
- `frontend/e2e/track-i/inspect-workspace.spec.ts` — drag handles, panel-size persistence, keyboard nav.

**Files to modify**:
- `frontend/package.json` — add `react-resizable-panels@^3` (verify latest stable version).
- `frontend/src/components/source/SourceDetailContent.tsx` — add "Open Inspect" button linking to new route; keep existing Chunks tab as fallback for now.
- `frontend/src/components/source/PdfChunkViewer.tsx` — accept `mode: "embed" | "fullscreen"` prop for layout adaptation; middle pane uses `fullscreen` mode.

**Acceptance criteria**:
1. New route `/sources/{id}/inspect` renders 3 panels separated by resize handles.
2. Drag handle resizes panels; assertion: layout reflows without overflow.
3. Resized panel sizes persist across navigation and reload (via Zustand persist).
4. Keyboard navigation: Tab between panels; Arrow keys to reorder selection.
5. ARIA: each panel has `role="region"` and `aria-label`; resize handles have `aria-orientation="vertical"`.
6. No regression in the existing Chunks tab.

**Tests required**:
- Unit: store reducer (set sizes, clamp to 10/80, persist).
- E2E (`inspect-workspace.spec.ts`): open route, drag handle 20% left, navigate away + back, assert size restored, drag handle via keyboard, assert focus trap stays inside workspace.
- Manual: visual smoke on a 100-page document; verify scroll perf.

**PR boundary**: ONE PR titled `feat(frontend): inspect workspace with resizable 3-pane layout (I.B)`.

**Effort estimate**: 2–3 days. Reviewer cycle budget: ×1.5.

**Risk mitigations**:
- Dep size: `react-resizable-panels` is ~14KB gzipped; documented in PR.
- Layout regression: keep old Chunks tab functional until I.E lands.

---

### Phase I.D — DS-parity sub-features (4 sub-PRs)

**Goal**: Close the inspection feature gap with DS. Split into 4 independently-reviewable PRs.

#### I.D-1 — LayersBar (element-type visibility toggles)

**Files to create**:
- `frontend/src/lib/constants/element-colors.ts` — ported verbatim from DS `frontend/src/features/document/elementColors.ts`. License: MIT.
- `frontend/src/components/source/inspect/LayersBar.tsx` — chip-row component.

**Files to modify**:
- `frontend/src/components/source/PdfChunkViewer.tsx` — accept `hiddenTypes: Set<string>` prop; skip bbox overlay for hidden types.
- `frontend/src/components/source/inspect/DocumentInspectWorkspace.tsx` — render `<LayersBar>` above middle pane.

**AC**: 10 element types each have toggle chip; toggling hides matching overlays; state lives in `document-workspace-store`; keyboard accessible (Space toggles focused chip).

**Effort**: 0.5 day.

#### I.D-2 — Full Docling conversion config

**Files to modify**:
- `frontend/src/components/source/PipelineConfigPanel.tsx` — extend with 5 toggles: `do_code_enrichment`, `do_formula_enrichment`, `do_picture_classification` (separated from existing combo), `generate_page_images`, `images_scale` (slider 1–4).
- Backend: relevant config-pass-through in the docling pipeline runner (`apps/app-main/src/app_main/services/parsing/docling_*`).
- Shared model: `packages/shared/src/shared/models/pipeline_config.py` (or equivalent) — add fields.

**AC**: All 5 fields round-trip via API; backend passes them to `DoclingDocumentConverter` options; UI shows sensible defaults so existing users see no behavior change (Q-I-D2-1).

**Effort**: 1 day.

#### I.D-3 — Chunk merge/split

**Files to create**:
- `apps/app-main/src/app_main/api/routers/chunks.py` (extend or new file): `POST /chunks/{id}/merge` and `POST /chunks/{id}/split` (body: `{cursorOffset: int}`).
- `apps/app-main/src/app_main/services/chunking/chunk_mutator.py` — service-level operations.
- `apps/app-main/tests/test_chunk_mutator.py` — atomicity test (merge two chunks → re-emit positions covering both; split at offset → two chunks with original positions split proportionally).
- `frontend/src/components/source/inspect/ChunkActionsToolbar.tsx` — UI buttons in active-chunk properties.

**Acceptance criteria**:
1. Merge: select 2 adjacent chunks; resulting chunk text = concat with separator; positions = union of bboxes.
2. Split: cursor offset in chunk text → 2 chunks; positions = proportional split of original bbox.
3. **Atomicity** (Q-I-D3-1): both ops wrap in `BEGIN TRANSACTION ... COMMIT` SurrealQL; rollback on any error.
4. Audit trail: each op appends to `chunk_edit` (deferred to I.H2; for now, log to existing observability).

**Effort**: 1 day.

#### I.D-4 — Result tabs (Markdown / Images / Structure)

**Files to create**:
- `frontend/src/components/source/inspect/MarkdownViewer.tsx` — `react-markdown` + `rehype-sanitize` + `remark-gfm`.
- `frontend/src/components/source/inspect/ImageGallery.tsx` — paginated grid; lazy-load.
- `frontend/src/components/source/inspect/StructureViewer.tsx` — placeholder (full tree-view in I.F).

**Files to modify**:
- `DocumentInspectWorkspace.tsx` — add tab strip in right pane: Properties / Markdown / Images / Structure / Config.

**Dependencies**: add `react-markdown`, `rehype-sanitize`, `remark-gfm` to `frontend/package.json` if not already present.

**AC**: Each tab loads without console error; Markdown XSS-safe (sanitized); Images lazy-load (assert via Network panel ≤6 in-flight).

**Effort**: 1 day.

**Combined I.D effort estimate**: 3.5 days across 4 PRs. Reviewer cycle budget: ×1.5 per PR.

---

### Phase I.E — Responsiveness & polish

**Goal**: Make the inspect workspace fully fluid (no media queries; `minmax()` grids; sleepbare panels); finish mono-numerics styling on page pills, token counts, bbox coords.

**Files to modify**:
- `frontend/src/components/source/inspect/*` — apply `min-h-0` / `overflow-auto` per `AppShell` pattern.
- `frontend/src/app/layout.tsx` — load IBM Plex Mono via `next/font/google` (deferred from I.A so the font load lands with its first consumer). Bind `--font-mono-numeric`.
- `frontend/src/app/globals.css` — `.mono-num` utility (already defined in I.A) now resolves to IBM Plex Mono via `--font-mono-numeric`.
- `frontend/src/components/source/PdfChunkViewer.tsx` — apply `.mono-num` to: `token-count`, `page-pill`, `bbox-coords` (deferred from I.A AC3). Page pill + bbox coord readout get `.mono-num`.

**AC**: Workspace adapts gracefully from 1024px wide to ultrawide; no horizontal scrollbars; bbox readout reads in IBM Plex Mono; `.mono-num` is applied to all three target consumers (`token-count`, `page-pill`, `bbox-coords`); visual smoke clean.

**Effort**: 1 day. Reviewer cycle budget: ×1.0.

---

### Phase I.F — Document structure graph (SurrealDB port)

**Goal**: Build a structural graph in SurrealDB alongside Track B's semantic entity graph. Nodes are DoclingDocument elements (sections, paragraphs, tables, figures); edges are parent_of, next_node (reading order), derived_from (chunk → element). Surfaces as a Structure tab in the inspect workspace using ON's existing Sigma.js/graphology stack (NO Cytoscape dep).

**Files to create**:
- `migrations/49.surrealql` + `migrations/49_down.surrealql`:
  ```surql
  DEFINE TABLE doc_node SCHEMAFULL;
  DEFINE FIELD source ON doc_node TYPE record<source>;
  DEFINE FIELD self_ref ON doc_node TYPE string;            -- "#/texts/12"
  DEFINE FIELD element_type ON doc_node TYPE string;        -- section_header|paragraph|table|picture|page|title|list_item|caption|formula|code
  DEFINE FIELD text ON doc_node TYPE option<string>;
  DEFINE FIELD page ON doc_node TYPE option<int>;
  DEFINE FIELD level ON doc_node TYPE option<int>;
  DEFINE FIELD sequence ON doc_node TYPE int;
  DEFINE FIELD bbox ON doc_node TYPE option<array<float>>;  -- [x1,y1,x2,y2] 0–1 (uses I.C output)
  DEFINE INDEX dn_source ON doc_node FIELDS source;
  DEFINE INDEX dn_ref ON doc_node FIELDS source, self_ref UNIQUE;
  DEFINE TABLE parent_of    SCHEMAFULL TYPE RELATION FROM doc_node TO doc_node;
  DEFINE TABLE next_node    SCHEMAFULL TYPE RELATION FROM doc_node TO doc_node;
  DEFINE TABLE derived_from SCHEMAFULL TYPE RELATION FROM chunk   TO doc_node;
  ```
- `apps/app-main/src/app_main/services/graph/doc_graph_builder.py` — reads `metadata.docling_document_json` (per `docs/docling_document_serialization.md`), upserts nodes + RELATEs. Idempotent: delete-then-rebuild per source.
- `apps/app-main/src/app_main/api/routers/structure_graph.py` — `GET /api/sources/{id}/structure-graph?page_limit=100` (default 100 per Q-I-F-2; cap 500).
- `frontend/src/components/source/inspect/StructureGraphView.tsx` — Sigma.js + graphology renderer; click node → highlight bbox in PdfChunkViewer via `self_ref`.
- `apps/app-main/tests/test_doc_graph_builder.py` — fixture: 50-page Docling document → 200+ doc_node rows + parent_of depth ≥3; re-ingest produces identical graph (idempotency).

**Files to modify**:
- `apps/app-main/src/app_main/services/ingestion/orchestrator.py` (or equivalent) — invoke `doc_graph_builder.build(source_id, docling_doc)` after parsing succeeds.
- `frontend/src/components/source/inspect/StructureViewer.tsx` (from I.D-4) — replace placeholder with `<StructureGraphView>`.

**Acceptance criteria**:
1. Migration 49 applies cleanly (forward) and reverts cleanly (down).
2. 50-page Docling fixture → 200+ `doc_node` rows + parent_of tree depth ≥3 + next_node chains.
3. `derived_from` links chunks to doc_nodes via `self_ref` — coverage ≥ 90% of chunks (Q-I-F-1: 1:many is allowed; primary self_ref required).
4. API endpoint returns subgraph for `?page=1` in < 200ms on 50-page fixture.
5. UI: clicking a node highlights bbox in middle pane.
6. Re-ingest produces same graph (delete-rebuild idempotency).
7. Page-limit guard enforced: `?page_limit=500` is the cap; higher requests rejected with 422.

**Tests required**:
- Unit (`test_doc_graph_builder.py`): 6 scenarios (basic tree, idempotency, derived_from coverage, empty document, missing metadata, missing docling_document_json).
- Integration: full ingest → assertion that `parent_of` traversal recovers the original DoclingDocument tree.
- E2E: load inspect workspace → Structure tab → graph renders without error; node click → bbox highlight.

**PR boundary**: ONE PR titled `feat(graph): document structure graph + Structure tab (I.F)`.

**Effort estimate**: 3–5 days. Reviewer cycle budget: ×1.5.

**Risk mitigations**:
- **Datavolume** (Risk 3): 10K-page document → 100K+ nodes. Mitigations: `dn_source` index; page-limit on query API; batched RELATE inserts (500/batch).
- **Sync**: handled by delete-rebuild on re-ingest.
- **No Cytoscape dep** — Sigma.js/graphology already in `package.json`.

---

### Phase I.G — ANN vector search (HNSW, Option 1)

**Goal**: Replace brute-force `vector::similarity::cosine` SELECT-over-all-rows with SurrealDB v2 native HNSW indexes. Pin embedding model to `nomic-embed-text` 768 dims (Q-I-G-1). Validate recall + latency vs brute-force baseline.

**Files to create**:
- `migrations/50.surrealql` + `migrations/50_down.surrealql`:
  ```surql
  -- Audit: reject migration if any row has dim != 768
  -- (handled in pre-flight check by migration runner)
  DEFINE INDEX idx_source_embedding_hnsw ON source_embedding
    FIELDS embedding HNSW DIMENSION 768 DIST COSINE TYPE F32 EFC 150 M 12;
  DEFINE INDEX idx_source_insight_embedding_hnsw ON source_insight
    FIELDS embedding HNSW DIMENSION 768 DIST COSINE TYPE F32 EFC 150 M 12;
  DEFINE INDEX idx_note_embedding_hnsw ON note
    FIELDS embedding HNSW DIMENSION 768 DIST COSINE TYPE F32 EFC 150 M 12;
  ```
- `apps/app-main/scripts/audit_embedding_dimensions.py` — pre-flight check (Q-I-G-3): SELECTs `array::len(embedding)` grouped; reports any non-768. Operator runs this before migration.
- `apps/app-main/scripts/backfill_embeddings.py` — re-embeds non-768 rows via the pinned `nomic-embed-text` model.
- `apps/app-main/tests/test_hnsw_search.py` — recall@5 ≥ 0.95 vs brute-force; latency < 50ms on 10K-row fixture.

**Files to modify**:
- `migrations/1.surrealql` — keep `fn::vector_search` as legacy fallback (don't drop yet); add new `fn::vector_search_knn` using KNN operator.
- `apps/app-main/src/app_main/repositories/search.py` (or equivalent) — switch primary query to KNN; brute-force becomes opt-in via env `USE_BRUTE_FORCE=true` for emergency rollback.
- `packages/shared/src/shared/config/embeddings.py` (or equivalent) — pin model to `nomic-embed-text:768`; reject other models with explicit error at config load.

**Acceptance criteria**:
1. Audit script reports 100% rows at dim 768 OR backfill script processes them; migration applies cleanly afterwards.
2. KNN query latency < 50ms p95 on 10K-row notebook (vs ~500ms brute-force baseline; verified via test).
3. Recall@5 ≥ 0.95 vs brute-force on a corpus of 1000 queries (Q-I-G-2: HNSW default; MTREE as documented fallback).
4. Rollback path: `USE_BRUTE_FORCE=true` env reverts query path within seconds (no migration revert needed).
5. Embedding-write paths reject non-768-dim embeddings with explicit error.

**Tests required**:
- Unit (`test_hnsw_search.py`): 4 scenarios (correctness vs brute-force, latency budget, recall budget, dim-rejection).
- Integration: full ingest → search → assert KNN returns expected ordering on labeled fixture.

**PR boundary**: ONE PR titled `feat(search): native SurrealDB HNSW vector indexes (I.G)`.

**Effort estimate**: 1–2 days + indexing operator time on prod. Reviewer cycle budget: ×1.5.

**Risk mitigations**:
- **Risk 2 (one-way migration)**: maintenance window. Rollback via `USE_BRUTE_FORCE=true` (no SurrealDB revert needed for hot rollback).
- **Mixed-dim audit**: pre-flight script + backfill before migration.
- **Cross-track**: Track D's `embedding`-exclude on export still works (not touched by this phase).

---

### Phase I.H2 — Chunk/version audit + frozen snapshots

**Goal**: Append-only audit log for chunk mutations + on-demand or ingest-completion frozen snapshots of the source state. Mirrors DS's `chunk_edits` / `chunk_pushes` / `document_versions`.

**Files to create**:
- `migrations/51.surrealql` + `migrations/51_down.surrealql`:
  ```surql
  DEFINE TABLE chunk_edit SCHEMAFULL;
  DEFINE FIELD chunk     ON chunk_edit TYPE record<chunk>;
  DEFINE FIELD op        ON chunk_edit TYPE string;        -- insert|update|delete|merge|split
  DEFINE FIELD before    ON chunk_edit TYPE option<object>;
  DEFINE FIELD after     ON chunk_edit TYPE option<object>;
  DEFINE FIELD actor     ON chunk_edit TYPE option<string>;
  DEFINE FIELD ts        ON chunk_edit TYPE datetime DEFAULT time::now();
  DEFINE INDEX ce_chunk  ON chunk_edit FIELDS chunk;
  DEFINE INDEX ce_ts     ON chunk_edit FIELDS ts;

  DEFINE TABLE document_snapshot SCHEMAFULL;
  DEFINE FIELD source    ON document_snapshot TYPE record<source>;
  DEFINE FIELD kind      ON document_snapshot TYPE string;  -- analysis|chunks
  DEFINE FIELD payload   ON document_snapshot TYPE object;
  DEFINE FIELD created   ON document_snapshot TYPE datetime DEFAULT time::now();
  DEFINE INDEX ds_source ON document_snapshot FIELDS source;
  ```
- `apps/app-main/src/app_main/services/audit/chunk_audit.py` — write-path hook for chunk CRUD endpoints.
- `apps/app-main/src/app_main/services/snapshot/document_snapshot.py` — snapshot writer.
- `apps/app-main/src/app_main/api/routers/audit.py` — `GET /api/sources/{id}/chunk-history`, `GET /api/sources/{id}/snapshots`, `POST /api/sources/{id}/snapshots`, `POST /api/sources/{id}/snapshots/{snap_id}/restore`.
- `apps/app-main/scripts/snapshot_cleanup.py` — soft-delete snapshots older than `SNAPSHOT_RETENTION_DAYS` (default 90 per Q-I-H2-1). Cron-runnable.
- `frontend/src/components/source/inspect/HistoryPanel.tsx` — timeline UI in inspect workspace.

**Files to modify**:
- `apps/app-main/src/app_main/api/routers/chunks.py` — wrap mutator calls in audit hook.
- `apps/app-main/src/app_main/services/ingestion/orchestrator.py` — snapshot on ingest-completion.

**Acceptance criteria**:
1. Migration 51 applies + reverts.
2. Every chunk CRUD endpoint writes a `chunk_edit` row with before/after JSON.
3. Snapshot on ingest-completion creates a `document_snapshot` row; manual snapshot via API works.
4. Restore: POST to `/snapshots/{id}/restore` replaces current state from payload; non-destructive (creates a new snapshot of the pre-restore state first).
5. Cleanup script soft-deletes snapshots older than `SNAPSHOT_RETENTION_DAYS`; hard-delete only via admin endpoint.
6. UI HistoryPanel shows timeline; clicking event displays diff.

**Tests required**:
- Unit: each op type produces correct `chunk_edit` row.
- Integration: ingest → mutate → restore → assert state recovered.
- E2E: HistoryPanel renders; restore round-trip in dev env.

**PR boundary**: ONE PR titled `feat(audit): chunk-edit log + frozen document snapshots (I.H2)`.

**Effort estimate**: 2–3 days. Reviewer cycle budget: ×1.5.

**Risk mitigations**:
- **Schrijfpad-dekking**: explicit hook on every endpoint (regression test asserts hooks fired).
- **Opslaggroei**: retention + soft-delete pattern.

---

### Phase I.H3 — External stores push + stale tracking (IN SCOPE, default-off)

**Goal**: Push chunks + embeddings to external OpenSearch (knn) or Neo4j; track which sources are Stale relative to upstream. Default-off via `EXTERNAL_STORES_ENABLED=false` (Q-I-H3-2). Port DS's Fernet `STORE_SECRET_KEY` secret-sealing pattern.

**Files to create**:
- `migrations/52.surrealql` + `migrations/52_down.surrealql`:
  ```surql
  DEFINE TABLE store SCHEMAFULL;
  DEFINE FIELD kind          ON store TYPE string;  -- opensearch|neo4j
  DEFINE FIELD config        ON store TYPE object;
  DEFINE FIELD credentials   ON store TYPE string;  -- Fernet-sealed
  DEFINE FIELD created       ON store TYPE datetime DEFAULT time::now();

  DEFINE TABLE document_store_link SCHEMAFULL;
  DEFINE FIELD source        ON document_store_link TYPE record<source>;
  DEFINE FIELD store         ON document_store_link TYPE record<store>;
  DEFINE FIELD state         ON document_store_link TYPE string;  -- Ingested|Stale|Failed
  DEFINE FIELD last_pushed   ON document_store_link TYPE option<datetime>;
  DEFINE INDEX dsl_source    ON document_store_link FIELDS source;
  DEFINE INDEX dsl_store     ON document_store_link FIELDS store;
  ```
- `apps/app-main/src/app_main/services/external_stores/sealer.py` — Fernet seal/unseal using `STORE_SECRET_KEY` env. Fail-fast at startup if missing AND `EXTERNAL_STORES_ENABLED=true`.
- `apps/app-main/src/app_main/services/external_stores/opensearch_pusher.py` — push chunks + embeddings to OpenSearch knn index.
- `apps/app-main/src/app_main/services/external_stores/neo4j_pusher.py` — push to Neo4j.
- `apps/app-main/src/app_main/handlers.py` (extend) — `JobType.PUSH_TO_EXTERNAL_STORE` handler (Q-I-H3-1: queue, not sync; reuses existing job-queue from B.1).
- `apps/app-main/src/app_main/api/routers/external_stores.py` — CRUD for stores; `POST /sources/{id}/push?store={store_id}`; mark Stale on re-ingest.
- `docs/operations/external-stores-runbook.md` — operator doc: Fernet key rotation procedure (Risk 4).
- `frontend/src/components/settings/ExternalStoresSettings.tsx` — admin UI for stores + per-source push button.

**Files to modify**:
- `apps/app-main/src/app_main/services/ingestion/orchestrator.py` — on re-ingest, mark linked `document_store_link` rows as Stale.
- `apps/app-main/src/app_main/config.py` — `EXTERNAL_STORES_ENABLED` (default `false`), `STORE_SECRET_KEY` (required when enabled).
- `packages/shared/src/shared/models/enums.py` — `JobType.PUSH_TO_EXTERNAL_STORE`.

**Acceptance criteria**:
1. Migration 52 applies + reverts.
2. With `EXTERNAL_STORES_ENABLED=false` (default): all external-store endpoints return 503; UI hides admin section.
3. With enabled + valid `STORE_SECRET_KEY`: CRUD on stores works; credentials sealed at rest, unsealed only at push time.
4. Push to OpenSearch: chunks + 768-dim embeddings indexed; knn search returns expected nearest.
5. Push to Neo4j: nodes + edges created; matches Track B entity graph shape.
6. Re-ingest of a source flips linked rows to `state=Stale`.
7. Operator runbook covers Fernet key rotation without downtime.

**Tests required**:
- Unit: Fernet seal/unseal round-trip; mock OpenSearch + Neo4j clients.
- Integration: docker-compose-up OpenSearch + Neo4j; full push; query.
- E2E: settings UI; manual push button.

**PR boundary**: ONE PR titled `feat(stores): external OpenSearch + Neo4j push with stale tracking (I.H3)`.

**Effort estimate**: 4–6 days. Reviewer cycle budget: ×2.0 (high complexity + new infra).

**Risk mitigations**:
- **Default-off**: zero impact on existing users.
- **Secret rotation**: documented operator runbook.
- **Re-introduces external-infra burden** (per source-plan caveat): explicitly documented in PR description; not the default.

---

## 4. Cross-track conflict analysis

| Affected track | Conflict surface | Mitigation |
|---|---|---|
| Track A (MinerU) | I.C touches `BoundingBox` model used by both parsers | I.C ships parity test on MinerU fixture |
| Track B (KG) | Both add SurrealDB graph tables | I.F's `doc_node`/`parent_of`/`next_node`/`derived_from` use distinct names; no field collision |
| Track D (export) | Track D excludes `embedding` from export; I.G pins to 768-dim | No conflict; exports stay internal-format-agnostic |
| Track H (DEFERRED, vision parser) | Will produce bboxes — must follow I.C's 0–1 convention | Document the convention in `BoundingBox.from_*` docstring during I.C |
| Migrations numbering | Track B used 44–48; Track D used none | I.F=49, I.G=50, I.H2=51, I.H3=52 |

## 5. Open questions & user-confirmed defaults

| ID | Question | Recommended default | Status |
|---|---|---|---|
| Q-I-A-1 | Fonts: `next/font` or `<link>` | `next/font` (subset, zero-CLS) | recommended |
| **Q-I-B-1** | Resizable impl | `react-resizable-panels` | **CONFIRMED** |
| Q-I-C-1 | Coord backfill strategy | In-place SurrealQL if `page_dimensions` present; else re-ingest | recommended |
| Q-I-C-2 | Keep `analyzeChunkFormat` as defensive log? | Remove (clarity > defensive dead weight) | recommended |
| Q-I-D2-1 | PipelineConfigPanel extension shape | Extend in place with sensible defaults | recommended |
| Q-I-D3-1 | Chunk merge/split atomicity | SurrealQL `BEGIN TRANSACTION` | recommended |
| Q-I-F-1 | `derived_from` chunk → doc_nodes cardinality | 1:many with primary self_ref + secondary refs | recommended |
| Q-I-F-2 | `GET /structure-graph` page-limit default | 100 (configurable, capped at 500) | recommended |
| **Q-I-G-1** | Embedding model pin | `nomic-embed-text` 768 | **CONFIRMED** |
| Q-I-G-2 | HNSW vs MTREE | HNSW default; MTREE fallback documented | recommended |
| Q-I-G-3 | Mixed-dim migration handling | Audit pre-flight + backfill before migration | recommended |
| Q-I-H1-1 | `RATE_LIMIT_RPM` default | 120 (DS uses 60; ON gets a little slack) | recommended |
| Q-I-H2-1 | Snapshot retention | 90 days soft-delete + admin hard-delete | recommended |
| Q-I-H3-1 | Push sync vs queue | Queue via existing job-queue (B.1 pattern) | recommended |
| **Q-I-H3-scope** | H3 in Track I scope? | **YES — final phase** | **CONFIRMED** |
| Q-I-H3-2 | `EXTERNAL_STORES_ENABLED` default | `false` (opt-in) | recommended |

User confirms remaining recommended defaults on autopilot unless explicitly contested.

## 6. Risk assessment

| # | Risk | Severity | Mitigation |
|---|---|---|---|
| 1 | I.C backfill on 10K-source corpus could take hours | Medium | Paginated batch script + progress reporting + idempotency |
| 2 | I.G HNSW migration is effectively one-way (drop+rebuild) | Medium | Maintenance window + recall validation pre/post + hot-rollback via env var |
| 3 | I.F `doc_node` table on 10K-page docs creates 100K+ rows | Medium | `dn_source` index + query-limit + batched RELATE inserts |
| 4 | I.H3 Fernet key rotation needs operator runbook | Medium | Runbook delivered with I.H3 |
| 5 | Reviewer rejection rate (B 47%, D 67%) | Medium | Budget ×1.5 attempts; small first PRs |
| 6 | `react-resizable-panels` bundle delta | Low | Measured in I.B PR description |
| 7 | I.G dimension-pin precludes mixing embedding models | Low | Documented limitation; future "multi-model" track if needed |
| 8 | I.H3 re-introduces external-infra burden Phase G's Option 3 rejected | Medium | Default-off; explicit caveat in PR description |

## 7. Ordering & effort

| Phase | Title | Effort (days) | PRs | Cumulative |
|---|---|---|---|---|
| I.A | Design tokens | 0.5–1 | 1 | 1d |
| I.H1 | Upload guards + rate limit | 1 | 1 | 2d |
| I.C | Coord canonicalization | 1–2 | 1 | 4d |
| I.B | Inspect workspace 3-pane | 2–3 | 1 | 7d |
| I.D | DS-parity sub-features (4 PRs) | 3.5 | 4 | 10.5d |
| I.E | Responsiveness & polish | 1 | 1 | 11.5d |
| I.G | ANN HNSW vector search | 1–2 | 1 | 13.5d |
| I.F | Document structure graph | 3–5 | 1 | 18.5d |
| I.H2 | Audit + snapshots | 2–3 | 1 | 21.5d |
| I.H3 | External stores | 4–6 | 1 | 27.5d |
| **Total (critical path)** | | **~25 days** | **12 PRs** | |
| **With ×1.5 reviewer budget** | | **~38 days** | | |

**Calendar estimate** (at 4 productive dev-days/week, sequential autopilot): **~6 weeks**.

**Recommended starting PR**: **I.A** — fastest visible win, low risk, fully reversible. Then **I.H1** (safety first), then **I.C** (correctness bug — must precede I.B's workspace).

## 8. References

- Source plan (Dutch): `docs/docling-studio-integration-plan.md`
- DoclingDocument serialization: `docs/docling_document_serialization.md`
- Track D plan template: `docs/tracks/D-output-richness/plan.md`
- Track B RETRO (review-cycle lessons): `docs/tracks/B-kg-quality/RETRO.md`
- Track D RETRO (filter-pipeline parity lessons): `docs/tracks/D-output-richness/RETRO.md`
- Donor repo: `scub-france/Docling-Studio` (MIT)
