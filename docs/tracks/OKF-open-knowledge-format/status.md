# Track OKF — status

## OKF.5 — UI export option + docs + RETRO · READY FOR REVIEW

**Branch**: `track/okf5-ui-docs` (off `track/okf4-mcp`).

### What landed

**1. OKF export UI** — an "Export OKF" button in the notebook header
(`frontend/src/app/(dashboard)/notebooks/components/NotebookHeader.tsx`) opens
a new zip-only dialog `frontend/src/components/notebooks/exports/OkfExportDialog.tsx`.
It mirrors the Track-D Obsidian dialog exactly (shared `ExportFilter` sliders +
switches + entity-types input + the live `/export-preview` parity widget) minus
the vault-path delivery-mode toggle (OKF is zip-only). New hook
`frontend/src/lib/hooks/use-okf-export.ts` posts to `POST /notebooks/{id}/export/okf`
with `{mode:'zip', filter}`, triggers the blob download, and parses the
`X-OKF-Export-Report` response header.

**2. Lossy-by-design surfaced** — the dialog shows an up-front `role="note"`
banner ("embeddings, chunk-level provenance, and verdict / contradiction edges
are not included") and, after export, itemises the server's omitted-field
ledger (`metadata.omitted_fields`) as a keyboard-operable `<details>`
disclosure. The 202 deferred-job path closes the dialog with a "queued" toast.

**3. Types** — `frontend/src/lib/types/exports.ts` gains `OkfExportRequest`,
`OkfExportReport`, `OkfExportMetadata`, `OkfOmittedFields`, `OkfExportDeferred`.

**4. Docs + close-out** — `ARCHITECTURE.md` OKF interchange section (near the
Track D export docs); `docs/FEATURE_ROADMAP.md` Track OKF ✅ SHIPPED entry with
the OKF.1–OKF.5 phase list; `RETRO.md`; `docs/tracks/_status.md` OKF row.

### Tests

`cd frontend`
- `npx vitest run src/lib/hooks/__tests__/use-okf-export.test.ts src/components/notebooks/exports/__tests__/OkfExportDialog.test.tsx` — **7 passed** (3 hook `parseOkfExportReport`, 4 dialog: default state + a11y + submit-forwards-filter + ledger surfaced).
- `npx tsc --noEmit` — clean.
- `npx eslint <changed files>` — clean.

### Not in scope / deferred (per plan)

- **Import UI** — deferred. Import is available over REST
  (`POST /notebooks/{id}/import/okf`) + MCP; no frontend surface yet (a minimal
  "import OKF bundle" entry is a small follow-up). Noted in RETRO.
- Durable artifact store + download-polling UI for the deferred (202) large-
  notebook path — the seam exists (`JobType.EXPORT_OKF`), the UI relies on the
  "queued" toast. Noted in RETRO follow-ups.
- No `# TODO`s left in the code.

## OKF.4 — Agent surface: MCP tools + REST import · READY FOR REVIEW

**Branch**: `track/okf4-mcp` (off `track/okf3-import`).

### What landed

**1. REST import endpoint (the OKF.3 deferral)** —
`apps/app-main/src/app_main/api/routers/exports.py`:
- `POST /notebooks/{id}/import/okf` — accepts a multipart-uploaded OKF v0.1
  bundle **zip** (`file`), a `?apply_dedup=` query flag, runs
  `OkfImportService.import_bundle(bytes, notebook_id=…, apply_dedup=…)`, and
  returns the `OkfImportReport` (imported/matched/skipped counts + skip/dangling
  ledger) as JSON. Mirrors `/export/okf`'s notebook-scoped auth (404 on missing
  notebook). A malformed bundle **container** (non-zip / unreadable archive) is
  a `422`; a malformed **concept** inside a valid bundle is skip-recorded in the
  `200` report (never a half-written graph). Inline only (no job-queue on the
  import path).

**2. MCP tools (the agent surface — Track W pattern)** —
`packages/surrealdb-service/src/surrealdb_service/mcp/server.py` (the graph MCP
server that already hosts `search` / `get_node` / `related` / `cite` /
`add_note` / `auto_link_note`):
- **`export_okf`** — args: `notebook_id` + the `ExportFilter` knobs
  (`min_connections` / `min_confidence` / `min_relation_confidence` /
  `entity_types` / `include_orphans` / `include_archived`). Returns JSON
  `{status:"exported", notebook_id, bundle:{path:text}, report:{…}}` — the
  bundle is the exporter's legible `{path: text}` mapping (unzipped from the
  service's deterministic archive), and `report` carries the OKF-D3
  omitted-field ledger.
- **`import_okf`** — args: `bundle` (the `{path: text}` mapping `export_okf`
  returns) **or** `zip_base64` (a base64 bundle zip), `notebook_id`,
  `apply_dedup`. Returns `{status:"imported", …OkfImportReport}`.
- Both **delegate to the existing app-main OKF services** (no reimplementation)
  via a **lazy import inside the tool body**
  (`app_main.dependencies.get_okf_export_service` / `get_okf_import_service`).
  This is the one **deliberate, documented layering exception** to this
  package's repo-direct / no-app-main rule (see the expanded module docstring):
  the OKF services are app-main orchestrators over these same repos + the
  K.5/K.3 dedup, module import stays app-main-free (the graph tools still load
  with no app-main present), and the OKF tools degrade to a typed
  `{status:"import_error"}` when app-main is absent. Precedent: this package's
  tests already reach app-main over the shared workspace venv
  (`test_entity_merge_roundtrip`, `test_relation_endpoint_resolution_roundtrip`).
- Typed, best-effort error shapes consistent with the other tools:
  `invalid_filter` / `bad_request` (neither/both inputs, bad base64) /
  `malformed_bundle` / `import_error` — never a raw crash out of the tool.

### Tests
- `apps/app-main/tests/test_okf_import_router.py` — 4 stubbed-service contract
  tests (happy path, dedup flag forwarded, 404, 422) + 1 `@requires_docker`
  exporter-bundle → HTTP (ASGITransport) → real-DB round-trip. **5 passed.**
- `packages/surrealdb-service/tests/test_mcp_okf_tools_roundtrip.py` — 6
  `@requires_docker` tests mirroring the W.3 `mcp_global_db` +
  independent-connection precedent: `export_okf` bundle/report shape, invalid
  filter, export→import round-trip (matched, not duplicated, asserted over an
  independent connection), base64-zip input, and the `bad_request` /
  `malformed_bundle` error shapes. **6 passed.**
- Regression: `test_okf_exports_router` + `test_exports_router` +
  `test_export_preview` + `test_okf_import_router` (offline) — **31 passed**;
  `test_mcp_graph_tools_roundtrip` (docker) — **11 passed**. No regressions.
- ruff: clean. mypy: no new errors (only the repo-wide baseline `import-untyped`
  notes for `shared.models.export`, unchanged from the OKF.1–OKF.3 modules and
  the pre-existing `exports.py` import).

### Not in scope (per plan)
- UI export-dialog option + ARCHITECTURE/roadmap/RETRO close-out → **OKF.5**.
- No `# TODO`s / deferrals left in the code.

## OKF.3 — OKF import as a source type (Backend) · READY FOR REVIEW

**Branch**: `track/okf3-import` (off `track/okf1-export`).

### What landed
- `apps/app-main/src/app_main/services/okf_import_service.py` — reads an OKF
  v0.1 Knowledge Bundle (a directory, a `.zip` path/bytes, or the
  `{path: text}` mapping OKF.1 emits) and reconstructs entities / relations /
  notes / sources into the KG. Two clean layers:
  - **Pure** `parse_okf_bundle` — splits `---` frontmatter (reusing
    `vault_sync_service._parse_frontmatter`, not a reinvented parser) from the
    body; routes each concept by `type` (`Note`→note, `Source`→source,
    `Index`/`Log` + reserved filenames→skip, else→entity); recovers entity→entity
    relations from body cross-links (absolute `/…` and relative `./…`, spec §5).
  - **Persistence** `import_bundle` — entities via
    `EntityRepository.upsert_entity` (deterministic `(name,type)` dedup),
    relations via the persistence service's injection-safe `type::thing`-bound
    RELATE primitive, notes/sources via their repositories.
- `get_okf_import_service()` factory in `dependencies.py`, wiring the K.5
  candidate-dedup proposer + K.3 recanonicalization apply over one shared
  `EntityRepository`.

### Round-trip result (OKF.1 ↔ OKF.3)
Reconstructs faithfully: **entity** canonical_name / type / description /
type_tags; **relations** (count-preserving); **notes** (title + body, leading
`# title` heading stripped); **sources** (title + resource + topics). Verified
offline (`parse`) and over a testcontainer (`import`).

**Lossy-by-design** (documented in-module, consistent with the OKF-D3 ledger):
- **Relation direction** — the exporter renders each edge on *both* endpoints as
  an undirected link, so import collapses the two links to one edge but cannot
  recover `in` vs `out`; edges are written sorted-endpoint deterministic.
- **Internal `resource` urn** — an entity without a real external id exports a
  synthesised `urn:open-notebook:…`; on import it is kept as an `okf_resource`
  property but NOT re-promoted to `external_ids` (only a genuine DOI/TOOI/http
  URI is), so a re-export regenerates a fresh urn.
- Everything in `OKF_OMITTED_FIELDS` (embeddings, chunk/verdict/similarity
  substrate) was never in the bundle, so it cannot be imported.

### Dedup + provenance
- Exact `(canonical_name, entity_type)` collision → matched via upsert (K.1),
  not duplicated. Fuzzy/near-duplicate against the persisted graph → K.5
  `propose_candidates` auto-merge band applied via K.3 `apply_merge`
  (relation repoint + provenance fold + soft tombstone); `review` band is left
  for a human, never auto-applied.
- Imported nodes tagged distinguishable: entities carry
  `extraction_method="okf-import"` + an `okf_import` property + an anchor-source
  in `source_documents`; sources carry an `okf_import` metadata flag.
- **Idempotent**: entities/relations dedup on natural keys; notes on
  `(title, content)` (the `note` table is SCHEMAFULL with no metadata bag).
- RELATE id-injection guard respected (endpoints go through the persistence
  service's `type::thing($id)` param-bound RELATE — no string interpolation).

### Malformed handling
A concept with broken/`type`-less frontmatter is skipped **non-silently**
(recorded in `OkfImportReport.skipped`) and never partially persisted; a
dangling relation link is recorded in `dangling_links`, never a crash (spec §5.3
tolerance). A malformed *container* (missing dir, non-zip file) raises.

### Tests — `apps/app-main/tests/test_okf_import_service.py`
`uv run --no-sync pytest apps/app-main/tests/test_okf_import_service.py`
- 14 offline (pure parse/round-trip + mocked wiring): **passed**.
- 3 `@requires_docker` (export→import round-trip, idempotent re-import, K.5/K.3
  fuzzy collapse): **passed**.
- Full OKF suite (import + export + router, offline) — 32 passed, no regressions.
- ruff: clean. mypy: clean (only the repo-wide baseline `import-untyped`
  notes for `shared.*` / `surrealdb_service.*` remain, as on the OKF.1 module).

### Deferred (per plan)
- REST import endpoint + source-type registration → OKF.4/OKF.5.
- MCP `import_okf` tool → OKF.4.
- No `# TODO`s left in the code.
