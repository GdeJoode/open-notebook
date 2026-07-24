# Track OKF — Open Knowledge Format export/import (PROPOSAL)

> **Status**: 📝 PROPOSED (2026-07-23) — **awaiting human approval**. This is a
> track-planner-style sprint plan, not yet approved for implementation. See the
> analysis it derives from in the session notes; grounded against the existing
> Track D exporter family.
>
> **Track ID**: `OKF` (single letters A–Z are exhausted/reserved; multi-letter
> mnemonic mirrors NS / PL / UX).

## Vision

Make curated open-notebook knowledge **portable to arbitrary AI agents** via
Google Cloud's **Open Knowledge Format** (OKF v0.1) — a vendor-neutral spec of
Markdown files + YAML frontmatter + inter-entry links from which a graph
projects. This realises the roadmap's "open-notebook is the *substrate*; Track G
makes it agent-callable" direction (constraint §0.8, agent-first API surface)
without coupling external consumers to our SurrealDB schema or MCP tools.

**One-line framing**: OKF is a standardised, bidirectional interchange adapter at
the *edges* of the system. It complements — never replaces — the SurrealDB graph,
hybrid search, and provenance that remain the substrate.

## Why this is mostly a conformance layer (honest scoping)

The repo **already emits OKF-shaped output** via Track D (✅ COMPLETE):

- `apps/app-main/src/app_main/services/obsidian_export_service.py` — one `.md`
  per entity, **YAML frontmatter + `[[wikilink]] (relation_type)` graph**,
  `min_connections`/`min_confidence` filters, `ExportReport` accounting of
  dropped/kept links.
- `apps/app-main/src/app_main/services/vault_sync_service.py` — **bidirectional**:
  parses `.md` frontmatter, registers aliases, queues new entities for export.
- `api/routers/exports.py` + `api/routers/vault.py` — REST surface already exists.

So the net delta of Track OKF is **spec-conformance** (OKF's required frontmatter
keys, its linking syntax, its bundle/tree + manifest conventions) plus the
agent-facing surface — **not** a new projection engine. This keeps the track
small and reuse-heavy.

## Scope & cut-line

**In scope (v1)**:
- OKF **export** of a notebook's entity/relation/note projection as a spec-valid
  bundle (reusing Track D's projection + filters).
- OKF **import** of an external bundle as a source type (reusing `vault_sync`).
- Agent surface: MCP `export_okf` / `import_okf` tools + REST endpoints.
- UI: an OKF option in the existing export dialog.

**Out of scope (documented as lossy-by-design)**:
- Embeddings, chunk-level provenance (Track X), verdict/contradiction edges
  (Track Z), hybrid-search signal (Track W) — these have **no OKF representation**
  and are intentionally dropped from the bundle. The `ExportReport` must list what
  was omitted so the loss is explicit, never silent.
- No re-architecture around OKF; it is an adapter, not a storage/retrieval change.

## Decision gates (resolve before / during OKF.1)

- **OKF-D1**: pin to OKF **SPEC v0.1** commit hash (the spec is explicitly "a
  starting point, not a finished standard" — vendor it or pin the ref so a spec
  bump can't silently break exports). ⚠ Risk driver.
- **OKF-D2**: bundle target = new sibling service `okf_export_service.py` (reuse
  obsidian projection) **vs** a shared projection extracted from
  `obsidian_export_service`. Rec: **extract the shared projection** into a pure
  helper both exporters call, to avoid a second drift surface.
- **OKF-D3**: import dedup authority = reuse **K.5 candidate-dedup** (fuzzy/
  embedding) + K.3 canonicalization, provenance-tagged `source: okf-import`.
- **OKF-D4**: lossy-field policy — drop-and-report (rec) vs stash as non-spec
  `x-` frontmatter extensions (risks non-portability). Rec: drop-and-report.

## Phases

Convention per `docs/tracks/README.md`: Backend → Export/API → Integration. Each
phase = one PR, its own branch `track/okf-<phase>`, green tests + AC before the
next.

### OKF.1 — Bundle mapper (Backend, pure) · reuse-heavy

**Deliverables**
- `apps/app-main/src/app_main/services/okf_export_service.py` — maps the
  (shared, OKF-D2) entity/relation/note projection → an in-memory OKF bundle:
  spec-required YAML frontmatter, OKF linking syntax between entries, bundle
  tree + manifest. Emits an `ExportReport` (kept/filtered + **omitted-field
  ledger** per out-of-scope §).
- If OKF-D2 = extract: `..._export_projection.py` pure helper, and
  `obsidian_export_service` refactored to call it (behaviour-preserving).

**Reuse vs new**: REUSE projection + `min_conn`/`min_conf` filters + ExportReport
from `obsidian_export_service`; NEW = frontmatter-key mapping + OKF link syntax +
bundle/manifest per SPEC.

**Acceptance criteria**
1. Bundle validates against the pinned OKF SPEC v0.1 (frontmatter required keys,
   linking syntax, tree/manifest present).
2. Entity→entry, relation→link, note→entry mapping is 1:1 with the notebook
   projection under identical filters as the Obsidian export.
3. Filtered-out link targets are accounted in `ExportReport` (no dangling links
   in the bundle — mirrors Q-D-4 wikilink accounting).
4. Omitted-field ledger lists every substrate field with no OKF representation
   (embeddings, chunk provenance, verdict edges, similarity scores).
5. Pure/deterministic: same input → byte-stable bundle (sorted keys, no clock in
   content); timestamps injected by the caller.
6. If OKF-D2 = extract: existing `test_obsidian_export*` stays green (behaviour
   preserved).

**Tests**: unit over a fixture notebook projection; a spec-conformance assert
(schema/linting of frontmatter + links); the omitted-field ledger assertion.
**PR boundary**: no API/UI, no DB writes — pure mapper only.

### OKF.2 — Export endpoint + packaging (Backend/API) · reuse-medium

**Deliverables**
- Extend `api/routers/exports.py`: `POST /notebooks/{id}/export/okf` → zipped
  bundle (mirror the obsidian-zip path).
- Large notebooks routed through the **job-queue** (constraint §5), streaming/
  artifact download like the existing zip export.

**Acceptance criteria**
1. Endpoint returns a spec-valid OKF bundle for a real notebook.
2. Large notebook (> threshold) runs as a job, not inline; status pollable via
   the existing pipeline/job surface.
3. `ExportReport` (incl. omitted-field ledger) returned in the response/metadata.
4. Auth via the existing API-key header (G-Q1); no new auth surface.

**Tests**: API test with a seeded notebook (testcontainers per repo pattern);
job-enqueue asserted at the seam (per the job-queue-singleton memory).
**PR boundary**: export direction only; no import, no UI.

### OKF.3 — OKF import as a source type (Backend) · reuse-heavy

**Deliverables**
- `okf_import_service.py` — parse an external OKF bundle (frontmatter + links) →
  entities/relations/notes; dedup via **K.5** candidate-dedup + **K.3**
  canonicalization; provenance `source: okf-import`.
- Reuse `vault_sync_service` frontmatter parsing + entity/alias queue.

**Acceptance criteria**
1. Importing a bundle produced by OKF.1 round-trips: entities/relations/notes
   created, matching the source projection (round-trip fidelity within the
   lossy-by-design bounds).
2. Dedup respected — an entity already present is matched, not duplicated
   (K.5 thresholds); re-import is **idempotent**.
3. Malformed/partial bundle → non-silent error + skip (no half-written graph);
   RELATE id-injection guards respected (per the SurrealDB RELATE memory).
4. Imported nodes are provenance-tagged and distinguishable from native ones.

**Tests**: round-trip (OKF.1 export → OKF.3 import → compare); dedup against a
seeded overlapping entity; idempotent re-import; malformed-bundle safety.
**PR boundary**: import direction; no UI.

### OKF.4 — Agent surface: MCP tools (Integration, Track W/G) · reuse-medium

**Deliverables**
- MCP `export_okf` / `import_okf` tools following the Track W pattern
  (`packages/surrealdb-service/src/surrealdb_service/mcp/server.py` or a
  dedicated graph-mcp member).

**Acceptance criteria**
1. An agent can export a notebook as an OKF bundle and import one over MCP.
2. Tool contracts match the Track W tool conventions (schema, error shape).
3. Exercised over a genuinely independent MCP connection (per the W.3 test
   precedent).

**Tests**: MCP tool integration test mirroring `w3-mcp-graph-tools`.
**PR boundary**: MCP only; assumes OKF.1–OKF.3 merged.

### OKF.5 — UI export option + docs + RETRO (Integration) · reuse-heavy

**Deliverables**
- Add "Open Knowledge Format" to the export dialog (reuse the Track D
  obsidian-dialog component + `exports` hooks).
- `ARCHITECTURE.md` OKF section; roadmap Track OKF entry; `RETRO.md`; close-out.

**Acceptance criteria**
1. UI export produces a downloadable OKF bundle; a11y + responsive per the UI
   quality gate.
2. Docs updated (ARCHITECTURE + roadmap); RETRO written; `_status.md` reflects
   Track OKF.
3. Full-suite regression green; track marked CLOSED.

**Tests**: e2e export-dialog flow (reuse Track D e2e harness).
**PR boundary**: UI + docs; assumes backend + MCP merged.

## Risks & mitigations

1. **OKF is v0.1 and will churn** → pin the spec ref (OKF-D1); isolate the
   mapping in one module so a spec bump is a localised change.
2. **Lossy vs the substrate** → explicit omitted-field ledger in `ExportReport`
   (AC OKF.1.4); documented as by-design, surfaced in UI/API, never silent.
3. **Two export code paths drift** (Obsidian vs OKF) → OKF-D2 extract-shared-
   projection so both call one pure helper.
4. **Adoption uncertainty** (Vertex-anchored ecosystem) → keep it a small,
   reuse-heavy track; do not re-architect. Value is realised even as one more
   interchange target alongside Obsidian/TTL/JSONL.

## Effort estimate

Small–medium. OKF.1/OKF.3/OKF.5 are reuse-heavy over Track D + K.3/K.5 + vault
sync; OKF.2/OKF.4 are thin wiring over existing exports-router + MCP patterns.
The genuinely new work is the SPEC-conformant frontmatter/link mapping (OKF.1).

## Open questions for the operator

- [ ] Approve the track at all, or shelve until OKF > v0.1 stabilises?
- [ ] OKF-D2: extract shared projection (rec) vs standalone service?
- [ ] Is **import** (OKF.3) wanted in v1, or export-only first?
- [ ] MCP surface (OKF.4) in v1, or REST/UI only initially?
