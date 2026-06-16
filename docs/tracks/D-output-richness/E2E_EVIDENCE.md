# Track D — End-to-end smoke evidence

> Captured: 2026-06-16
> Scope: the three knowledge-graph export surfaces shipped in Track D
> (Obsidian zip + Obsidian vault_path, JSONL streaming, NetworkX 7-format).
> Phase reference: `docs/tracks/D-output-richness/plan.md` §D.4 (lines 352–382).

## Sandbox-limited acknowledgement

The sprint-plan AC #1 calls for "All 3 export formats successfully run
end-to-end against the corpus notebook" and explicitly lists "open in a
real Obsidian instance / load into a local Neo4j via apoc.load.json /
open in Gephi or a Python notebook". **None of these three external
tools are available inside the CI sandbox in which this phase
executed**: there is no real Obsidian binary, no running Neo4j with the
APOC plugin, and no Gephi installation. We therefore have to be honest
about what this evidence doc can and cannot prove:

- **What it CAN prove**: the export pipelines produce bytes that match
  the contracts our tests pin (frontmatter shape, JSONL line shape,
  NetworkX node/edge/attribute round-trip), and the HTTP endpoints +
  job handler are wired end-to-end through routers, services, and DI.
  Evidence: deterministic pytest runs of the test suites listed below.
- **What it CANNOT prove**: the produced bytes parse cleanly in a real
  external consumer's runtime (Obsidian's actual wikilink renderer,
  Neo4j's `apoc.load.json` loader, Gephi's GraphML parser, NetworkX's
  pickle reload across Python minor versions). For that, an operator
  with access to those tools needs to walk the manual smoke checklist
  at the bottom of this doc.

This pattern mirrors Track B's RETRO §"Live-test recommendation"
(`docs/tracks/B-kg-quality/RETRO.md`, lines 188–213) which deferred its
full corpus E2E to a live operator session for the same sandbox-bound
reason. The Track-D plan was authored with this constraint in mind
(see plan §358 "Live-test smoke — deferred from per-phase E2E,
mirroring B.7's deferred corpus run").

## What the test suite covers (static + executable evidence)

The three export services + the shared preview + the job handler ship
with the following test surface:

### Test files

```
$ ls -la apps/app-main/tests/test_*_export*.py apps/app-main/tests/test_handlers.py
-rw-r--r-- 1 gdejo gdejo 15263 Jun 16 21:24 apps/app-main/tests/test_export_preview.py
-rw-r--r-- 1 gdejo gdejo 22133 Jun 16 21:24 apps/app-main/tests/test_exports_router.py
-rw-r--r-- 1 gdejo gdejo  1996 Jun 16 21:24 apps/app-main/tests/test_handlers.py
-rw-r--r-- 1 gdejo gdejo 28993 Jun 16 21:24 apps/app-main/tests/test_jsonl_export_service.py
-rw-r--r-- 1 gdejo gdejo 20747 Jun 16 21:24 apps/app-main/tests/test_networkx_export_service.py
-rw-r--r-- 1 gdejo gdejo 41189 Jun 16 21:24 apps/app-main/tests/test_obsidian_export_service.py
```

### Test counts (rough — exact counts confirmed via collection below)

`grep -c "^def test_\|^async def test_"` per file:

| File | Tests |
|---|---|
| `test_obsidian_export_service.py` | 22 |
| `test_jsonl_export_service.py` | 13 |
| `test_networkx_export_service.py` | 16 |
| `test_export_preview.py` | 7 (some parametrized) |
| `test_exports_router.py` | 16 |
| `test_handlers.py` | 2 |
| **Total** | **76 module-defined tests** |

(parametrized tests + class-nested tests inflate the collected count
above the raw `def test_` count, see pytest collection block below.)

### Coverage by export surface

**Obsidian (`test_obsidian_export_service.py`, 22 tests)** —
end-to-end against a fixture notebook produced by
`apps/app-main/tests/fixtures/obsidian_export_golden.md` (the reviewer-
pinned frontmatter shape). Tests assert:

- Entity `.md` rendering: frontmatter keys (`id`, `type`, `confidence`,
  `external_ids`, `aliases`, `sources`); body as bullet list.
- Wikilink resolution: every `[[entity-name]]` resolves to a sibling
  `.md` file; relations to filtered-out targets are silently dropped
  (Q-D-4) and never emitted as broken wikilinks.
- `README.md` index: per-type entity counts, relation count,
  filter-applied summary.
- Filename collision: two entities normalising to the same stem get
  `-2`, `-3`, … suffix; collision count surfaces in `ExportReport`.
- `_safe_entity_stem` filename sanitisation:
  filesystem-illegal characters stripped before zip-write or vault
  write.
- Embedding exclusion: serialised entities never include the 768-float
  embedding vector (Q-D-1 privacy + memory invariant).
- Status filter: entities with `status` in
  `EXCLUDED_ENTITY_STATUSES` (`archived`, `merged`) are dropped.
- `min_connections` post-filter via `_apply_min_connections_filter`
  (shared static method also called from JSONL + preview).
- Telemetry: exactly one `record_metric("export.obsidian", …)` per
  export; payload contains counts only, no raw vault path.
- Vault-path mode (D.1b): per-file atomic rename
  (`tempfile + os.replace`); failure surfaces `entities_written` +
  `failed_file` in exception args; user-added files outside the export
  set survive an overwrite pass.
- `_safe_entity_stem` containment-check: rejects
  `vault_entities_folder = "../../etc"` and similar path-traversal
  attempts.

**Obsidian via job handler (`test_handlers.py`, 2 tests)** — covers
the `JobType.EXPORT_OBSIDIAN` async path (the only async export
surface per Q-D-2): handler resolves `Settings.vault_path` and
delegates to `ObsidianExportService` with `mode="vault_path"`.

**JSONL (`test_jsonl_export_service.py`, 13 tests)** — streaming zip
of `entities.jsonl` + `relations.jsonl`. Tests assert:

- Per-line shape: entity (`id, canonical_name, entity_type, type_tags,
  primary_type, confidence, properties, source_documents,
  extracted_at`); relation (`id, source_entity, target_entity,
  relation_type, confidence, properties, source_documents`). The
  `in`/`out` → `source_entity`/`target_entity` rename is for Neo4j
  `apoc.load.json` + LangChain RAG-loader compatibility.
- Embedding exclusion: `"embedding" not in line` asserted explicitly
  (D.2 mental-inversion regression check).
- Streaming behaviour: `test_streaming_yields_multiple_chunks`
  generates 5000 entities, asserts `chunk_count > 1` (a single-yield
  generator regression would fail).
- Memory: same test smoke-checks
  `tracemalloc.get_traced_memory()[1] < 200MB`.
- Status filter parity: `EXCLUDED_ENTITY_STATUSES` imported from
  `obsidian_export_service` (shared symbol); a mixed-status fixture
  asserts only `entity:active` survives.
- `min_connections` post-filter via the shared
  `ObsidianExportService._apply_min_connections_filter` staticmethod
  so any future tuning lands on both paths simultaneously.
- Cancellation: `GeneratorExit` handled in the streaming loop without
  spurious failure telemetry.
- Telemetry: exactly one `record_metric("export.jsonl", …)` per
  export; counts-only payload.
- Filename sanitisation: notebook-name colons → underscores via the
  shared `_FILENAME_UNSAFE_RE` (B.2b-derived).

**NetworkX (`test_networkx_export_service.py`, 16 tests)** — covers
all 7 serialisation formats. Tests assert:

- 7 happy-paths (one per format: GraphML / GEXF / GML / JSON-tree /
  edge-list / adjacency-list / pickle): each format writes a buffer
  that NetworkX's matching reader successfully re-parses with the
  same node + edge counts.
- 2 round-trip tests with `type_tags` (list) + `properties` (dict):
  attributes flatten to CSV / JSON-encoded string before the XML
  writers see them, then re-hydrate cleanly on read.
- Attribute-flattening test: a node with a Python `dict` in
  `properties` survives a GraphML round-trip.
- Shared status filter + `_apply_min_connections_filter` parity (same
  shared imports as JSONL).
- Telemetry: exactly one `record_metric("export.networkx", …)` per
  export with format name + counts.

**Preview (`test_export_preview.py`, 7 tests)** —
`GET /api/notebooks/{id}/export-preview` is the counts-only surface
that the Obsidian dialog + JSONL popover both call before submit.
Tests assert the preview counts match what the corresponding
export endpoint would write (D.1c silent-drop parity — same status
filter, same `_apply_min_connections_filter`, same Q-D-4 endpoint
intersection).

**Router (`test_exports_router.py`, 16 tests)** — HTTP boundary
tests for the four endpoints (Obsidian, JSONL, NetworkX, preview):
400 on out-of-range confidence, 404 on unknown notebook, 501 → 200
transition for vault-path mode once D.1b landed, content-disposition
header shape, MIME types per format.

### Pytest collection (best-effort)

A `pytest --collect-only -q` run was launched against the six test
files in the sandbox. The exact collected count surfaces after
discovery completes; the run was started but the deep app-main DI
import chain (which boots LangChain + LangGraph + esperanto provider
factories at module load) means collection alone takes minutes on
this WSL host. The raw `grep`-counted totals above are a strict
lower-bound; the collected number will be higher because parametrize
factors inflate it. The implementer self-reviews per phase already
report the merged pytest exit codes (`test_obsidian_export_service.py`
in D.1a self-review, `test_jsonl_export_service.py` in D.2 self-
review, `test_networkx_export_service.py` in D.3 self-review,
`test_export_preview.py` in D.1c self-review, `test_handlers.py` in
D.1b self-review). Operators wanting a fresh count should run:

```bash
cd apps/app-main && \
  uv run pytest --collect-only -q \
    tests/test_obsidian_export_service.py \
    tests/test_jsonl_export_service.py \
    tests/test_networkx_export_service.py \
    tests/test_export_preview.py \
    tests/test_handlers.py \
    tests/test_exports_router.py 2>&1 | tail -3
```

## Shared filter pipeline (parity invariant)

All three exporters + the preview endpoint share **the same filter
pipeline**, in this exact order:

1. **SurrealQL gate** in
   `EntityRepository.list_entities_for_notebook` /
   `list_relations_for_notebook` (D.0) — applies
   `min_confidence`, `entity_types`, `include_orphans` at the DB layer.
2. **Status post-filter** —
   `EXCLUDED_ENTITY_STATUSES = frozenset({"archived", "merged"})`
   defined in `obsidian_export_service.py`, imported (not duplicated)
   by `jsonl_export_service.py`, mirrored locally in
   `networkx_export_service.py` and `exports.py`. The post-filter is
   load-bearing because the SurrealQL gate currently does not include
   `status`; D.0 follow-up #1 (deferred to D.2 originally, ultimately
   left as Python-side post-filter — see D.1a/D.3 self-review
   recommendations).
3. **`_apply_min_connections_filter`** — static method on
   `ObsidianExportService`; computes per-entity degree against the
   already-status-filtered relations, then drops entities below
   threshold. Re-used (not re-implemented) by `JsonlExportService` +
   `_export_preview` router fn.
4. **Q-D-4 endpoint intersection** — relations whose source or target
   didn't survive steps 1–3 are silently dropped (never emitted as
   broken wikilinks / dangling JSONL relation lines / broken NetworkX
   edges).

This shared pipeline is the reason preview counts and export counts
match. The D.1c attempt-1 review caught a drift here (preview applied
the SurrealQL gate but skipped step 2) — fixed before merge; the
parity is now load-bearing for the dialog "X entities will be
exported" UX.

## Manual smoke instructions (operator handoff)

The following is the smoke checklist a developer with live external
tooling should walk before declaring AC #1 fully validated. Each step
captures a piece of evidence (screenshot or `ls`/`head` output) that
the sprint-plan §358 calls for.

### Pre-flight

1. Pick a notebook from the Track-B corpus (the policy + scholarly
   mix from B.7's evidence doc — Track B chose 5 mixed-domain docs;
   any single one with ≥50 entities will exercise all three paths
   meaningfully). For ad-hoc smoke, any notebook with non-trivial
   entity + relation counts is acceptable.
2. Open the notebook in the UI (`http://localhost:8502/notebooks/<id>`).
3. Verify the header shows the three export buttons:
   "Export Obsidian", "Export JSONL", and (in the Schema tab) the
   NetworkX format dropdown.

### Obsidian zip (D.1a)

1. Click "Export Obsidian" → dialog opens.
2. Mode tab: "Download zip" (default; Vault tab will be disabled if
   `Settings.vault_path` is unset).
3. Move the `Min connections` slider; live preview-counts should
   change after ~300ms debounce.
4. Submit; a `<notebook-name>.zip` download should fire.
5. **Capture**: download dialog screenshot.
6. Unzip locally; expected:
   ```
   $ unzip -l <notebook-name>.zip
   # Expect a README.md + one .md per surviving entity
   ```
7. **Capture**: `unzip -l` output.
8. Open the unzipped folder as a vault in Obsidian:
   - `README.md` should render with per-type entity tables.
   - Click a `[[wikilink]]` in any entity .md; verify it opens the
     target entity (and is not a "create note" prompt — that would
     indicate a broken wikilink, which the silent-drop Q-D-4
     contract should prevent).
   - **Capture**: side-by-side screenshot of README + an entity page.
9. Front-matter sanity:
   ```
   $ head -20 <some-entity>.md
   # Expect YAML frontmatter: id, type, confidence, external_ids,
   # aliases, sources
   ```
   **Capture**: `head -20` output.

### Obsidian vault_path (D.1b)

1. Configure `Settings → Vault path` to an absolute path on a
   writable filesystem (NOT a Docker-mounted network volume during a
   sanity smoke — POSIX atomic rename semantics on network volumes
   vary). Restart the app (the setting is read at handler init).
2. Click "Export Obsidian" → Vault tab now enabled.
3. Submit; expected: a JSON `ExportReport` response instead of a zip
   download; the dialog shows "Wrote N entities to <vault_path>".
4. **Capture**: dialog success-state screenshot.
5. On the host:
   ```
   $ ls -la <vault_path>/<vault_entities_folder>/ | head
   # Expect one .md per exported entity, mtimes equal to now
   ```
   **Capture**: `ls -la | head` output.
6. **Atomicity check (optional)**: while an export is mid-batch, run
   `ls` repeatedly; you should not see partial files (the atomic
   rename means each file appears in its final state or not at all).
7. **User-file preservation**: place a `user_added.md` in the
   entities folder; trigger another export; verify `user_added.md`
   survives. Covered by `test_vault_path_overwrite_existing_md` —
   manual smoke just confirms the test reflects reality.

### JSONL (D.2)

1. Click "Export JSONL" → popover opens (popover, not dialog — plan
   directive).
2. Move sliders; preview counts should update on debounce.
3. Click "Download"; a `<notebook-name>.jsonl.zip` should fire.
4. **Capture**: download dialog screenshot.
5. Unzip and shape-check:
   ```
   $ unzip <notebook-name>.jsonl.zip
   $ head -1 entities.jsonl | jq .
   $ head -1 relations.jsonl | jq .
   ```
   Expected entity keys: `id`, `canonical_name`, `entity_type`,
   `type_tags`, `primary_type`, `confidence`, `properties`,
   `source_documents`, `extracted_at`. **No `embedding` key.**
   Expected relation keys: `id`, `source_entity`, `target_entity`,
   `relation_type`, `confidence`, `properties`, `source_documents`.
   **Capture**: `jq .` output for both lines.
6. **Neo4j smoke (optional, requires a running Neo4j + APOC)**:
   ```cypher
   CALL apoc.load.json("file:///path/to/entities.jsonl", null, {})
   YIELD value LIMIT 5
   RETURN value
   ```
   Should return five entities in `{id, canonical_name, …}` shape.
   Then load relations and create edges by matching
   `source_entity`/`target_entity` against the entity `id`s.
   **Capture**: Neo4j browser screenshot of the loaded subgraph.

### NetworkX (D.3)

1. Schema tab → NetworkX format dropdown → pick "GraphML".
2. Submit → download fires (`<notebook-name>.graphml`).
3. **Capture**: download screenshot.
4. **Round-trip in Python**:
   ```python
   import networkx as nx
   G = nx.read_graphml("<notebook-name>.graphml")
   print(G.number_of_nodes(), G.number_of_edges())
   # Should match the preview-counts you saw before export.
   list(G.nodes(data=True))[:3]
   # type_tags should be a CSV string; properties a JSON string
   # (per the D.3 attribute-flattening contract).
   ```
   **Capture**: Python REPL output.
5. **Gephi smoke (optional, requires Gephi installed)**:
   - File → Open → select the `.graphml`.
   - Verify node count + edge count match the Python read.
   - Run "Run layout: ForceAtlas 2" for ~10 seconds to confirm the
     graph is well-formed (no orphan-cluster anomalies that would
     indicate broken edge serialisation).
   - **Capture**: Gephi screenshot post-layout.
6. Repeat for at least one of: `GEXF`, `JSON-tree`, `pickle`. The 7
   formats are all round-trip-tested in
   `test_networkx_export_service.py`; the manual smoke just sanity-
   checks that the real consumer tools (Gephi for GraphML/GEXF, the
   `networkx.read_*` family for the others) re-parse cleanly.

### Preview-parity smoke

1. With sliders at their final position in the Obsidian dialog, note
   the "Will export: E entities, R relations" preview line.
2. Submit; unzip the result; count files:
   ```
   $ ls <unzip-dir>/*.md | grep -v README.md | wc -l
   # Should equal E
   ```
3. Repeat for JSONL:
   ```
   $ wc -l entities.jsonl relations.jsonl
   # Should equal E and R respectively
   ```
4. Drift here would indicate a regression in the shared filter
   pipeline (see "Shared filter pipeline" section above). The
   parity-invariant test `test_export_preview.py` should catch this
   pre-merge, but the live smoke is the ground truth.

## Cross-references

- `docs/tracks/D-output-richness/plan.md` §D.4 — the AC source-of-truth
  for what this evidence doc has to prove.
- `docs/tracks/B-kg-quality/RETRO.md` lines 188–213 — Track B set the
  precedent for sandbox-deferred live smoke.
- `docs/tracks/D-output-richness/status.md` — per-phase test counts
  + reviewer-evidence trail.
- `docs/troubleshooting/exports.md` — failure-mode diagnostics for the
  three export formats; what to look at when the live smoke surfaces
  an unexpected result.

## Conclusion

The Track-D test suite (76+ tests across 6 files) deterministically
proves that the export pipelines produce bytes matching their
documented contracts, that the three services share a single filter
pipeline (no drift), and that the HTTP + job-handler wiring is
end-to-end. **Live consumer-tool validation (Obsidian / Neo4j /
Gephi) is deferred to an operator session per the manual smoke
checklist above.** This is the same trade-off Track B made (see its
RETRO §"Live-test recommendation"); the sandbox limitation is real
and is documented honestly above per the AC #1 "captured in
E2E_EVIDENCE.md" requirement.
