# Knowledge graph export troubleshooting

Open Notebook ships three knowledge-graph export surfaces (Track D —
output richness): **Obsidian** (zip download or direct-write-to-vault),
**JSONL** (Neo4j + RAG-pipeline ready), **NetworkX** (7 formats:
GraphML, GEXF, GML, JSON-tree, edge-list, adjacency-list, pickle).
This page covers the operational issues you are most likely to hit.

> Shared filter pipeline — every export endpoint applies, in order:
> the SurrealQL gate (D.0), the `EXCLUDED_ENTITY_STATUSES` post-filter
> (`{archived, merged}` — drops tombstones), the
> `_apply_min_connections_filter` static method, and the Q-D-4
> endpoint intersection (drops relations whose source or target
> didn't survive the entity filter). The `GET /export-preview`
> surface applies the same pipeline so dialog counts match export
> counts byte-for-byte. If you see drift, look here first.

---

## Obsidian — zip mode

### Download fails / browser shows network error

**Symptom**: clicking "Download" in the Obsidian dialog hangs or the
browser shows `net::ERR_*` after several seconds.

**Causes** (ordered most-likely-first):
1. The notebook has so many entities that the in-memory
   `BytesIO` zip exceeds the FastAPI request-timeout. The
   `BytesIO`-then-stream pattern (Q-D-7) peaks memory at zip size
   and emits the response only once the whole archive is built.
2. The reverse proxy in front of FastAPI (Nginx/Traefik) has a
   `proxy_read_timeout` lower than the build time.
3. A live `Settings.vault_path` write was attempted in zip mode by
   accident — check the request payload's `mode` field.

**Fix**:
1. Tighten the export filter (raise `min_confidence` or
   `min_connections`) so the entity count drops; the dialog's live
   preview-counts shows the effect in real time.
2. Raise the proxy read timeout to ≥ 60 seconds for very large
   notebooks.
3. Check the live response in DevTools → Network: a `application/zip`
   response with `Content-Disposition: attachment; filename=…` is
   the success shape. A `application/json` response with mode
   `vault_path` indicates a client-side bug routing the request to
   the wrong branch.

### Wikilink in entity .md shows "create note" prompt instead of resolving

**Symptom**: opening an exported entity in Obsidian, a `[[entity-name]]`
appears as a clickable "create" prompt instead of resolving to a
sibling page.

**Causes**:
1. The target entity was filtered out by `min_connections` or
   `min_confidence`. Q-D-4's silent-drop contract means relations to
   filtered-out targets are *never* emitted as broken wikilinks —
   so seeing the prompt indicates either a regression OR a
   `vault_path` mode where you wrote into an existing vault that
   contains a stale entity name (Obsidian resolves wikilinks across
   the entire vault, not just the export folder).
2. Filename-stem mismatch: the wikilink uses
   `_derive_filename(entity)` and the actual file uses the same —
   if a Q-D-5 collision suffix (`-2`, `-3`) landed, both wikilink
   and filename should match. A mismatch indicates a regression.

**Fix**:
1. Verify the target entity is in the export: check the
   `ExportReport.entities` field or just `unzip -l <archive.zip>`
   and grep for the expected filename.
2. If the wikilink target doesn't exist as a file but you expected
   it to, file the wikilink-target name + the entity it linked from
   + the active filter settings — this is a Q-D-4 regression and
   needs a unit-test reproducer.
3. For `vault_path` mode: name-collisions with pre-existing files
   are by design preserved — the user-added file wins, but the
   wikilink from a newly-exported entity now resolves to the user
   file, which may surprise. Move user-added files outside
   `vault_entities_folder` if you don't want the export to interact
   with them.

### Frontmatter parse error in Obsidian

**Symptom**: Obsidian's properties pane shows "Invalid YAML" or a
specific key shows as `<empty>` when the file actually contains data.

**Causes**:
1. A `:` or `"` character in an entity `canonical_name` leaked into
   an unquoted YAML scalar.
2. The `aliases` or `external_ids` list contains a value that
   re-renders ambiguously (e.g. a bare URL).
3. (V1) `external_ids` is always `[]` because the Track-M4 stub is
   still in place — this is documented in ARCHITECTURE.md §7 and
   is *not* a regression.

**Fix**:
1. Open the affected `.md` in a plain editor (VS Code) and look at
   the literal bytes between the `---` frontmatter delimiters.
   YAML 1.2 quoting rules: scalars containing `:` followed by space,
   `#`, leading `[/{/"/'/&/*/?/|/>/<`, or trailing whitespace must be
   quoted. The Obsidian writer wraps user-derived values
   (canonical_name, aliases) in double quotes; if you see an
   unquoted value, file with the entity_id so we can repro.
2. Empty `external_ids: []` is expected in V1 until Track M4 (Q9)
   lands TOOI + Crossref resolution.

---

## Obsidian — `vault_path` (direct-write) mode

The vault-path branch returns JSON (`ExportReport`) instead of a zip;
the response status code is the diagnostic anchor.

### 400 — Vault path not configured

**Symptom**: `POST /export-obsidian` with `mode="vault_path"` returns
`400 {"detail": "Vault path not configured"}`.

**Cause**: `Settings.vault_path` or `Settings.vault_entities_folder`
is not set in the app config.

**Fix**: Settings → Vault path → enter an absolute path on a
writable filesystem. Restart the app — the setting is read at handler
init for the `JobType.EXPORT_OBSIDIAN` async path.

### 400 — Vault path must be absolute

**Symptom**: 400 with detail mentioning "absolute".

**Cause**: D.1b's `_validate_vault_path` rejects `./relative/vault`,
`~/myvault` (the `~` isn't expanded at this layer), and anything not
starting with `/` (POSIX) or a drive letter (Windows).

**Fix**: provide a real absolute path. If you have `~` in your
config, expand it at the call site (e.g. in the env-file loader).

### 400 — Vault path does not exist / not a directory

**Symptom**: 400 with detail mentioning "does not exist" or "not a
directory".

**Cause**: D.1b validates the path with `Path.is_dir()` + writable
check before opening the first tmpfile.

**Fix**: `mkdir -p <vault_path>/<vault_entities_folder>` on the host.
Verify the FastAPI process user can write there (container vs host
permission mismatches are the common gotcha here).

### 500 — Permission denied / mid-batch failure

**Symptom**: `POST /export-obsidian` with `mode="vault_path"` returns
`500` with body `{"detail": "...", "entities_written": <N>,
"failed_file": "<name>.md"}`.

**Cause**: D.1b's atomicity model is **per-file, not whole-batch**.
A mid-batch permission denied (e.g. another process locked a target
file) leaves the first `N` files written and propagates the
exception. The body lists where the batch stopped.

**Fix**:
1. Fix the underlying filesystem issue (permission, disk full,
   target file locked by Obsidian's own open viewer).
2. Re-run the export. The export is idempotent for the entity set
   (same entities → same filenames → same content); re-writing the
   first `N` files is harmless.
3. If you need a hard rollback to a pre-export state, you must do
   it manually (a vault snapshot before export is the recommended
   workflow; Track D explicitly chose not to ship rollback because
   it would require an additional staging directory and a
   transactional rename across the whole batch).

### 422 — Filter validation failed

**Symptom**: `422 {"detail": [...]}` with a Pydantic validation
error in the body.

**Cause**: The `ExportFilter` Pydantic model bounds each slider
(`min_confidence: 0-1`, `min_connections: int >= 0`, etc.). The
422 is the standard FastAPI Pydantic-422 surface.

**Fix**: check the request payload against the documented bounds in
`packages/shared/src/shared/models/export.py`.

### Status code reference

| Symptom | Status | Meaning |
|---|---|---|
| `Vault path not configured` | 400 | `Settings.vault_path` unset |
| `Vault path must be absolute` | 400 | Relative path or `~` not expanded |
| `Vault path does not exist` / `not a directory` | 400 | Path validation failed |
| `Vault path is not writable` | 400 | Permission check failed pre-write |
| Filter validation error | 422 | Pydantic bound exceeded |
| `entities_written: N, failed_file: …` | 500 | Mid-batch filesystem error |
| Notebook not found | 404 | `notebook_id` doesn't match a row |
| Out-of-range confidence (preview) | 400 | `Query()` bound check failed |

---

## JSONL — streaming export

### Empty file / 0 bytes downloaded

**Symptom**: download fires, but `entities.jsonl` is 0 bytes.

**Causes**:
1. Filters dropped every entity. Check the dialog's preview-counts
   *before* submitting — if "Will export: 0 entities, 0 relations"
   you'll still get the zip but with empty `.jsonl` files inside.
2. The download was cancelled mid-stream (browser tab closed,
   client timeout). The Track-D D.2 attempt-2 fix catches
   `GeneratorExit` separately and emits no failure metric for
   cancellation; the file you receive is the partial chunk(s) up to
   cancel point.

**Fix**:
1. Relax `min_connections` / `min_confidence`; re-check preview;
   re-submit.
2. For very large notebooks where you suspect timeout cancellation,
   raise the client timeout (curl `--max-time`, browser tab kept
   in foreground).

### Partial download / truncated last line

**Symptom**: `head -10 entities.jsonl | jq .` succeeds for the first
9 lines but errors on line 10 ("Unexpected EOF").

**Cause**: The stream was cut mid-chunk. The 16KB chunk boundary
isn't aligned to line boundaries — D.2's streaming yields raw bytes
from the in-memory `BytesIO` ZipFile member, so a cancel at chunk N
leaves the last chunk's tail line truncated.

**Fix**:
1. Re-trigger the export (D.2's pipeline is idempotent for the
   entity set).
2. If the truncation is reproducible without a client-side cancel,
   that's a regression; capture the response headers
   (`Content-Length`, `Transfer-Encoding`) and file with the
   notebook ID + filter settings.

### Neo4j `apoc.load.json` parse error

**Symptom**: `apoc.load.json("file:///path/to/entities.jsonl", null, {})`
returns a parse error or 0 rows.

**Causes** (most-likely-first):
1. `apoc.load.json` expects one JSON value per file by default;
   JSONL is one JSON value per line. Use `apoc.load.json` with the
   `path` parameter as `"$"` and call once per line, OR use
   `apoc.load.jsonArray` against a wrapped `[…]` array (you'd need
   to pre-process the JSONL into a JSON array).
2. APOC plugin not installed in your Neo4j instance.
3. Path traversal: `file:///` URIs in `apoc.load.json` require the
   directory to be allow-listed in `neo4j.conf`
   (`apoc.import.file.enabled=true`,
   `dbms.security.allow_csv_import_from_file_urls=true`).

**Fix**:
1. Per-line load (the canonical pattern for JSONL):
   ```cypher
   CALL apoc.load.json("file:///path/to/entities.jsonl")
   YIELD value
   CREATE (e:Entity)
   SET e = value
   ```
   Some `apoc` versions require explicit line-mode; consult your
   version's docs.
2. Install APOC core if missing
   (`https://neo4j.com/labs/apoc/installation/`).
3. Allow-list the import directory in `neo4j.conf`.

### JSONL line shape reference

Each entity line:
```json
{
  "id": "entity:abc",
  "canonical_name": "Some Entity",
  "entity_type": "person",
  "type_tags": ["person", "researcher"],
  "primary_type": "person",
  "confidence": 0.92,
  "properties": {"affiliation": "ACME"},
  "source_documents": ["source:xyz"],
  "extracted_at": "2026-06-12T10:11:12Z"
}
```

Each relation line:
```json
{
  "id": "relation:def",
  "source_entity": "entity:abc",
  "target_entity": "entity:ghi",
  "relation_type": "AUTHORED",
  "confidence": 0.88,
  "properties": {},
  "source_documents": ["source:xyz"]
}
```

The `in`/`out` → `source_entity`/`target_entity` rename is
deliberate for Neo4j `apoc.load.json` + LangChain RAG-loader
compatibility (both want named fields, not SurrealDB's `in`/`out`
graph-edge convention). **No `embedding` field is ever emitted**
(Q-D-1 privacy + memory invariant).

---

## NetworkX — 7-format export

### GraphML XML parse error in downstream tool (Gephi, networkx.read_graphml)

**Symptom**: Gephi shows "Invalid XML" or
`networkx.read_graphml()` raises `KeyError` on a node attribute.

**Causes**:
1. A pre-D.3 export (impossible if you're on current main, but
   relevant if you mixed exports from a stale branch) didn't apply
   the attribute-flattening contract: `type_tags` must be a CSV
   string, `properties` must be a JSON-encoded string. GraphML
   doesn't accept native Python lists or dicts as attribute values.
2. A user-supplied entity name contains characters illegal in XML
   1.0 (e.g. raw `\x00`-`\x08`). The writer is supposed to strip
   these but if you hit a case, file a reproducer.

**Fix**:
1. Re-export from current main; the flatten/unflatten contract is
   pinned by `test_networkx_export_service.py` round-trip tests.
2. In Python, the round-trip:
   ```python
   import networkx as nx, json
   G = nx.read_graphml("<file>.graphml")
   # type_tags is now a CSV string; split if you want a list:
   for _, attrs in G.nodes(data=True):
       if "type_tags" in attrs:
           attrs["type_tags"] = attrs["type_tags"].split(",")
       if "properties" in attrs:
           attrs["properties"] = json.loads(attrs["properties"])
   ```

### Pickle unpickle fails with `AttributeError` or version warning

**Symptom**: `pickle.load(open("<file>.gpickle", "rb"))` raises
`AttributeError` or warns about Python version mismatch.

**Cause**: NetworkX pickles use Python's `pickle` module with the
default protocol; cross-version unpickling is best-effort. Pickling
also captures the NetworkX class hierarchy at write-time, so an
unpickle in a different NetworkX version may fail.

**Fix**: prefer GraphML / GEXF / JSON-tree for cross-environment
portability. Pickle is the fastest format but the least portable —
documented in module docstring of `networkx_export_service.py` and
in the dropdown tooltip.

### GEXF rejects an attribute type

**Symptom**: `nx.write_gexf` raises `TypeError: Unsupported attribute
type`.

**Cause**: GEXF is stricter than GraphML; some attribute types
(notably `set`, `tuple`, nested `dict`) get rejected even after the
flatten step.

**Fix**: D.3 flattens `properties` to JSON-string and `type_tags` to
CSV-string, which should cover the standard attribute payloads. If
you hit a rejection, the offending attribute is likely a per-entity
`properties` value that itself contains a non-JSON-serialisable
type. Inspect with:
```python
import networkx as nx, json
G = build_graph_from_jsonl(...)  # your local round-trip
for n, attrs in G.nodes(data=True):
    try:
        json.dumps(attrs)
    except TypeError as e:
        print(n, e)
        break
```
File the offending value and we'll fix the writer's flatten step.

### JSON-tree format: "Graph is not a tree" error

**Symptom**: NetworkX format dropdown set to JSON-tree, the export
returns a 400 with `detail` mentioning "tree" or "single root".

**Cause** (D.3 attempt-2 fix): `networkx.readwrite.json_graph.tree_data`
requires the graph to be a **rooted tree** (single root, no
cycles). The KG projection for a notebook is generally not a tree
— it's a multi-rooted DAG at best. D.3 attempt-2 added a pre-check
that rejects multi-rooted graphs with a 400 + remediation hint
instead of silently picking an arbitrary root.

**Fix**:
1. Pick a different format. GraphML / GEXF / JSON-edges /
   adjacency-list all handle arbitrary digraphs.
2. If you specifically need JSON-tree, pre-filter your entities to a
   single root by tightening the filter or restricting
   `entity_types`. The dialog preview will tell you the surviving
   shape.

### Format-by-format known limitations

| Format | Notes |
|---|---|
| `graphml` | Best portability. Attribute flatten contract required. |
| `gexf` | Stricter than GraphML; some nested attr types rejected. |
| `gml` | Plain-text format; large graphs produce big files. |
| `json-tree` | Requires single-rooted tree (D.3 pre-check enforces). |
| `edge-list` | No node attributes; only edges. Isolate entities NOT in any edge are dropped — documented per-entity in D.3 attempt-2's edge-list isolate-doc note. |
| `adjacency-list` | Same isolate caveat as edge-list. |
| `pickle` | Fastest. Least portable. Python + NetworkX version sensitive. |

---

## Preview counts mismatch the export

**Symptom**: The dialog says "Will export: 250 entities", the
downloaded zip contains 247 `.md` files.

**Cause**: This used to be a real bug in D.1c attempt-1 (preview
applied the SurrealQL gate but skipped the `EXCLUDED_ENTITY_STATUSES`
post-filter, so it over-counted by the tombstone count). D.1c
attempt-2 fixed this by applying the same status filter in the
preview surface. If you see drift on current main, file a regression
issue with the notebook ID + filter settings — the parity is
load-bearing and protected by
`test_export_preview.py::test_status_filter_parity`.

**Less-common causes if it's not a regression**:
1. The notebook was modified (an entity was archived or merged)
   between preview fetch and export submit. The preview hook
   debounces 300ms and caches 30s, so you can race against a
   server-side state change. Re-fetch the preview by tweaking a
   slider and re-submit.
2. A Q-D-4 silent-drop of relations to filtered-out targets caused
   relation count drift, not entity count drift. The preview
   surface applies the same Q-D-4 endpoint intersection as the
   exporters, so this shouldn't drift; if it does, capture the
   filter settings.

---

## Cross-references

- `ARCHITECTURE.md` §7 — full architecture of the export surfaces
  + shared filter pipeline.
- `docs/tracks/D-output-richness/RETRO.md` — what went wrong and
  what the fixes were (the four BLOCKERs above mostly map to
  attempt-2 fixes documented there).
- `docs/tracks/D-output-richness/E2E_EVIDENCE.md` — manual smoke
  checklist for an operator with live Obsidian / Neo4j+APOC / Gephi
  available.
- `packages/shared/src/shared/models/export.py` — Pydantic models
  for the request + response shapes referenced above.
