# Track OKF — status

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
