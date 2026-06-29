# Track NS — Status

## Phase NS.1 — Note→source similarity + `note_about` edge + helper (Backend) — READY FOR REVIEW

**Branch**: `track/ns1-note-source-similarity` (off `main` @ `56cd093`, Tracks W/X/Y/Z merged)
**Commits**: `b75604b` (migration 70), `73303d1` (repo methods + tests)

### Cross-type embedding validity (verified first, per the plan's STOP gate)
The note↔source cosine is meaningful — **same model, same dimension**:
- `source.embedding` = mean-pool (R.0) of the source's chunk vectors; each chunk
  vector comes from `EmbeddingService` → `embedding_model.aembed`.
- `note.embedding` = `_embed_with_retry([note.content])` → the SAME
  `embedding_model.aembed`.
- **Read-only staging probe** (`SURREAL_DATABASE=staging`): `source.embedding`
  len = **1024** (6/6 sources), `source_embedding` (chunk) len = **1024**.
  Staging holds **0 notes**, so a live note dim couldn't be probed there — but the
  note vector is produced by the identical model call as the 1024-dim chunk
  vectors, so it is 1024-dim by construction (and [[embedding-model-pin-1024]]
  pins mxbai-embed-large/1024). The `array::len(embedding) = array::len($q)`
  guard is the hard net: any dim mismatch is excluded, never crashed.
- Conclusion: cosine(note vector, source mean-pool) is the standard
  query↔document retrieval signal. **Proceeded** (no STOP).

### Deliverables
1. **`NoteRepository.find_related_sources_by_embedding(note_id, k)`**
   (`packages/surrealdb-service/src/surrealdb_service/repositories/notebook.py`):
   cosine of `note.embedding` vs every `source.embedding`; `WHERE embedding != NONE
   AND array::len(embedding) = array::len($q)`; `ORDER BY score DESC, id ASC LIMIT $k`;
   returns `[{id, title, score}]`. No-embedding note → `[]` (graceful); errors → `[]`
   + log. No hardcoded dim.
2. **Migration 70** (`migrations/70.surrealql` + `70_down.surrealql`):
   `DEFINE TABLE OVERWRITE note_about SCHEMAFULL TYPE RELATION FROM note TO source`,
   strict fields `similarity_score float DEFAULT 0.0`, `method string DEFAULT
   "embedding"`, `created_at datetime DEFAULT time::now()`, index
   `idx_note_about_score`. Non-destructive (null-endpoint DELETE + OVERWRITE), S.4-safe.
   Down = documented no-op. **Edge name** `note_about`: mirrors the lowercase
   snake_case `related_note`/`source_verdict` convention + schema.org/about;
   distinct from the ontology's uppercase entity-relation `ABOUT`.
3. **`NoteRepository.relate_note_source(note_id, source_id, *, similarity_score, method)`**:
   strict `_validate_record_id` on BOTH raw ids BEFORE interpolation; enforces
   `note:` (in) / `source:` (out); clear-before-relate per `(in, out)`; returns bool.

### Per-criterion test evidence (`@requires_docker`, fresh container)
`test_note_source_similarity_roundtrip.py` (8) + `test_migration_70_note_about_relation.py` (6) — **14 passed in 7.35s**:
- **AC1** cross-type ranking: `test_find_related_sources_ranks_by_cosine` (near>mid>far),
  `_respects_k`, `_empty_note_embedding_returns_empty` (no crash),
  `_excludes_no_embedding_sources`.
- **AC2** `note_about` is `TYPE RELATION` note->source on FRESH container
  (`test_migration_70_discovered_and_note_about_is_note_to_source`); strict fields
  default on bare RELATE (`_defaults_populate_on_bare_relate`); drift conversion;
  idempotent; healthy edges preserved.
- **AC3** idempotency (one row, latest score), note→source typing enforced
  (swapped/wrong-table refused), **injection-safe**
  (`test_relate_note_source_refuses_sql_injection_id`: drop-table payload in either
  position → `False`; `note`/`source`/`note_about` counts unchanged).
- **AC4** canonical `note`/`source` rows untouched (`_canonical_note_and_source_rows_untouched`);
  migrations roundtrip + Y.1/Z.1/68/69 suites green (**41 passed**, no regressions).

`mypy`: clean on the new code (only the pre-existing `shared.models` missing-stubs
note on the import line).

**Next**: NS.2 — extend `NoteAutoLinkService` orchestrator + triggers to also produce
`note_about` edges (depends on NS.1).

---

## Phase NS.2 — Extend the auto-link orchestrator + triggers (Backend) — READY FOR REVIEW

**Branch**: `track/ns2-orchestrator` (off `main` @ `6ee89bb`, NS.1 merged)
**Commits**: `9639443` (filter alignment), `ca1ab40` (orchestrator + schema),
`599dacd` (MCP tool), `cea5ce0` (endpoint/job carry-through tests)

### How `auto_link` was extended (combined summary)
`NoteAutoLinkService.auto_link` embeds the note **once**, then runs **two passes**
over that one embedding, each `rank → gate → idempotent RELATE`:
1. **notes** → `find_related_by_embedding` → `relate_note` (Y.2, unchanged);
2. **sources** → `find_related_sources_by_embedding` → `relate_note_source` (NS.2).

Both passes share the **same** `min_similarity` + top-`k` gate (`DEFAULT_MIN_SIMILARITY=0.75`,
`DEFAULT_K=5`; `k` bounds each pass independently). The source pass is **purely
additive** — appended after the note pass; the note→note behaviour is byte-for-byte
unchanged. `needs_embedding` / `not_found` short-circuit **both** passes symmetrically.

`AutoLinkResult` (and `to_dict()` / `NoteAutoLinkResponse`) gained a parallel,
distinctly-named source counter family so the two link types never conflate:
`source_links_created`, `source_skipped_existing`, `source_below_threshold`,
`source_candidates_considered`, `linked_source_ids` — alongside the existing
`created` / `skipped_existing` / `below_threshold` / `candidates_considered` /
`linked_note_ids`. One `auto_link` call now reports BOTH.

### Trigger carry-through (endpoint / MCP / job all link sources)
- **Endpoint** `POST /notes/{id}/auto-link`: unchanged route logic — it flattens
  `result.to_dict()`, so the same call now returns the source counts too. No new
  endpoint, no new param (the existing `k`/`min_similarity` gate both passes).
  Route validation intact (bad id/bounds → 422 — existing tests still green).
- **After-embed job** `handle_note_auto_link`: unchanged — it calls the orchestrator
  and returns `**result.to_dict()`, so the source pass carries through automatically.
  Confirmed by a DB-backed test (both `related_note` + `note_about` edges written;
  source set idempotent on re-run).
- **MCP `auto_link_note`** (repo-direct, option (a)): extended to run the source
  pass too (`find_related_sources_by_embedding` + `relate_note_source` both live
  in surrealdb-service, so the embedding-free layer covers sources without an
  embedding model). JSON result carries both counter families. `needs_embedding`
  / `not_found` / `invalid_id` statuses unchanged.

### Folded-in NS.1 minor (filter alignment)
Both ranking queries now use the **identical** no-embedding predicate
`embedding != NONE AND array::len(embedding) > 0`. Finding: `array::len(NONE)`
**raises** in this SurrealDB version (it does not return NONE), so the bare
`array::len(embedding) > 0` form the plan suggested crashes on a source whose
aggregate embedding is genuinely NONE (unaggregated). The `!= NONE` clause is
therefore load-bearing for the source path and short-circuits before the length
check; the `> 0` clause is what the source path was missing for empty-array rows.
Correctness-neutral for the note path (strict `[]`), one shared form.

### Per-criterion test evidence (LLM not involved — pure embedding cosine)
- **AC1** (on-demand source linking, threshold/top-k/idempotency, note links still
  produced, summary reports both):
  - service unit (`test_note_auto_link_service.py`, 18): `test_source_pass_threshold_gate_only_links_above`,
    `test_source_top_k_forwarded_to_source_ranking`, `test_source_relate_refused_counts_as_source_skipped`,
    `test_both_passes_run_in_one_call_and_counts_are_separate` (both edge types,
    distinct counters), `test_note_pass_unaffected_when_no_sources` (additivity),
    `test_needs_embedding_runs_neither_pass`.
  - MCP container (`test_mcp_auto_link_note.py`, 9): `test_auto_link_note_links_sources_above_threshold`
    (above-threshold `note_about` only, accounting balances, persisted set matches),
    `test_auto_link_note_reports_both_note_and_source_counts`, `test_auto_link_note_source_top_k_caps`
    (k=2 caps source links), `test_auto_link_note_source_idempotent_on_rerun`.
- **AC2** (endpoint + MCP + job link sources; no 500 on no-embedding; route validation
  intact): router (`test_notes_auto_link_router.py`, 9) `test_auto_link_summary_includes_source_counts`
  + existing 422/404/needs_embedding tests unchanged; DB job (`test_handle_note_auto_link_db.py`)
  `test_job_links_both_notes_and_sources`.
- **AC3** (idempotency — identical `note_about` + `related_note` sets, no dup rows):
  MCP `test_auto_link_note_source_idempotent_on_rerun` + job `test_job_source_links_are_idempotent_on_rerun`
  + the existing note→note idempotency tests.
- **AC4** (tests for the source path, summary, endpoint, MCP, job): see above.

Suites: auto-link app-main suite **41 passed**; surrealdb-service NS suites
(MCP+roundtrip+migration70) **23 passed**; neighboring Y/Z/NS.1 suites green
file-by-file; full app-main **1420 passed** (only the 3 known docling failures +
2 skips, no NS regressions). `mypy` clean on all new code (only the pre-existing
`shared.models` / `surrealdb_service.repositories` missing-stub import notes).

**Caveat for the reviewer**: running both `test_note_similarity_roundtrip.py` and
`test_note_source_similarity_roundtrip.py` in **one** pytest invocation cross-pollutes
the shared container (those exact-ranking tests aren't isolation-safe) — this
pre-exists on `main` (confirmed by stashing). Run them file-by-file.
