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
