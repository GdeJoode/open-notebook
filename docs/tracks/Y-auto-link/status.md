# Track Y — Status

## Phase Y.1 — Note-level similarity + note↔note edge table (Backend) — Ready for review

**Branch**: `track/y1-note-similarity` (off `main`, Tracks W + X merged)
**Commits**:
- `cc59115` feat(notes): note-level similarity + related_note edge + idempotent relate
- `89eb428` test(notes): Y.1 fresh-container tests + relate_note endpoint-binding fix

### Deliverables

1. **`NoteRepository.find_related_by_embedding(note_id, k)`**
   (`packages/surrealdb-service/src/surrealdb_service/repositories/notebook.py`) —
   note-level mirror of `SourceRepository.find_related_by_embedding`. Cosine over
   `note.embedding` via `vector::similarity::cosine`, excludes self (`id != $id`),
   excludes empty-embedding notes (`array::len(embedding) > 0` + the dim guard),
   ranks `score DESC, id ASC`, `LIMIT $k`. Returns `[{id, title, score}]`.
   A query note with the strict-but-empty `[]` embedding → `[]` (the
   needs-embedding signal Y.2 acts on), never a crash. Errors → `[]` + logged.

2. **Migration 68** (`migrations/68.surrealql` + `68_down.surrealql`) —
   `DEFINE TABLE OVERWRITE related_note SCHEMAFULL TYPE RELATION FROM note TO note`,
   fields `similarity_score: float DEFAULT 0.0`, `method: string DEFAULT "embedding"`,
   `created_at: datetime DEFAULT time::now()` (all strict WITH defaults — S.4),
   index `idx_related_note_score`. Non-destructive: null-endpoint-only DELETE +
   OVERWRITE (mirrors 66/67); healthy edges preserved. Down is a documented no-op.

3. **`NoteRepository.relate_note(from_note, to_note, *, similarity_score, method)`** —
   idempotent **clear-before-relate** (DELETE the exact `(in, out)` pair, then
   RELATE once) — RELATE is not idempotent (W.2/W.3). Refuses a self-edge,
   validates both ids are `note:<id>` records. The RELATE *endpoints* are
   interpolated as validated literal record ids (the `BaseRepository.relate`
   pattern); SurrealDB's RELATE graph syntax does **not** bind a `$param` in the
   in/out position (a parameterized `RELATE $from->...->$to` writes nothing).

### Acceptance — per-criterion evidence

All `@requires_docker`, run on a FRESH migration container (`surrealdb/surrealdb:v2`,
memory engine, full 1..68 migration apply):

| AC | Test | Result |
|----|------|--------|
| 1 cosine DESC + self-excluded | `test_find_related_ranks_by_cosine_and_excludes_self`, `test_find_related_respects_k` | pass |
| 1 no-embedding graceful | `test_find_related_empty_embedding_returns_empty`, `test_find_related_excludes_empty_embedding_candidates` | pass |
| 2 RELATION note->note on fresh container | `test_migration_68_discovered_and_related_note_is_note_to_note` | pass |
| 2 strict fields carry defaults | `test_related_note_defaults_populate_on_bare_relate` | pass |
| 2 drift conversion + idempotency + preservation | `test_drifted_any_related_note_converted`, `test_migration_68_idempotent`, `test_healthy_related_note_edges_preserved` | pass |
| 3 idempotent single edge | `test_relate_note_idempotent_single_edge` | pass |
| 3 self-edge refused | `test_relate_note_refuses_self_edge` | pass |
| 3 fields round-trip + non-note id rejected | `test_relate_note_fields_round_trip`, `test_relate_note_rejects_non_note_id` | pass |
| 4 canonical note rows untouched | `test_canonical_note_rows_untouched` | pass |

```
test_note_similarity_roundtrip.py + test_migration_68_related_note_relation.py
  → 14 passed
```

Regression check: `test_migrations_roundtrip.py`, `test_repositories.py`,
`test_migrations.py` → 47 passed (the roundtrip dynamically discovers + applies
68 up/down on a fresh container).

### Notes worth a memory

- **RELATE endpoint params don't bind**: a parameterized `RELATE $from->edge->$to`
  silently writes nothing on SurrealDB v2. The endpoints must be literal
  (validated) record ids interpolated into the query string — only the `SET`
  values can be `$params`. This is why `BaseRepository.relate` interpolates ids,
  and why the first `relate_note` cut returned True but wrote 0 rows.
- **RecordID-vs-string in WHERE**: `WHERE in = $param` only matches when `$param`
  is a `RecordID` (via `ensure_record_id`), not a `"note:x"` string. Test helpers
  that compare edge endpoints must convert.
