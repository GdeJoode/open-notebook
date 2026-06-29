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
   RELATE once) — RELATE is not idempotent (W.2/W.3). Refuses a self-edge and a
   wrong-table id. Both endpoint ids are strict-validated against
   `_RECORD_ID_RE` (the `BaseRepository.relate` validator) **before any
   interpolation** — a `;`-bearing / `REMOVE TABLE`-bearing / malformed id is
   refused, never interpolated. The RELATE *endpoints* are then interpolated as
   these validated literal record ids; SurrealDB's RELATE graph syntax does
   **not** bind a `$param` in the in/out position (a parameterized
   `RELATE $from->...->$to` writes nothing), and interpolation is only safe
   because the accepted id is `table:id` with no SurrealQL metacharacters.
   (Note: `RecordID.parse` splits on the first colon and round-trips an
   injection payload verbatim, so the SDK's parsing is **not** a validator — the
   regex is.)

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
| 3 SurrealQL-injection id refused (table intact) | `test_relate_note_refuses_sql_injection_id` | pass |
| 4 canonical note rows untouched | `test_canonical_note_rows_untouched` | pass |

```
test_note_similarity_roundtrip.py + test_migration_68_related_note_relation.py
  → 15 passed
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
- **`RecordID.parse` is NOT an injection validator**: it splits on the first
  colon and round-trips the rest verbatim, so `note:x; REMOVE TABLE note; --`
  parses to table `note` + that whole id, and `str()` reproduces it. Any code
  that interpolates a record id into a query (RELATE endpoints can't be
  parameterized) MUST strict-validate the raw string against `_RECORD_ID_RE`
  first — `startswith("note:")` is bypassable and was a data-destroying
  injection in the first `relate_note` cut (review attempt 1).

---

## Phase Y.2 — Auto-link orchestrator + on-demand triggers (Backend) — Ready for review

**Branch**: `track/y2-autolink-ondemand` (off `main`, Y.1 merged)
**Commits**:
- `c5dd27d` feat(notes): auto-link orchestrator + on-demand HTTP/MCP triggers
- `272a0a8` test(notes): Y.2 auto-link service + endpoint + MCP tool tests
- `c952f52` test(notes): make MCP auto-link threshold test shared-DB-robust

### Deliverables

1. **`NoteAutoLinkService`**
   (`apps/app-main/src/app_main/services/note_auto_link_service.py`) —
   flow: `_ensure_embedding` (embed via injected `EmbeddingService.embed_note`
   when `note.embedding` is `[]`/missing; embed failure or no-content → clean
   `needs_embedding`, never a crash) → `NoteRepository.find_related_by_embedding`
   (top-`k` cosine, self-excluded — Y.1) → keep candidates with `score >=
   min_similarity` → idempotent `relate_note` per pair (carries the cosine
   `similarity_score`). Returns an `AutoLinkResult` summary
   (`status`/`created`/`skipped_existing`/`below_threshold`/
   `candidates_considered`/`embedded`/`linked_note_ids`/`k`/`min_similarity`).
   Canonical `note` rows untouched (only edges written; the embed step
   backfills the strict `embedding` field non-destructively).
   **Conservative defaults**: `min_similarity = 0.75`, `k = 5`. Params clamped
   to `[1, 50]` / `[-1, 1]` in the service too (defense-in-depth).

2. **HTTP trigger** `POST /notes/{id}/auto-link`
   (`apps/app-main/src/app_main/api/routers/notes.py`, params `k`,
   `min_similarity`) → drives the service, returns `NoteAutoLinkResponse`.
   **Route-layer validation**: the note id is strict-validated via
   `_validate_record_id` + `note:` prefix → a malformed/injection/wrong-table id
   is a **422** (not a 500, never reaches the service/DB); `k`/`min_similarity`
   bounded by `Query(ge/le=…)` → out-of-range is a 422. A missing note → 404; a
   no-content note → 200 `needs_embedding`. DI factory
   `get_note_auto_link_service` (async — resolves the embedding model via
   `get_embedding_service`).

3. **MCP `auto_link_note` tool** (`surrealdb-service/.../mcp/server.py`).
   **Layering: option (a)** — the tool calls the repo primitives
   (`find_related_by_embedding` + `relate_note`) DIRECTLY and REQUIRES the note
   to already be embedded; an empty/absent embedding → `{"status":
   "needs_embedding"}`. **Why (a) over (b)**: keeps surrealdb-service
   embedding-free, the exact contract its `search` / `search_similar` /
   `add_note` tools already establish (the embed model lives in app-main's
   local-Ollama pipeline, deliberately out of this package). Option (b) —
   calling the app-main endpoint — would invert the layering (the low-level
   graph package depending on the app service over HTTP) and add a network
   dependency for a pure-DB operation. The orchestrator with the embed step
   stays in app-main (the endpoint); the MCP tool is the embedding-free sibling.
   Boundary-validates the id (bad id → `invalid_id`), clamps `k`/`min_similarity`.
   **W.3 stdio-write follow-up applies**: `auto_link_note` is another unconditionally-
   registered WRITE tool — documented in the server module docstring that
   write-tool registration should be gated to stdio (or add auth) before any HTTP
   exposure. It writes only `related_note` edges and is idempotent.

### Acceptance — per-criterion evidence

| AC | Test(s) | Result |
|----|---------|--------|
| 1 threshold (only ≥ min_similarity) | `test_below_threshold_candidates_not_linked`, `test_threshold_boundary_is_inclusive` (service); `test_auto_link_note_threshold_and_self_exclusion` (MCP, real DB) | pass |
| 1 top-k cap | `test_top_k_passed_to_ranking_and_capped`, `test_k_clamped_to_bounds` (service); `test_auto_link_note_top_k_caps_links` (MCP, real DB) | pass |
| 1 no self-link | `test_never_links_to_self` (service); `test_auto_link_note_threshold_and_self_exclusion` (MCP, real DB) | pass |
| 1 idempotent re-run (no dup edges) | `test_relate_refused_counts_as_skipped` (service drives idempotent relate); `test_auto_link_note_idempotent_on_rerun` (MCP, real DB — identical edge set + count) | pass |
| 2 endpoint drives + summary | `test_auto_link_happy_path_returns_summary`, `test_auto_link_forwards_query_params` | pass |
| 2 endpoint no-embedding → 200 (not 500) | `test_auto_link_needs_embedding_is_200_not_500` | pass |
| 2 endpoint embed-first | `test_unembedded_note_is_embedded_then_linked` (service) + endpoint happy | pass |
| 2 MCP no-embedding → needs_embedding (no 500) | `test_auto_link_note_needs_embedding` (MCP, real DB) | pass |
| 3 route id validation (injection/wrong-table → 422) | `test_injection_note_id_rejected_422_before_service`, `test_wrong_table_id_rejected_422` | pass |
| 3 route param bounds (k/min_similarity → 422) | `test_out_of_range_k_rejected_422`, `test_out_of_range_min_similarity_rejected_422` | pass |
| — service no-content/embed-fail/not-found | `test_note_that_cannot_be_embedded_returns_needs_embedding`, `test_embed_failure_degrades_to_needs_embedding_no_crash`, `test_missing_note_returns_not_found` | pass |

```
app-main:   test_note_auto_link_service.py (12) + test_notes_auto_link_router.py (8) → 20 passed
surrealdb:  test_mcp_auto_link_note.py (5, @requires_docker) → 5 passed
combined regression w/ Y.1 + W.3 roundtrip → 31 passed; app-main adjacent → 43 passed
```

### Notes worth a memory

- **MCP tool tests share one DB** (notes accumulate across the suite). A
  threshold assertion must NOT assume a specific note is "ranked-then-dropped":
  with `k=10` other high-similarity notes can push it out of the top-k entirely
  (so `below_threshold` for it is 0, not 1). Assert the invariant instead — the
  high-cosine note is linked, the low ones / self are never linked, and the
  `created + below_threshold + skipped == candidates_considered` accounting
  balances.
- **Layering**: surrealdb-service MCP stays embedding-free. Auto-link's MCP tool
  (option a) requires a pre-embedded note; the embed-then-link convenience is the
  app-main endpoint only. This is the same split as W.3 `search`/`add_note`.

### Review log
- Attempt 1 → REVISIONS_NEEDED: 1 blocker — `relate_note` used
  `startswith("note:")` (bypassable) then interpolated the id literally, a
  data-destroying SurrealQL injection (`REMOVE TABLE note`). Fixed: both
  endpoints strict-validated via `_validate_record_id` (`_RECORD_ID_RE`) before
  interpolation; added `test_relate_note_refuses_sql_injection_id` (payload in
  both positions → refused, `note`/`related_note` counts unchanged); corrected
  the false "injection-safe" claims. Migration 68 / `find_related_by_embedding` /
  persistence-idempotency-self-edge were verified correct in attempt 1 and are
  unchanged.
