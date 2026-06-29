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

---

## Phase Y.3 — Background job + integration + docs + RETRO (Integration) — Ready for review

**Branch**: `track/y3-autolink-job` (off `main`, Y.1 + Y.2 merged)
**Commits**:
- `2a56ef2` feat(notes): background auto-link job chained after note embed (Y.3)
- `ea3dbad` test(notes): Y.3 background auto-link job + isolation tests
- `<docs>` docs(track-y): ARCHITECTURE §12 + FEATURE_ROADMAP + Y.3 status/RETRO; Track Y CLOSED

### Deliverables

1. **Background auto-link job** — a new `JobType.NOTE_AUTO_LINK`
   (`packages/shared/.../enums.py`) + `handle_note_auto_link`
   (`apps/app-main/.../handlers.py`) running the Y.2 orchestrator. **The trigger
   is the embed job**: a note can only be ranked by similarity once it HAS an
   embedding, so `_handle_embed_single_item` best-effort enqueues an
   `auto_link_note` job after a NOTE is successfully embedded
   (`result.embeddings_created > 0`) — mirroring the R.0 `DOCUMENT_PARSE` →
   `embed_source` chaining. Command mapping `auto_link_note → NOTE_AUTO_LINK`
   in `command_service.py`. The chained enqueue omits `k`/`min_similarity` so
   the orchestrator applies the conservative Y.2 defaults (0.75 / 5).
   - **Idempotent** — `auto_link` clears-before-relates each pair (Y.1/Y.2), so
     re-running the job yields the identical edge set, no duplicate rows.
   - **Best-effort / isolated** — a SEPARATE job from the embed. The note + its
     embedding are persisted by the upstream embed job before the auto-link job
     runs, so a linking failure cannot corrupt the note. The **enqueue seam** is
     best-effort (a queue hiccup never fails the embed — wrapped in try/except
     that logs, like the R.0 chaining). The **auto-link job** raises on a hard
     failure so the worker records it `FAILED` (not silently swallowed), but it
     only ever writes `related_note` edges (no note CRUD), so a raise leaves the
     note + embedding intact and no half-written graph state.

2. **Y.2 minors folded in** (both clean):
   - **(a)** `_ensure_embedding` now returns `(found, has_embedding,
     embedded_now)`; the not_found path no longer double-`get`s the note — the
     `found` flag is carried out instead of re-fetching.
   - **(b)** `invalid_id` documented on `NoteAutoLinkResponse.status` as
     MCP-only: the HTTP route 422s a malformed/wrong-table id at the boundary
     (before the service runs), so the HTTP response can never carry `invalid_id`.
     The `status` field is a free-form `str` (no enum to extend), so this is a
     doc clarification, not a new enum value the HTTP path would never emit.

3. **ARCHITECTURE note** — `ARCHITECTURE.md` §12 "Note auto-link: embedding →
   similarity → `related_note` edges (Track Y)": the three-layer flow, the
   app-main-vs-surrealdb-service embed split, the two-half trigger (on-demand +
   background job), the **isolation** design, the **sync model** (new notes;
   edit re-link is a noted follow-up), the data reality, and the note↔source
   extension point. Existing sections untouched (new §12, "Further reading"
   renumbered §13).

4. **RETRO + roadmap** — this RETRO (below); FEATURE_ROADMAP Track Y entry marked
   ✅ CLOSED; ARCHITECTURE Further-reading link added.

### Acceptance — per-criterion evidence

| AC | Test(s) | Result |
|----|---------|--------|
| 1 job links a NEW embedded note (job path, not endpoint) | `test_job_links_a_new_embedded_note` (DB, @requires_docker) | pass |
| 1 embed enqueues the auto-link job | `test_autolink_enqueued_after_note_embed`; not for 0-embeddings / source: `test_no_autolink_when_note_produced_no_embedding`, `test_no_autolink_for_source_embed` | pass |
| 1 on-demand still works + both idempotent | Y.2 router/service/MCP suites (unchanged, re-run green); `test_job_is_idempotent_on_rerun` (DB) | pass |
| 2 enqueue failure never fails the embed | `test_autolink_enqueue_failure_does_not_fail_embed` | pass |
| 2 job failure isolated (note + edges intact, no half-write) | `test_job_failure_leaves_note_and_edges_intact` (DB) + `test_autolink_failure_is_isolated_to_this_job` (unit) | pass |
| 2 needs_embedding is a status, not an error | `test_needs_embedding_is_not_an_error` | pass |
| 3 handler registered | `test_note_auto_link_handler_registered`, `test_existing_handlers_still_registered` (Y.3 added) | pass |

```
app-main job tests: test_handle_embed_note_autolink.py (4) +
  test_handle_note_auto_link.py (4) + test_handle_note_auto_link_db.py (3, docker)
  + test_handlers.py (2) → 13 passed
regression (auto-link service/router/MCP + R.0 chaining + handlers) → 34 passed
job-queue package → 38 passed
```

The 3 known docling failures + pre-existing top-level `tests/` import errors are
the documented baseline, untouched by Y.3.

### Notes worth a memory

- **The trigger is the embed job, not note creation.** Note creation
  (`POST /notes`) does NOT itself embed — embedding is its own job/route. So the
  correct, race-free hook for "auto-link a NEW note" is *after the embed
  completes* (`_handle_embed_single_item`, item_type == "note",
  embeddings_created > 0), exactly where the embedding the similarity needs has
  just landed. A new job (`NOTE_AUTO_LINK`) rather than inlining keeps the
  failure isolated from the embed's recorded result.
- **Isolation = a separate downstream job.** Two seams: the *enqueue* is
  best-effort (try/except, logs, never fails the embed); the *job itself* raises
  on hard failure (worker marks it FAILED) but touches only `related_note` edges,
  so it can't corrupt the note. Idempotent `relate_note` makes a re-run safe.

---

## Track Y — RETROSPECTIVE (CLOSED 2026-06-29)

**What Track Y delivered.** Note↔note auto-link (Constella Feature 2): a note,
once embedded, is linked to its most-related notes by embedding similarity, the
links persisted as `related_note` RELATE edges. Both halves of the phased
trigger shipped — on-demand (HTTP endpoint + MCP tool, Y.2) and a background job
chained off the embed (Y.3) — over an idempotent, precision-gated orchestrator
(Y.2) on a fresh-container-safe edge table + similarity primitive (Y.1).

**Design highlights / lessons:**

1. **Note↔note similarity mirrors the source layer.** The whole feature is the
   source-level `find_related_by_embedding` lifted to notes — same cosine, same
   RELATE discipline, a second edge table. Reusing the proven mechanism (rather
   than inventing a note-specific one) is why Y.1 was small and Y.2/Y.3 could
   focus on the orchestration + trigger.

2. **The SurrealQL-injection blocker (Y.1, review attempt 1).** The first
   `relate_note` cut used `startswith("note:")` to "validate" an id then
   interpolated it literally into a `RELATE` query — a **data-destroying
   injection** (`note:x; REMOVE TABLE note; --`). Two compounding facts made it
   sharp: (a) `RELATE`'s graph syntax **cannot bind a `$param`** in the in/out
   position (a parameterized `RELATE $from->edge->$to` silently writes *nothing*),
   so the endpoints *must* be interpolated; (b) `RecordID.parse` is **not** a
   validator — it splits on the first colon and round-trips the payload verbatim.
   The fix: strict-validate the raw string against `_RECORD_ID_RE` **before** any
   interpolation. The lesson generalises to every RELATE in the codebase
   (the §9 materializers carry the same note).

3. **Conservative threshold / no graph explosion.** `min_similarity=0.75` +
   top-`k=5` (clamped in both the route and the service). The alternative —
   link-everything-above-zero — would make the graph near-complete and useless.
   This is the R.6 discipline: a sparse, meaningful graph beats a dense, noisy
   one. Defaults are conservative by intent and tunable per call.

4. **Isolation as a first-class property (Y.3).** Auto-link is best-effort and
   downstream of the embed: it can never block note creation or corrupt the note.
   The two-seam design (best-effort enqueue + isolated, idempotent job) means the
   worst case is "a note has no related-note edges yet" — recoverable by the
   on-demand endpoint or a re-run, never a 500 or a half-written graph.

5. **Data reality — a mechanism, honestly framed.** The live corpus is
   source-heavy with few notes, so today auto-link is a built-and-tested
   *mechanism* that lights up as notes accumulate — exactly like the Track U
   `cites` edges (a clean no-op until the data arrives). The tests seed notes to
   prove the path; production value scales with note count.

**Follow-ups (noted, not Y core):**
- **Edit re-link** — re-linking a note when its content/embedding changes (stale
  edges). Idempotent `auto_link` makes this a drop-in (re-run after the edit's
  embedding settles), but the upkeep loop on mutation is the next increment.
- **note↔source extension** — the same embedding-similarity mechanism with a
  second edge type (`related_source`), promotable from the documented extension
  point. Y core is note↔note.

**Status: Track Y CLOSED.** Y.1 + Y.2 merged to `main`; Y.3 on
`track/y3-autolink-job`, ready for review. ARCHITECTURE §12 + FEATURE_ROADMAP
updated.
