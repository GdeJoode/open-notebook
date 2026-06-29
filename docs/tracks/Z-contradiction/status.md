# Track Z — status

## Phase Z.1 — Verdict edge schema + idempotent helper (Backend) — READY FOR REVIEW

**Branch**: `track/z1-verdict-schema` (off `main` @ `96ccb3d`, Tracks W/X/Y merged)
**Commits**:
- `8237734` feat(migrations): migration 69 — `source_verdict` source↔source edge
- `1e2aede` feat(source-repo): idempotent, injection-safe `relate_verdict`
- `58ae6c3` test(source-verdict): container tests for 69 + `relate_verdict`

### What landed
- **Migration 69** (`migrations/69.surrealql` + `69_down.surrealql`): asserts
  `source_verdict` as `SCHEMAFULL TYPE RELATION FROM source TO source`, strict
  fields WITH defaults (S.4): `verdict` (`"neutral"`), `confidence` (`0.0`),
  `reasoning` (`""`), `judge_model` (`""`), `created_at` (`time::now()`); index
  `idx_source_verdict_verdict` on `verdict`. Non-destructive: null-endpoint-only
  `DELETE` + `DEFINE TABLE OVERWRITE` (mirrors 66/67/68). Down = documented no-op.
- **`SourceRepository.relate_verdict`** (`repositories/source.py`): mirrors
  `NoteRepository.relate_note` (Y.1). Strict-validates BOTH raw id strings via
  `_validate_record_id` (`_RECORD_ID_RE`) BEFORE interpolation; refuses self-edges
  and non-`source` ids; clear-before-relate per `(in,out)`; only SET values are
  parameterized.

### `claim` / `contradicts` reconciliation (decision)
Probed staging (`SURREAL_DATABASE=staging`, read-only `INFO FOR DB`):
- `claim` = `TYPE ANY SCHEMALESS`, ~13 rows — app-side, NOT migration-managed.
- `contradicts` = `TYPE RELATION IN source OUT claim SCHEMAFULL`, 0 rows — a
  **source→claim** edge (claim-level; fields `strength`/`quote`/`evidence_type`).

That is a **different unit** (a source contradicting a *claim*) and **different
shape** (source→claim) from Z's source↔source verdict. Per plan Decision 2 & 5,
claim-level contradiction is a **documented extension**, not Z core. **Decision:
introduce a separate, cleanly-named `source_verdict` (source→source) and do NOT
touch `claim`/`contradicts`.** A container test recreates the app-side shape,
applies 69, and asserts `claim`/`contradicts` survive (typed + a source→claim
RELATE still works).

### Acceptance criteria — evidence
1. **`source_verdict` is `TYPE RELATION FROM source TO source` on a FRESH
   container; strict fields default** —
   `test_migration_69_discovered_and_source_verdict_is_source_to_source`
   (asserts `TYPE RELATION`, `source`, NOT `claim`) +
   `test_source_verdict_defaults_populate_on_bare_relate` (bare RELATE, no SET,
   succeeds; defaults populate). PASS.
2. **`relate_verdict` idempotent / self-edge / id-validation** —
   `test_relate_verdict_idempotent_single_edge` (re-relate → 1 row, latest
   verdict `contradicts`/`0.92`), `test_relate_verdict_refuses_self_edge`,
   `test_relate_verdict_fields_round_trip`,
   `test_relate_verdict_rejects_non_source_id`. PASS.
3. **Injection-safe (`_validate_record_id`-before-interpolate)** —
   `test_relate_verdict_refuses_sql_injection_id`: a `;`/`REMOVE TABLE`-bearing
   id in **from, to, AND both** positions is refused; `source` AND
   `source_verdict` row counts UNCHANGED. PASS.
4. **`claim`/`contradicts` not broken; canonical `source` rows untouched** —
   `test_claim_contradicts_scaffolding_not_broken`,
   `test_canonical_source_rows_untouched`, plus
   `test_healthy_source_verdict_edges_preserved` (OVERWRITE preserves N edges).
   PASS.
5. **Tests + migrations roundtrip green** — see below.

### Test runs (Docker)
- `test_migration_69_source_verdict_relation.py` + `test_source_verdict_relate.py`
  → **12 passed**.
- `test_migrations_roundtrip.py` (up/down all migrations incl. 69) → **19 passed**.
- No regression: `test_migration_68_related_note_relation.py` +
  `test_note_similarity_roundtrip.py` → **15 passed**.
  - NOTE: this "no regression" was a targeted-subset run. The FULL
    `packages/surrealdb-service` suite was red at the time due to a pre-existing
    Y.1 test-isolation flake in
    `test_note_similarity_roundtrip.py::test_find_related_ranks_by_cosine_and_excludes_self`
    (session-scoped fixture, no per-test cleanup → notes from other suites
    evicted the seeded notes from top-k). Not introduced by Track Z; fixed on
    `fix/relate-cites-injection` (5-dim isolation subspace). Full suite now green.

### Warnings
- mypy on `repositories/source.py`: 2 pre-existing errors (`shared.models`
  missing stubs; `get_embedding_count` `no-any-return`) — both present on `main`,
  unrelated to this change. `relate_verdict` is mypy-clean.

---

## Phase Z.2 — Candidate generation + pairwise LLM judge (Backend) — READY FOR REVIEW

**Branch**: `track/z2-judge` (off `main` @ `80e860a`, Z.1 + relate_cites security fix merged)
**Commits**:
- `cc6030b` feat(judge): Z.2 contradiction judge — candidate gen + precision-first pairwise judge
- `6f81652` test(judge): Z.2 unit + @requires_docker coverage (LLM mocked)

### What landed
- **`apps/app-main/.../services/contradiction_judge_service.py`**:
  - `build_candidate_pairs(source_id, related_ids, k)` — pure/testable: forms
    `(source, related)` pairs, self-excluded, deduped, bounded by top-k,
    rank-order preserved. Candidates come from the Track R substrate, never O(n²).
  - `parse_verdict(raw)` — robust parse to a normalised `JudgeVerdict`. Strips
    fences/prose to the first balanced `{...}`; any malformed/missing/unknown
    label / non-numeric confidence degrades to `neutral`/`0.0`. A parse failure
    can only SUPPRESS an edge, never fabricate one.
  - `ContradictionJudgeService.judge_pair` / `judge_source` — compact context
    (titles + bounded `full_text` snippets, capped at 2000 chars/source), routed
    LLM via the injected `(system, user) -> str` caller (json_mode), precision
    gate, idempotent persistence via Z.1 `relate_verdict`. `judge_source` returns
    a `{judged, contradicts, reinforces, neutral, below_threshold, edges_written,
    candidates_considered, verdict_pairs, ...}` summary.
  - `JUDGE_SYSTEM_PROMPT` — a careful fact-checking judge; strict-JSON output;
    repeated, explicit bias toward `neutral` when unsure (precision-first).
- **Precision gate**: persist ONLY `verdict ∈ {contradicts, reinforces}` AND
  `confidence >= min_confidence`. Default `min_confidence = 0.7` (conservative;
  lower only once trusted). Default `k = 5`, `MAX_K = 50`.
- **DI**: `get_contradiction_judge_service()` (async) wires the
  `HybridRetrievalService` related substrate + Z.1 `relate_verdict` + the J.4
  routed caller (`default_chat_model`, json_mode). `related_service` accepts
  hybrid (`find_related_hybrid`) OR dense-only (`find_related`).

### Test runs
- `test_contradiction_judge_service.py` (unit, LLM mocked, no DB) → **40 passed**.
- `test_contradiction_judge_db.py` (`@requires_docker`, real SourceService +
  SourceRepository, LLM mocked) → **5 passed** (one edge on confident contradicts;
  idempotent re-judge; neutral/below-threshold → no edge; canonical rows untouched).
- No regression: `test_routed_summarization` + `test_hybrid_retrieval_service` +
  `test_note_auto_link_service` → **25 passed**; `test_routing_e2e` +
  `test_routed_extraction` + `test_source_related_hybrid_router` → **18 passed**.

### Per-AC evidence
1. Structured `{verdict, confidence, reasoning}` + correct gate — each verdict
   class tested: confident contradicts → edge, confident reinforces → edge,
   neutral → no edge, low-confidence contradicts → no edge, malformed → no edge
   (no crash). (unit + DB).
2. Candidates from the related substrate (hybrid first, dense fallback); no
   self-pair; `(a,b)` judged once; bounded by top-k. (unit).
3. Idempotent (re-judge → one edge, latest verdict) + injection-safe via Z.1. (DB).
4. All tests with the LLM mocked; `@requires_docker` only on the DB-write file.

### Notes / follow-ups (for Z.3)
- `min_confidence` default 0.7 is the precision dial; expose as an endpoint param.
- `judge_model` provenance is stamped from the served chat model id.

### Z.2 review round 1 — REVISIONS addressed (commit `10606fe`)
**Blocker (false `contradicts` edge from a JSON array)** — FIXED. The parser now
accepts ONLY a top-level JSON **object**:
- `_parse_top_level_object`: fence-strip → `json.loads` → accept ONLY a `dict`;
  a top-level array/scalar → neutral. Never descends into an array element.
- Prose fallback (`"…: {…}"`) kept but guarded: if the first JSON-structural char
  is `[` (bare `[{…}]` OR prose-wrapped `"Result: [{…}]"`) → refuse, since the
  `{` is nested inside the array. A legit object whose `reasoning` contains `[`
  (object's `{` precedes the `[`) still parses.

**Major (missing adversarial parse tests)** — ADDED. New cases, each asserted at
BOTH `parse_verdict` and end-to-end `judge_pair` (relate_verdict NOT awaited):
array containing a confident verdict object (bare/fenced/prose-wrapped), scalar
array, `}` inside the reasoning string (genuine-or-neutral, never wrong), two
JSON blocks (first object only), string confidence. Plus a DB test that the
array case writes no real `source_verdict` row.

**Minor (string confidence)** — folded in: `_coerce_confidence` is strict-numeric;
a STRING confidence (`"0.9"`) → 0.0 (no edge), `bool` excluded.

Re-run: judge unit + DB → **66 passed**; no regression (summarization/hybrid/
auto-link → 25 passed).

---

## Phase Z.3 — On-demand trigger + integration + docs + RETRO (Integration → CLOSE) — READY FOR REVIEW

**Branch**: `track/z3-judge-ondemand` (off `main` @ `5885465`, Z.1 + Z.2 merged)
**Commits**:
- `eaff248` feat(judge): Z.3 on-demand endpoint `POST /sources/{id}/judge-contradictions`
- `b2dea56` docs(mcp): Z.3 — defer `judge_contradiction` MCP tool, document the layering rationale
- `5d4a07a` docs(architecture): Z.3 — contradiction flow section + Track Z roadmap (CLOSED)
- _(this commit)_ docs(track-z): Z.3 status + Track Z RETRO; Track Z CLOSED

### What landed
- **HTTP endpoint** `POST /sources/{id}/judge-contradictions`
  (`api/routers/sources_processing.py`): params `k` (`Query(ge=1, le=MAX_K)`) and
  `min_confidence` (`Query(ge=0.0, le=1.0)`); drives Z.2's
  `ContradictionJudgeService.judge_source` and returns the precision-first summary
  `{judged, contradicts, reinforces, neutral, below_threshold, edges_written,
  candidates_considered, min_confidence, k, verdict_pairs}` via the new
  `SourceJudgeContradictionsResponse` schema.
- **Route-layer validation (Y.2 discipline)**: `_validate_source_id` strict-validates
  the id via `_validate_record_id` BEFORE the service (a `;`-bearing injection / a
  wrong-table id → **422**, never a 500); `k`/`min_confidence` bounded by `Query`
  → 422; a missing source → **404**. The judge/DB is never reached on a bad id.

### MCP-tool decision — (b) DEFER, with rationale
The deliverable offered (a) an MCP tool that calls the app-main endpoint, or (b)
deferring it. **Chose (b).** The judge requires the **app-main LLM routing layer**
(`RoutedLLMCaller` / `make_default_llm_caller`); the surrealdb-mcp server
(`packages/surrealdb-service/.../mcp/server.py`) is intentionally a **thin,
repo-direct layer with NO app-main / LLM dependency**, and **no app-main base URL
is available to it** (its only `httpx` use is a test-fixture health check). Y.2's
`auto_link_note` tool *could* live there because all its primitives (cosine
ranking + `relate_note`) are in surrealdb-service; the judge's are not. Option (a)
would either pull the LLM stack into that package or invert the layering with a
new surrealdb-service→app-main HTTP coupling. So the HTTP endpoint is the on-demand
trigger; the MCP tool is a **documented follow-up** (recorded in the server module
docstring, next to the W.3/Y.2 tool inventory), to be done once an app-main base
URL is cleanly available to that server. The HTTP endpoint fully covers the Z.3
on-demand trigger acceptance.

### Acceptance criteria — evidence
1. **Endpoint drives the judge over related pairs + returns the summary; route-layer
   validation (bad id/bounds → 422 not 500; missing source → 404); LLM mocked** —
   `test_sources_judge_contradictions_router.py`:
   - `test_judge_happy_path_returns_summary` (judge service mocked → summary keys
     + params forwarded with route defaults `k=5`, `min_confidence=0.7`),
   - `test_judge_forwards_query_params` (`?k=3&min_confidence=0.9` forwarded),
   - `test_judge_below_threshold_no_edges_is_200` (all-neutral/below run → clean 200,
     `edges_written=0`),
   - `test_judge_missing_source_returns_404` (judge NOT awaited),
   - `test_injection_source_id_rejected_422_before_service` (a `;`/`REMOVE TABLE`
     id → 422; neither `source_svc.get` nor the judge awaited),
   - `test_wrong_table_id_rejected_422` (`note:abc` → 422),
   - `test_out_of_range_k_rejected_422` (`k=9999`, `k=0` → 422),
   - `test_out_of_range_min_confidence_rejected_422` (`2.0`, `-1.0` → 422).
   **8 passed.** The LLM is mocked (the judge service is an `AsyncMock`).
2. **MCP-tool decision implemented per (b) with justification; HTTP endpoint covers
   the on-demand trigger** — see the decision above; the deferral is documented in
   the server docstring + this status + the RETRO.
3. **ARCHITECTURE note + RETRO; Track Z CLOSED; roadmap updated; extensions noted;
   all-5-features milestone** — `ARCHITECTURE.md` §13 (the flow + precision-first +
   cost framing + background-job + claim-level extensions); `FEATURE_ROADMAP.md`
   Track Z CLOSED + the 5/5 Constella milestone; the RETRO below.
4. **Suites green** — see below; the 3 docling failures are the known baseline.

### Test runs
- `test_sources_judge_contradictions_router.py` (endpoint, judge mocked) → **8 passed**.
- No regression: `test_contradiction_judge_service.py` +
  `test_notes_auto_link_router.py` → **68 passed**.
- Sources router aggregation registers
  `/sources/{source_id}/judge-contradictions` (import check).
- Pre-existing top-level `tests/` import errors + the 3 known docling failures are
  the documented baseline (unrelated to Z.3).

### Warnings
- None new. The endpoint reuses the Z.2 service + Z.1 helper unchanged.

---

## Track Z — RETRO (CLOSED 2026-06-29)

Track Z delivered the last Constella feature: contradiction detection over related
source pairs. Three phases, precision-first throughout. **Track Z is CLOSED.**

### What worked
- **Precision-first as the load-bearing design choice.** "A false contradiction is
  worse than a missed one" drove every decision: a `neutral`-by-default judge prompt,
  a conservative `min_confidence=0.7` gate persisting ONLY confident
  contradicts/reinforces, and a parser whose every failure mode degrades to
  `neutral` (so a parse problem can only ever *suppress* an edge, never fabricate
  one). The graph stays trustworthy even on a noisy model. Same discipline as U.3's
  `cites` membership gate — proven, reused.
- **Candidate-bounding (Track R related, not O(n²)).** Judging every pair is an
  LLM-cost explosion. Taking candidates from the existing Track R related substrate
  (`find_related_hybrid`/`find_related`, top-`k`) means we only ever judge
  *plausibly-related* pairs — O(top-`k`) per source. The feature reuses the relatedness
  layer instead of re-deriving it, and the cost stays bounded by construction.
- **Reusing the Y.1/Y.2 patterns wholesale.** Z.1's `relate_verdict` mirrors Y.1's
  `relate_note` (idempotent clear-before-relate, strict id validation before
  interpolation); Z.3's endpoint + route-layer validation mirror Y.2's
  `POST /notes/{id}/auto-link` (strict id → 422, bounds via `Query(ge/le)`, missing
  → 404). The phased on-demand-now / background-job-later trigger is the same shape
  Y took. Two tracks, one validated edge-write template.

### What we caught (the review wins)
- **The JSON-array fabrication blocker (Z.2 review).** The first parser would lift a
  verdict out of `[{"verdict":"contradicts",...}]` — a *common* "respond with JSON"
  model failure mode — fabricating a confident contradiction edge from a malformed
  response. The fix is a hard invariant: **accept ONLY a top-level JSON object**; a
  top-level array/scalar degrades to `neutral`, and the prose-fallback scan refuses
  to descend into an array (`[` before `{` → reject). The lesson: for a
  precision-critical parser, *parse only the exact shape you trust* — never recover a
  "best effort" value out of the wrong container, because in a precision-first system
  the wrong recovery is worse than no recovery.
- **The `relate_cites` injection (Z.1 review, cross-track).** The Z.1 review surfaced
  that the analogous `relate_cites` path interpolated ids into a RELATE without strict
  validation (RELATE can't `$param`-bind the in/out endpoints) — a data-destroying
  SurrealQL-injection vector. Found while building Z.1's `relate_verdict` (which
  validates), fixed on its own branch, and Z.1 then rebased on the fix. A
  cross-track security win that Track Z's discipline (validate-before-interpolate)
  flushed out of a *neighbouring* feature.

### The honest caveats
- **Data reality.** Few sources today, so this is a built-and-tested *mechanism* that
  lights up as the corpus grows — like U.3's `cites`. The DB tests seed pairs and
  prove the machinery; production value scales with the corpus.
- **The MCP tool is deferred, not done.** The on-demand trigger is the HTTP endpoint;
  the MCP tool waits on a clean app-main-URL seam for the repo-direct mcp server (see
  the Z.3 decision above). An honest deferral, not a silent gap.
- **On-demand only; the background job is a noted follow-up.** Judging is O(pairs) of
  LLM calls; automating it over new/edited sources waits until the judge is trusted
  in practice — the same staging auto-link took. Claim-level contradiction
  (source→`claim`) is the other documented extension.

### Constella adoption — COMPLETE (5 / 5)
Track Z closes the Constella adoption work. **All five features done**:
- **Feature 1** — shared graph memory → **Track W**
- **Feature 2** — auto-link → **Track Y**
- **Feature 3** — contradiction detection → **Track Z** ← this track
- **Feature 4** — citations to source → **Track X**
- **Feature 5** — MCP graph-tools substrate → **Track W**

**Track Z: CLOSED. Constella adoption: COMPLETE.**
