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
