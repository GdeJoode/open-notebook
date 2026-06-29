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
