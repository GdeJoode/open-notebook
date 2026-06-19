# Phase I.D-3 — adversarial review (attempt 1)

> Branch: `track/i-d3-chunk-mutate`, diff `main...HEAD`
> Reviewer: adversarial-reviewer agent
> Plan: `docs/tracks/I-docling-studio/plan.md` §I.D-3 (4 ACs)
> Date: 2026-06-19

## VERDICT: REVISIONS_NEEDED → resolved

Transaction structure correct (single BEGIN/COMMIT, one call, parameterized, no injection).
Two MAJORs: atomicity error-propagation unproven + rollback test theater; multi-box split
geometrically wrong. Both resolved.

## Findings

### 1. MAJOR — atomicity error-propagation unverified; rollback test theater
`chunk_mutator.py` sent the transaction via `execute_query`, whose underlying `query()` returns
only the FIRST statement's result and never inspects per-statement `status:"ERR"`. So a failure in
statements 2-4 (DELETE / CREATE / order-UPDATE) could be swallowed → the op reports success on a
rolled-back (unchanged) DB. The rollback tests used a FakeDB that re-implemented merge/split
all-or-nothing, so they'd pass even if the code were non-transactional — they proved nothing.

### 2. MAJOR — multi-box split positions geometrically incorrect
`_split_positions` cut EVERY bbox at the same `fraction`. For a multi-line chunk (N>1 boxes), a 50%
split gave each half the top/bottom slice of every box, so each half's overlay highlighted regions
belonging to the other. Correct only for the single-box case (the only one tested).

### Minors
3. Malformed bbox dropped from the union (split). 4. Split on malformed chunk_id → 500 not 400.
5. cursorOffset 422-vs-400 boundary split across layers. 6. Frontend offset is a free-typed number,
not a rendered-text cursor (UX).

## High-risk spot assessment
1. Atomicity — structure REAL; error-propagation UNPROVEN + test THEATER → MAJOR 1.
2. Merge correctness/adjacency — CORRECT (server-side adjacency, concat by order, lower-order
   survives, order compacted in-txn).
3. Split correctness/bounds — bounds CORRECT, text CORRECT, positions WRONG multi-box → MAJOR 2.
4. order resequencing — CORRECT (in-txn).
5. chunk_id `:path` + injection — SAFE (parameterized); malformed-id 500 nit (Minor 4).
6. tests — merge/split happy-path genuine; rollback theater (Major 1); multi-box untested.
7. frontend wiring — disabled states correct, invalidates on success, tsc clean.

## AC scorecard (pre-revision)
- AC1 Merge: PASS. AC2 Split: PARTIAL (multi-box wrong). AC3 Atomicity: UNPROVEN. AC4 Audit: PASS.

---

## Attempt 1 — revisions

Mutator tests: **17 passed** (was 12). ruff clean.

| # | Severity | Resolution |
|---|---|---|
| 1 | MAJOR | Added `execute_transaction` + pure `_check_transaction_response` to `surrealdb_service.connection`: runs the BEGIN/COMMIT block via `query_raw` and inspects EVERY statement's status, raising on any `ERR` (or top-level error) — so a rolled-back transaction always propagates. Mutator now uses `execute_transaction`. Killed the theater: added 3 direct unit tests of `_check_transaction_response` (OK → results; statement ERR → raises; top-level error → raises) that don't touch the fake, plus the rollback tests now assert the mutator *propagates* the failure. Docstring made honest (live server rollback = documented SurrealQL guarantee, integration run deferred). |
| 2 | MAJOR | Rewrote `_split_positions` to map the text fraction onto BOX units: whole line-boxes go to their side; only the single straddling box is sliced vertically. Single-box reduces to the prior top/bottom cut. Added 2 multi-box tests (whole-box partition; straddling-box slice). |
| 3 | Minor | Malformed boxes now kept with the first half explicitly (union conserved), documented. |
| 4 | Minor | `_get_chunk` catches `ensure_record_id` parse errors → returns None → 400 "not found" (was 500). |
| 5/6 | Minor | Left as documented follow-ups (boundary is harmless; cursor-offset UX is a polish item for I.E). |
