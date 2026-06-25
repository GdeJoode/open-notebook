# Track P — sprint plan (P.2 completion)

> P.1 (store entity embeddings + backfill primitives + script + tests) is merged.
> P.2 is the remaining operator step: run the backfill against the live corpus so K.5
> semantic dedup activates, then verify the embedding band fires.

**Workflow**: track methodology — `implementer` preps verification harness/runbook;
the **live backfill is orchestrator-run with a checkpoint** (dry-run shown first, single
confirm). Main tree, no worktree.

---

## Phase P.2 — Live entity-embedding backfill + K.5-band verify (Integration · LIVE CHECKPOINT)

**Why**: entities persisted before the P.1 fix stored `embedding=[]`, degrading K.5 to
fuzzy-only. Backfilling vectors re-enables the semantic dedup band on the existing corpus.

**Deliverables**
- Backfill run evidence (dry-run count, dimension log, post-run count).
- Confirmed/added verification that K.5 `CandidateDedupService` embedding band fires when vectors are present (extend/confirm `test_backfill_entity_embeddings.py` + the K.5 band test).

**Procedure**
1. `python scripts/backfill_entity_embeddings.py --dry-run` → report N entities missing a vector. Show you the count + confirm the embedding model is reachable. **← single confirm.**
2. Optional smoke: `--limit 50` first, confirm dimension, then full run.
3. Full `python scripts/backfill_entity_embeddings.py`.

**Acceptance criteria**
1. Dry-run reports N missing before the run.
2. Backfill logs observed embedding dimension == **1024** (the REAL live pin: `mxbai-embed-large:latest` via Ollama — resolved via `ModelManager` default, never hardcoded). NOTE: the "768-dim I.G pin" referenced throughout the codebase/docs is **stale/incorrect** — live entity vectors are 1024-dim; see [[embedding-model-pin-1024]].
3. Post-run `--dry-run` reports **0** missing (idempotent / resumable proven).
4. K.5 embedding band fires on a vector-present pair where it previously degraded to fuzzy-only (test green or live probe).
5. Entity `count()` unchanged; only the `embedding` field is populated (no row churn).

**Mandatory commands**: `uv run --project apps/app-main pytest -k "backfill_entity_embeddings or dedup_embedding_band"`.

**Evidence**: dry-run N, dimension log line, post-run 0-missing, band-fires assertion.

**Branch**: `track/p2-embedding-backfill`. **Depends on**: none (P.1 merged). Independent of Track O — can run in parallel, but sequenced **after O** per your decision.

---

## Phase Z — Track closeout (Polish & Harden)

**Why**: close both tracks cleanly per methodology.

**Deliverables**
- `docs/tracks/O-relation-persist/status.md` + `escalations.md`: O.1 marked resolved via migration 62; O.2a/O.2b ledgered.
- `docs/tracks/P-entity-embeddings/status.md`: P.2 ledgered.
- RETRO notes for both tracks; roadmap/status pointers updated where O/P are referenced.

**Acceptance criteria**
1. Both status.md files show the track CLOSED with branch + commit refs.
2. O escalation marked resolved (not open).
3. Final audit: aggregated mandatory commands re-run clean; every O.2/P.2 acceptance criterion re-checked; deliverable files present.

**Depends on**: O.2a, O.2b, P.2.
