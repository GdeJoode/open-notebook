# Track P — status

## P.2 — COMPLETE (2026-06-25): live backfill executed + verified
User authorized live writes (with checkpoint). Ran `scripts/backfill_entity_embeddings.py`
against staging:
- Dry-run: **135 active entities** missing a vector (the script scopes to active entities — the
  set K.5 dedup compares; the archived tail is excluded).
- Backfill: **135 backfilled, 0 failed**, observed embedding **dimension = 1024** via
  `ollama/mxbai-embed-large:latest` (**local**, never cloud — the real pin is 1024-dim, not the
  stale "768" in docs).
- Idempotency: re-run dry-run → **0 missing**.
- Verification: `test_backfill_entity_embeddings.py` 4/4, `test_candidate_dedup_service.py` 21/21
  (K.5 embedding band fires when vectors present).

**Track P CLOSED.** Forward fix (P.1) merged; live corpus backfilled; K.5 semantic dedup now has
vectors for every active entity.

---

## P.1 — store entity embeddings so K.5 semantic dedup works (2026-06-23)
**State**: forward fix + backfill + tests COMPLETE. Ready for review. Live
backfill run against the corpus is the remaining operator step (run the script —
see below); it was not executed here (no write to the live corpus without
confirmation).

**Branch**: `track/p-entity-embeddings`
**Commits**: `4492b7e..97b6ba3`
- `4492b7e` fix(persist): store computed entity embedding instead of [] (P.1)
- `d63eadb` feat(entity-repo): backfill primitives for entity embeddings (P.1)
- `cc539f0` feat(scripts): backfill entity embeddings for K.5 dedup (P.1)
- `97b6ba3` test(dedup): K.5 embedding band fires only when vectors present (P.1)

### Root cause (confirmed)
`EntityPersistenceService.persist_filtered_result` deliberately discarded the
entity's semantic vector: it stripped `properties["embedding"]` and stored
`embedding=[]` on the `Entity(...)`. The vector IS computed during extraction —
`entity_extraction_service._embed_entities` embeds `entity.text` through the
configured embedding model and stores it in `entity.properties["embedding"]`,
which rides through `model_dump()` into the persist input. K.5's
`CandidateDedupService` documents the consequence: an entity with `embedding=[]`
degrades to fuzzy-only, so the embedding band never fires.

### Embedding-flow verification
- **Source**: `properties["embedding"]`, set by `_embed_entities`
  (`apps/app-main/.../entity_extraction_service.py:626`) over `entity.text`.
  At persist, `canonical_name = text`, so the forward vector and the backfill
  vector (over `canonical_name`) are embeddings of the same string.
- **Model**: resolved through `get_embedding_service()` → `ModelManager` → the
  DB `default_embedding_model`. Never hardcoded. The forward fix and the backfill
  both use this exact path, so the dimension is whatever that model emits — the
  I.G 768-dim pin. The pin is unchanged (no model/dimension edit anywhere).
- **Dimension**: not re-shaped or validated in code; the backfill logs the
  observed dimension (`Embedding dimension = N`) on its first batch for operator
  confirmation that it matches 768.
- **Always present?** No — handled gracefully: `properties.get("embedding") or []`
  on the forward path; the dedup band already skips vectorless entities.

### Forward fix
`persist_filtered_result` now lifts `properties.get("embedding") or []` onto
`Entity(embedding=...)` (was `embedding=[]`), still excluding it from the stored
properties bag (no double-store of the ~3KB vector). No repo change — 
`upsert_entity` already persists `embedding = $embedding` (entity.py CREATE path).

### Backfill — `scripts/backfill_entity_embeddings.py`
Embeds the `canonical_name` of every active entity whose `embedding` is
empty/missing, through the SAME configured model, and writes it via
`EntityRepository.update_entity_embedding`. Idempotent (only loads the
still-missing set; re-runs skip done rows), resumable (commits each batch then
re-queries), batched, logs progress + the observed dimension.

Run it (inside the app environment — app container or `uv run --project apps/app-main`):
```
python scripts/backfill_entity_embeddings.py            # backfill all
python scripts/backfill_entity_embeddings.py --dry-run  # count only, no write
python scripts/backfill_entity_embeddings.py --batch-size 128
python scripts/backfill_entity_embeddings.py --limit 200  # cap one run
```

### New repo primitives (`EntityRepository`)
- `list_active_entities_missing_embedding(limit=None)` — active rows with
  `array::len(embedding ?? []) == 0`, ordered by `created_at` (stable cursor).
- `update_entity_embedding(entity_id, embedding)` — sets only `embedding` +
  `updated_at`; never touches the B.8 dedup key. Rejects an empty vector.

### Tests
- `test_entity_persistence_service.py` (54 → 56): embedding now reaches the
  first-class field when present; `[]` when absent; not double-stored.
- `test_candidate_dedup_service.py`: added `test_p1_embedding_band_fires_only_when_vectors_present`
  — same pair yields 0 proposals with `embedding=[]`, 1 (embedding-method) with
  vectors. Existing embedding-band coverage unchanged.
- `test_backfill_entity_embeddings.py` (new, DB-free): full-backfill, idempotent
  skip, `--limit` cap, dry-run-writes-nothing.
- `test_entity_repository_roundtrip.py` (requires_docker): list-missing →
  update → re-list idempotency; empty-vector rejection.

Run results (this branch):
- `pytest test_entity_persistence_service.py test_candidate_dedup_service.py -q`
  → **75 passed**.
- `pytest test_backfill_entity_embeddings.py -q` → **4 passed**.
- `pytest test_entity_repository_roundtrip.py -q` (docker) → **7 passed**.
- `python -c "from app_main.api.app import create_app"` → OK.
- ruff on all changed files → clean.

### Contracts held
- I.G 768-dim pin: model + dimension untouched; resolved via the existing layer.
- B.8 dedup/hash_id: unchanged (embedding is not part of the dedup key; the
  backfill writes only `embedding`/`updated_at`).
- K.5 over-merge guards (`must_not_merge`, discriminator guard): untouched — only
  the embedding band is re-enabled by giving it vectors.
- Relation/typing logic: not touched.

### Outstanding
- Live backfill execution against the ~1044-entity corpus (operator step; run the
  script). After it completes, `CandidateDedupService.propose_candidates(
  notebook_id=...)` should return more than the ~1 it returns today, as the
  embedding band now has vectors to compare.
