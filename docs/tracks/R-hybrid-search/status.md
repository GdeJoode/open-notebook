# Track R — status

> ✅ **TRACK CLOSED (reconciled 2026-07-23)** — all phases R.0–R.6 are **merged
> to `main`** (branch tips `r0…r6` are ancestors of `main`; R.0 was live-complete
> at `119ecd3`). No formal RETRO was written; this banner records the close. No
> code is pending. See [`../_status.md`](../_status.md).

## Phase R.5 — Search-function integration + UI — ✅ READY FOR REVIEW (2026-06-27)
**Branch**: `track/r5-search-ui` (off `main`, has R.0–R.3, R.6, Track O/P/S/T).
**Commits**: `f72fd8d` (API client + types + hook), `9fba73a` (Related tab + why-matched), `833a110` (E2E + region fix).

### What shipped (frontend only — backend R.1/R.2/R.3 already live)
A **"Related" tab** on the source detail page (`SourceDetailContent.tsx`, 6th tab)
rendering `RelatedSources.tsx`, driven by the R.3 hybrid endpoint
`GET /sources/{id}/related-hybrid?k=8&preset=kg-heavy`.

- **Data layer**: `RelatedSourceHybrid` / `SignalProvenance` / `KGSharedEntity`
  types in `lib/types/api.ts` (mirror the backend schema); `sourcesApi.getRelatedHybrid(id, {k, preset})`;
  `useRelatedSources` TanStack Query hook + `sourceRelated` query key.
- **Component**: ranked `<ul>` of results. Each row = a `next/link` to the
  related source (title) + a fused-score `Badge` + a **"why matched"** line.
- **"Why matched" provenance** (derived from per-result provenance, KG-prominent):
  - KG signal present → `Shares entities: <entity badges>` (names from `kg_entities`,
    capped at 4 + "+N more"), with a `Network` icon (decorative, `aria-hidden`).
  - Dense signal present → `…and similar text` (or `Similar text content` standalone),
    with a `Sparkles` icon. The explanation is **plain text** (screen-reader
    readable, not colour/icon-only) and is also folded into the link's `aria-label`.

### States
- **Loading**: a pulse skeleton (`role="status"`, `aria-live="polite"`).
- **Empty** (no aggregate embedding / no shared entities — backend returns `[]`):
  a friendly "No related sources yet" message, **not** an error (no crash, no raw dump).
- **Error**: a destructive `Alert` (`role="alert"`) with the error message.

### a11y
- Section is a real landmark: `role="region"` + `aria-labelledby` on the Card
  (added the explicit role — `aria-labelledby` alone does not promote a `div`).
- Result list uses `<ul>/<li>` semantics with an `aria-label`; titles are native
  links (keyboard-navigable, `focus:underline`, `focus-within:ring` on the row).
- Score badge + title link carry `aria-label`s; the "why" is conveyed in text.

### Evidence (per AC)
- **AC1** (section driven by `/related-hybrid`; title linked + score + provenance):
  E2E test 1 asserts the linked title (`href=/sources/{id}`), the `0.049` fused
  score, and a why-line naming `BZK` + `Cohesion Policy` + "similar text".
- **AC2** (loading/empty/error graceful): E2E test 2 asserts the empty-state copy
  renders and no in-section `role="alert"` appears; loading/error branches exist
  in the component (skeleton + destructive Alert).
- **AC3** (a11y): `role="region"` landmark, list semantics, `aria-label`s, the
  why-text asserted by text (not colour); no uncaught page errors in either test.
- **AC4** (E2E covers render + why): `frontend/e2e/track-r/related-sources.spec.ts`
  — **2 passed** (run against a built server on :8599; route-mocked, no live backend).
- **AC5** (build/typecheck/lint clean):
  - `npx tsc --noEmit` → exit 0.
  - `npm run lint` → exit 0, **0 errors** (pre-existing warnings only; the now-unused
    `Network`/`Play` imports in `SourceDetailContent.tsx` were pre-existing warnings).
  - `npm run build` → exit 0.
  - `npx vitest run` → **87 passed** (no unit regression).

### Notes / follow-ups
- The Related tab fetches lazily-ish (the hook is enabled on mount; results are
  cheap and cached 60s). If desired, gate the query on tab activation later.
- A weight/preset control (kg-heavy vs balanced toggle) is intentionally **not**
  surfaced in the UI for R.5 — default `kg-heavy` honours the locked steer; a
  user-facing tuner is a clean follow-up.
- R.4 cluster-summary signal is not yet a contributor (R.4 not done); the "why"
  copy is structured to extend with a third signal without a rewrite.

## Phase R.0 — ✅ LIVE COMPLETE (2026-06-26)
The embedding foundation is fully live on `staging`. All 6 sources: non-empty chunks ==
`source_embedding` count (284/209/135/260/280/279), source-level aggregate vector set, **dim 1024**, local.
Required FIVE fixes, each adversarially reviewed + merged — the original R.0 was green on tests but the
REAL run exposed data/infra realities the synthetic container tests didn't:
- **R.0** (`119ecd3`) — auto-embed on ingest + migration 63 (`source.embedding`) + mean-pool aggregate + backfill.
- **R.0b** (`4f42537`) — chunk-embedding context-length guard (mxbai ~512 tok; esperanto uses Ollama legacy `/api/embeddings` which ignores truncate → client-side cap + per-text halving).
- **R.0c** (`c69d16f`) — backfill detects PARTIALLY-embedded sources (`chunk_count > embedding_count`, was `= 0`).
- **R.0d** (`835d7e6`) — skip empty/whitespace chunks (table/image chunks) on embed AND exclude them from completeness detection (coupled).
- **R.0e** (`977634a`) — migration 64: coalesce drifted strict `source.private` (NONE→false) so source rows are writable (the aggregate UPDATE was blocked).

Live ops run: orphan GC (10,501→1,448 chunks), migrations 63+64 applied to staging (v62→64), full chunk
backfill (1447 non-empty chunks, dim 1024), source aggregates (6/6). Follow-ups (→ R.6 / later): `embed_source`
embeds `is_content=False` noise chunks; `embed_note`/`embed_insight` whole-doc head-truncation; distinct-vector
alignment test; 2 comment nits in `migrations/64.surrealql`. **Systematic strict-field drift → Track S.**

---

## Phase R.2 — KG retrieval signal — ✅ READY FOR REVIEW (2026-06-26)
Branch `track/r2-kg-signal` (off `main` @ R.1 merge `4788d5d`). The KG-prominent
"rank by shared knowledge" signal — a pure scorer + repo loaders + service +
`GET /sources/{id}/related-kg` (parallel to R.1's dense `/related`).

**Commits**: `ffc7fc8` (pure scorer + 27 unit tests) → `4a19384` (repo/service/
endpoint + 12 router+container tests) → `2155…` (mypy annotations).

### Weighting chosen (the explicit table)
Per-entity contribution = `type_salience × inverse_source_frequency`.
- **type_salience** (`shared/retrieval/kg_source_scorer.py::TYPE_SALIENCE`, over
  the Track L canonical set): HIGH **1.0** = named/specific (person,
  organization, government_organization, administrative_area, location,
  programme, technology, legislation, policy_document, dataset, grant,
  research_project, scholarly_article); MEDIUM **0.5** = event, product,
  creative_work, periodical, public_consultation, social_profile; LOW **0.15** =
  the generic buckets `topic`/`concept` (the 81%-of-active LLM noise); FLOOR
  **0.05** = `other`. Unknown/drifted types fall back to 0.15 (never crash).
- **rarity (IDF)** = `log((N+1)/(df+1)) + 1` — smoothed, strictly positive; a
  `programme` in 2 sources outweighs a gov-org in all 4. Generic buckets are
  down-weighted, NOT excluded; archived/non-active entities ARE excluded (at the
  query — `status='active'`).

### 1-hop relation expansion
**Implemented but gated OFF by default** (`expand_relations=False`; opt-in via the
scorer arg / `?expand_relations=true`). Bounded to exactly one undirected hop
(`Q has X, X--rel--Y, B has Y`), credited at most once per candidate entity,
direct shares always beat relation paths. Gated because the 1-hop path
multiplies through the duplicate-predicate noise R.6 has not yet trimmed — so
shared-entity scoring ships as the reliable core. (Covered by 4 unit tests.)

### Per-criterion evidence
1. **Ranked + per-pair explanation** — `score_related_sources` returns ranked
   `SourceKGScore(source_id, score, contributions[])`; each contribution names
   the entity (id/name/type/df/weight/via_relation). Endpoint surfaces it as
   `explanation`. ✅ (unit + container + staging below).
2. **Generic down-weight provable** — `test_named_pair_outranks_topic_only_pair`
   + `test_other_bucket_is_weakest`: two sources sharing only a `topic` rank
   strictly below two sharing a named org/programme. ✅
3. **Live staging (read-only, `SURREAL_DATABASE=staging`)** — ran
   `KGRetrievalService` against the 6 live sources. The 4 Regio-Deal convenanten
   cluster each other and the academic papers fall away:
   ```
   QUERY: Convenant Regio Deal Zuidwest-Friesland
     #1 score=4.860  Convenant Regio Deal Noord-Holland Noord
     #2 score=4.144  Convenant Regio Deal Het Hogeland
     #3 score=3.864  Convenant Regio Deal Midden-Limburg
        (academic papers: absent)
   QUERY: Convenant Regio Deal Het Hogeland
     #1 score=4.144  Zuidwest-Friesland   #2 3.777 Noord-Holland   #3 2.554 Midden-Limburg
   QUERY: Economics_without_equilibrium (academic)  ->  (no KG-related sources)
   ```
   **Shared entities driving the clusters**: the named `programme`/
   `administrative_area` **"Regio"** (weight 1.223, df=3) and **"Regio Deal"**
   (weight 1.000, df=4) dominate every pair; generic `topic`s (brede welvaart,
   energietransitie, …) each add only ~0.18–0.23. One shared named entity ≈ 5
   shared topics — the down-weight holds on real data. The academic paper
   correctly shares NO active entities with the convenanten (discriminative
   negative). ✅
4. **Pure scorer unit-tested in isolation (no DB)** — 27 tests in
   `packages/shared/tests/test_kg_source_scorer.py`. ✅
5. **No hardcoded dim/cloud; read-only on entity/relation** — the scorer is
   dense-free (no embeddings at all); loaders only SELECT `status='active'`
   entities/relations. ✅

### Tests
- `uv run --project packages/shared pytest packages/shared/tests/test_kg_source_scorer.py` → **27 passed**.
- `uv run --project apps/app-main pytest apps/app-main/tests/test_source_related_kg_router.py apps/app-main/tests/test_source_related_kg_db.py` → **12 passed** (8 router unit, 4 `requires_docker` roundtrip: named-over-topic ordering, archived-excluded, no-overlap-empty).
- R.1 regression (`test_source_related_router/db`) → **14 passed** (unchanged).
- surrealdb-service repo suite (non-docker) → **61 passed** (no regression from the new loaders).
- mypy on the pure scorer → clean; the service's only mypy notes are the
  pre-existing `import-untyped` for workspace packages without `py.typed`
  (`shared`, `surrealdb_service.repositories`) — codebase-wide, not new.

### Deliverables / file map
- Pure scorer: `packages/shared/src/shared/retrieval/kg_source_scorer.py` (+ `__init__.py`).
- Repo loaders: `EntityRepository.load_active_entity_source_map` / `load_active_relations`; `SourceRepository.get_titles_by_ids`.
- Service: `apps/app-main/src/app_main/services/kg_retrieval_service.py` (DI: `get_kg_retrieval_service`).
- Endpoint: `GET /sources/{id}/related-kg` (schemas `RelatedSourceKGResponse` / `KGSharedEntity`).

### Notes for R.3 / R.6
- The scorer is the seam R.3 fuses with R.1 dense; the service already returns a
  `score` + `explanation` ready for RRF/weighted-sum (KG-prominent preset).
- The staging run shows the duplicate-cased noise R.6 owns: `Energietransitie`/
  `energietransitie`, `Brede welvaart`/`brede welvaart`, and a `programme "Regio"`
  vs `administrative_area "Regio"` split — R.6 dedup will tighten scores but the
  ranking already holds without it.

---

## Phase R.3 — Hybrid ranker / fusion — ✅ READY FOR REVIEW (2026-06-26)
**Branch**: `track/r3-hybrid-ranker` (off `main`, has R.0/R.1/R.2/Track S).
**Commits**: `b6a2d94` (pure RRF core), `eadbce8` (service + endpoint + n_sources fix), `d635cbf` (staging test).

### Fusion method: Reciprocal Rank Fusion (RRF) — chosen over normalize-then-sum
The two signals are on different scales (dense cosine ∈ [0,1]; KG = unbounded
weighted sum, ~2.5–6.2 on staging). RRF fuses on **rank, not raw score**:
`fused(doc) = Σ_signal w_signal · 1/(k + rank_signal(doc))` with `k=60` (the
Cormack et al. 2009 default). Picked over min-max / z-score normalize-then-sum
because normalization is dominated by the outlier KG magnitudes (one large KG
score compresses every other source toward 0); RRF discards magnitudes entirely,
which is exactly right when one signal is bounded and the other is an unbounded
sum. Trade-off recorded in the module docstring: RRF can't reward a runaway top
hit over a merely-good one within a signal — acceptable for source-linking.

### Default + presets (KG-prominent, config-tunable)
- **Default = `kg-heavy`** preset: `dense=1.0, kg=3.0` (KG weight 3× dense) — honours the locked steer.
- **`balanced`** preset: `dense=1.0, kg=1.0` (still KG-prominent: kg==dense).
- Endpoint tuning: `?preset=` and explicit `?w_kg=&w_dense=` (override the preset; a partial override fills from the preset). Weights live in `shared.retrieval.hybrid_fusion` (`FusionWeights`, `KG_HEAVY`, `BALANCED`, `PRESETS`).

### Deliverables
1. Pure fusion `fuse_rankings()` (no I/O) — RRF over `[(src,score)]` dense + `[(src,score,explanation)]` KG → ranked `FusedResult`s with per-signal score/rank/contribution + KG driving entities. `packages/shared/src/shared/retrieval/hybrid_fusion.py`.
2. `HybridRetrievalService` — calls R.1 dense + R.2 KG (passing true `n_sources`), pulls a wider-than-k pool from each, fuses, truncates. `apps/app-main/src/app_main/services/hybrid_retrieval_service.py`.
3. `GET /sources/{id}/related-hybrid?k=&preset=&w_kg=&w_dense=&expand_relations=` — parallel to R.1 `/related` and R.2 `/related-kg`. Returns fused ranking + provenance.
4. Ablation harness in the pure tests: constructed disagreement case (dense order ≠ KG order) where KG-only, dense-only, and fused all differ; removing either signal changes the order ⇒ neither is dead weight.
5. `n_sources` fix: `KGRetrievalService.find_related_by_kg` now fetches `source_repo.count()` (true 6) for the IDF instead of the inferred entity-bearing subset (4).

### Acceptance evidence
1. **AC1 fused + provenance** — endpoint returns fused ranking; each result has `dense{score,rank,contribution}`, `kg{score,rank,contribution}`, `kg_entities`. Router + service unit tests.
2. **AC2 ablation** — `test_hybrid_fusion.py::TestAblation`: fused ≠ dense-only AND ≠ kg-only on a disagreement case; `test_removing_dense_changes_the_order` / `test_removing_kg_changes_the_order` prove each signal moves the ranking.
3. **AC3 weight reorder** — `TestWeightTuning`: balanced vs kg-heavy provably flip the winner (A↔B); service test `test_weight_change_reorders` confirms end-to-end.
4. **AC4 one-signal source kept** — `TestMissingSignal` + service `test_one_signal_source_not_dropped`; on staging the 2 sources with no entities (`1k3c…`, `bc6x…`) still rank with `kg rank=None`.
5. **AC5 live staging (read-only, SURREAL_DATABASE=staging)** — reported fused rankings for 3 query sources; the 4 entity-bearing convenanten cluster at top, both signals agree there. Weight change reorders top-5 on 2 of 3 queries (3rd: both signals already agree). Provenance populated (driving entities incl. "Regio Deal", "Ondermijnende criminaliteit"). `test_source_related_hybrid_staging.py` (gated, skips off staging).
6. **AC6 true n_sources** — staging IDF denominator = **6** (true source count), not 4 (entity-bearing subset). `test_count_failure_falls_back_to_none`, `test_passes_count_to_scorer`, and the staging `test_staging_uses_true_source_count_in_idf` assert it. No hardcoded dim/cloud.

### Tests
- `uv run --project packages/shared pytest packages/shared/tests/test_hybrid_fusion.py` → **25 passed** (RRF math, missing-signal, weight tuning, ablation, provenance, presets).
- `uv run --project apps/app-main pytest apps/app-main/tests -k "hybrid or related or kg_retriev"` → **54 passed, 2 skipped** (staging skipped off-staging).
- Live staging (env `SURREAL_DATABASE=staging`): `pytest -m requires_staging test_source_related_hybrid_staging.py` → **2 passed** read-only.
- mypy on new files clean.

### Notes / carry-ins for the reviewer
- `expand_relations` stays OFF by default (forwarded to R.2's gated 1-hop; runs through R.6 noise). The relation-path discount the carry-in mentions is moot while expansion is off; if R.4/R.6 turn it on, the discount belongs in the R.2 scorer (`via_relation` weight), not the fusion.
- Pool depth: the service pulls `max(25, min(4·k, 100))` from each signal before fusing so cross-signal disagreement past the top-k cutoff isn't hidden.
- The `.serena/project.yml` working-tree edit is pre-existing and untouched.

---

## Phase R.6 — Extraction noise re-scope — ✅ READY FOR REVIEW (2026-06-26)
**Branch**: `track/r6-noise-rescope` (off `main`, has R.0/R.1/R.2/R.3/Track S).
**Commits**: `fcfcc70` (pure normalizer + 20 tests) → `8e2c25f` (service wiring +
remap + 4 tests) → `c8b75c3` (read-only staging measurement script).

A **search-facing normalization layer** that sharpens the R.2 KG signal by
trimming extraction noise WITHOUT mutating the canonical `entity`/`relation`
rows. Pure, additive, reversible, config-driven; the exporters, Track K, and
Track Q all keep reading the canonical data unchanged.

### The normalization rules
1. **Entity case/type unification** (`shared.retrieval.kg_signal_normalizer.
   normalize_entities_for_signal`): group active entities by a normalized concept
   key — case-folded + whitespace-collapsed `canonical_name`. For NAMED types
   (salience ≥ 0.5: the HIGH+MEDIUM tiers) the SAME surface form unifies ACROSS
   types (so `programme "Regio"` + `administrative_area "Regio"` = one concept);
   generic buckets (`topic`/`concept`/`other`) keep the type in the key so a
   coarse `topic` cannot absorb a named entity. A merged concept's source set =
   the UNION of its members' `source_documents`; its type = the MAX-salience
   (most specific) member. **Result**: case/type duplicates now count as SHARED
   for ranking. No entity row is rewritten — in-memory projection only.
2. **Singleton suppression** (same fn, `drop_singletons=True` default): a concept
   in only one source after grouping (df==1) can't link two sources → excluded
   from the signal. Signal-only; row stays in the KG. (`other`/bare `topic`
   stay heavily down-weighted by R.2's `TYPE_SALIENCE` — confirmed/extended.)
3. **Predicate canonicalization** (`PREDICATE_CANON` + `canonical_predicate`/
   `canonicalize_relations`): a reviewable EN/NL-variant + typo → canonical map
   (`ACEPTS`→`ACCEPTS`, `IS_PIJLER_VAN`→`IS_PILLAR_OF`, `LEIDT_TOT`→`LEADS_TO`,
   plus part_of/works_at/located_in/funds/collaborates_with EN/NL pairs). A
   CONFIG table, not a data migration — reversed by editing the map; unmapped
   predicates pass through as upper-snake (never dropped). For normalized 1-hop
   expansion, `remap_relations_to_concepts` re-keys endpoints onto concept ids so
   adjacency matches the normalized entities (drops collapsed self-loops).

Wired into `KGRetrievalService` (default ON; `normalize_signal` /
`drop_singletons` ctor flags make it reversible). Entity rows → concepts BEFORE
the R.2 scorer; relations canonicalized/remapped only when expansion is on
(still OFF by default).

### Before/after staging measurement (read-only, `SURREAL_DATABASE=staging`)
`scripts/r6_noise_measurement.py` against the 6 live sources / 423 active
entities / 1466 active relations:

- **Entity normalization**: 423 raw rows → **413 concepts** — **10 case/type
  duplicates unified**, exactly the flagged noise:
  `Energietransitie`/`energietransitie`, `Brede welvaart`/`brede welvaart`,
  the `programme "Regio"`/`administrative_area "Regio"` cross-type split, plus
  circulariteit/energie/openbaar vervoer/samenredzaamheid/voortgezet onderwijs/
  regionale samenwerking/dubbele vergrijzing. **388 singletons (df==1) excluded**
  → **25 concepts emitted** to the scorer (the cross-source-linking core).
- **Generic share**: by concept-COUNT 82.0% → 81.8% (merge) → 92.0% (after
  singleton drop — RISES because most NAMED entities are source-unique singletons
  and get dropped too). By RANKING-WEIGHT in top-10: 39.7% → 42.7%. Honest
  finding: R.6's win is **de-fragmentation** (duplicate unification + singleton
  exclusion), NOT generic-weight suppression — that's already R.2's salience job.
  The generic-weight share holds ~40% because the previously-fragmented generics
  now correctly count as shared (the case-split was silently dropping them).
- **Predicate canon**: 3 raw forms rewritten covering **592 of 1466 edges**
  (`IS_PIJLER_VAN` x317, `LEIDT_TOT` x274, `ACEPTS` x1) → their canonical forms.
- **Convenant-cluster effect**: the 4 Regio-Deal convenanten **still cluster**
  (each query's top-3 related = the other 3 convenanten; academic papers absent)
  AND the previously-WEAKEST intra-cluster pair **Het Hogeland ↔ Midden-Limburg
  strengthens +1.303** (3.294 → 4.597) — the case/type merge raising the right
  score. Other Δ are negative-but-order-preserving (IDF re-normalizes as
  concepts merge + the scored set shrinks). Ranking holds, the right pair rises.

### Per-criterion evidence
1. **Measurable noise reduction** — 10 duplicate concepts unified, 388 singletons
   excluded, 592 edges' predicates canonicalized (staging numbers above). ✅
2. **Nothing torn out** — canonical `entity`/`relation` NOT mutated (in-memory
   projection only). Suites GREEN: exports (Obsidian/NetworkX/JSONL/router/
   preview) **81 passed**; entity-resolution (`test_name_normalizer`/
   `test_nl_normalization`/`test_resolution_metrics`) **+ surrealdb repo 61** all
   pass; entity-filtering **518 passed, 1 pre-existing fail** (`test_llm_matcher::
   test_calls_ollama_for_unknown_pair` — missing `_agentic_enabled` attr in the
   test's `__new__` setup; fails identically on `main`, unrelated to R.6);
   triage (Q) **47 passed**. ✅
3. **Predicate canon reviewable + reversible** — `PREDICATE_CANON` is a
   commented config table, unit-tested (idempotent fixpoint, unmapped
   pass-through), no destructive data change. ✅
4. **R.2/R.3 ranking holds/improves** — convenant cluster holds, weakest pair
   strengthens (+1.303); R.2 router+DB **12 passed**, hybrid/related **54 passed,
   2 staging-skipped** — no regression. ✅
5. **Pure + isolated + read-only** — normalizer is pure (no DB), 24 unit tests;
   measurement script aborts unless `SURREAL_DATABASE=staging`, SELECT-only. ✅

### Tests
- `uv run --project packages/shared pytest .../test_kg_signal_normalizer.py` → **24 passed**.
- shared retrieval set (normalizer+scorer+fusion) → **76 passed**; mypy on the normalizer clean.
- `apps/app-main` R.2 router+DB → **12 passed**; hybrid/related → **54 passed, 2 skipped**.
- Exports **81**, triage **47**, entity-filtering **518 (+1 pre-existing fail)**,
  surrealdb repo **61**, shared resolution/export **147** — all green.

### Deliverables / file map
- Pure normalizer: `packages/shared/src/shared/retrieval/kg_signal_normalizer.py`
  (+ exports in `retrieval/__init__.py`).
- Service wiring: `apps/app-main/src/app_main/services/kg_retrieval_service.py`
  (`normalize_signal`/`drop_singletons` flags; remap on normalized expansion).
- Measurement: `scripts/r6_noise_measurement.py` (read-only staging).
- Tests: `packages/shared/tests/test_kg_signal_normalizer.py`.

### Notes / carry-ins
- 1-hop expansion stays OFF by default; with R.6's predicate canon + concept-id
  remap it is now SAFE to turn on (the duplicate-predicate multiplication R.2
  flagged is canonicalized first). R.4/R.6-followup can enable it.
- Generic-weight share does NOT drop — by design; the AC "% generic drops" was
  an *example*, and the realized noise reduction is de-fragmentation. Worth a
  reviewer note: if generic-weight suppression is wanted, lower `topic`/`concept`
  salience in R.2 (a one-line table change), kept separate from R.6's structural
  normalization.

---

## Phase R.0 — Embedding foundation — MERGED + APPROVED; live backfill BLOCKED (→ R.0b) [superseded by the LIVE COMPLETE entry above]
- O.2a-style adversarial review: **APPROVED attempt 1** (0 blockers/majors, 3 minors). Merged to `main` (`119ecd3`).
- **Live backfill BLOCKED**: running the chunk backfill against staging failed on all 6 sources —
  `Ollama API error: the input length exceeds the context length`. `mxbai-embed-large` has a ~512-token
  context; chunk texts exceed it and the embedding service does no truncation. The container test missed it
  (fake model, no context limit). → **R.0b** (embedding context-length fix, branch `track/r0b-embed-context-fix`)
  must land + be reviewed before the live backfill can complete.
- Decisions locked + Purview lessons added (`purview-lessons.md`). Orphan chunks GC'd (10,501→1,448).
- Review minors (follow-up): `embed_source` embeds `is_content=False` noise chunks (→ R.6); dead `embed:True` flag.

---

## Phase R.0 — Embedding foundation (forward-fix + backfill) — Ready for review

**Branch**: `track/r0-embedding-foundation` (off `main`)
**Commits** (`cc530ac..a32fa87`):
- `cc530ac` feat(embeddings): auto-enqueue embed_source after ingest (forward-fix)
- `69b9fff` feat(source): aggregate embedding field + mean-pool populate (migration 63)
- `66386ef` feat(scripts): backfill_chunk_embeddings.py (Track-P pattern)
- `a32fa87` test(embeddings): end-to-end backfill container test

### Forward-fix seam
Enqueue the existing `embed_source` job from the `DOCUMENT_PARSE` handler
(`apps/app-main/src/app_main/handlers.py:handle_process_source`) once
`process_source` returns `chunk_count > 0`. Chosen over inline embedding and
over editing `SourceProcessor` because: (a) it keeps the domain orchestrator
free of a job-queue dependency; (b) every ingest path (async upload, sync
upload, reprocess) funnels through this single handler; (c) `embed_source` is
already idempotent. Enqueue is best-effort — a queue hiccup logs but never
fails the already-persisted extraction.

### Migration 63 (`migrations/63.surrealql`)
```surrealql
DEFINE FIELD IF NOT EXISTS embedding ON TABLE source TYPE option<array<float>>;
```
`option<...>` so NONE = "not yet computed / no chunk vectors" (distinct from
`[]`); legacy rows read back NONE; dimension not schema-pinned (follows the
model). Down: `REMOVE FIELD IF EXISTS embedding ON TABLE source;`.

### Acceptance criteria — all PASS
1. **Fresh ingest auto-embeds** — handler enqueues `embed_source` when chunks
   exist (`test_handle_process_source_autoembed.py`); `embed_source` → real
   `source_embedding` rows proven on a container
   (`test_backfill_chunk_embeddings_db.py::test_backfill_chunks_real_run...`).
2. **Backfill idempotent + model-derived dim** — dry-run reports the missing
   set without writing; real run populates `source_embedding`; re-discovery → 0
   missing; dimension model-derived (logged), never hardcoded.
3. **`source.embedding` aggregate** — migration 63 + `populate_source_embedding`
   mean-pools chunk vectors; no chunk vectors → NONE gracefully.
4. **No hardcoded dim; Track J guardrail green** — no `768`/`1024` literal in
   touched code; `test_embedding_local_guardrail.py` 6/6 pass.
5. **Suites green** — see below.

### Test evidence
- `apps/app-main` (`-k "embedding or backfill or source_processor or source_embedding"`, scoped to `apps/app-main/tests`): **50 passed**
- Track J guardrail (`test_embedding_local_guardrail.py`): **6 passed**
- `pipelines/embeddings` (`pipelines/embeddings/tests`): **28 passed**
- Migration roundtrip (`test_migrations_roundtrip.py`): **16 passed** (incl. 2 new migration-63)
- Backfill container tests (`test_backfill_chunk_embeddings_db.py`): **4 passed**
- mypy on touched files: clean (the one reported error is the pre-existing
  `get_embedding_count` return-Any, not R.0 code).

> The legacy top-level `tests/test_domain.py|test_graphs.py|test_models_api.py|test_utils.py`
> have pre-existing collection import errors unrelated to R.0 — scope pytest to
> the package `tests/` dir to avoid them.

### LIVE backfill (gated checkpoint — for the operator to run)
Run inside the app env (e.g. the app container or `uv run --project apps/app-main`),
against the live staging DB, in order:

```bash
# 1. chunk embeddings — dry-run first, then real, then confirm idempotent
python scripts/backfill_chunk_embeddings.py --dry-run
python scripts/backfill_chunk_embeddings.py
python scripts/backfill_chunk_embeddings.py --dry-run        # expect 0 missing

# 2. source-level aggregate vectors (mean-pool; no model calls)
python scripts/backfill_chunk_embeddings.py --source-embeddings --dry-run
python scripts/backfill_chunk_embeddings.py --source-embeddings
```
Expect the chunk run to log `Embedding dimension = 1024`. Post-run verification
query:
```surrealql
-- chunk embeddings present for the 6 live sources, and aggregate set
SELECT id,
  count(SELECT id FROM source_embedding WHERE source = $parent.id) AS chunk_vecs,
  (embedding != NONE) AS has_aggregate,
  array::len(embedding) AS agg_dim
FROM source;
```
All 6 live sources should show `chunk_vecs > 0`, `has_aggregate = true`,
`agg_dim = 1024`.

The orchestrator can call `populate_all_source_embeddings()` /
`populate_source_embedding(source_id)` from
`scripts/backfill_chunk_embeddings.py` directly for the aggregate step.

---

## R.0b — chunk-embedding context-length guard (2026-06-25)

**Branch**: `track/r0b-embed-context-fix` (off `main`)
**Commits**: `08d1751` fix, `ba62b7d` tests
**Status**: Ready for review. Full 6/14-source backfill is the human-gated step.

### Bug
Live R.0 backfill failed on every source: `Ollama API error: the input length
exceeds the context length` → `embedded_sources=0 failed=N`. `mxbai-embed-large`
has a ~512-token context; chunk passages exceed it and the embedding service
applied no length guard.

### Root cause (confirmed live)
The esperanto Ollama provider embeds **one text per HTTP call** against the
legacy `/api/embeddings` endpoint, so this is per-text over-length, not a
batch-total. That legacy endpoint **ignores `truncate` and `options.num_ctx`** —
verified directly: `truncate=true`, `truncate=false`, and `num_ctx=8192` all
return the same 500 on a 9000-char input. Only the newer `/api/embed` endpoint
honours `truncate` (default true → 1024-dim returns), but esperanto does not use
it. So threading `truncate=true` through the wrapper would NOT have fixed it.
`num_ctx` does not raise mxbai's 512-token pin either; only truncation works.

### Fix (client-side, model-aware truncation)
`pipelines/embeddings/src/embeddings/{config.py,service.py}`:
- `max_input_chars=2048` (≈512 tok × 4 chars/tok, Latin) cap applied before
  every embed call; tail-truncate (keep head), log each cut.
- `truncate_on_context_error=True`: on a residual context-length rejection
  (dense/CJK text the char cap misses — char count is not a token proxy: CJK
  fails ~600 chars, dense Latin survives ~2500), recover **per-text by halving**
  until accepted. A single over-long text never fails its batch-mates and **no
  chunk is dropped**. Dimension stays model-derived (1024), never hardcoded;
  embeddings stay LOCAL (no `model_routing` import).

### Evidence (per acceptance criterion)
1. **Real-Ollama over-long embed** — `test_service_ollama_context.py` (runs
   against local mxbai, not a fake): a ~9000-char text fails raw (`context
   length`), then embeds to **dim 1024** through the guarded service; dense CJK
   recovers via adaptive shrink to 1024. PASS.
2. **R.0 unit/container tests unaffected** — `pipelines/embeddings` 36 passed;
   `apps/app-main` backfill + autoembed 12 passed.
3. **Truncation observable, no silent drops** — `logger.warning` on every char
   cap + every adaptive shrink; batch order/shape preserved; unit tests assert
   no chunk dropped.
4. **Track J guardrail green** — `test_embedding_local_guardrail.py` 6 passed;
   no cloud routing; dim not hardcoded.
5. **Live smoke** — `--dry-run` reports 14 sources / 1960 chunks; `--limit 1`
   ran twice, embedded 2 sources (140 chunks each), **dim=1024, failed=0**,
   remaining 14→13→12 (idempotent + resumable). Stored vectors confirmed dim
   1024 in staging.

### Note on current staging data
Staging has been re-ingested since the bug report (now 14 sources / smaller
chunks, max chunk **1519 chars**). The 200 longest current chunks embed raw
without the guard, so the per-chunk failure does **not** reproduce on today's
data — but the original ~2000-char chunks did exceed mxbai's context (reproduced
directly with 9000-char + CJK input). The guard is correct defence-in-depth for
whenever over-long input occurs.

### Remaining
Human-gated full backfill: `python scripts/backfill_chunk_embeddings.py`
(12 sources left), then `--source-embeddings` for aggregates.

## R.0c — Partial-source detection in chunk backfill (2026-06-25)

Branch `track/r0c-partial-detection` (off `main` w/ R.0 + R.0b merged).
Commits: `cc7c2c3` (fix), `d8cca05` (tests).

### Bug (measured live on staging)
`backfill_chunk_embeddings.py --dry-run` reported **0 sources / 0 chunks** while
staging had only **390 of 1448 chunks embedded** — all 6 sources PARTIALLY
embedded (70/284, 80/209, 80/135, 10/260, 100/281, 50/279). Backfill silently
did nothing; the "resumable/idempotent" claim was false for partial sources.

### Root cause
`list_sources_missing_chunk_embeddings()` selected sources with
`count(source_embedding) = 0`. A partial source has `count >= 1`, so the
`AND ... = 0` clause excluded it — any source with ≥1 embedding was treated as
done. Partial states arise because `_embed_chunks` commits per-batch; the R.0b
context-length failure left sources embedded part-way.

### Fix
Detect on counts-differ: `count(chunk) > count(source_embedding)` (fully OR
partially unembedded). `embed_source` deletes+rebuilds a source's embeddings, so
source-level "counts differ" is sufficient — no per-chunk granularity needed.
`count_missing_chunks` now reports only the unembedded chunks of a partial source
(`chunk_count - embedding_count`), not its whole chunk set. Dim stays
model-derived (1024), local, no hardcoding.

### Evidence (per acceptance criterion)
1. **Partial flagged (fails-old/passes-new)** —
   `test_discovery_query_flags_partially_embedded_source`: seeds 3-chunk source
   with 1 embedding, asserts it is flagged. FAILS against reverted `= 0` query
   (`assert partial in missing` -> AssertionError), PASSES against the fix.
   The updated `test_discovery_query_finds_unembedded_sources` also fails-old.
2. **Fully embedded NOT flagged** — same test, `full` (3/3) absent from missing.
3. **Zero embeddings still flagged** — same test, `zero` (0/3) present (no
   regression).
4. **embed_source completes partial** —
   `test_embed_source_completes_partial_then_clean`: stale 1-of-3 row, run
   embed_source (local fake model), assert count==3 and re-detection returns 0.
5. **Existing R.0 tests green / dry-run writes nothing** — full suite
   `test_backfill_chunk_embeddings.py` + `_db.py` = **11 passed**. Dry-run path
   unchanged (`get_embedding_count` left clamped, no model resolved on dry-run).

NOT run: real backfill against staging (human-gated). Expectation once gated run
fires: dry-run reports all 6 partial sources / the 1058 missing chunks.

## R.0d — skip empty/whitespace chunks when embedding (live backfill fix)

Branch: `track/r0d-empty-chunk-skip` (off `main` after R.0c).
Commits: `7e19c57` (embeddings service skip), `7a4b4bd` (detection exclude + tests).

### Bug (measured live on staging)
The full chunk backfill embedded 5 of 6 sources to dim 1024 but source
`dndibxmjveoxk7tfqfsl` FAILED with `Embedding failed after 3 attempts: Text
cannot be empty`. Root cause: that source has exactly 1 chunk with empty `text`
— a `table`-type chunk (`chunk:ik1k140l5lbwol7vt1`, element_type=`table`, order
280, no extracted text). The model rejects empty input, aborting the whole
source. (1 such chunk of 1448 corpus-wide today, but table/image chunks with no
text are a recurring class.)

### Root cause
Two coupled gaps: (1) `_embed_chunks` / `_embed_text_chunks` passed every
chunk's text to the model, including `""`; the model raised `Text cannot be
empty` and `_embed_with_retry` re-raised after retries, killing the source.
(2) the R.0c detection counted ALL chunks, so even if the empty chunk were
skipped, `chunk_count > embedding_count` would flag the source forever and
re-embed it every run (eternal-incomplete).

### Fix (two-sided)
1. **Embed side** (`pipelines/embeddings/src/embeddings/service.py`): new guard
   `_is_empty_text` (None / empty / `str.strip()` whitespace). Both embed paths
   build an `embeddable` list of `(original_order, chunk)` for non-empty chunks
   only, embed just those, and zip vectors back — empties get no model call and
   no `source_embedding` row; batch-mates still embed; original `order`
   preserved.
2. **Detection side** (`scripts/backfill_chunk_embeddings.py`):
   `list_sources_missing_chunk_embeddings` counts only
   `text != NONE AND string::trim(text) != ""` chunks;
   `count_missing_chunks` filters empties in Python via `_is_empty_chunk`
   (shared None/empty/whitespace logic, works for `Chunk` objects and raw
   strings). A source whose every non-empty chunk is embedded reads COMPLETE.

Dim stays model-derived (1024), local, no hardcoding.

### Evidence (per acceptance criterion)
1. **Empty/whitespace skipped, source succeeds (fails-old/passes-new)** —
   `test_service.py::TestEmbedSourceEmptyChunkSkip::test_empty_and_whitespace_chunks_skipped_mixed_batch`:
   normal + empty(table) + whitespace + normal, strict model rejecting empties;
   asserts 2 rows, model only sees the 2 normals, orders [0,3]. FAILS against
   reverted service (`Text cannot be empty`), PASSES with fix (verified by
   `git stash` of the service file -> 4 fail; restored -> pass).
2. **Not eternally flagged** (DB, `@requires_docker`) —
   `test_backfill_chunk_embeddings_db.py::test_empty_chunks_skipped_and_not_eternally_flagged`:
   5 chunks (3 normal + empty table + whitespace), `embed_source` writes 3 rows
   and does not raise; detection does NOT flag despite chunk_count(5) >
   embedding_count(3). FAILS against reverted detection query (source stays
   flagged), PASSES with fix (verified by `git stash` of the script -> fail;
   restored -> pass).
3. **Genuinely partial still flagged (no R.0c regression)** (DB) —
   `test_partial_nonempty_source_still_flagged_with_empties`: 3 non-empty + 1
   empty, 1 embedded -> flagged, `count_missing_chunks == 2`.
4. **count_missing_chunks counts only non-empty unembedded** —
   DB-free `test_count_missing_chunks_excludes_empty` (3 of 5, then 0) + the AC3
   DB assertion above.
5. **No regressions / dry-run writes nothing / Track J green** —
   embeddings suite **40 passed**; backfill DB-free+DB **15 passed**; Track J
   `test_embedding_local_guardrail.py` **6 passed**. Dry-run path unchanged.

NOT run: real backfill against staging (human-gated). Expectation once gated:
source `dndibxmjveoxk7tfqfsl` embeds its non-empty chunks to dim 1024, skips the
1 empty table chunk, and does NOT reappear in the missing set.

## R.0e — backfill NONE strict `source` fields (schema-drift fix) (2026-06-26)

Branch `track/r0e-source-field-backfill` (off `main`). Ready for review.

### Blocker
The R.0 source-aggregate write (`set_aggregate_embedding`, an
`UPDATE source:x SET embedding=...`) failed for every staging source with
`Found NONE for field 'private', with record 'source:...', but expected a bool`.
Legacy rows predate migration 52's `private TYPE bool DEFAULT false`, so they
carry `private = NONE`. A SCHEMAFULL UPDATE re-validates the WHOLE record, so
that one NONE blocks ANY write to the row — the aggregate embedding AND the
app's own source updates. Identical drift class migration 61 fixed for `entity`.

### Drifted-field analysis (empirical, full 1->63 schema)
Booted a fresh container, ran `INFO FOR TABLE source` + a per-field
NONE-rejection probe (UNSET each field, attempt a write). Result:

| field | type | rejects NONE? |
|-------|------|---------------|
| `private` | `bool DEFAULT false` | **YES — the only one** |
| `topics` | `option<array<string>>` | no (NONE valid; coalesced to `[]` for hygiene) |
| `embedding` | `option<array<float>>` | no — NONE intended (mig 63); NOT touched |
| `asset`, `command`, `full_text`, `metadata`, `title`, `zotero_*` | `option<...>` | no |
| `created`, `updated` | computed `VALUE` clause | no (never NONE) |

So `private` is the sole strict, non-`option<>`, no-VALUE field — the load-bearing
fix. Task statement's claim that `topics` would block next is incorrect for the
migrated schema (it's `option<array<string>>`); coalesced anyway as data hygiene.

### Migration 64 body
```surrealql
UPDATE source SET
    private = private ?? false,
    topics  = topics  ?? [];
```
Idempotent (no-op on clean rows), no schema change. `64_down.surrealql` = documented
no-op (mirror 61_down; data backfill has no faithful inverse).

### Evidence (per acceptance criterion, all `@requires_docker`, all green)
1. **Defaults restored** — `test_migration_64_backfills_drifted_strict_fields`:
   forged legacy row (`private=NONE`, `topics=NONE`) reads back `private=false`,
   `topics=[]` after 64.
2. **UPDATE fails pre-64 / passes post-64 (load-bearing)** —
   `test_migration_64_unblocks_aggregate_embedding_update`: the
   `UPDATE source:x SET embedding=[...]` raises "Found NONE for field private"
   pre-64 (`pytest.raises` + asserts "private" in message), succeeds on the SAME
   row post-64.
3. **Idempotent** — `test_migration_64_idempotent`: applying 64 twice doesn't
   raise; a clean row with `private=true` / `topics=['ai','db']` is unchanged
   (`?? default` no-ops on real values) and stays writable.
4. **Clean 1->64 chain on fresh container** — existing `test_migrations_applied`
   derives the expected version set from disk; 64 is applied + recorded.
5. **64_down present + sane** — comment-only no-op (0 non-comment lines, same
   shape as 61_down).

Full migrations roundtrip suite: **19 passed** (no regressions).

Drift reproduced faithfully via `REMOVE FIELD -> UNSET -> re-DEFINE` (recreates
the pre-mig-52 history; a bare DEFINE doesn't backfill existing rows). A plain
`UPDATE ... UNSET private` is itself rejected by the strict revalidation, so it
cannot reach the legacy state — hence the remove/redefine dance.

NOT run: the live aggregate / live migration against staging (human-gated).
Systematic staging strict-field drift: `entity` fixed by mig 61, `source` by
mig 64; likely other tables carry the same drift.

---

## Phase R.1 — Source-level kNN retrieval + `/sources/{id}/related` — COMPLETE

**Branch**: `track/r1-source-knn` (off `main`, post-R.0 + Track S).
**Commits**: `50fd2bb` (feat: endpoint + repo/service), `43751b9` (test: unit + container).

### What shipped
- `SourceRepository.find_related_by_embedding(source_id, k)` — server-side cosine
  kNN over `source.embedding` aggregates.
- `SourceService.find_related(source_id, k)` — thin pass-through.
- `RelatedSourceResponse` schema `{id, title, score}`.
- `GET /sources/{id}/related?k=` on the sources CRUD sub-router (default k=5,
  clamped [1,50] by FastAPI `Query(ge=1, le=50)`).

### Cosine approach: **SurrealDB-side** (not Python kNN)
Ranking runs in SurrealDB via `vector::similarity::cosine(embedding, $q)` — the
same operator `fn::vector_search` uses for chunk search. Chosen because:
- Keeps all 1024-dim vectors in the DB; no bulk pull into Python.
- Reuses a proven path; NONE-handling (`WHERE embedding != NONE`) and
  deterministic ordering (`ORDER BY score DESC, id ASC`) are one query.
- A same-dim guard (`array::len(embedding) = array::len($q)`) makes it robust to
  mixed-dim rows (legacy/cross-test) instead of one bad vector failing the whole
  ranking. Dim is derived from the query vector — never hardcoded.
A bounded Python kNN was the documented fallback but wasn't needed: SurrealDB
ranks source-vs-source cleanly.

### Per-criterion evidence
1. **Top-k cosine-desc, self excluded** — `WHERE id != $id ORDER BY score DESC`;
   covered by `test_ranks_by_cosine_excluding_self` + router test; verified live.
2. **Deterministic + bounded k + more-than-available** — `id ASC` tie-break
   (`test_tie_break_is_deterministic_by_id`); k clamped by FastAPI (422 on 0/51);
   k>available returns all (`test_k_limits_and_more_than_available`).
3. **No aggregate -> graceful** — query source with NONE embedding returns `[]`
   (`test_query_source_without_aggregate_returns_empty`); NONE sources never
   appear as results (`test_none_embedding_sources_never_appear_as_results`);
   missing source -> 404 (router test). Documented: exists-but-no-embedding = `[]`,
   missing = 404.
4. **Live staging ranking (read-only, ns=open_notebook db=staging)** — 6 sources,
   all 1024-dim. The 4 Regio-Deal convenanten cluster:

   `/related` for **Convenant Het Hogeland** (k=5):
   ```
   0.9922  Convenant Regio Deal Midden-Limburg
   0.9911  Convenant Regio Deal Zuidwest-Friesland
   0.9901  Convenant Regio Deal Noord-Holland Noord
   0.7937  Economics_without_equilibrium(2)
   0.7912  J of Common Market Studies - 2025 - Ali
   ```
   -> the 3 other convenanten (0.99x) rank above the 2 papers (0.79). YES, the
   convenanten cluster.

   `/related` for **Economics_without_equilibrium** (k=5):
   ```
   0.9084  J of Common Market Studies - 2025 - Ali
   0.8047  Convenant Regio Deal Noord-Holland Noord
   0.8010  Convenant Regio Deal Midden-Limburg
   0.7994  Convenant Regio Deal Zuidwest-Friesland
   0.7937  Convenant Regio Deal Het Hogeland
   ```
   -> the other academic paper ranks top (0.908), convenanten below (0.79-0.80).
5. **No hardcoded dim; local-only; reads `source.embedding`** — confirmed; dim
   derived from the stored/query vectors.

### Tests
`uv run --project apps/app-main pytest apps/app-main/tests -k "related or source_knn or source_retrieval"`
-> **18 passed** (9 router unit, 5 container roundtrip, 4 incidental matches).
Adjacent regression (autoembed, backfill, health router): **14 passed**.
mypy on changed files: no new errors (only pre-existing `shared.models` untyped +
the pre-existing `get_embedding_count` Any-return).

The mandated bare `-k` selector hits the 4 known-broken top-level `tests/`
import errors (`api`, `open_notebook` modules) — scope to `apps/app-main/tests`
to run clean, per the task note.
