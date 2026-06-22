# Track K — Entity resolution & deduplication — status

Append-only ledger. One row per phase attempt.

| Phase | Title | PR | Branch | Date | Status |
|---|---|---|---|---|---|
| — | (planning) | — | — | 2026-06-22 | track-planner producing plan.md |
| K.1 | NL-aware normalizer + precision guard + measurement harness | (pending) | `track/k1-nl-normalizer` | 2026-06-22 | implemented — all 9 ACs green, ready for review |
| K.1 rev3 | Option A: no cross-type NAME collisions; name-only false-merge gate | (pending) | `track/k1-nl-normalizer` | 2026-06-22 | implemented — 0 name-only false-merges, ready for review |
| K.1 rev4 | Articles + spelling only; NO content-prefix strip (ministerie van collides cross-type) | (pending) | `track/k1-nl-normalizer` | 2026-06-22 | implemented — 0 name-only false-merges, −14 fragmentation, ready for review |
| K.2 | Curated NL gov-org alias table + extensible override config | (pending) | `track/k2-org-aliases` | 2026-06-22 | implemented — all 7 ACs green; full-form keys (no blind strip); 0 name-only false-merges over combined corpus; K.1 1346 → K.2 1337 (BZK 6, VRO 3); ready for review |
| K.3 | Retroactive canonicalization — dry-run plan + reviewable merge | (pending) | `track/k3-retroactive-merge` | 2026-06-22 | implemented — all 8 ACs green; dry-run-default + idempotency + soft-merge guards; ID-based relation re-pointing; person/org separation verified; 43 tests green (11 unit + 5 roundtrip + B.8 regression + migration-54). Ready for review |
| K.4 | TOOI + Crossref vocabulary lookup → external_ids + aliases | (pending) | `track/k4-vocabulary` | 2026-06-22 | implemented — all 7 ACs green; precision guard (single high-confidence match only); resolve_external_ids signature unchanged + 45 Track D exporter tests green; migration 55 idempotent; TOOI source = K-D2 (verified seed shipped, exact bulk URL/cadence flagged). 63 tests green. Ready for review |
| K.5 | Fuzzy/embedding candidate dedup + review queue + alias overlay | (pending) | `track/k5-fuzzy-dedup` | 2026-06-22 | implemented — all 7 ACs green; type-aware bucketing + review band (queued, never auto-applied) + force-split hard veto / force-merge per-notebook include; thresholds tuned over must-NOT corpus (fuzzy auto 0.93 / review 0.86, embedding 0.95 / 0.90) → typo class auto-merges (~0.95), every must-NOT near-miss rejects (<0.86), 0 false auto-merges; migration 56 idempotent; B.8 + K.1-K.4 contracts intact. 38 K.5 tests green (18 unit candidate-dedup+overlay, 4 overlay roundtrip, 16 migration roundtrip). Ready for review |

## Phase K.4 — TOOI + Crossref vocabulary reconciliation → external_ids/aliases — 2026-06-22

**Branch**: `track/k4-vocabulary` (off main, with K.1-K.3). Commits per logical unit (model+migration → providers → repository → reconciler → external_ids body → router/DI → entity-repo helper → lint).

**Delivered**
- `migrations/55.surrealql` (+`_down`) — `DEFINE FIELD IF NOT EXISTS external_ids ON entity FLEXIBLE TYPE array DEFAULT [];` + defensive `idx_ref_name`. ADDITIVE/`IF NOT EXISTS` only — does not touch `canonical_name`/`hash_id`/`entity_type` (B.8 drift caution). `reference_entity` already carried all provider fields from migration 41, so no field additions there.
- `packages/shared/src/shared/models/entity.py` — `external_ids: list[str] = Field(default_factory=list)` + `ensure_list` validator coverage.
- `packages/shared/src/shared/vocabulary/{__init__,provider,http_client,tooi_provider,crossref_provider}.py` — `VocabularyProvider` protocol + `VocabMatch`; a disciplined HTTP client (timeout + rate-limit + per-authority cache + fail-soft); TOOI provider (verified-live ministry seed → `reference_entity`, documented bulk-file ingest, name/alias lookup); Crossref provider (polite-pool DOI resolution, scholarly-types only).
- `packages/surrealdb-service/.../repositories/reference_entity.py` — `ReferenceEntityRepository` (`upsert` idempotent on the migration-41 `(canonical_name, source_vocabulary)` UNIQUE key, `bulk_load`, `lookup_by_name`, `lookup_by_alias`, `count`, `last_validated`). Registered in repos `__init__`.
- `apps/app-main/.../services/entity_resolution/vocabulary_reconciler.py` — `reconcile_entity(entity) -> ReconcileResult`; **PRECISION GUARD: auto-links external_ids/aliases ONLY on a single high-confidence candidate**; ≥2 distinct-URI candidates → recorded, NO write. Same-URI agreement collapses to one (still links). Fail-soft per provider.
- `packages/surrealdb-service/.../repositories/entity.py` — `update_external_ids(id, ext, aliases)` union helper (the reconciler's persistence seam; never touches the B.8 key).
- `packages/shared/src/shared/utils/external_ids.py` — body implemented: returns `entity.external_ids` (reconciled) as a deduped defensive copy. **Signature/return type UNCHANGED** (`Entity -> List[str]`); empty-input still `[]`.
- `apps/app-main/.../api/routers/vocabulary.py` — `POST /api/vocabulary/refresh` (idempotent TOOI sync; Crossref reported on-demand), `GET /api/vocabulary/status` (per-provider row count + `last_validated`). Registered in `app.py`; DI factory `get_reference_entity_repo`.

**Network discipline (the NIM/fair-use lesson)**: every external call goes through `VocabularyHTTPClient` — bounded timeout, min-interval rate limit, per-authority TTL cache, and fail-soft (any transport/HTTP/parse error → `None` → caller treats as no-match, NEVER raises). Crossref uses the polite pool (`User-Agent` with a `mailto:` contact). ONE gentle live Crossref sanity call confirmed the response shape (`message.items[].DOI/score/title`); ALL TESTS MOCK HTTP via `httpx.MockTransport` — no live network in CI.

**TOOI source decision (K-D2)**: NON-BLOCKING. TOOI verified reachable as content-negotiated RDF/Turtle per identifier (`identifier.overheid.nl/tooi/id/ministerie/<code>`, e.g. `mnre1034` = BZK) with `afkorting`/`officieleNaam*` fields. The exact full-vocabulary **bulk-download URL** was not discoverable without docs, so the provider ships (a) a documented `tooi_organisations.json` bulk-file loader (`TOOI_BULK_SOURCE` env) and (b) a verified seed of real ministry records as the default so AC1-2 pass offline. Open question (exact bulk URL/format + refresh cadence/trigger) flagged in `escalations.md` (K-D2).

**Precision-guard behaviour (AC3)**: two equally-confident matches (e.g. a `Groningen` province vs municipality reference) → `linked=False`, `external_ids` stays `[]`, both kept on `result.candidates`, `reason="ambiguous_multiple_candidates"`. A single match clearing the threshold links; below-threshold does not. This is the K-style over-merge backstop applied to vocabulary linking (over-linking into the wrong entry is silent bad data).

**Tests (63 green)**
- `packages/shared/tests/test_tooi_provider.py` (9): seed load, refresh idempotency (no dup rows, AC6), canonical + alias lookup with non-null URI (AC1-2), miss/empty/fail-soft, bulk-file ingest, missing-file→seed fallback.
- `packages/shared/tests/test_crossref_provider.py` (8): DOI resolution (AC4), non-scholarly skips network, low-score drop, two-strong-hits surface both, transport/5xx fail-soft, refresh no-op, cache.
- `apps/app-main/tests/test_vocabulary_reconciler.py` (7): single-match auto-link (AC2), two-candidate NO link (AC3, precision), same-URI collapse links, below-threshold, no-match, failing-provider ignored, repo persistence.
- `packages/surrealdb-service/tests/test_reference_entity_repository.py` (5, `@requires_docker`): bulk_load + name/alias lookup roundtrip (AC1), refresh idempotency (AC6), `last_validated` stamp, `update_external_ids` union + idempotency.
- `packages/shared/tests/test_external_ids_stub.py` (12, upgraded): reconciled entity emits URIs; empty-canonical/un-reconciled → `[]` (AC5 contract preserved); result-is-a-copy; signature pinned.
- Migration `test_migrations_roundtrip.py` (12 green) — 55 + 55_down idempotent.
- **B.8 regression**: `test_entity_persistence_service.py` + `test_entity_repository_roundtrip.py` + `test_entity_merge_roundtrip.py` green unmodified (hash_id/upsert contract intact). **Track D regression**: 45 exporter tests (`test_obsidian_export_service.py` + `test_export_preview.py` + `test_exports_router.py`) green — `resolve_external_ids` contract preserved.

**Validation**: the plan's command suite + migration roundtrip + `from app_main.api.app import create_app` (app imports, both `/api/vocabulary/*` routes registered) → all green. ruff clean on changed src.

**Notes for reviewer**
- `httpx` added to `packages/shared` deps (providers live in shared; previously app-main-only). `respx` added to shared's dev extra but tests use `httpx.MockTransport` (no respx runtime dep on the test path).
- The reconciler is a standalone service (no auto-call at persist time — the plan's optional persist-time hook is left default-off / out of scope for K.4; batch reconcile is a K.5/maintenance step). `entity_persistence_service.py` was NOT modified.
- No live TOOI bulk refresh or live reconcile was run against a real DB — testcontainers only; the live smoke is a K-D2 follow-up once the bulk source is confirmed.

## Phase K.2 — Government-org abbreviation alias table + extensible alias config — 2026-06-22

**Branch**: `track/k2-org-aliases` (off main, which has APPROVED K.1). Commits: source modules → corpora + tests.

**Spec correction applied**: the plan's "strip `ministerie van` first then expand `bzk`" ordering is stale (K.1 rev4 removed ALL content-prefix stripping — it collided cross-type). K.2 instead keys the **full surface forms directly** in the alias table (`ministerie van bzk` is its own key), never relying on prefix stripping, so it cannot collapse a ministry onto a bare different-typed token.

**Delivered**
- `packages/shared/src/shared/utils/org_aliases.py` — `_GOV_ORG_ALIASES: dict[str,str]` (normalized full-form keys → normalized canonical) + `expand_org_alias(name, *, aliases=None)` (exact-whole-string match only; unknown → pass through untouched, no fuzzy guessing). Curated 13-ministry set: BZK, VRO, EZK, OCW, VWS, IenW, JenV, SZW, LNV, BuZa, Financiën, Defensie, AZ. Per ministry: abbreviation + `ministerie van <abbrev>` + bare full-form + `ministerie van <full-form>` → one canonical. `minister van …` (person) role forms deliberately NOT keyed.
- `packages/shared/src/shared/config/alias_overrides.py` (+ `config/__init__.py`) — 3-layer loader (built-in floor → file `ONB_ALIAS_OVERRIDES_PATH` JSON → DB-overlay seam `set_db_overlay` for K.5). Validates on load: empty key/value rejected (logged, skipped); keys/values normalized through the same pre-alias pipeline (`normalize_alias_key`). Cached; `reload_alias_overrides()` rebuilds.
- `packages/shared/src/shared/utils/nl_normalization.py` — added `normalize_alias_key` (shared pre-alias pipeline; placed here to break the name_normalizer↔alias_overrides import cycle).
- `name_normalizer.py` — composes `expand_org_alias(name, aliases=get_resolved_aliases())` as the FINAL stage, after article-strip + spelling (alias keys are post-K.1 forms). Public signature unchanged; docstrings + doctests updated.
- `packages/shared/tests/test_org_aliases.py` — expansion, exact-match-only (noisy `vro (...)` untouched), distinct-canonical precision, type-safety (OCW≠onderwijs, no person-role keys), override config (AC4 honoured, empty rejected, malformed-file ignored, DB-overlay seam).

**Curated alias set** (abbrev → canonical full-form, each a DISTINCT org name, never a bare concept):
BZK→binnenlandse zaken en koninkrijksrelaties · VRO→volkshuisvesting en ruimtelijke ordening · EZK→economische zaken en klimaat · OCW→onderwijs cultuur en wetenschap · VWS→volksgezondheid welzijn en sport · IenW→infrastructuur en waterstaat · JenV→justitie en veiligheid · SZW→sociale zaken en werkgelegenheid · LNV→landbouw natuur en voedselkwaliteit · BuZa→buitenlandse zaken · Fin→financien · Def→defensie · AZ→algemene zaken.

**Measured over the frozen 1402-entity Convenant fixture**
- Distinct-canonical: **K.1 1346 → K.2 1337** = **further −9** (V1 1360 → −23 cumulative). Drop comes from the BZK + VRO org-form merges.
- BZK cluster (`binnenlandse zaken en koninkrijksrelaties`): **6 surface forms → 1** (`BZK`, `ministerie van BZK`, `Ministerie van Binnenlandse Zaken en Koninkrijk(s)relaties` ×2 spellings, both bare spellings).
- VRO cluster (`volkshuisvesting en ruimtelijke ordening`): **3 → 1** (`VRO`, `Ministerie van Volkshuisvesting en Ruimtelijke Ordening`, bare full-form).
- **Name-only false-merges over the FULL combined must_not_merge corpus: 0.** Unmerged over must_merge: 0.
- Org/concept non-collision confirmed: `Ministerie van Onderwijs` (`ministerie van onderwijs`) ≠ `Onderwijs` (`onderwijs`); `Ministerie van OCW` (`onderwijs cultuur en wetenschap`) ≠ `Onderwijs`. `BZK` ≠ `EZK`.

**AC ledger**: AC1✓ (BZK forms collapse) AC2✓ (VRO) AC3✓ (BZK≠EZK; `expand_org_alias("xyz")=="xyz"`) AC4✓ (override `{"min az":"algemene zaken"}` honoured, no code change) AC5✓ (0 false-merges over combined corpus) AC6✓ (further drop; BZK/VRO each single canonical) AC7✓ (B.8 hash derive-rule unchanged).

**Corpus changes**
- `must_merge.jsonl`: +5 org-form pairs (`Ministerie van BZK`↔`BZK`↔`Binnenlandse Zaken en Koninkrijksrelaties`; `VRO`↔full-form; `Ministerie van VRO`↔full-form).
- `must_not_merge.jsonl`: +`BZK`↔`EZK`, +unknown-abbrev (`XYZ`↔`Onderwijs`), +`Ministerie van OCW`↔`Onderwijs` (org/concept), +`Minister van VRO`↔`Ministerie van VRO` (person/org). **Removed** the stale K.1 line `Ministerie van BZK`(org)↔`BZK`(person) — K.2 now legitimately merges that org-form pair (it is in must_merge), and the person/org role split stays protected by `Minister van BZK`↔`Ministerie van BZK` (kept) and the new `Minister van VRO`↔`Ministerie van VRO`. See escalations.md note.

**Regression**: `packages/shared/tests/` 299 passed; combined validation (shared + persistence + entity-repo roundtrip + notebook-merge) **322 passed**. B.8 `TestB8HashContract` green (derive-rule unchanged). ruff + mypy clean on changed src + test files. (Pre-existing unrelated failure `test_llm_matcher.py::test_calls_ollama_for_unknown_pair` confirmed present on clean tree — not a K.2 regression.)

## Phase K.1 — NL-aware normalizer + precision guard + measurement harness — 2026-06-22

**Branch**: `track/k1-nl-normalizer` (off main). Commits: NL rules (`nl_normalization.py` + `name_normalizer.py` compose) → harness + fixtures + tests.

**Delivered**
- `packages/shared/src/shared/utils/nl_normalization.py` — `_LEADING_ARTICLES`, `_ROLE_ORG_PREFIXES` (ordered longest-first), `strip_leading_noise` (tail-preserving guard, `_MIN_TAIL_LEN=2`), `canonicalize_spelling` (curated dict: `koninkrijkrelaties`→`koninkrijksrelaties`).
- `packages/shared/src/shared/utils/resolution_metrics.py` — `measure_fragmentation` → `FragmentationReport` (distinct-canonical count + size histogram + merged-cluster member lists), `count_false_merges`, `count_unmerged_must_merge`. DB-free; normalizer is injected so the same code measures baseline vs candidate; `entity_type` folded into the cluster key (homograph guard).
- `name_normalizer.py` — composes V1 → strip_leading_noise → canonicalize_spelling. Public signature unchanged; docstring rewritten (V1-stub framing dropped). `Apple Inc.` doctests/tests still green.
- Fixtures: `tests/fixtures/entity_resolution/{convenant_entities.jsonl (1402 entities, frozen live dump from the 4 measured Convenant sources), must_merge.jsonl (9 pairs), must_not_merge.jsonl (8 adversarial pairs)}`.
- Tests: `test_nl_normalization.py`, `test_resolution_metrics.py` (incl. per-pair parametrized over-merge canary), extended `test_name_normalizer.py` (NL ACs + B.8 hash-contract).

**Measured (AC8) over the frozen 1402-entity Convenant fixture**
- Distinct-canonical: **1360 (V1) → 1314 (K.1)** = **−46**.
- BZK full-form cluster (`binnenlandse zaken en koninkrijksrelaties`, type `other`): **6 surface forms → 1** canonical key (article/role-prefix/spelling variants collapsed). The `organization`-typed slice collapses a further 3→1. Across both type-keys the BZK/Binnenlandse-Zaken family drops from ~14 surface forms to 2 keys (split only by `entity_type`).
- False-merges over `must_not_merge.jsonl`: **0** (AC7). Unmerged over `must_merge.jsonl`: **0** (AC6).

**AC ledger**: AC1✓ AC2✓ AC3✓ AC4✓ AC5✓ AC6✓ (0 unmerged) AC7✓ (0 false-merge) AC8✓ (−46, BZK 6→1) AC9✓ (hash derive-rule unchanged).

**Regression**: `packages/shared/tests/` 264 passed. B.8 `test_entity_persistence_service.py` + `test_entity_repository_roundtrip.py` 15 passed. ruff clean on changed files.

**Notes / tricky pairs**
- The K.1 must_merge corpus deliberately does NOT include the `BZK ↔ Binnenlandse Zaken en Koninkrijksrelaties` (abbreviation↔full-form) pair — that crosses two distinct normalized keys (`bzk` vs `binnenlandse zaken...`) and is K.2's curated-abbreviation job, not K.1's prefix-strip job. Including it here would have falsely failed AC6.
- `Provincie Groningen` ↔ `Groningen` (the city): both normalize to the bare name `groningen`; distinctness is carried by `entity_type` (`organization` vs `location`), which the harness folds into the cluster key. Fixture pair encodes the differing types; documented in `test_nl_normalization.py::TestTailPreservationCanary`.
- `_MIN_TAIL_LEN=2` is the guard that keeps `Gemeente a`-style 1-char tails from being stripped to noise while letting `bzk` (3) through.

## Phase K.1 rev3 — Option A: no cross-type NAME collisions — 2026-06-22

**Decision** (after 3 review cycles): K.1 normalization must NOT create cross-type
name collisions. Relations resolve endpoints by name ALONE
(`WHERE canonical_name = $name`, no type filter), so a name shared by two
different-typed real entities corrupts the graph regardless of the
`(canonical_name, entity_type)` dedup key. The rev2 over-merge canary checked at
`(name, type)` and so missed name-only collisions; rev3 fixes the criterion.

**The fix**
- `nl_normalization.py::_ROLE_ORG_PREFIXES` reduced to the org-only set:
  **`("het ministerie van ", "ministerie van ")`**. REMOVED: `minister van`,
  `staatssecretaris van`, `de minister van`, `de staatssecretaris van` (person
  leaders — `Minister van BZK` is a PERSON), and `gemeente`, `provincie`
  (municipality leaders — `Gemeente Groningen` the ORG ≠ `Groningen` the city).
  Articles (`de`/`het`/`een`) and `ministerie van` kept (org → org-tail, same
  type, no cross-type collision).
- `resolution_metrics.py::count_false_merges` now compares at the **NAME level**
  (ignores `entity_type`) — the criterion relations actually depend on. Added
  `count_false_merges_with_type` for `(name, type)` fragmentation diagnostics;
  `measure_fragmentation` still keys on `(name, type)` (mirrors persistence
  dedup key).
- `must_not_merge.jsonl`: added/encoded the cross-type pairs that must stay
  distinct at name-only — `Minister van BZK` ↔ `Ministerie van BZK`,
  `Gemeente Groningen` ↔ `Groningen`, `Provincie Groningen` ↔ `Groningen`.
  Gate: **0 name-only collisions** across all of them.
- `must_merge.jsonl`: removed the person-minister↔org pairs that are now
  (correctly) distinct (`Minister van BZK` ↔ `BZK`; `De Minister van … KR` ↔ …).
  Kept org forms + article variants.
- `notebook_merge_service.py`: KEEPS the rev2 `(name, type)` bucket key (correct
  + consistent with persistence). With zero name-only cross-type collisions the
  `name_to_canon` relation-endpoint rewrite is now safe. Regression test
  `test_same_name_different_type_stays_distinct` kept (now the two normalize to
  DIFFERENT strings → trivially distinct); added
  `test_relation_endpoints_do_not_cross_types` confirming the endpoint rewrite
  routes minister vs ministry to distinct endpoints.

**Final strip list**: leading articles `de` / `het` / `een`; org leaders
`het ministerie van` / `ministerie van`. (No person-role or municipality
leaders.)

**Re-measured over the frozen 1402-entity Convenant fixture**
- Distinct-canonical: **1360 (V1) → 1341 (K.1 rev3)** = **−19**. Smaller than
  rev2's −46 BY DESIGN — rev2's extra merges were partly WRONG cross-type
  merges (person/municipality leaders stripped onto org/location names).
- BZK full-form cluster (`binnenlandse zaken en koninkrijksrelaties`, type
  `other`): **4 surface forms → 1** (two bare spelling variants + two
  `Ministerie van …` forms via ministerie-strip + spelling). Person-role forms
  no longer fold in.
- **Name-only false-merges over `must_not_merge.jsonl`: 0.** Unmerged over
  `must_merge.jsonl`: 0.
- `Minister van BZK` → `minister van bzk`; `Ministerie van BZK` → `bzk` —
  confirmed DIFFERENT strings.

**Regression**: `packages/shared/tests/` 273 passed; merge + persistence +
roundtrip suites green (296 total across the validation command). B.8 hash_id
derive-rule unchanged (`TestB8HashContract` green). ruff clean on changed src +
shared test files (pre-existing import-sort finding in
`test_notebook_merge_service.py` left untouched — not introduced here).

## Phase K.1 rev4 — articles + spelling only, NO content-prefix strip — 2026-06-22

**Decision** (final, after 4 review cycles): K.1 strips ONLY collision-safe
leading articles + curated spelling variants. **No blind content-prefix
stripping at all** — not even `ministerie van`. Attempt 4 found that
`ministerie van` collides cross-type in the live data: `Ministerie van Onderwijs`
→ `onderwijs` == the bare `onderwijs` concept, and since relations resolve
endpoints by name alone (`WHERE canonical_name = $name`) that corrupts the graph.
The org-form merge (`Ministerie van BZK` ↔ `BZK`) moves to **K.2**'s curated,
type-aware alias table.

**The fix**
- `nl_normalization.py`: `_ROLE_ORG_PREFIXES` / `_MIN_TAIL_LEN` and all org-prefix
  stripping logic **removed**. `strip_leading_noise` now strips ONLY a leading
  article (`de`/`het`/`een`) when a non-empty remainder survives (article-only
  string like `de` returned unchanged). `canonicalize_spelling` unchanged
  (curated `koninkrijk(s)relaties` map). Module docstring rewritten around the
  collision-safety rationale.
- `name_normalizer.py`: docstring de-staled — dropped the "V1 stub / Q9 will
  replace" framing and the `ministerie van → bzk` examples; now describes the
  actual article + curated-spelling behaviour and states that type-aware
  org-form / abbreviation merging lives in Track K.2. `normalize_entity_name`
  signature + pipeline order unchanged (V1 → strip article → canonicalize
  spelling).
- Harness (`resolution_metrics.py`) and the name-only `count_false_merges` gate:
  **unchanged**. `notebook_merge_service.py` `(name, type)` bucket key:
  **unchanged** (correct, consistent with persistence) — its rev3 regression
  tests stay green because `Minister van BZK` (→ `minister van bzk`) and
  `Ministerie van BZK` (→ `ministerie van bzk`) still normalize distinct.

**Final strip behaviour**: leading articles `de` / `het` / `een` only. No
content-bearing prefix is ever stripped.

**Corpora**
- `must_merge.jsonl` (6 pairs): article variants (`De Regio Deal` ↔ `Regio Deal`,
  `Een Regio Deal` ↔ whitespace), case-only (`Ministerie van BZK` ↔
  `ministerie van BZK`, `Provincie Drenthe` ↔ `provincie drenthe`), spelling
  (`Koninkrijk(s)relaties`), and `De Gemeente Groningen` ↔ `Gemeente Groningen`
  (article strip, gemeente kept). **Removed** the org-form pairs
  (`Ministerie van BZK` ↔ `BZK`, `Ministerie van Binnenlandse Zaken … ` ↔ bare
  form) — those are K.2's job and (correctly) do NOT merge in K.1.
- `must_not_merge.jsonl` (13 pairs): all rev3 cross-type pairs kept, **plus**
  the rev4 collision attempts `Ministerie van Onderwijs` ↔ `Onderwijs` and
  `Ministerie van BZK` ↔ `BZK`. Gate: **0 name-only false-merges** across the
  full corpus (now trivially satisfied since no content token is stripped — but
  asserted, incl. the per-pair parametrized canary).

**Re-measured over the frozen 1402-entity Convenant fixture**
- Distinct-canonical: **1360 (V1) → 1346 (K.1 rev4)** = **−14**. Smaller than
  rev3's −19 BY DESIGN — rev4 strips no content prefix, so the drop is purely
  the collision-safe article + curated-spelling consolidation. The bigger
  org-form merges land in K.2.
- BZK full-form cluster (`binnenlandse zaken en koninkrijksrelaties`, type
  `other`): the two bare `Binnenlandse Zaken en Koninkrijk(s)relaties` spelling
  variants collapse → 1 key (size 2). `Ministerie van …` forms no longer fold in.
- **Name-only false-merges over `must_not_merge.jsonl`: 0** (13 pairs).
  Unmerged over `must_merge.jsonl`: 0 (6 pairs).
- `Ministerie van Onderwijs` → `ministerie van onderwijs`; `Onderwijs` →
  `onderwijs` — confirmed DIFFERENT strings.

**Regression**: `packages/shared/tests/` 273 passed; full validation command
(`packages/shared/tests/` + merge + persistence + roundtrip) **296 passed**. B.8
hash_id derive-rule unchanged (`TestB8HashContract` green). ruff + mypy clean on
changed src + shared test files.

## Basis
- B.8c assessment: V1 normalizer resolves identical surface forms (107 cross-doc) but fragments variants (BZK 8-way, "minister" 23-way). See `../B-kg-quality/reviews/phase-B.8c-resolution-assessment.md`.
- Swap-point: `packages/shared/src/shared/utils/name_normalizer.py::normalize_entity_name` (persistence dedup key + filtering both call it).
- Two layers: K.1-K.2 cheap NL normalization (quick wins) → K.3+ full Q9/M4 vocabulary resolution.

| K.1 | collision-safe normalizer (articles+spelling) + harness + corpora | — | `track/k1-nl-normalizer` | 2026-06-22 | adversarial-reviewer APPROVED (attempt 5). Name-only 0 false-merges over 13-pair corpus; frag −14; 296 tests green. The gate caught the cross-type collision class (person/org, org/location, org/concept) across 4 prior attempts — would have silently corrupted relations. FU: pinned live-collision allow-list test. |

| K.2 | curated NL gov-org alias table + extensible override config | — | `track/k2-org-aliases` | 2026-06-22 | adversarial-reviewer APPROVED (attempt 2; 1 blocker fixed: Financiën diacritic mismatch). BZK/VRO/Financiën collapse to one canonical each; 0 name-only false-merges (16-pair corpus); 324 tests green. Override config (last-wins, validated) seam for K.5. |

## Phase K.3 — Retroactive canonicalization (dry-run plan + reviewable merge) — 2026-06-22

**Branch**: `track/k3-retroactive-merge` (off main, which has K.1 + K.2).
Commits `7890298..7647ef9` (4): migration+model → repo primitives → service+router → tests.

**Goal**: dedup the ALREADY-PERSISTED KG. K.1+K.2 sharpened `normalize_entity_name`
for NEW extractions; existing entities keep their old fragmented `canonical_name`s.
K.3 re-normalizes them, groups collisions, merges duplicates — reversibly.

**Delivered**
- `migrations/54.surrealql` (+`_down`) — `DEFINE FIELD IF NOT EXISTS aliases ON entity FLEXIBLE TYPE array DEFAULT [];` (first-class denormalized alias list, K-D1). `status`/`merged_into`/`idx_entity_status` already exist from migration 39, re-declared defensively with `IF NOT EXISTS` (never OVERWRITE). ADDITIVE only — does not touch `canonical_name`/`hash_id`/`entity_type` (B.8 drift caution honoured).
- `packages/shared/src/shared/models/entity.py` — `aliases: list[str] = Field(default_factory=list)` + `ensure_list` validator coverage.
- `packages/surrealdb-service/.../repositories/entity.py` — `repoint_relations(loser_id, winner_id)` (ID-based; delete-and-recreate edges with parallel-edge dedup + self-loop drop), `mark_merged(loser_id, winner_id)` (soft `status='merged'` + `merged_into`, never hard delete), `merge_into_winner(winner_id, losers)` (reuses the upsert `_union_preserve_order` merge math), `list_active_entities(source_ids=None)`.
- `apps/app-main/src/app_main/services/entity_resolution/recanonicalization_service.py` — `plan_merges(notebook_id=None)` (dry-run, zero writes), `apply_merge(cluster)` (one logical op, idempotent), `apply_plan(plan, *, dry_run=True)` (hard dry-run default).
- `apps/app-main/.../api/routers/entity_resolution.py` — `POST /api/entity-resolution/plan` (dry-run MergePlan), `POST /api/entity-resolution/apply` (explicit reviewed cluster ids only — NO implicit apply-all). Registered in `app.py`.

**Design**
- **Plan/apply split**: `plan_merges` loads active entities, computes `new_canonical = normalize_entity_name(canonical_name)` each, groups by `(new_canonical, ENTITY_TYPE)`, returns clusters of size ≥2. Pure read — zero writes. `apply_merge` is the destructive half, gated behind the explicit router call and `apply_plan`'s hard `dry_run=True` default.
- **Winner-selection rule**: highest `confidence` → tie-break most `source_documents` → tie-break oldest `created_at` → final total-order tie-break lowest record id (deterministic).
- **Reused upsert merge math**: `merge_into_winner` calls the existing `_union_preserve_order` helper (the exact function `upsert_entity` uses) for `source_documents`/`type_tags`/`provenance_chain` union + `confidence` max + `properties` overlay; not reimplemented. Aliases union the loser surface forms.
- **ID-based relation repointing**: edges are re-pointed loser_id→winner_id by entity ID (delete-and-recreate, since SurrealDB v2 RELATE `in`/`out` are not safely mutable in place), with `(in,out,relation_type)` parallel-edge dedup and winner self-loop drop. Never name-based → no cross-type ambiguity. Post-apply assertion: NO relation references a merged entity's id.
- **Idempotency**: `apply_merge` operates only on losers still `status='active'`; an already-merged cluster is a no-op (no double-merge, no duplicate aliases — alias repo also dedups on `(entity, alias_text)`).
- **Dry-run guard**: `apply_plan` default does ZERO writes (unit test asserts a write-spy list == [] and entities stay active).
- **Soft + reversible**: losers are `status='merged'` + `merged_into=<winner>`, never hard-deleted.

**Tests (43 green)**
- Unit `apps/app-main/tests/test_recanonicalization_service.py` (11): cluster grouping (BZK 3-way), winner rule (confidence / source-count / oldest-created_at tie-breaks), type-aware must-NOT separation (person/org homograph + full `must_not_merge.jsonl` corpus → 0 false clusters), dry-run zero-writes (plan + apply_plan), apply happy-path + idempotency.
- Roundtrip `packages/surrealdb-service/tests/test_entity_merge_roundtrip.py` (5, `@requires_docker`): aliases reads back `[]`; plan-no-write; full apply contract (merged status, source/alias union, **2 relations re-pointed, 0 dangling refs to merged entities**, entity_alias rows); idempotent re-apply (no duplicate aliases); person/org homograph never shares a cluster.
- Migration `test_migrations_roundtrip.py::test_migration_54_aliases_roundtrip_and_idempotent` — `aliases=[]` + `status='active'` default, replay is a no-op, field still writable.
- **B.8 regression**: `test_entity_persistence_service.py` + `test_entity_repository_roundtrip.py` green unmodified (hash_id/upsert contract intact).

**Validation**: `apps/app-main/tests/test_recanonicalization_service.py` + `test_entity_merge_roundtrip.py` + `test_entity_persistence_service.py` + `test_entity_repository_roundtrip.py` + `test_migrations_roundtrip.py` → **43 passed**. ruff clean on changed src + new test files.

**Notes for reviewer**
- The plan's "strip ministerie van first" framing is stale (K.1 rev4 dropped content-prefix stripping); K.3 inherits whatever `normalize_entity_name` does — it just groups by its output, so it stays correct as the normalizer evolves.
- Pre-existing `shared.config` package/module name collision (`MIN_APPLICABLE_CONFIDENCE` import) breaks the *full* `create_app()` import chain via `ontology_extraction`; unrelated to K.3 — router/service import cleanly in isolation and per-package tests pass. Flagged, not fixed (out of scope).
- No apply was run against the live DB (user-gated). Tests use the testcontainer only.

| K.3 | retroactive canonicalization — dry-run plan + reviewable merge | — | `track/k3-retroactive-merge` | 2026-06-22 | adversarial-reviewer APPROVED (attempt 2; blockers fixed: shared.config app-startup [K.2 hotfix→main] + relation-provenance loss on repoint). Dry-run-default, ID-based repoint, soft+reversible merge, idempotent. 34 tests green. |
| hotfix | shared.config package shadowed config.py → create_app broke (K.2 regression) | — | `fix/shared-config-collision` | 2026-06-22 | merged to main 9478235. K.2's gate missed it (tests imported the submodule, not the app chain). |

| K.4 | TOOI + Crossref vocabulary reconciliation → external_ids/aliases | — | `track/k4-vocabulary` | 2026-06-22 | adversarial-reviewer APPROVED (attempt 2; 1 major fixed: export projection dropped external_ids/aliases). Provider-pluggable, fail-soft HTTP (cache+rate-limit), single-match precision guard, Crossref title-overlap gate. K-D2 (TOOI bulk URL) flagged non-blocking w/ verified seed. 208+81 tests green. |

## Phase K.5 — Fuzzy/embedding candidate dedup + review queue + alias overlay — 2026-06-22

**Branch**: `track/k5-fuzzy-dedup` (off main, with K.1-K.4). Commits per logical unit (migration+repo+config → service+API+tests → status).

**Delivered**
- `migrations/56.surrealql` (+`_down`) — `DEFINE TABLE IF NOT EXISTS alias_overlay SCHEMAFULL` with `scope` (global|notebook), `notebook option<record<notebook>>`, `kind` (merge|split), `name_a`/`name_b` (non-empty asserts), `entity_type`, `created_at`, + scope/notebook indexes. ADDITIVE only (new table, never touches `entity`/`canonical_name`/`hash_id`/`entity_type` → B.8 key intact). Down drops the table. Idempotent (validated by the migration roundtrip harness).
- `packages/surrealdb-service/.../repositories/alias_overlay.py` — `AliasOverlayRepository` (create/get/list/delete). `list_overlays(notebook_id)` returns global rules + that notebook's rules (the union the dedup service evaluates).
- `apps/app-main/.../services/entity_resolution/overlay_service.py` — `OverlayService` CRUD + `split_pairs()`/`merge_rules()` resolution + `sync_alias_overrides()` bridging force-merge rules into the K.2 `alias_overrides` DB seam (`set_db_overlay`). `OverlayRule.matches` is type-aware and order-insensitive.
- `apps/app-main/.../services/entity_resolution/candidate_dedup_service.py` — `CandidateDedupService.propose_candidates(notebook_id=None)`. Reuses the entity-filtering `FuzzyResolver` scoring (`_compute_similarity`) + a cosine embedding pass; **never reimplements matching**. Type-aware bucketing, review-band partition, force-split veto, force-merge include. Auto-merge candidates project onto a K.3 `MergeCluster` (`to_merge_cluster`) so the destructive apply reuses `RecanonicalizationService` (relation repoint, provenance fold, alias rows, soft status).
- `packages/surrealdb-service/.../repositories/entity.py` — `list_active_entities_with_embeddings` (separate from K.3's `list_active_entities` so that row contract stays byte-identical).
- `pipelines/entity-filtering/.../config.py` — additive `auto_merge_threshold`/`review_threshold` on `FuzzyDedupConfig` (0.93/0.86) + `EmbeddingDedupConfig` (0.95/0.90). Defaults documented with the corpus evidence.
- API: `GET /api/entity-resolution/candidates` (review queue) + `POST/DELETE /api/entity-resolution/overlay`. Wired into the existing K.3 router.
- `tests/fixtures/entity_resolution/must_not_merge.jsonl` — extended with 5 fuzzy near-miss pairs (Regio Deal Groningen↔Drenthe, Zuid↔Noord-Limburg, Provincie Groningen↔Drenthe, Gemeente Stadskanaal↔Veendam, Min Financiën↔EZK).

**Over-merge guards (the ×2.0 surface)**
1. **Type-aware** — entities bucketed by `entity_type`; only same-type pairs are ever compared (a person and an org named X never propose, verified by `test_cross_type_homograph_never_proposed`).
2. **Review band, nothing silent** — `auto_merge` (≥ auto), `review` (review ≤ s < auto → queued, NEVER auto-applied), `reject` (< review, dropped). `propose_candidates` is read-only (the apply path is K.3, opt-in). AC2 asserts the review-band pair is not written.
3. **must-NOT gate** — over the extended corpus at the tuned threshold, auto-merge proposals contain **0** must-NOT pairs (AC3/AC6).
4. **force-split = hard veto** — a split rule removes a pair from every band even at similarity 1.0 (AC4); split also beats a contradictory force-merge.
5. **force-merge per-notebook** — a notebook merge rule fires only within that notebook (AC5).
6. Embedding degrades to fuzzy-only when `embedding=[]` (no fabricated score).

**Threshold tuning + measurement (AC6)** — measured the `FuzzyResolver` levenshtein similarity of the OCR-typo must-merge pair vs every must-NOT near-miss:

| pair | score | band @ (0.93/0.86) |
|---|---|---|
| Koninkrijksrelaties ↔ Koninkrijksreiaties (typo, MUST MERGE) | 0.9474 | **auto** |
| Regio Deal Zuid-Limburg ↔ Noord-Limburg (MUST NOT) | 0.8333 | reject |
| Regio Deal Groningen ↔ Drenthe (MUST NOT) | 0.7000 | reject |
| Provincie Groningen ↔ Drenthe (MUST NOT) | 0.6842 | reject |
| BZK ↔ EZK (MUST NOT) | 0.6667 | reject |
| Ministerie van Financiën ↔ Economische Zaken (MUST NOT) | 0.5938 | reject |
| Gemeente Stadskanaal ↔ Veendam (MUST NOT) | 0.5500 | reject |

The tuned auto=0.93 catches the typo class; review=0.86 sits comfortably above the closest must-NOT (0.8333) → **0 false auto-merges, 0 false reviews** over the corpus. Embedding band is higher (0.95/0.90) because cosine over short-name embeddings is noisier than exact string distance.

**Tests (green)**
- `apps/app-main/tests/test_candidate_dedup_service.py` (10): typo caught (AC1), review-band-not-applied (AC2), zero must-NOT auto-merges over the corpus (AC3/AC6), cross-type guard, force-split veto (AC4), force-merge include + split-beats-merge (AC5), embedding pair caught, embedding fallback, `to_merge_cluster` winner/loser.
- `apps/app-main/tests/test_overlay_service.py` (8): per-notebook isolation (AC5), global-visible-everywhere, scope union, split/merge partition, validation, delete, alias_overrides bridge.
- `packages/surrealdb-service/tests/test_alias_overlay_roundtrip.py` (4, `@requires_docker`): global roundtrip, per-notebook isolation, delete, schema enum reject (migration 56 SCHEMAFULL ASSERT).
- Regression: 348 shared+K.5 unit, 18 K.3/K.4 unit, 10 B.8 persistence, 50 entity-filtering config/fuzzy, 12 migration roundtrip — all green. `create_app()` imports clean; `/candidates` + `/overlay` routes registered. Lint clean (ruff).

**Decision points honoured**: K-D3 (review-only for the uncertain band; auto only ≥ auto-threshold, applied via explicit K.3 op) and K-D4 (notebook > global; force-split is an absolute veto). No new escalations.
