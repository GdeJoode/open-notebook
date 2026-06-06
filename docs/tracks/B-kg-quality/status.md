# Track B — KG quality: rolling status

## Phase B.1c — Pass-1 schema validation module — attempt 2 (2026-06-05)

**Branch**: `track/b-pass1-module`
**Commits added in attempt 2**: `9a2e3fc` → `bf01589` → `339907d` → `8076722`
**State**: attempt 1 rejected (REVISIONS_NEEDED). All 3 majors + 6 minors addressed; ready for re-review.

### Attempt 2 changes (vs attempt 1)

- **Major 1 (`bf01589`)**: malformed-JSON paths now degrade gracefully per plan AC #3 — return empty `Pass1Output` (detected_schema="", coverage=0.0, confidence=0.0, empty lists) + WARNING log. `Pass1ParseError` reclassified to transport-only.
- **Major 2 (`bf01589`)**: `Pass1Output.alternative_schemas` is now `List[Dict[str, Any]]` matching `Pass1Result` — no more ValidationError at B.1f persistence time. LLM contract carries `{"name", "confidence"}` per entry.
- **Major 3 (`9a2e3fc`)**: schema-summary auto-compression at > 30 types + output-format trim. 100-type ontology + 1500-token sample now fits at ~2192 tokens (cap 2400). Stress tests pin the contract.
- **Minor 1 (`8076722`)**: coverage 100% on both `pass1_schema_validation.py` AND `prompts/pass1.py` (was 89%, target ≥ 90%).
- **Minor 2 (`9a2e3fc`)**: candidate-schema list extended with `base`, `policy_themes`, `social_profiles` — full 11-ontology surface.
- **Minor 3 (`bf01589`)**: brace-extraction salvage for prose-wrapped JSON.
- **Minor 4 (`bf01589`)**: dead `ModelManager()` instantiation removed; replaced with import-only check.
- **Minor 5 (`9a2e3fc`)**: system prompt moved to `prompts/pass1.PASS1_SYSTEM_PROMPT`.
- **Minor 6 (self-review edit)**: scholarly.yaml type count corrected (8, not "~30").
- **Bonus (`339907d`)**: LLMExtractor `LLMManager` → `ModelManager` rename — Option B (TODO marker) chosen because the proper fix requires DI plumbing that belongs in B.1f.

### Quality gates (attempt 2)

```
pipelines/ontology-extraction : 98 → 124, all green
  Coverage pass1_schema_validation.py: 100%
  Coverage prompts/pass1.py:           100%
packages/shared              : 145 (unchanged)
apps/app-main                : 368 (no regression)

Token budgets (attempt 2):
  scholarly.yaml + 1500-tok sample:  ~1947 / 2400 tokens (18.9% headroom)
  synth 100 types + 1500-tok sample: ~2192 / 2400 tokens (8.7% headroom)
  synth 150 types + 1500-tok sample: ~2367 / 2400 tokens (1.4% headroom)
```

---

## Phase B.1c — Pass-1 schema validation module (2026-06-05)

**Branch**: `track/b-pass1-module`
**Commits**: `1b82c48` (name_normalizer stub) → `aa7d02f` (Pass-1 module)
**State**: code complete, all quality gates green, ready for review.

### Delivered

- `packages/shared/src/shared/utils/name_normalizer.py` — V1 stub
  (`lowercase + collapse-whitespace + strip-trailing-punctuation`)
  behind a single import point `shared.utils.name_normalizer.
  normalize_entity_name`. Q9 (Track M4) replaces this with TOOI +
  Crossref lookups; until then, the single import point keeps the
  upgrade invisible to downstream callers.
- `pipelines/ontology-extraction/src/ontology_extraction/
  pass1_schema_validation.py` — `Pass1SchemaValidator` with async
  `run(text_sample, ontology)` returning a fully-validated
  `Pass1Output`. Coarse `len(text)//4` token-budget guard fires
  pre-LLM-call at the 2400-token cap (3000 plan budget minus 20 %
  safety margin per Q-B-2); raises `TokenBudgetExceeded`.
- `pipelines/ontology-extraction/src/ontology_extraction/prompts/
  pass1.py` — three-section prompt template (schema summary / text
  sample / output JSON schema). Real-world headroom for
  `scholarly.yaml` (~30 entity types) + 1500-token sample is
  ~2132 / 2400 tokens (11.2 %).
- `Pass1Output.model_dump()` keys are field-compatible with
  `shared.models.notebook_schema.Pass1Result` — a guard test
  (`TestPass1OutputCompatibility::test_model_dump_keys_match_…`)
  fails if the two models drift, blocking B.1f-style persistence
  bugs at PR time.
- Defensive output parser: tolerates markdown code fences,
  percentage-style scalars (`87` → `0.87`), `null` arrays, extra
  fields; raises `Pass1ParseError` on structurally bad responses.
- LLM caller is **injected** (not lazy-imported by default) — see
  the self-review for the trade-off. Tests pass canned callables;
  B.1f wires the real one. `EntityExtractionService.run_extraction`
  gained a TODO marker only — no behaviour change.

### Tests added

- `packages/shared/tests/test_name_normalizer.py` — 17 tests
  (transformations, idempotence, Unicode passthrough, public API).
- `pipelines/ontology-extraction/tests/
  test_pass1_schema_validation.py` — 37 tests (token budget at
  boundary, prompt template renderings, malformed JSON parsing,
  field-validator edge cases, end-to-end with mocked sync + async
  LLM callers, real-world scholarly-ontology budget headroom,
  `Pass1Output` ↔ `Pass1Result` field compatibility).

### Quality gates

```
packages/shared           : 128 → 145 (+17), all green
pipelines/ontology-extraction : 61 → 98 (+37), all green
apps/app-main             : 368 → 368, no regressions
```

### Decisions worth flagging (full detail in self-review)

- **Injected LLM caller > lazy import default**: the existing
  `LLMExtractor` imports `LLMManager` (which does not exist in
  `llm-manager`), always hits the ImportError fallback, and returns
  empty results. Pass-1 chose injection so unit tests do not
  inherit that broken-default behaviour. B.1f wires the production
  caller.
- **Pass1Output.alternative_schemas is `List[str]`** (the LLM-facing
  contract), while `Pass1Result.alternative_schemas` stays
  `List[Dict[str, Any]]` (the DB-side FLEXIBLE shape). The B.1f
  persistence wrapper lifts strings into `{"name": s}` dicts.
- **Percentage rescaling on coverage_pct / confidence_in_choice**:
  values > 1.5 are divided by 100. Defensive against LLM
  inconsistency; tested + easy to revert if a reviewer prefers
  strict rejection.

### Outstanding follow-ups for downstream phases

- **B.1d (Pass 2)**: import `Pass1SchemaValidator`, `Pass1Output`
  from `ontology_extraction` — re-exports are already in place.
- **B.1e (multi-schema)**: this phase shipped only the
  single-schema validator. B.1e adds the orchestrator that runs
  Pass-1 against several candidate schemas and picks the best fit.
- **B.1f (service integration)**: replace the TODO marker in
  `EntityExtractionService.run_extraction` with the actual
  sample-→-validate-→-persist path. The default lazy LLM caller
  in `Pass1SchemaValidator._default_llm_caller` is the swap point.
- **B.4 (telemetry)**: when the metrics table lands, the validator
  should emit `pass1_runs`, `pass1_token_estimate`,
  `pass1_token_budget_exceeded` counters. Currently we have only
  `loguru` WARNING-level observability.
- **Track M4 Q9**: replace `normalize_entity_name` body with the
  full TOOI + Crossref pipeline. The import point stays at
  `shared.utils.name_normalizer` — no caller rewiring needed.

## Phase B.1b — notebook_schema + pass1_results tables + repos (2026-06-05)

**Branch**: `track/b-models-notebook-schema`
**Commits**: `5fc4859` → `e7f0310` → `997ad8f`
**State**: code complete, all quality gates green, ready for review.

### Delivered

- `migrations/45.surrealql` + `migrations/45_down.surrealql` — two new
  SCHEMAFULL tables (`notebook_schema`, `pass1_results`) following the
  migration-43 FLEXIBLE-extension-bag pattern. `UNIQUE` index on
  `notebook_schema.notebook` enforces one row per notebook;
  `idx_pass1_source` covers the hot read path. All `DEFINE` statements
  use `IF NOT EXISTS` so the migration is idempotent.
- `packages/shared/src/shared/models/notebook_schema.py` —
  `NotebookSchema` and `Pass1Result` Pydantic models. Both carry
  bounded confidence/coverage fields, a defensive
  `ensure_metadata_dict` validator on the FLEXIBLE bag, and
  `List[Dict[str, Any]]` for extension-shaped arrays so the dict
  shape can evolve without further migrations.
- `packages/surrealdb-service/src/surrealdb_service/repositories/notebook_schema.py`
  — `NotebookSchemaRepository` (singleton-per-notebook with
  rewrite-on-conflict upsert; plus
  `add_pending_extension` / `accept_pending_extension` /
  `reject_pending_extension`) and `Pass1ResultRepository` (append-only
  + source-scoped / notebook-scoped reads).
- `packages/shared/tests/test_notebook_schema_model.py` — 11 unit
  tests covering construction, full roundtrip, bounds, metadata
  coercion.
- `packages/surrealdb-service/tests/test_notebook_schema_repo_roundtrip.py`
  — 10 `requires_docker` tests covering migration record-keeping +
  idempotence, full roundtrip, UNIQUE-rewrite semantic, direct-CREATE
  blocking, extension lifecycle, and empty-list handling.
- `packages/shared/src/shared/models/__init__.py` + repository
  `__init__.py` — additive exports only. **Coordination note**: B.1a
  (`track/b-models-entity`) touches the same two files to add
  `Entity` / `Relation` and their repos. Both branches are additive
  in distinct sections of `__all__`; merge is expected to be clean
  three-way without semantic conflicts.

### Decisions taken (all per autopilot defaults Q-B-8, Q-B-9)

- **Q-B-9**: migration 45 is reserved for B.1b. (B.1a takes 44.)
- **Q-B-8**: shared `notebook_event` table is NOT introduced here —
  deferred to B.3b as planned.
- **UNIQUE-index handling**: rewrite-on-conflict semantic in the
  repository's `upsert`. Detailed rationale in
  `reviews/phase-B.1b-self-review.md` and inline near `upsert()`.

### Test results

| Suite | Before | After | Note |
|---|---|---|---|
| `packages/shared` | 105 | 116 (+11) | new model tests |
| `packages/surrealdb-service` (not requires_docker) | 52 | 52 | no new non-docker tests; no regressions |
| `packages/surrealdb-service` (requires_docker) | 5 pass, 1 xfail | 15 pass, 1 xfail (+10) | new repo roundtrips |
| `apps/app-main` | 367 | 367 | no regressions |

Final `requires_docker` run summary: `15 passed, 52 deselected, 1 xfailed in 17.63s`.

### Ready for review

PR title: `feat(shared,surrealdb): notebook_schema + pass1_results tables + repos (B.1b)`

## Phase B.0 — Testcontainers SurrealDB harness (2026-06-05)

**Branch**: `track/b-kg-foundation`
**State**: code complete, local tests green, ready for review.

### Delivered

- `packages/surrealdb-service/src/surrealdb_service/testing/` — new subpackage
  exposing the `live_surrealdb` pytest fixture. Boots
  `surrealdb/surrealdb:v2` via generic `testcontainers.DockerContainer` (no
  official SurrealDB adapter exists as of 2026-06), waits for `/health`,
  resets the connection-pool singleton, applies all discovered migrations via
  the canonical `AsyncMigrationManager`, and yields a `SurrealDBConfig`. The
  fixture is importable from any workspace member as
  `from surrealdb_service.testing import live_surrealdb`.
- `packages/surrealdb-service/tests/conftest.py` — re-exports the fixture for
  the local test suite (and serves as a template downstream packages can
  copy).
- `packages/surrealdb-service/tests/test_migrations_roundtrip.py` — five
  canary tests:
  - migrations-applied smoke (asserts version ≥ 43);
  - `entity` roundtrip (canonical_name, entity_type, defaults);
  - `entity_alias` roundtrip;
  - `relation` RELATE roundtrip;
  - `source` roundtrip including the migration-43 `metadata` bag;
  - **xfail** for the legacy `entity_persistence_service` write shape
    (`name`/`weight`/`source_ids`) — confirms the bug B.1a will fix and
    documents the exact source location (lines 132-156).
- `.github/workflows/db-integration.yml` — new workflow runs the harness on
  every PR touching `migrations/`, `packages/surrealdb-service/`, or
  `packages/shared/src/shared/models/`. Verifies Docker availability up
  front (Track A's GPU mishap is the cautionary tale).
- `docs/tracks/B-kg-quality/TESTCONTAINERS_GUIDE.md` — usage guide.
- `packages/surrealdb-service/pyproject.toml` — added `testcontainers>=4.0.0`
  to dev deps, registered the `requires_docker` marker.
- Workspace `pyproject.toml` — also registered `requires_docker` so other
  packages can use the marker without re-defining it.

### Decisions taken (all per autopilot defaults)

- **Q-B-1**: legacy persistence drift is *surfaced* via an `xfail(strict=True)`
  test, not fixed here. Strictness means if B.1a accidentally over-fixes, the
  test will turn XPASS and force us to delete/promote it.
- **Storage engine**: `memory` (not rocksdb). Each container is throwaway and
  faster to boot.
- **Session scope**: one container per pytest session. Tests that touch the
  same table use unique IDs (`_unique()` helper) to avoid cross-test
  interference rather than re-applying migrations per test.

### Test results

- `packages/surrealdb-service`: **45 passed, 6 skipped** (the 6 are the
  `requires_docker` tests skipping cleanly because no Docker daemon is
  available in the sandbox where this was authored).
- `apps/app-main`: **367 passed** — no regressions.

### Open items / hand-off notes

- The `requires_docker` tests have not been executed end-to-end yet (no
  Docker in the authoring sandbox). They are designed to skip cleanly when
  Docker is absent and the CI workflow verifies Docker is reachable on the
  runner before running them. **First CI run on the PR is the validation
  gate**.
- B.1a inherits the xfail test in `test_migrations_roundtrip.py` — its
  acceptance criterion #4 should explicitly delete or invert it.

## Phase B.0 — attempt 2 (2026-06-05)

**State**: revisions addressed, verified end-to-end against a real
SurrealDB container, ready for re-review.

### Fixes vs attempt 1

Reviewer rejected attempt 1 with REVISIONS_NEEDED (review at
`docs/tracks/B-kg-quality/reviews/phase-B.0-attempt-1.md`). Attempt 2
addresses every blocker and major plus several minors. Full per-blocker
table with commit SHAs lives at
`docs/tracks/B-kg-quality/reviews/phase-B.0-self-review.md` → "Attempt 2
fixes".

Highlights:

- **Blocker #1** (migrations-dir off-by-one) → `fixtures.py` now walks
  up from `__file__` looking for a `migrations/` dir sibling to a
  workspace-marker `pyproject.toml`. Robust to file moves.
- **Blocker #2** (no non-Docker safety net) → new
  `tests/test_testing_fixtures.py` (7 tests, no marker) catches
  path-drift, missing migrations files, and dead-code regressions on
  every `pytest -q` run.
- **Major #3** (pool-lifecycle across `asyncio.run`) → pool is now
  reset *after* the migration block, before `yield config`.
- **Major #4** (stale docstring) → rewritten; engine is `memory`, file
  count is "43+".
- **Major #5** (`live_surrealdb_async` dead code) → deleted.
- **Minors #6, #7, #9, #10** → all addressed (see self-review for
  details).

### Verification (attempt 2)

End-to-end run with Docker:

```
cd packages/surrealdb-service && uv run pytest -m requires_docker -v
5 passed, 52 deselected, 1 xfailed in 12.58s
(real 24s including container boot — well under 90s budget)
```

Gating without Docker:

```
cd packages/surrealdb-service && uv run pytest -q -m "not requires_docker"
52 passed, 6 deselected in 0.91s
```

App-main regression check:

```
cd apps/app-main && uv run pytest -q
367 passed in 51.38s
```

### New issue surfaced while running end-to-end

`SCHEMAFULL entity` requires `embedding` to be supplied at CREATE time
(migration 39 declares it as `FLEXIBLE TYPE array` with no DEFAULT).
Tests now pass `embedding = []` to mirror production-correct callers.
**Implication for B.1a**: every `EntityRepository.upsert_entity` write
must include `embedding` — keep this in mind when routing
`entity_persistence_service` through the repository.

### Commit hashes (attempt 2)

- `d2342bb` — `fix(surrealdb-service): robust migrations-dir lookup + pool reset`
- `5de7ed8` — `test(surrealdb-service): non-docker safety net for fixture path drift`
- `37bd30f` — `test(surrealdb-service): roundtrip canaries pass end-to-end against real DB`

## Phase B.1a — Entity/Relation models + persistence drift fix (2026-06-05)

**Branch**: `track/b-models-entity`
**State**: code complete, all gates green, ready for review.

### Delivered

- `packages/shared/src/shared/models/entity.py` — new `Entity(ObjectModel)`
  mirroring migration-39 fields plus the multi-type-tagging additions
  (`type_tags`, `primary_type`) introduced by migration 44. New
  `Relation(ObjectModel)` mirroring the `relation` RELATE table; DB-side
  `in`/`out` surface as `in_entity`/`out_entity` to dodge the Python
  keyword. Both exported from `shared.models`.
- `migrations/44.surrealql` + `_down.surrealql` — additive
  `DEFINE FIELD IF NOT EXISTS type_tags ... FLEXIBLE TYPE array DEFAULT []`
  and `primary_type ... TYPE option<string>`. Idempotent. **Note**:
  `FLEXIBLE` is required — without it SCHEMAFULL silently drops list
  values on this SurrealDB version (confirmed via repro script).
- `packages/surrealdb-service/src/surrealdb_service/repositories/entity.py`
  — new `upsert_entity(entity: Entity) -> str`. Lookup by
  `(canonical_name, entity_type)` (the migration-39 unique index);
  Python-side merge of confidence (max) / source_documents / type_tags /
  provenance_chain (union) / properties (dict overlay). Merge moved
  client-side because `object::extend` is unavailable in this SurrealDB
  version (Parse error).
- `apps/app-main/src/app_main/services/entity_persistence_service.py`
  — entity-upsert routed through `EntityRepository.upsert_entity`. Field
  names align to migration 39 (`canonical_name`, `source_documents`,
  explicit `embedding=[]`). Relation block also fixed: lookup uses
  `canonical_name` (was `name`), edge carries `source_documents` (was
  legacy `source_id` scalar). DI: optional `entity_repository=` argument
  for test injection.
- `packages/shared/tests/test_entity_model.py` — 11 Pydantic roundtrip
  tests covering construction, defaults, None coercion (legacy rows),
  confidence bounds, multi-type roundtrip.
- `packages/surrealdb-service/tests/test_entity_repository_roundtrip.py`
  — 3 docker-gated tests: create with type_tags, merge-on-second-call
  (asserts union/max semantics), empty-embedding contract.
- `packages/surrealdb-service/tests/test_migrations_roundtrip.py` — the
  former `test_entity_persistence_drift_xfail` was renamed to
  `test_entity_persistence_field_alignment` and **flipped to PASSING**.
  It exercises `EntityRepository.upsert_entity` directly (no
  app_main dependency) and also asserts the legacy field shape IS still
  rejected — drift regression guard.
- `apps/app-main/tests/test_entity_persistence_service.py` — refactored
  to patch the injected repository for the entity path, while keeping
  `execute_query` mocks for the relation path. Added a guard test
  asserting the `Entity` passed to the repo uses the migration-39
  canonical field names.

### Verification

```
cd packages/shared && uv run pytest -q
116 passed in 2.30s

cd packages/surrealdb-service && uv run pytest -m "not requires_docker" -q
52 passed, 9 deselected in 2.80s

cd packages/surrealdb-service && uv run pytest -m requires_docker -v
9 passed, 52 deselected in 6.35s
  (incl. test_entity_persistence_field_alignment — formerly xfail)

uv run pytest apps/app-main/tests -q
368 passed in 57.29s   (baseline 367 + 1 new alignment guard)
```

### Notes for the next phase (B.1b/B.1e merge step)

- `EntityRepository.upsert_entity` is now the canonical write-path. Pass
  `embedding=[]` if the vector isn't computed yet (the SCHEMAFULL column
  has no DB default).
- For multi-type merges, `type_tags` accumulates via union and
  `primary_type` is overwritten with the new value when supplied
  (otherwise the existing one is preserved). The B1.e merge step gets
  union semantics for free.
- Schema-level rule: any new SCHEMAFULL array field that may need to
  hold non-typed elements MUST use `FLEXIBLE TYPE array`. Plain
  `TYPE array` silently coerces to `[]` on write — verified in this
  phase, worth recording for migrations 45+.

### Commit hashes

- `445c072` — `feat(shared): Entity/Relation models with type_tags + primary_type (B.1a)`
- `c0127f7` — `feat(surrealdb-service): EntityRepository.upsert_entity canonical write-path (B.1a)`
- `c459fe8` — `fix(app-main): align entity_persistence_service to migration-39 schema (B.1a)`

## Phase B.1a — attempt 2 (2026-06-05)

**State**: revisions addressed, docker-gated suite re-verified
end-to-end, ready for re-review.

### Fixes vs attempt 1

Reviewer rejected attempt 1 with `REVISIONS_NEEDED` (1 major + 6
minors). Per-issue fix table with commit SHAs lives at
`docs/tracks/B-kg-quality/reviews/phase-B.1a-self-review.md` →
"Attempt 2 fixes".

Highlights:

- **Major** (timestamp drop): `Entity` now carries explicit
  `created_at` + `updated_at` (option A from the review). `Relation`
  carries `created_at` only (schema declares no `updated_at` on the
  `relation` table). Net-new models, no caller-side breakage.
- **Minor 1** (in/out aliases): `Relation.in_entity`/`out_entity` now
  have `Field(alias="in"/"out")` + `populate_by_name=True`. Unit test
  added.
- **Minor 2** (race window): documented inline in `upsert_entity` —
  B.1e must lock or transact.
- **Minor 3** (embedding docstring): softened wording.
- **Minor 4** (inaccurate test-failure claim): removed from
  self-review.
- **Add-on**: `EntityRepository.get_entity(record_id) -> Optional[Entity]`
  added (typed read-path; B.1e merge will use it).
- **Add-on**: docker-gated `test_upsert_roundtrips_created_at_and_updated_at`
  added — regression guard for the major.

### Verification (attempt 2)

```
cd packages/shared && uv run pytest -q
  116 passed in 1.14s

cd packages/surrealdb-service && uv run pytest -m "not requires_docker" -q
  52 passed, 10 deselected in 2.04s

cd packages/surrealdb-service && uv run pytest -m requires_docker -v
  10 passed, 52 deselected in 6.31s
  (incl. test_upsert_roundtrips_created_at_and_updated_at — new)
```

### Commit hashes (attempt 2)

- `4486aee` — `fix(shared): Entity/Relation surface schema-side timestamps + in/out aliases (B.1a r2)`
- `6621e76` — `feat(surrealdb-service): get_entity + timestamp roundtrip test + race note (B.1a r2)`

### Known follow-ups

These are pre-existing issues that B.1a flagged but does not fix
(reviewer minors 5 + 6). All are read-side or counter-side and not on
the canonical write-path B.1a hardened.

- **Read-side entity drift in `EntityRepository`**: `find_by_type`,
  `list_entities`, `search_entities`, and
  `get_all_entities_and_relations` still `SELECT id, name,
  entity_type, weight` — but migration 39 doesn't carry `name` or
  `weight` columns. These read paths silently return empty rows for
  the missing fields. Symmetric counterpart to the write-side drift
  B.1a fixed. **Fix in B.1e or earlier** (the merge step touches these
  paths anyway).
- **`relations_created` over-counts in `entity_persistence_service.py`
  lines 184-207**: the counter increments per RELATE call without
  deduping when the same entity-pair fires multiple relation_types
  back-to-back. Pre-existing, low-impact (purely a telemetry skew),
  not on the write-path. Fix when the relation block gets its
  upsert-equivalent (B.1c).

---

## Phase B.2a — TTL/RDFS exporter fix + roundtrip test (2026-06-06)

**Branch**: `track/b-ttl-exporter-fix`
**Commits**: `aa61bb1` (fix) → `150135f` (tests)
**State**: code complete, all quality gates green, ready for review.

### Delivered

- `packages/ontology-manager/src/ontology_manager/rdf_owl_shacl.py`
  — fixed the module-load `NameError: name 'Namespace' is not defined`
  bug. Pre-fix, the module raised at import time when rdflib was missing
  because `ON`/`ONR`/`DTYPE_MAP` at module scope referenced `Namespace`
  and `XSD` imported inside a `try:` block. Fix uses the sentinel
  pattern from the RETRO: `RDFLIB_AVAILABLE = True/False` flag, all
  rdflib-referencing constants guarded behind `if RDFLIB_AVAILABLE:`,
  and a `_require_rdflib()` helper that raises a clear `ImportError`
  with install hint at every public entry point.
- `packages/ontology-manager/pyproject.toml` — added `rdflib>=7.0.0`
  to runtime deps (was missing — the legacy try/except was masking
  the missing-dep bomb). Added `pyshacl>=0.25.0` to dev deps for the
  roundtrip parsability check.
- `packages/ontology-manager/tests/test_ttl_roundtrip.py` — four
  new tests:
  - `test_yaml_to_ttl_roundtrip_preserves_triples_scholarly` (set
    equality on `(s, p, o)` triples for `scholarly.yaml`)
  - `test_yaml_to_ttl_roundtrip_preserves_triples_policy` (same for
    `policy.yaml` — second ontology per plan)
  - `test_ttl_output_parses_with_pyshacl` (Protégé surrogate —
    skips cleanly if pyshacl unavailable)
  - `test_rdflib_imports_succeed_at_module_load` (permanent
    regression guard for the original NameError)

### Quality gates

- `cd packages/ontology-manager && uv run pytest -q` → **192 passed**
  (188 pre-fix + 4 new, zero regressions).
- `cd packages/shared && uv run pytest -q` → **128 passed**, no regressions.
- `cd apps/app-main && uv run pytest tests/ -q` → **368 passed**
  (311 core + 57 parser tests); `tests/test_ontology_service.py`
  passes 7/7 explicitly. app-main only imports
  `ontology_manager.manager` / `ontology_manager.schema`, so the
  `rdf_owl_shacl.py` changes are isolated.
- Coverage on changed lines: 100% on reachable paths; defensive
  branches (rdflib-missing else, ImportError raise) are unreachable
  in CI by design (rdflib IS installed) but validated by inspection.

### Self-review

See `docs/tracks/B-kg-quality/reviews/phase-B.2a-self-review.md` for
the full acceptance-criteria walkthrough, exact bug reproduction,
and REFACTOR_PLAN follow-up notes (untested SHACL/SKOS functions,
silent exception-swallowing in `load_all_ontologies`, hardcoded
demo path).

### Follow-ups for later phases

- `generate_shacl_shapes`, `validate_entities`, `create_skos_scheme`,
  `_demo` remain untested. Out of scope for B.2a; flag for B.2c/B.3.
- `load_all_ontologies` silently swallows `Exception` per YAML file.
  Logged for future cleanup; may mask data-quality regressions.
- `_demo` hardcodes a Windows-style path as `PROJECT_ROOT` default.
  Cosmetic.
