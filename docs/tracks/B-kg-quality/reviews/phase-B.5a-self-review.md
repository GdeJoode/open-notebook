# Phase B.5a — self-review

> Author: implementer agent, 2026-06-09
> Branch: `track/b-orphan-connector`
> Goal: port myKG's `orphan_connector.py` core algorithm into
> `pipelines/entity-filtering`. Detect entities with no relations,
> propose new relations via chunk co-occurrence, LLM-confirm proposals.

## Acceptance criteria check

| # | Criterion | Verified? |
|---|---|---|
| 1 | 5 entities, 2 orphans → `find_orphans` returns 2 | YES — `TestFindOrphans::test_returns_only_orphans`. The repo mock returns the 2 orphan rows for `source:demo`; `find_orphans` passes them through verbatim. |
| 2 | 2 orphans co-occur with another entity in 1 chunk → ≥1 proposal each | YES — `TestProposeConnections::test_simple_cooccurrence`. Alice + Bob + MIT in one chunk yields 4 directed proposals (Alice→Bob, Alice→MIT, Bob→Alice, Bob→MIT). Each orphan emits at least one proposal. The directed-pair set is unique. |
| 3 | Mock LLM confirms all → `ExtractedRelation`s with confidence ≥ `min_confidence`; relation_type from LLM | YES — `TestConfirmConnections::test_llm_confirms_all_proposals`. Returns 2 relations, `relation_type="KNOWS"`, confidence=0.85, `properties.extraction_method="orphan_connector"`. |
| 4 | Mock LLM denies all (`relation_type=null`) → 0 new relations | YES — `TestConfirmConnections::test_llm_denies_all_yields_no_relations`. LLM is still called (denial is a parse result, not a malformed response); zero `ExtractedRelation` rows. |
| 5 | Token-budget guard: per LLM call ≤ 1500 tokens | YES — `TOKEN_BUDGET_TARGET = 1500`. `TestConfirmConnections::test_token_budget_exceeded_raises_before_call` builds a proposal whose evidence text is 8 000 chars (~2 000 tokens), asserts `OrphanTokenBudgetExceeded` is raised **before** the LLM is invoked (`llm.calls == []`). The check fires on `_estimate_tokens(system_prompt + "\n" + user_prompt)`, matching the Pass-2 envelope. |
| 6 | ≥ 85% line coverage | YES — **98%** (`196 stmts, 3 missed`) on `orphan_connector.py`; **100%** on `orphan_prompts.py`. Missed lines are defensive branches inside `_strip_code_fence` (the non-`json` code-fence path) and two unreachable post-condition returns in `propose_connections`. |
| 7 | Malformed LLM JSON degrades to empty list + warning log | YES — `TestConfirmConnections::test_malformed_json_degrades_gracefully` queues a broken-JSON response then a valid one; only the second proposal yields a relation, no exception escapes. Negative-shape variants covered: top-level array (`test_non_object_top_level_dropped`), non-string `relation_type` (`test_relation_type_non_string_dropped`), empty/whitespace `relation_type` (`test_empty_string_relation_type_treated_as_deny`), empty response (`test_empty_response_dropped`). |

## Pre-resolved decisions honoured

| Q | Decision | Implementation |
|---|---|---|
| Q-B-2 | Token budget heuristic = `len(text) // 4`; 1500 cap per LLM call | `_estimate_tokens` in `orphan_connector.py` matches Pass-2 verbatim. `TOKEN_BUDGET_TARGET = 1500` (lower than Pass-2's 2400 — see module docstring on `orphan_prompts.py` for the rationale: orphan-connector fires many times per source so a tighter cap surfaces oversize evidence early). |
| Q-B-7 | Reuse existing `shared.utils.name_normalizer.normalize_entity_name` stub | Yes — imported in `orphan_connector.py` and used to key the orphan-lookup map and the chunk-entity dedup set. No new normalisation code in this phase. |

## Files created

- `pipelines/entity-filtering/src/entity_filtering/resolution/orphan_connector.py`
  (~600 lines): the three async stages (`find_orphans`,
  `propose_connections`, `confirm_connections`) plus a convenience
  `run(...)` wrapper for the workflow.
  - `OrphanEntityRepoProtocol` — narrow DI surface (only
    `list_orphans_for_source`) so tests need a 5-line mock.
  - `OrphanProposal` — frozen dataclass with `orphan`, `candidate_partner`,
    `evidence_chunk_id`, `evidence_text`.
  - `OrphanTokenBudgetExceeded` — mirrors the Pass-2 exception shape so
    the workflow can swallow it in a single `except` clause.
- `pipelines/entity-filtering/src/entity_filtering/resolution/orphan_prompts.py`
  (~80 lines): `ORPHAN_CONFIRM_SYSTEM_PROMPT` + `build_orphan_confirm_prompt`.
- `pipelines/entity-filtering/tests/test_orphan_connector.py`
  (38 tests across 7 classes — well over the 6+ target).

## Files modified

- `pipelines/entity-filtering/src/entity_filtering/config.py` — added
  `OrphanConnectorConfig{enabled=True, max_proposals_per_orphan=3,
  min_confidence=0.6}` and wired the field on `FilteringConfig`.
- `pipelines/entity-filtering/src/entity_filtering/workflow.py` — added
  Stage 14 (orphan-connector) after edge prediction. The stage runs only
  when **all four** DI inputs (`source_id`, `chunks`, `orphan_entity_repo`,
  `orphan_llm_caller`) are present **and** the config is enabled. This
  preserves backward compatibility — every existing `workflow.process(...)`
  call still works without modification. The new orphan relations are
  appended to `filtered_relations` (strict ADD-ONLY; never removes rows).
  Token-budget breaches log + skip rather than failing the whole pipeline.
- `pipelines/entity-filtering/src/entity_filtering/resolution/__init__.py`
  — re-exports the new public surface.
- `packages/surrealdb-service/src/surrealdb_service/repositories/entity.py`
  — added `list_orphans_for_source(source_id)`. Implemented as two SQL
  steps (1× SELECT entities by source, then N× SELECT relation LIMIT 1
  per entity) because SurrealDB's sub-query-COUNT on RELATE tables is
  awkward. The N+1 cost is acceptable on the orphan-detection path
  because the orphan set is by definition small.

## Typical orphan-proposal flow (worked example)

**Input** (a source with 1 orphan):

- Repo state: `entity:alice` (no relations) and 2 other connected
  entities (`entity:bob`, `entity:mit`) for `source:demo`.
- Chunks fed to the stage:
  ```python
  [{"id": "c-1", "text": "Alice met Bob at MIT.",
    "entities": ["Alice", "Bob", "MIT"]}]
  ```
- LLM stub returns `{"relation_type": "MET", "confidence": 0.85}` for
  the Alice→Bob call and `{"relation_type": null, "confidence": 0.92}`
  for Alice→MIT.

**Stage 1** — `find_orphans("source:demo", repo)` →
`[{"id": "entity:alice", "canonical_name": "Alice", ...}]`.

**Stage 2** — `propose_connections([...], chunks,
max_proposals_per_orphan=3)` →
```
[OrphanProposal(orphan="Alice", candidate_partner="Bob",
                evidence_chunk_id="c-1", evidence_text="Alice met..."),
 OrphanProposal(orphan="Alice", candidate_partner="MIT",
                evidence_chunk_id="c-1", evidence_text="Alice met...")]
```

**Stage 3** — `confirm_connections(proposals, llm, min_confidence=0.6)` →
```
[ExtractedRelation(source_entity="Alice", target_entity="Bob",
                   relation_type="MET", confidence=0.85,
                   source_chunk_id="c-1",
                   properties={"extraction_method": "orphan_connector"})]
```

The Alice→MIT proposal is dropped because the LLM returned
`relation_type=null` (negative confirmation). The Alice→Bob relation
appears in `FilteredResult.relations`; downstream persistence picks it
up via the existing relation-write path.

## Quality gates

```
cd packages/shared && uv run pytest -q               # 154 passed
cd packages/surrealdb-service && uv run pytest -q    # 77 passed
cd pipelines/entity-filtering &&
    uv run --all-extras pytest tests/test_orphan_connector.py
    --cov=entity_filtering.resolution.orphan_connector
    --cov=entity_filtering.resolution.orphan_prompts -q
# 38 passed, 98% / 100% coverage
cd pipelines/entity-filtering &&
    uv run --all-extras pytest -q                    # 488 passed, 1 pre-existing failure
```

The pre-existing failure is
`test_llm_matcher.py::TestMatchPair::test_calls_ollama_for_unknown_pair`
(missing `_agentic_enabled` attribute on `LLMMatcher`). `git diff main
HEAD -- pipelines/entity-filtering/src/entity_filtering/resolution/llm_matcher.py`
is empty — this branch does not touch the matcher, so the failure is
not a B.5a regression.

## Notes / handoffs

- **Workflow integration is opt-in by DI presence**. To exercise Stage 14,
  the caller (B.1f's `EntityExtractionService` or B.5b's prune-lifecycle
  worker) must pass `source_id`, `chunks`, `orphan_entity_repo`,
  `orphan_llm_caller` as kwargs. The plan is explicit that this lands
  before final persistence; the workflow comment block points at the
  Stage 14 location for future readers.
- **B.5b dependencies**: the `OrphanEntityRepoProtocol` is intentionally
  narrow (only `list_orphans_for_source`). B.5b's prune-lifecycle work
  will extend it with `mark_pending_reconnect` and friends — additive,
  no rename needed.
- **No DI container wiring in this PR**. Per the plan, `apps/app-main`
  injection lands when the production caller (B.1f) is wired; this PR
  ships the stage + tests only. The `dependencies.py` factory will
  thread the entity repo + LLM caller through to the workflow when the
  entity-extraction service starts using the orphan-connector.
- **Pre-existing failure has been flagged once before** (B.4 / B.1f
  reviews mentioned `_agentic_enabled`). Not in scope for B.5a.

Ready for review.
