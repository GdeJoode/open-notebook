# Track J — Cloud/local model routing — status

Append-only ledger. One row per phase attempt.

| Phase | Title | PR | Branch | Date | Status |
|---|---|---|---|---|---|
| — | (planning) | — | — | 2026-06-21 | track-planner producing plan.md |
| J.1 | Provider-chain config + privacy-aware ordered-route resolver | (pending) | track/j1-route-resolver | 2026-06-21 | Implemented — all 7 ACs met, tests green, ready for review |

### J.1 implementation notes (2026-06-21)
- **Built**: migration 51 + 51_down (`model_route` SCHEMAFULL table, one row per task, ordered `provider_chain` + local-only `private_chain`, UNIQUE index on `task`); `shared.models.llm.ModelRoute`; `ModelRouteRepository` (get_by_task / upsert singleton-per-task / list_all); `services/model_routing/route_resolver.py` (`LLMTask`{ENTITY_EXTRACTION, SUMMARIZATION, CHAT}, `PrivacyMode`{CLOUD, PRIVATE}, `ModelCandidate`, `ResolvedRoute`, `RouteResolver` + module-level `resolve_route`); DI factories `get_model_route_repo()` + `get_route_resolver()`; `resolve_default_model_id` reworked into a thin shim over the route head with an inline B.8-precedence fallback.
- **Tests**: `apps/app-main/tests/test_route_resolver.py` — 12 passed (the 6 plan ACs + disabled-entry/disabled-model/configured-chain-override coverage). `packages/surrealdb-service/tests/test_model_route_repository.py` — 6 passed (`requires_docker`; migration 51 applied + idempotent, provider_chain ordering roundtrip, singleton-per-task). B.8 regression: `test_entity_persistence_service.py` (10) + `test_entity_extraction_service.py` (23) green unmodified — shim intact, `resolve_default_model_id` still returns a single id.
- **Lint**: changed `*.py` files ruff-clean for the rules I touched (auto-fixed I001 on my files only). Pre-existing repo I001 debt + the 2 pre-existing F821 forward-ref annotations in `entity_extraction_service.py` left untouched per scope.
- **Decision points hit**:
  - **J-Q3 (Anthropic model id)**: the `claude-api` skill/reference is NOT installed in this sandbox (the plan flagged this). Used a sensible placeholder `claude-sonnet-4-5` in `_DEFAULT_PROVIDER_MODEL_NAME` with a `TODO(J-Q3)`. NON-BLOCKING for J.1: the placeholder is only used as a synthesized id when no DB-backed default model id is configured; J.4 seeds the real Anthropic `model` row + final id.
  - **J-D1 (table vs DefaultModels)**: implemented as a dedicated `model_route` table per the plan recommendation.
- **No live behavior change**: the resolver is wired but only the head candidate (`ordered_candidates[0]`) is consumed via the shim; failover execution is deferred to J.2.
- **Deferred (per plan scope)**: failover executor + circuit breaker (J.2); privacy-flag plumbing + embedding-local guardrail test (J.3); cloud provider call layer + provenance of served provider (J.4).

## Decisions (from intake 2026-06-21)
- J-Q1 granularity: layered (global → notebook → document; `private` wins, sticky).
- J-Q2 scope: LLM stages only (extraction/summarization/chat); embeddings + parsing stay local/fixed.
- J-Q3 providers: track-planner proposes default chain (Anthropic → OpenAI → local), configurable.
- J-Q4 failover: per-document (whole doc fails over to next provider; local last in cloud mode).

| J.1 | provider-chain config + route resolver | — | `track/j1-route-resolver` | 2026-06-21 | adversarial-reviewer APPROVED (0 blockers/majors; 5 minor test-coverage nits → J.2). 51 tests green; B.8 shim safe; I.G guardrail holds. Ready for merge sign-off. |

**J.1 deferred to J.2** (reviewer minors): route-path shim test (currently exercises the fallback), 51_down execution test, convert the PRIVATE-mode `assert` to an explicit raise (python -O strips asserts), test the empty-private_chain derive branch.
**J-Q3 RESOLVED (2026-06-21, user)**: cloud provider = **NVIDIA NIM** (OpenAI-compatible REST).
- base_url `https://integrate.api.nvidia.com/v1`, auth `Bearer`, env var **`NVIDIA_API_KEY`**.
- cloud model: **`mistralai/mistral-medium-3.5-128b`**. Validated live (single call, returns OK).
- Key lives in gitignored `NIM info.txt` (never committed). J.4 wires the NIM provider into the registry + seeds the `model`/`model_route` rows; J.1's hard-coded default chain gets repointed to `[nvidia-nim → local]`.
- **Fair-use**: "don't overload the endpoint" → J.2 must rate-limit (conservative default, e.g. ~20 req/min, configurable) + exponential backoff on 429, per-document sequential (no parallel hammering). NIM hosts multiple models, so a same-endpoint model fallback (e.g. mistral → llama-3.3-70b) is an option for the 'multiple cloud providers' redundancy the user wanted.
