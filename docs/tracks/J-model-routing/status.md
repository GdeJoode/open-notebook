# Track J — Cloud/local model routing — status

Append-only ledger. One row per phase attempt.

| Phase | Title | PR | Branch | Date | Status |
|---|---|---|---|---|---|
| — | (planning) | — | — | 2026-06-21 | track-planner producing plan.md |
| J.1 | Provider-chain config + privacy-aware ordered-route resolver | (pending) | track/j1-route-resolver | 2026-06-21 | Implemented — all 7 ACs met, tests green, ready for review |
| J.2 | Per-provider circuit-breaker + per-document failover executor + fair-use rate limiter | (pending) | track/j2-failover | 2026-06-21 | Implemented — all 6 plan ACs + fair-use additions met, 29 J.2 tests green, B.8/J.1 regression green, ready for review |
| J.2 rev2 | Adversarial-review revisions (1 major + 1 minor) | (pending) | track/j2-failover | 2026-06-21 | Major + minor fixed; 32 J.2 tests green (3 new), 35-test regression green, changed files ruff-clean. Ready for re-review |

### J.2 rev2 revision notes (2026-06-21)
Addresses the J.2 adversarial review (REVISIONS_NEEDED: 1 major + 1 worth-fixing minor).

- **MAJOR fixed — `RateLimitTimeout` now fails over to local WITHOUT a breaker penalty** (`failover_executor.py`): previously a saturated *local* rate limiter raised `RateLimitTimeout`, which is not a `FailoverEligibleError`, so it propagated and hard-failed the whole document stage — defeating the J-Q4 "local is the last resort" guarantee. Now `execute_with_failover` catches `RateLimitTimeout` in a dedicated branch *before* the eligibility classifier: it records a new `AttemptOutcome.SKIPPED_RATE_LIMITED` attempt + fallback log event and advances to the next candidate (ultimately local, which is exempt from throttling), but does **NOT** call `breaker.record_failure()` — our throttle saturating is not evidence the provider is down, so opening its breaker would be wrong. The three categories are now distinct: local-acquire-timeout → failover, no breaker penalty; provider 429 → bounded backoff-retry-same-provider then failover + breaker-failure (unchanged); hard 5xx/timeout/auth → immediate failover + breaker-failure (unchanged).
- **MINOR fixed — backoff retries now re-acquire a rate-limit slot per request** (`failover_executor._invoke`): the `acquire()` was hoisted out of the backoff loop, so a sustained-429 stage issued up to ~4 endpoint requests against 1 consumed slot, under-counting the fair-use cap. The acquire now lives inside the `_do` closure so the initial call AND every backoff retry each consume a slot. If a retry's acquire times out, the resulting `RateLimitTimeout` (not a 429) is not retried by `backoff_retry` — it propagates to the executor's saturation branch (failover-to-local, no breaker penalty), keeping behavior consistent with the major fix.
- **New tests** (`test_failover_executor.py`, +3, total 32 green):
  - `test_rate_limit_saturation_drains_to_local_without_breaker_penalty` — the pinning test: a pre-saturated cloud limiter (`cloud_rpm=1`, `max_wait_seconds=0`, slot pre-consumed) drains to the exempt local candidate → `served_provider == "ollama"`, `was_failover True`, the cloud attempt recorded `SKIPPED_RATE_LIMITED`, and the cloud provider's breaker asserted **still CLOSED**.
  - `test_rate_limit_saturation_alone_exhausts_to_all_failed` — saturation with no local fallback drains every candidate → `AllProvidersFailedError`, all attempts `SKIPPED_RATE_LIMITED`, all breakers still CLOSED.
  - `test_backoff_retries_each_consume_a_rate_limit_slot` — pins the minor fix: `cloud_rpm=2`, `max_wait=0`; the initial call + 1 retry each acquire a slot, the next retry's acquire saturates → failover-to-local without breaker penalty (`cloud_calls == 2`, breaker CLOSED).
- **Validation**: `test_failover_executor.py` + `test_provider_rate_limiter.py` + `test_circuit_breaker.py` = 32 passed. Regression `test_route_resolver.py` + `test_entity_extraction_service.py` = 35 passed. `ruff check` on the two changed files = clean. Fake clock throughout — no real sleeps.

**Deferred to follow-ups (reviewer minors #2/#3 — NOT fixed in rev2 per scope):**
- **FU-J2-1 (reviewer minor #2)**: `route_resolver.is_provider_healthy(provider)` is an untested, lock-free advisory read of the breaker registry. Add coverage and decide whether it needs to participate in the breaker's asyncio lock (currently a best-effort snapshot; acceptable for advisory use but unpinned by a test).
- **FU-J2-2 (reviewer minor #3)**: the DI singletons (`get_circuit_breaker_registry` / `get_rate_limiter_registry` / `get_failover_executor`, `@lru_cache`) are a test-isolation hazard — tests sharing the process cache can leak breaker/limiter state. Add a `.cache_clear()` fixture (or per-test factory override) before these are exercised through the DI path.

### J.2 implementation notes (2026-06-21)
- **Built**:
  - `services/model_routing/circuit_breaker.py` — `ProviderCircuitBreaker` (CLOSED/OPEN/HALF_OPEN; opens after N=3 consecutive failures, cools down T=60s, one HALF_OPEN probe; asyncio-lock thread-safe; injectable clock) + `CircuitBreakerRegistry` (process singleton, lazy per-provider). In-memory V1 with the **J-D3 multi-worker caveat in the module docstring**.
  - `services/model_routing/rate_limiter.py` — `ProviderRateLimiter` (per-provider sliding-window; conservative cloud default **20 rpm**, local effectively unlimited **100_000 rpm**; `acquire()` blocks until a slot or raises `RateLimitTimeout` past `max_wait` default 30s; per-provider rpm overrides; injectable clock/sleep) + `backoff_retry` (exponential 1s ×2 cap 30s, default 3 retries; 429→retry-same-provider, hard error→re-raise immediately).
  - `services/model_routing/failover_executor.py` — `FailoverExecutor.execute_with_failover(route, call) -> FailoverResult`. Skips OPEN breakers (`skipped_open_circuit` attempt, `call` not invoked), rate-limits + backoff-wraps cloud candidates BEFORE the call, records breaker failure + advances on failover-eligible exceptions, records breaker success + served provider on success. Raises `AllProvidersFailedError(attempts)` only when exhausted. `FailoverResult` carries `value/served_provider/served_model_id/attempts` (+ `was_failover`/`fallback_from` for J-Q7). Failover-eligibility is an **injectable whitelist predicate** (default markers `RateLimitError`/`ProviderUnavailableError`/`ProviderTimeoutError`/`ProviderAuthError`); non-eligible (`TypeError`/`ValueError`) propagate immediately and do NOT trip the breaker.
  - `route_resolver.py` — added `is_provider_healthy(provider)` advisory helper reading the breaker registry.
  - `dependencies.py` — `get_circuit_breaker_registry()` + `get_rate_limiter_registry()` (`@lru_cache` process singletons) + `get_failover_executor()`.
- **Tests** (29 passed): `test_circuit_breaker.py` (open-after-N, skip-when-open, half-open probe success/failure, independent per-provider state, configurable threshold); `test_failover_executor.py` (failover-on-eligible-error, skip-open-circuit, all-failed→`AllProvidersFailedError`, non-eligible-propagates ×2, single served-provider, cloud-rate-limited, local-not-limited, 429-backoff-then-success, 429-exhaust-then-failover, injectable predicate); `test_provider_rate_limiter.py` (cap blocks throughput, local unthrottled, max_wait→`RateLimitTimeout`, per-provider override, backoff success/hard-error/exhaust/cap). Deterministic via fake clock + fake sleep (no real waiting/hangs).
- **Regression**: `test_route_resolver.py` (12) + `test_entity_extraction_service.py` (23) green = 35 passed. J.1 + B.8 intact.
- **Lint**: all 6 changed/new `*.py` files ruff-clean.
- **Rate-limiter defaults chosen**: cloud **20 rpm** (conservative NIM fair-use), local **100_000 rpm** (effectively unlimited), `max_wait` **30s**, window **60s**, backoff base **1s** / factor **2** / cap **30s** / **3** retries. All overridable via constructor / `per_provider_rpm`.
- **Naming deviation**: plan named the rate-limiter test `test_rate_limiter.py`, but that name is already taken by Track I.H1's per-IP HTTP rate-limit tests — used `test_provider_rate_limiter.py` to avoid clobbering.
- **Not wired into live extraction**: the executor is built + tested but NOT yet called from `run_extraction` — that wiring + the real NIM/esperanto error→marker mapping is **J.3/J.4** (guardrail: no live behavior change in J.2). `get_failover_executor()` uses the default whitelist; J.4 injects concrete predicates.
- **Deferred to J.4**: error_mapping (`is_failover_eligible` over real SDK/NIM errors → the J.2 marker types), `call_candidate` (esperanto dispatch), provenance of served provider, summarization-path unification, embeddings/parsing untouched.

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

| J.2 | failover executor + circuit-breaker + fair-use rate limiter | — | `track/j2-failover` | 2026-06-21 | adversarial-reviewer APPROVED (attempt 2; 1 major fixed: RateLimitTimeout→failover-to-local w/o breaker penalty). 67 tests green. FU-J2-1/2 deferred. |
