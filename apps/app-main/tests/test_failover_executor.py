"""Unit tests for the Track J.2 per-document failover executor.

Covers the plan ACs:
  1. First candidate fails (eligible) -> second succeeds; served_provider is the
     second, attempts has 2 records, first marked failed.
  2. After 3 failures provider P is OPEN -> next execute skips P WITHOUT calling
     it; records a ``skipped_open_circuit`` attempt.
  3. (covered in test_circuit_breaker) half-open probe transitions.
  4. All candidates fail -> AllProvidersFailedError carrying attempt records.
  5. A non-failover-eligible exception (TypeError) propagates immediately,
     without advancing to the next provider and without tripping the breaker.
  6. Exactly one served_provider per successful stage.

PLUS the fair-use additions:
  * Cloud candidate is rate-limited (acquire called) before the call.
  * A 429 (RateLimitError) triggers bounded backoff-retry on the SAME provider
    before that candidate is considered failed and failed over.
  * Local candidates are not rate-limited.

A no-sleep fake clock keeps the rate limiter + backoff instant.
"""

from __future__ import annotations

import pytest
from app_main.services.model_routing.circuit_breaker import (
    CircuitBreakerRegistry,
    CircuitState,
)
from app_main.services.model_routing.failover_executor import (
    AllProvidersFailedError,
    AttemptOutcome,
    FailoverExecutor,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from app_main.services.model_routing.rate_limiter import ProviderRateLimiter
from app_main.services.model_routing.route_resolver import (
    LLMTask,
    ModelCandidate,
    PrivacyMode,
    ResolvedRoute,
)


class FakeTime:
    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    async def sleep(self, dt: float) -> None:
        self.slept.append(dt)
        self.t += dt


def _route(*candidates: ModelCandidate, mode=PrivacyMode.CLOUD) -> ResolvedRoute:
    return ResolvedRoute(
        task=LLMTask.ENTITY_EXTRACTION,
        mode=mode,
        ordered_candidates=list(candidates),
    )


def _cloud(provider: str, model_id: str = "m") -> ModelCandidate:
    return ModelCandidate(
        provider=provider, model_id=model_id, is_local=False
    )


def _local(provider: str = "ollama") -> ModelCandidate:
    return ModelCandidate(provider=provider, model_id="local-m", is_local=True)


def _executor(ft: FakeTime | None = None, **kw) -> FailoverExecutor:
    ft = ft or FakeTime()
    return FailoverExecutor(
        circuit_breakers=CircuitBreakerRegistry(clock=ft.now),
        rate_limiter=ProviderRateLimiter(clock=ft.now, sleep=ft.sleep),
        sleep=ft.sleep,
        **kw,
    )


# AC#1 — failover on eligible error ----------------------------------------


async def test_failover_to_second_provider_on_eligible_error():
    ex = _executor()
    calls: list[str] = []

    async def call(c: ModelCandidate) -> str:
        calls.append(c.provider)
        if c.provider == "nvidia-nim":
            raise ProviderUnavailableError("500")
        return f"ok-{c.provider}"

    route = _route(_cloud("nvidia-nim"), _cloud("openai"))
    result = await ex.execute_with_failover(route, call)

    assert result.value == "ok-openai"
    assert result.served_provider == "openai"
    assert calls == ["nvidia-nim", "openai"]
    assert len(result.attempts) == 2
    assert result.attempts[0].outcome == AttemptOutcome.FAILED
    assert result.attempts[1].outcome == AttemptOutcome.SUCCESS
    assert result.was_failover is True
    assert result.fallback_from == ["nvidia-nim"]


# AC#2 — open circuit is skipped without calling ---------------------------


async def test_open_circuit_is_skipped_without_calling():
    ft = FakeTime()
    ex = _executor(ft)

    async def always_fail(c: ModelCandidate) -> str:
        raise ProviderUnavailableError("500")

    # Drive nvidia-nim to OPEN via 3 failed routes (default threshold 3).
    route = _route(_cloud("nvidia-nim"), _local())

    async def call_fail_cloud_ok_local(c: ModelCandidate) -> str:
        if c.is_local:
            return "local-ok"
        raise ProviderUnavailableError("500")

    for _ in range(3):
        await ex.execute_with_failover(route, call_fail_cloud_ok_local)

    breaker = await ex._breakers.get("nvidia-nim")
    assert await breaker.state() == CircuitState.OPEN

    # Next execution must SKIP nvidia-nim without invoking call for it.
    called_providers: list[str] = []

    async def spy(c: ModelCandidate) -> str:
        called_providers.append(c.provider)
        return "local-ok" if c.is_local else "cloud"

    result = await ex.execute_with_failover(route, spy)
    assert "nvidia-nim" not in called_providers
    assert result.served_provider == "ollama"
    assert result.attempts[0].outcome == AttemptOutcome.SKIPPED_OPEN_CIRCUIT
    assert result.attempts[0].provider == "nvidia-nim"


# AC#4 — all providers fail -> AllProvidersFailedError ---------------------


async def test_all_providers_failed_raises_with_records():
    ex = _executor()

    async def always_fail(c: ModelCandidate) -> str:
        raise ProviderTimeoutError("timeout")

    route = _route(_cloud("nvidia-nim"), _cloud("openai"), _local())
    with pytest.raises(AllProvidersFailedError) as ei:
        await ex.execute_with_failover(route, always_fail)

    attempts = ei.value.attempts
    assert [a.provider for a in attempts] == ["nvidia-nim", "openai", "ollama"]
    assert all(a.outcome == AttemptOutcome.FAILED for a in attempts)


# AC#5 — non-eligible exception propagates immediately ---------------------


async def test_non_eligible_exception_propagates_without_failover():
    ex = _executor()
    calls: list[str] = []

    async def call(c: ModelCandidate) -> str:
        calls.append(c.provider)
        raise TypeError("programming bug")  # NOT failover-eligible

    route = _route(_cloud("nvidia-nim"), _cloud("openai"))
    with pytest.raises(TypeError):
        await ex.execute_with_failover(route, call)

    # Did NOT advance to the second provider.
    assert calls == ["nvidia-nim"]
    # Breaker NOT tripped — this is a bug, not a provider fault.
    breaker = await ex._breakers.get("nvidia-nim")
    assert await breaker.state() == CircuitState.CLOSED


async def test_value_error_also_propagates():
    ex = _executor()

    async def call(c: ModelCandidate) -> str:
        raise ValueError("bad input")

    route = _route(_cloud("nvidia-nim"), _cloud("openai"))
    with pytest.raises(ValueError):
        await ex.execute_with_failover(route, call)


# AC#6 — exactly one served provider ---------------------------------------


async def test_single_served_provider_recorded():
    ex = _executor()

    async def call(c: ModelCandidate) -> str:
        return "ok"

    route = _route(_cloud("nvidia-nim"), _cloud("openai"), _local())
    result = await ex.execute_with_failover(route, call)
    assert result.served_provider == "nvidia-nim"
    assert result.served_model_id == "m"
    assert result.was_failover is False
    assert result.fallback_from == []
    # Only the served candidate produced a SUCCESS record.
    successes = [
        a for a in result.attempts if a.outcome == AttemptOutcome.SUCCESS
    ]
    assert len(successes) == 1


# Fair-use — rate limiter is consulted before cloud calls ------------------


async def test_cloud_candidate_is_rate_limited():
    ft = FakeTime()
    # cap 1/60s so the 2nd cloud call within one stage must wait.
    ex = FailoverExecutor(
        circuit_breakers=CircuitBreakerRegistry(clock=ft.now),
        rate_limiter=ProviderRateLimiter(
            cloud_rpm=1, max_wait_seconds=120.0, clock=ft.now, sleep=ft.sleep
        ),
    )

    async def ok(c: ModelCandidate) -> str:
        return "ok"

    route1 = _route(_cloud("nvidia-nim"))
    await ex.execute_with_failover(route1, ok)  # t=0
    await ex.execute_with_failover(route1, ok)  # second call must wait
    assert ft.now() == pytest.approx(60.0)


async def test_local_candidate_not_rate_limited():
    ft = FakeTime()
    ex = FailoverExecutor(
        circuit_breakers=CircuitBreakerRegistry(clock=ft.now),
        rate_limiter=ProviderRateLimiter(
            cloud_rpm=1, clock=ft.now, sleep=ft.sleep
        ),
    )

    async def ok(c: ModelCandidate) -> str:
        return "ok"

    route = _route(_local())
    for _ in range(10):
        await ex.execute_with_failover(route, ok)
    assert ft.slept == []  # never throttled a local provider


# Fair-use — 429 triggers bounded backoff-retry before failover ------------


async def test_rate_limit_429_retries_same_provider_then_succeeds():
    ft = FakeTime()
    ex = _executor(ft)
    attempts_per_provider: dict[str, int] = {}

    async def call(c: ModelCandidate) -> str:
        n = attempts_per_provider.get(c.provider, 0) + 1
        attempts_per_provider[c.provider] = n
        if c.provider == "nvidia-nim" and n <= 2:
            raise RateLimitError("429")
        return f"ok-{c.provider}"

    route = _route(_cloud("nvidia-nim"), _cloud("openai"))
    result = await ex.execute_with_failover(route, call)

    # Served by nvidia-nim after 2 backoff-retries on the SAME provider —
    # did NOT fail over to openai.
    assert result.served_provider == "nvidia-nim"
    assert attempts_per_provider["nvidia-nim"] == 3
    assert "openai" not in attempts_per_provider
    assert ft.slept == [1.0, 2.0]  # bounded exponential backoff
    # Exactly one attempt record (the eventual success), not one per retry.
    assert len(result.attempts) == 1


async def test_rate_limit_exhausts_retries_then_fails_over():
    ft = FakeTime()
    ex = _executor(ft)

    async def call(c: ModelCandidate) -> str:
        if c.provider == "nvidia-nim":
            raise RateLimitError("429")  # always rate-limited
        return f"ok-{c.provider}"

    route = _route(_cloud("nvidia-nim"), _cloud("openai"))
    result = await ex.execute_with_failover(route, call)

    # After exhausting backoff retries, nvidia-nim is treated as failed and we
    # fail over to openai.
    assert result.served_provider == "openai"
    assert result.attempts[0].provider == "nvidia-nim"
    assert result.attempts[0].outcome == AttemptOutcome.FAILED
    # default max_retries=3 -> sleeps 1,2,4
    assert ft.slept == [1.0, 2.0, 4.0]


async def test_injectable_eligibility_predicate():
    """J.4 supplies its own whitelist; the executor honors it."""
    ft = FakeTime()

    class _MyTransient(Exception):
        pass

    ex = FailoverExecutor(
        circuit_breakers=CircuitBreakerRegistry(clock=ft.now),
        rate_limiter=ProviderRateLimiter(clock=ft.now, sleep=ft.sleep),
        is_failover_eligible=lambda e: isinstance(e, _MyTransient),
    )

    async def call(c: ModelCandidate) -> str:
        if c.provider == "nvidia-nim":
            raise _MyTransient()
        return "ok"

    route = _route(_cloud("nvidia-nim"), _cloud("openai"))
    result = await ex.execute_with_failover(route, call)
    assert result.served_provider == "openai"
