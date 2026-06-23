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
    default_is_rate_limit,
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


async def test_backoff_retries_each_consume_a_rate_limit_slot():
    """Fair-use: every actual request to the cloud endpoint — the initial call
    AND each backoff retry — must consume a rate-limit slot, so a sustained-429
    stage isn't under-counted against the rpm cap. A 429 here is retried up to
    max_retries times; with a cap below that count, a later retry's acquire must
    saturate, which fails the candidate over WITHOUT a breaker penalty."""
    ft = FakeTime()
    # cap 2 over the window, max_wait 0: first two acquires (initial + 1 retry)
    # succeed instantly; the 3rd request's acquire saturates and times out.
    ex = FailoverExecutor(
        circuit_breakers=CircuitBreakerRegistry(clock=ft.now),
        rate_limiter=ProviderRateLimiter(
            cloud_rpm=2, max_wait_seconds=0.0, clock=ft.now, sleep=ft.sleep
        ),
        sleep=ft.sleep,
    )

    cloud_calls = {"n": 0}

    async def call(c: ModelCandidate) -> str:
        if c.provider == "nvidia-nim":
            cloud_calls["n"] += 1
            raise RateLimitError("429")  # always rate-limited
        return f"ok-{c.provider}"

    route = _route(_cloud("nvidia-nim"), _local("ollama"))
    result = await ex.execute_with_failover(route, call)

    # Two endpoint requests fit the cap (initial + 1 retry, each acquiring a
    # slot); the 3rd retry's acquire saturated -> RateLimitTimeout -> failover.
    assert cloud_calls["n"] == 2
    assert result.served_provider == "ollama"
    # Saturation path: recorded as rate-limited, breaker stays CLOSED.
    assert result.attempts[0].outcome == AttemptOutcome.SKIPPED_RATE_LIMITED
    breaker = await ex._breakers.get("nvidia-nim")
    assert await breaker.state() == CircuitState.CLOSED


# Fair-use — saturated LOCAL rate limiter fails over WITHOUT breaker penalty --


async def test_rate_limit_saturation_drains_to_local_without_breaker_penalty():
    """The "don't overload" saturation scenario (J-Q4): when OUR cloud rate
    limiter cannot grant a slot within max_wait, the document must fail over to
    the local candidate (exempt from throttling) and the cloud provider's
    breaker must stay CLOSED — a saturated throttle is not evidence the provider
    is down, so opening its breaker would be wrong."""
    ft = FakeTime()
    # Cloud cap 1, and max_wait 0 so any contended acquire times out instantly.
    limiter = ProviderRateLimiter(
        cloud_rpm=1, max_wait_seconds=0.0, clock=ft.now, sleep=ft.sleep
    )
    # Pre-consume the only cloud slot so the executor's acquire saturates.
    await limiter.acquire("nvidia-nim")

    ex = FailoverExecutor(
        circuit_breakers=CircuitBreakerRegistry(clock=ft.now),
        rate_limiter=limiter,
        sleep=ft.sleep,
    )

    called: list[str] = []

    async def call(c: ModelCandidate) -> str:
        called.append(c.provider)
        return f"ok-{c.provider}"

    route = _route(_cloud("nvidia-nim"), _local("ollama"))
    result = await ex.execute_with_failover(route, call)

    # Drained to local; the cloud call was never even issued (slot never granted).
    assert result.served_provider == "ollama"
    assert result.was_failover is True
    assert called == ["ollama"]

    # The cloud attempt is recorded as rate-limit saturated, NOT failed.
    assert result.attempts[0].provider == "nvidia-nim"
    assert result.attempts[0].outcome == AttemptOutcome.SKIPPED_RATE_LIMITED
    assert result.attempts[1].outcome == AttemptOutcome.SUCCESS

    # Breaker for the cloud provider must still be CLOSED: our throttle
    # saturating did NOT record a failure against it.
    breaker = await ex._breakers.get("nvidia-nim")
    assert await breaker.state() == CircuitState.CLOSED


async def test_rate_limit_saturation_alone_exhausts_to_all_failed():
    """A saturated cloud limiter with no local fallback drains every candidate
    and raises AllProvidersFailedError — still without any breaker penalty."""
    ft = FakeTime()
    limiter = ProviderRateLimiter(
        cloud_rpm=1, max_wait_seconds=0.0, clock=ft.now, sleep=ft.sleep
    )
    await limiter.acquire("nvidia-nim")
    await limiter.acquire("openai")

    ex = FailoverExecutor(
        circuit_breakers=CircuitBreakerRegistry(clock=ft.now),
        rate_limiter=limiter,
        sleep=ft.sleep,
    )

    async def call(c: ModelCandidate) -> str:
        return "ok"

    route = _route(_cloud("nvidia-nim"), _cloud("openai"))
    with pytest.raises(AllProvidersFailedError) as ei:
        await ex.execute_with_failover(route, call)

    assert all(
        a.outcome == AttemptOutcome.SKIPPED_RATE_LIMITED
        for a in ei.value.attempts
    )
    for provider in ("nvidia-nim", "openai"):
        breaker = await ex._breakers.get(provider)
        assert await breaker.state() == CircuitState.CLOSED


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


# default_is_rate_limit classifier — 429-as-status, not 429-in-a-number --------


class TestDefaultIsRateLimit:
    """The default 429 classifier must match 429 as an HTTP STATUS, never as a
    substring of a longer number (the old bare ``" 429"`` marker false-matched
    bodies like '4290 tokens')."""

    def test_typed_rate_limit_error(self):
        assert default_is_rate_limit(RateLimitError("nope")) is True

    def test_phrase_markers(self):
        assert default_is_rate_limit(RuntimeError("RESOURCE_EXHAUSTED")) is True
        assert default_is_rate_limit(RuntimeError("rate limit reached")) is True
        assert default_is_rate_limit(RuntimeError("exceeded your current quota"))
        assert default_is_rate_limit(RuntimeError("Quota exceeded for model"))

    def test_status_429_variants_match(self):
        assert default_is_rate_limit(RuntimeError("HTTP 429: Too Many Requests"))
        assert default_is_rate_limit(RuntimeError("status 429"))
        assert default_is_rate_limit(RuntimeError("error code: 429"))
        assert default_is_rate_limit(RuntimeError("got 429 from provider"))

    def test_429_inside_longer_number_is_not_a_rate_limit(self):
        # The regression: a non-rate-limit error whose body merely CONTAINS the
        # digits 429 in a larger token must NOT be misclassified.
        assert default_is_rate_limit(RuntimeError("prompt is 4290 tokens")) is False
        assert default_is_rate_limit(RuntimeError("token id 14290 invalid")) is False
        assert default_is_rate_limit(ValueError("model dim 4296 mismatch")) is False

    def test_unrelated_error_is_not_a_rate_limit(self):
        assert default_is_rate_limit(ValueError("bad json")) is False
