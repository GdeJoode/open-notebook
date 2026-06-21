"""Unit tests for the Track J.2 per-provider circuit breaker.

Covers the 5 plan scenarios:
  1. Opens after N consecutive failures.
  2. ``allow`` returns False while OPEN (executor skips without calling).
  3. After cooldown -> HALF_OPEN; one probe success -> CLOSED.
  4. HALF_OPEN probe failure -> OPEN again (cooldown restarts).
  5. Per-provider state is independent in the registry.

A fake monotonic clock drives the cooldown deterministically — no real sleeping.
"""

from __future__ import annotations

import pytest
from app_main.services.model_routing.circuit_breaker import (
    CircuitBreakerRegistry,
    CircuitState,
    ProviderCircuitBreaker,
)


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def _breaker(clock: FakeClock, threshold: int = 3, cooldown: float = 60.0):
    return ProviderCircuitBreaker(
        "nvidia-nim",
        failure_threshold=threshold,
        cooldown_seconds=cooldown,
        clock=clock,
    )


# AC#1 — opens after N consecutive failures --------------------------------


async def test_opens_after_threshold_failures():
    clock = FakeClock()
    cb = _breaker(clock, threshold=3)

    assert await cb.state() == CircuitState.CLOSED
    await cb.record_failure()
    await cb.record_failure()
    assert await cb.state() == CircuitState.CLOSED  # 2 < 3
    await cb.record_failure()
    assert await cb.state() == CircuitState.OPEN


async def test_success_resets_failure_count():
    clock = FakeClock()
    cb = _breaker(clock, threshold=3)

    await cb.record_failure()
    await cb.record_failure()
    await cb.record_success()  # resets the streak
    await cb.record_failure()
    await cb.record_failure()
    assert await cb.state() == CircuitState.CLOSED  # only 2 since reset


# AC#2 — allow() is False while OPEN (skip without calling) -----------------


async def test_allow_false_when_open():
    clock = FakeClock()
    cb = _breaker(clock, threshold=2, cooldown=60.0)

    await cb.record_failure()
    await cb.record_failure()
    assert await cb.state() == CircuitState.OPEN
    assert await cb.allow() is False


# AC#3 — cooldown -> HALF_OPEN -> probe success -> CLOSED -------------------


async def test_half_open_probe_success_closes():
    clock = FakeClock()
    cb = _breaker(clock, threshold=2, cooldown=60.0)

    await cb.record_failure()
    await cb.record_failure()
    assert await cb.state() == CircuitState.OPEN
    assert await cb.allow() is False

    clock.advance(60.0)  # cooldown elapsed
    # First allow() in HALF_OPEN admits exactly one probe.
    assert await cb.allow() is True
    assert await cb.state() == CircuitState.HALF_OPEN
    # A second concurrent caller is NOT admitted while a probe is in flight.
    assert await cb.allow() is False

    await cb.record_success()
    assert await cb.state() == CircuitState.CLOSED
    assert await cb.allow() is True


# AC#4 — HALF_OPEN probe failure -> OPEN again -----------------------------


async def test_half_open_probe_failure_reopens():
    clock = FakeClock()
    cb = _breaker(clock, threshold=2, cooldown=60.0)

    await cb.record_failure()
    await cb.record_failure()
    clock.advance(60.0)
    assert await cb.allow() is True  # probe admitted
    assert await cb.state() == CircuitState.HALF_OPEN

    await cb.record_failure()  # probe fails
    assert await cb.state() == CircuitState.OPEN
    assert await cb.allow() is False

    # Cooldown restarts from the re-open instant.
    clock.advance(59.0)
    assert await cb.allow() is False
    clock.advance(1.0)
    assert await cb.allow() is True  # half-open probe again


# AC#5 — independent per-provider state ------------------------------------


async def test_registry_independent_provider_state():
    clock = FakeClock()
    registry = CircuitBreakerRegistry(
        failure_threshold=2, cooldown_seconds=60.0, clock=clock
    )

    a = await registry.get("nvidia-nim")
    b = await registry.get("ollama")

    await a.record_failure()
    await a.record_failure()
    assert await a.state() == CircuitState.OPEN
    # ollama untouched.
    assert await b.state() == CircuitState.CLOSED
    assert await b.allow() is True

    # Same instance returned for the same provider.
    assert await registry.get("nvidia-nim") is a


async def test_registry_get_nowait_does_not_create():
    registry = CircuitBreakerRegistry()
    assert registry.get_nowait("openai") is None
    created = await registry.get("openai")
    assert registry.get_nowait("openai") is created


@pytest.mark.parametrize("threshold", [1, 5])
async def test_threshold_is_configurable(threshold):
    clock = FakeClock()
    cb = _breaker(clock, threshold=threshold)
    for _ in range(threshold - 1):
        await cb.record_failure()
        assert await cb.state() == CircuitState.CLOSED
    await cb.record_failure()
    assert await cb.state() == CircuitState.OPEN
