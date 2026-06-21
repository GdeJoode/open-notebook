"""Unit tests for the Track J.2 fair-use provider rate limiter + backoff helper.

NOTE on filename: the plan names this ``test_rate_limiter.py``, but that name is
already taken by Track I.H1's per-IP HTTP rate-limit tests. This file holds the
J.2 ProviderRateLimiter tests under a non-colliding name.

Covers the addition-beyond-plan acceptance points:
  * Rate limiter caps throughput: with a small cap the next immediate acquire
    blocks until a slot opens (driven by a fake clock that advances inside the
    injected ``sleep`` — no real waiting, no hang).
  * Local providers are NOT throttled (high cap window).
  * ``max_wait`` raises :class:`RateLimitTimeout` rather than hanging.
  * 429 -> bounded backoff-retry on the same provider before failover; a hard
    error re-raises immediately; retries exhaust then re-raise.

All timing is deterministic via a fake clock + a fake sleep that advances it.
"""

from __future__ import annotations

import pytest
from app_main.services.model_routing.rate_limiter import (
    ProviderRateLimiter,
    RateLimitTimeout,
    backoff_retry,
)


class FakeTime:
    """Fake monotonic clock whose ``sleep`` advances the clock instead of
    waiting — keeps the sliding-window math exact and the test instant."""

    def __init__(self, t: float = 0.0) -> None:
        self.t = t
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    async def sleep(self, dt: float) -> None:
        self.slept.append(dt)
        self.t += dt


def _limiter(ft: FakeTime, **kw) -> ProviderRateLimiter:
    return ProviderRateLimiter(clock=ft.now, sleep=ft.sleep, **kw)


# --- throughput cap --------------------------------------------------------


async def test_cloud_acquire_blocks_when_cap_reached():
    """cap 2 over a 60s window: after 2 immediate grants the 3rd waits the full
    window before a slot opens."""
    ft = FakeTime()
    # max_wait must exceed the window so the blocked acquire can eventually
    # proceed rather than time out.
    limiter = _limiter(ft, cloud_rpm=2, max_wait_seconds=120.0)

    await limiter.acquire("nvidia-nim")  # t=0, slot 1
    await limiter.acquire("nvidia-nim")  # t=0, slot 2 (cap reached)

    await limiter.acquire("nvidia-nim")  # must wait for slot 1 to age out
    assert ft.slept, "third acquire should have slept for a slot"
    assert ft.now() == pytest.approx(60.0)


async def test_second_immediate_acquire_waits_with_tight_cap():
    """With a cap of 1 per 60s the 2nd immediate acquire waits ~60s."""
    ft = FakeTime()
    limiter = _limiter(ft, cloud_rpm=1, max_wait_seconds=120.0)

    await limiter.acquire("nvidia-nim")  # t=0
    await limiter.acquire("nvidia-nim")  # waits
    assert ft.now() == pytest.approx(60.0)


# --- local providers not throttled ----------------------------------------


async def test_local_provider_not_throttled():
    ft = FakeTime()
    limiter = _limiter(ft, cloud_rpm=1)  # cloud cap tiny; local must ignore it

    for _ in range(50):
        await limiter.acquire("ollama", is_local=True)
    assert ft.now() == 0.0  # never slept
    assert ft.slept == []


# --- max_wait bound --------------------------------------------------------


async def test_acquire_raises_on_max_wait_exceeded():
    ft = FakeTime()
    # cap 1 over a 60s window but max_wait only 5s -> the 2nd acquire can't get
    # a slot in time and raises instead of hanging.
    limiter = _limiter(ft, cloud_rpm=1, max_wait_seconds=5.0)

    await limiter.acquire("nvidia-nim")
    with pytest.raises(RateLimitTimeout):
        await limiter.acquire("nvidia-nim")


async def test_per_provider_rpm_override():
    ft = FakeTime()
    limiter = _limiter(ft, cloud_rpm=1, per_provider_rpm={"nvidia-nim": 3})
    for _ in range(3):
        await limiter.acquire("nvidia-nim")  # override allows 3 immediate
    assert ft.now() == 0.0


# --- backoff_retry ---------------------------------------------------------


class _RateLimited(Exception):
    pass


class _HardError(Exception):
    pass


def _is_rl(exc: Exception) -> bool:
    return isinstance(exc, _RateLimited)


async def test_backoff_retries_on_rate_limit_then_succeeds():
    ft = FakeTime()
    calls = {"n": 0}

    async def call():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise _RateLimited()
        return "ok"

    result = await backoff_retry(
        call, is_rate_limit=_is_rl, sleep=ft.sleep, max_retries=3
    )
    assert result == "ok"
    assert calls["n"] == 3
    assert ft.slept == [1.0, 2.0]  # exponential: 1s then 2s


async def test_backoff_hard_error_propagates_immediately():
    ft = FakeTime()

    async def call():
        raise _HardError()

    with pytest.raises(_HardError):
        await backoff_retry(call, is_rate_limit=_is_rl, sleep=ft.sleep)
    assert ft.slept == []  # never backed off a non-rate-limit error


async def test_backoff_exhausts_retries_then_reraises():
    ft = FakeTime()
    calls = {"n": 0}

    async def call():
        calls["n"] += 1
        raise _RateLimited()

    with pytest.raises(_RateLimited):
        await backoff_retry(
            call, is_rate_limit=_is_rl, sleep=ft.sleep, max_retries=2
        )
    assert calls["n"] == 3  # initial + 2 retries
    assert ft.slept == [1.0, 2.0]


async def test_backoff_delay_is_capped():
    ft = FakeTime()

    async def call():
        raise _RateLimited()

    with pytest.raises(_RateLimited):
        await backoff_retry(
            call,
            is_rate_limit=_is_rl,
            sleep=ft.sleep,
            base_delay=10.0,
            factor=2.0,
            max_delay=30.0,
            max_retries=4,
        )
    assert ft.slept == [10.0, 20.0, 30.0, 30.0]  # capped at 30
