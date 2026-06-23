"""Track M.2 — per-provider RPM caps + env override + pace-not-skip.

Drives :class:`ProviderRateLimiter` with the resolved per-provider caps under a
fake clock (no real sleeping) and asserts:

  * google paces at its low cap while nvidia admits a higher burst;
  * a local provider (ollama) is never throttled;
  * the cap PACES (the over-cap acquire blocks then succeeds within max_wait)
    rather than instantly skipping — the existing J behaviour M keeps (M-D1);
  * env vars (GOOGLE_RPM / NVIDIA_RPM) override the defaults (M-D2).
"""

import pytest
from app_main.services.model_routing.rate_limiter import (
    DEFAULT_PER_PROVIDER_RPM,
    ProviderRateLimiter,
    RateLimitTimeout,
    resolve_per_provider_rpm,
)


class FakeTime:
    """Fake monotonic clock whose ``sleep`` advances the clock instantly."""

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


class TestResolveCaps:
    def test_defaults(self):
        caps = resolve_per_provider_rpm(environ={})
        assert caps["google"] == DEFAULT_PER_PROVIDER_RPM["google"]
        assert caps["nvidia"] == DEFAULT_PER_PROVIDER_RPM["nvidia"]

    def test_env_override(self):
        caps = resolve_per_provider_rpm(
            environ={"GOOGLE_RPM": "9", "NVIDIA_RPM": "40"}
        )
        assert caps["google"] == 9
        assert caps["nvidia"] == 40

    def test_bad_env_keeps_default(self):
        caps = resolve_per_provider_rpm(
            environ={"GOOGLE_RPM": "not-an-int", "NVIDIA_RPM": "-5"}
        )
        assert caps["google"] == DEFAULT_PER_PROVIDER_RPM["google"]
        assert caps["nvidia"] == DEFAULT_PER_PROVIDER_RPM["nvidia"]

    def test_ollama_override_only_when_set(self):
        # No default ollama entry; a high override is honoured when set.
        caps = resolve_per_provider_rpm(environ={"OLLAMA_RPM": "100000"})
        assert caps["ollama"] == 100000


@pytest.mark.asyncio
class TestPerProviderCapsWired:
    async def test_google_paces_at_low_cap(self):
        ft = FakeTime()
        caps = resolve_per_provider_rpm(environ={})
        limiter = _limiter(ft, per_provider_rpm=caps, max_wait_seconds=120.0)

        # Admit the full google cap immediately (no sleep).
        for _ in range(caps["google"]):
            await limiter.acquire("google")
        assert ft.now() == 0.0

        # The next google acquire must PACE — block (sleep) until a slot ages
        # out of the 60s window, then succeed (not instant-skip).
        await limiter.acquire("google")
        assert ft.slept, "over-cap google acquire should have paced (slept)"
        assert ft.now() == pytest.approx(60.0)

    async def test_nvidia_admits_higher_burst_than_google(self):
        ft = FakeTime()
        caps = resolve_per_provider_rpm(environ={})
        limiter = _limiter(ft, per_provider_rpm=caps, max_wait_seconds=120.0)

        # nvidia's cap is higher than google's — a burst the size of google's
        # cap is admitted immediately on nvidia with no pacing.
        for _ in range(caps["google"] + 1):
            await limiter.acquire("nvidia")
        assert ft.now() == 0.0
        assert caps["nvidia"] > caps["google"]

    async def test_ollama_local_never_throttled(self):
        ft = FakeTime()
        caps = resolve_per_provider_rpm(environ={})
        limiter = _limiter(ft, per_provider_rpm=caps, max_wait_seconds=120.0)
        for _ in range(200):
            await limiter.acquire("ollama", is_local=True)
        assert ft.now() == 0.0
        assert ft.slept == []

    async def test_pace_then_timeout_past_max_wait(self):
        ft = FakeTime()
        # google cap 6, but max_wait only 5s -> the 7th acquire cannot get a
        # slot within max_wait and raises (-> SKIPPED_RATE_LIMITED failover).
        limiter = _limiter(
            ft, per_provider_rpm={"google": 6}, max_wait_seconds=5.0
        )
        for _ in range(6):
            await limiter.acquire("google")
        with pytest.raises(RateLimitTimeout):
            await limiter.acquire("google")
