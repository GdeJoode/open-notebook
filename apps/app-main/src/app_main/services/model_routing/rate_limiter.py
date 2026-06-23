"""Per-provider fair-use rate limiter for cloud calls (Track J.2).

Motivation: the NVIDIA NIM endpoint (and cloud providers generally) are
rate-limited and "must not be overloaded". Before the failover executor issues
a call to a *cloud* candidate it must ``await acquire(provider)`` so the
aggregate request rate to that provider stays under a conservative cap.

Design:
  * One sliding-window limiter per provider (a deque of recent grant
    timestamps). ``acquire`` blocks until the oldest grant in the window has
    aged out enough to admit a new request, then records the new grant.
  * Local providers (ollama / llamacpp) are effectively unthrottled — running
    against your own GPU has no fair-use concern — so their limiter has a very
    high cap.
  * A ``max_wait`` bound prevents an unbounded hang if a provider is badly
    backed up; exceeding it raises :class:`RateLimitTimeout` rather than
    blocking forever.

The window/clock are injectable so tests can drive the limiter deterministically
with a fake clock and small windows (no real sleeping in unit tests).

This module is provider-agnostic on purpose: it knows only "cloud vs local"
(via the route's ``is_local`` signal supplied by the caller). Mapping concrete
provider names to caps lives in the registry config, not in NIM/esperanto code
(that is J.4).
"""

from __future__ import annotations

import asyncio
import os
import time
from collections import deque
from typing import Awaitable, Callable, Deque, Dict, TypeVar

from loguru import logger

T = TypeVar("T")

# Conservative cloud default: 20 requests/minute. Chosen deliberately low for
# the NIM fair-use ask ("don't overload"); operators raise it per provider via
# the registry config once they know their quota.
DEFAULT_CLOUD_RPM = 20
# Local providers are not a fair-use concern; give them an effectively
# unlimited cap so ``acquire`` never blocks for ollama/llamacpp.
DEFAULT_LOCAL_RPM = 100_000

_WINDOW_SECONDS = 60.0

# Track M.2: per-provider request-rate caps (requests / 60s). These pace each
# CLOUD provider under its real quota; the per-MODEL context window is a
# separate axis (handled by the context packer), so caps live on the provider.
#
#   google ~ 6   — Gemini free tier is ~10/min; stay conservatively under it so
#                  the sliding window itself respects the cap rather than
#                  relying on 429 backoff after the fact.
#   nvidia ~ 30  — NIM fair-use ceiling is ~40/min; leave headroom.
#   ollama unlimited — local GPU has no fair-use concern (uses DEFAULT_LOCAL_RPM
#                  via the high-cap path; no entry needed, listed for clarity).
DEFAULT_PER_PROVIDER_RPM: Dict[str, int] = {
    "google": 6,
    "nvidia": 30,
}

# Env var per provider that overrides its cap at construction (Decision M-D2:
# DI/env config map, no registry schema change, ops-tunable).
_RPM_ENV_VARS: Dict[str, str] = {
    "google": "GOOGLE_RPM",
    "nvidia": "NVIDIA_RPM",
    "ollama": "OLLAMA_RPM",
}


def resolve_per_provider_rpm(
    base: Dict[str, int] | None = None,
    *,
    environ: Dict[str, str] | None = None,
) -> Dict[str, int]:
    """Resolve the per-provider RPM cap map, applying env overrides (M.2).

    Starts from :data:`DEFAULT_PER_PROVIDER_RPM` (or ``base``) and overlays any
    ``GOOGLE_RPM`` / ``NVIDIA_RPM`` / ``OLLAMA_RPM`` env var that parses as a
    positive int. An unset or unparseable var leaves the default in place. This
    keeps the cap source in the DI/env layer (no registry/DB schema change) and
    lets an operator tune a provider's quota without a code change.
    """
    env = environ if environ is not None else os.environ
    resolved = dict(base if base is not None else DEFAULT_PER_PROVIDER_RPM)
    for provider, var in _RPM_ENV_VARS.items():
        raw = env.get(var)
        if raw is None:
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            logger.warning(
                f"resolve_per_provider_rpm: {var}={raw!r} is not an int; "
                f"keeping default for {provider}."
            )
            continue
        if value <= 0:
            logger.warning(
                f"resolve_per_provider_rpm: {var}={value} must be > 0; "
                f"keeping default for {provider}."
            )
            continue
        resolved[provider] = value
    return resolved


class RateLimitTimeout(Exception):
    """Raised when :meth:`ProviderRateLimiter.acquire` cannot get a slot
    within ``max_wait`` seconds."""


class _SlidingWindow:
    """Sliding-window limiter for a single provider.

    Admits at most ``rpm`` grants within any ``window`` seconds. Thread-safe
    via an asyncio lock; cooperative ``sleep`` is used to wait for a slot so
    other tasks make progress while one is throttled.
    """

    def __init__(
        self,
        rpm: int,
        *,
        window: float = _WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._rpm = max(1, rpm)
        self._window = window
        self._clock = clock
        self._sleep = sleep
        self._grants: Deque[float] = deque()
        self._lock = asyncio.Lock()

    def _evict(self, now: float) -> None:
        cutoff = now - self._window
        while self._grants and self._grants[0] <= cutoff:
            self._grants.popleft()

    async def acquire(self, max_wait: float) -> None:
        deadline = self._clock() + max_wait
        while True:
            async with self._lock:
                now = self._clock()
                self._evict(now)
                if len(self._grants) < self._rpm:
                    self._grants.append(now)
                    return
                # Wait until the oldest grant exits the window.
                wait_for = self._grants[0] + self._window - now
            if self._clock() + wait_for > deadline:
                raise RateLimitTimeout(
                    f"rate limiter: no slot within {max_wait:.1f}s "
                    f"(cap {self._rpm}/{self._window:.0f}s)"
                )
            # Small floor so a fake clock that doesn't advance via sleep still
            # yields rather than spins.
            await self._sleep(max(wait_for, 0.0))


class ProviderRateLimiter:
    """Per-provider sliding-window rate limiter.

    Callers ``await acquire(provider, is_local=...)`` before each cloud call.
    Local providers resolve to a high-cap window so they are never throttled.
    """

    def __init__(
        self,
        *,
        cloud_rpm: int = DEFAULT_CLOUD_RPM,
        local_rpm: int = DEFAULT_LOCAL_RPM,
        per_provider_rpm: Dict[str, int] | None = None,
        max_wait_seconds: float = 30.0,
        window: float = _WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._cloud_rpm = cloud_rpm
        self._local_rpm = local_rpm
        self._per_provider_rpm = per_provider_rpm or {}
        self._max_wait = max_wait_seconds
        self._window = window
        self._clock = clock
        self._sleep = sleep
        self._windows: Dict[str, _SlidingWindow] = {}
        self._lock = asyncio.Lock()

    def _rpm_for(self, provider: str, is_local: bool) -> int:
        if provider in self._per_provider_rpm:
            return self._per_provider_rpm[provider]
        return self._local_rpm if is_local else self._cloud_rpm

    async def _window_for(self, provider: str, is_local: bool) -> _SlidingWindow:
        async with self._lock:
            win = self._windows.get(provider)
            if win is None:
                win = _SlidingWindow(
                    self._rpm_for(provider, is_local),
                    window=self._window,
                    clock=self._clock,
                    sleep=self._sleep,
                )
                self._windows[provider] = win
            return win

    async def acquire(self, provider: str, *, is_local: bool = False) -> None:
        """Block until a request slot is available for ``provider``.

        Local providers pass through a high-cap window (effectively
        non-blocking). Raises :class:`RateLimitTimeout` if no slot opens within
        ``max_wait_seconds``.
        """
        win = await self._window_for(provider, is_local)
        await win.acquire(self._max_wait)


async def backoff_retry(
    call: Callable[[], Awaitable[T]],
    *,
    is_rate_limit: Callable[[Exception], bool],
    base_delay: float = 1.0,
    factor: float = 2.0,
    max_delay: float = 30.0,
    max_retries: int = 3,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Run ``call`` with bounded exponential backoff on rate-limit (429) errors.

    Distinguishes a *rate-limit* error (``is_rate_limit(exc)`` True) — which is
    retried on the SAME provider after a growing delay — from any other
    exception, which is re-raised immediately so the failover executor can move
    to the next provider. After ``max_retries`` exhausted rate-limit attempts
    the last rate-limit exception is re-raised (the candidate is then treated as
    failed and failed over).

    Delays: base * factor**attempt, capped at ``max_delay`` (1s, 2s, 4s, … ≤30s).
    The ``sleep`` callable is injectable so tests advance a fake clock without
    real waiting.
    """
    attempt = 0
    while True:
        try:
            return await call()
        except Exception as exc:  # noqa: BLE001 — classify then re-raise
            if not is_rate_limit(exc):
                raise
            if attempt >= max_retries:
                logger.warning(
                    f"backoff_retry: rate-limited {attempt + 1}x, giving up "
                    "(failing over)"
                )
                raise
            delay = min(base_delay * (factor**attempt), max_delay)
            logger.info(
                f"backoff_retry: rate-limited, retry {attempt + 1}/{max_retries} "
                f"after {delay:.1f}s"
            )
            await sleep(delay)
            attempt += 1
