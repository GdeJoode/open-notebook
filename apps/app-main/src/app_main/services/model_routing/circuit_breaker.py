"""Per-provider circuit breaker for the failover executor (Track J.2).

A :class:`ProviderCircuitBreaker` tracks the recent health of a single
provider so the failover executor can skip a dead provider fast instead of
paying its (slow, failing) call cost on every document.

State machine (one instance per provider, keyed in
:class:`CircuitBreakerRegistry`):

    CLOSED ──N consecutive failures──▶ OPEN
    OPEN ──cooldown elapsed──▶ HALF_OPEN (allows ONE probe)
    HALF_OPEN ──probe success──▶ CLOSED
    HALF_OPEN ──probe failure──▶ OPEN (cooldown restarts)

CLOSED is the healthy steady state. OPEN means "skip this provider without
calling it" until the cooldown elapses. HALF_OPEN admits exactly one probe
request to test whether the provider has recovered; while that probe is in
flight further callers are told the breaker is still effectively open (no
second concurrent probe).

Thread-safety: every state read/transition is guarded by an ``asyncio.Lock``.
The executor calls :meth:`allow` (decide skip/probe), then exactly one of
:meth:`record_success` / :meth:`record_failure` per attempt.

J-D3 caveat (multi-worker): state is **in-memory and per-process**. With more
than one app-main worker process each worker keeps its own breaker state, so a
provider can be OPEN in one worker while CLOSED in another. This is acceptable
for the V1 single-worker deployment; persisting breaker state to SurrealDB for
multi-worker correctness is the deferred J-D3 decision.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum
from typing import Callable, Dict

from loguru import logger

DEFAULT_FAILURE_THRESHOLD = 3
DEFAULT_COOLDOWN_SECONDS = 60.0


class CircuitState(str, Enum):
    """Health state of a single provider's breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class ProviderCircuitBreaker:
    """Circuit breaker for one provider.

    Args:
        provider: Provider name this breaker guards (for logging only).
        failure_threshold: Consecutive failures that trip CLOSED -> OPEN.
        cooldown_seconds: Seconds OPEN stays open before allowing a HALF_OPEN
            probe.
        clock: Monotonic time source (injectable for deterministic tests).
    """

    def __init__(
        self,
        provider: str,
        *,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.provider = provider
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock

        self._state = CircuitState.CLOSED
        self._consecutive_failures = 0
        self._opened_at: float = 0.0
        # True while a HALF_OPEN probe has been handed out but not yet
        # resolved, so a second concurrent caller is not also admitted.
        self._probe_in_flight = False
        self._lock = asyncio.Lock()

    @property
    def provider_name(self) -> str:
        return self.provider

    async def state(self) -> CircuitState:
        """Return the *effective* state, applying any due OPEN->HALF_OPEN
        transition first. Read-only from the caller's perspective."""
        async with self._lock:
            self._maybe_half_open()
            return self._state

    async def allow(self) -> bool:
        """Decide whether the executor may attempt a call to this provider.

        Returns ``True`` for CLOSED (normal) or for the single admitted
        HALF_OPEN probe. Returns ``False`` while OPEN (cooldown not elapsed) or
        while a HALF_OPEN probe is already in flight — the executor records a
        ``skipped_open_circuit`` attempt in that case.
        """
        async with self._lock:
            self._maybe_half_open()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.OPEN:
                return False

            # HALF_OPEN: admit exactly one probe.
            if self._probe_in_flight:
                return False
            self._probe_in_flight = True
            return True

    async def record_success(self) -> None:
        """Record a successful call: reset to CLOSED."""
        async with self._lock:
            self._consecutive_failures = 0
            self._probe_in_flight = False
            if self._state != CircuitState.CLOSED:
                logger.info(
                    f"circuit_breaker[{self.provider}]: {self._state.value} -> closed "
                    "(success)"
                )
            self._state = CircuitState.CLOSED

    async def record_failure(self) -> None:
        """Record a failed call.

        From HALF_OPEN a probe failure re-opens immediately (cooldown
        restarts). From CLOSED, opens once ``failure_threshold`` consecutive
        failures accumulate.
        """
        async with self._lock:
            self._probe_in_flight = False

            if self._state == CircuitState.HALF_OPEN:
                self._open_locked("half-open probe failed")
                return

            self._consecutive_failures += 1
            if (
                self._state == CircuitState.CLOSED
                and self._consecutive_failures >= self._failure_threshold
            ):
                self._open_locked(
                    f"{self._consecutive_failures} consecutive failures"
                )

    def _open_locked(self, reason: str) -> None:
        """Transition to OPEN. Caller MUST hold ``self._lock``."""
        self._state = CircuitState.OPEN
        self._opened_at = self._clock()
        self._probe_in_flight = False
        logger.warning(
            f"circuit_breaker[{self.provider}]: -> open ({reason}); cooldown "
            f"{self._cooldown_seconds}s"
        )

    def _maybe_half_open(self) -> None:
        """If OPEN and the cooldown has elapsed, move to HALF_OPEN. Caller MUST
        hold ``self._lock``."""
        if self._state != CircuitState.OPEN:
            return
        if self._clock() - self._opened_at >= self._cooldown_seconds:
            self._state = CircuitState.HALF_OPEN
            self._probe_in_flight = False
            logger.info(
                f"circuit_breaker[{self.provider}]: open -> half_open "
                "(cooldown elapsed)"
            )


class CircuitBreakerRegistry:
    """Process-singleton registry of per-provider breakers.

    Lazily creates one :class:`ProviderCircuitBreaker` per provider name with
    shared threshold/cooldown config. Wired as a process singleton via
    :func:`app_main.dependencies.get_circuit_breaker_registry`.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = DEFAULT_FAILURE_THRESHOLD,
        cooldown_seconds: float = DEFAULT_COOLDOWN_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._clock = clock
        self._breakers: Dict[str, ProviderCircuitBreaker] = {}
        self._lock = asyncio.Lock()

    async def get(self, provider: str) -> ProviderCircuitBreaker:
        """Return (creating if needed) the breaker for ``provider``."""
        async with self._lock:
            breaker = self._breakers.get(provider)
            if breaker is None:
                breaker = ProviderCircuitBreaker(
                    provider,
                    failure_threshold=self._failure_threshold,
                    cooldown_seconds=self._cooldown_seconds,
                    clock=self._clock,
                )
                self._breakers[provider] = breaker
            return breaker

    def get_nowait(self, provider: str) -> ProviderCircuitBreaker | None:
        """Non-async peek for synchronous health helpers; does not create."""
        return self._breakers.get(provider)
