"""Per-document failover executor (Track J.2).

Executes a :class:`~app_main.services.model_routing.route_resolver.ResolvedRoute`
by walking its ordered candidates: try the first available provider, and on a
**failover-eligible** failure advance the WHOLE document's LLM stage to the next
candidate (J-Q4: per-document failover, local last in cloud mode).

Per attempt the executor:
  1. Skips a candidate whose circuit breaker is OPEN (records a
     ``skipped_open_circuit`` attempt without calling it).
  2. For cloud candidates, ``await``s the per-provider rate limiter BEFORE the
     call (fair-use) AND before every backoff retry, so each actual request to
     the endpoint counts against the rpm cap. The call is wrapped in bounded
     exponential backoff so a 429 *from the provider* is retried on the same
     provider a few times before that candidate is considered failed.
  3. Invokes ``call(candidate)``. On success records a breaker success + the
     served provider and returns. On a failover-eligible exception records a
     breaker failure + a fallback telemetry event and advances.
  4. A NON-eligible exception (e.g. a ``TypeError``/``ValueError`` programming
     bug) propagates immediately — we never mask real bugs as failover (AC5).

Three distinct throttle/fault categories drive different behavior:
  * **Local rate-limiter saturation** (``RateLimitTimeout`` from ``acquire``):
    OUR throttle could not grant a slot within ``max_wait``. This is not
    evidence the provider is down, so the executor fails over to the next
    candidate (ultimately local, which is exempt) WITHOUT recording a circuit
    breaker failure. The attempt is recorded as ``skipped_rate_limited`` and a
    fallback telemetry event is emitted (J-Q4: local is the last resort even
    under saturation; opening the provider's breaker here would be wrong).
  * **Provider 429** (``RateLimitError`` from the call): bounded backoff-retry
    on the same provider, then (if still failing) failover + breaker-failure.
  * **Hard fault** (5xx/timeout/auth): immediate failover + breaker-failure.

When every candidate is exhausted, raises :class:`AllProvidersFailedError`
carrying the per-attempt records (never a generic exception).

Failover-eligibility is an injectable **whitelist** predicate. J.2 ships a
default set of marker exception types (:class:`RateLimitError`,
:class:`ProviderUnavailableError`, :class:`ProviderTimeoutError`,
:class:`ProviderAuthError`); J.4 maps concrete NIM/esperanto/SDK errors onto
these markers via its own ``is_failover_eligible``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Generic, List, Optional, TypeVar

from loguru import logger

from app_main.services.model_routing.circuit_breaker import CircuitBreakerRegistry
from app_main.services.model_routing.rate_limiter import (
    ProviderRateLimiter,
    RateLimitTimeout,
    backoff_retry,
)
from app_main.services.model_routing.route_resolver import (
    ModelCandidate,
    ResolvedRoute,
)

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Marker exception whitelist (J.2 defaults; J.4 maps real errors onto these)
# ---------------------------------------------------------------------------


class FailoverEligibleError(Exception):
    """Base marker for exceptions that SHOULD trigger failover to the next
    provider. J.4 raises subclasses (or maps SDK errors) so transient/provider
    faults fail over while programming bugs propagate."""


class RateLimitError(FailoverEligibleError):
    """Provider returned a rate-limit (HTTP 429) response."""


class ProviderUnavailableError(FailoverEligibleError):
    """Provider returned a server error (5xx) or is otherwise unavailable."""


class ProviderTimeoutError(FailoverEligibleError):
    """Connection/read timeout talking to the provider."""


class ProviderAuthError(FailoverEligibleError):
    """Auth failure (401/403) — treated as a provider misconfiguration: drop
    this provider and fail over (also trips its breaker)."""


def default_is_failover_eligible(exc: Exception) -> bool:
    """Default whitelist predicate: only :class:`FailoverEligibleError`
    subclasses fail over. Everything else (TypeError, ValueError, KeyError, …)
    propagates so real bugs are never masked."""
    return isinstance(exc, FailoverEligibleError)


def default_is_rate_limit(exc: Exception) -> bool:
    """Default 429 classifier used to drive backoff-retry on the same provider
    before failing over.

    Recognizes the typed :class:`RateLimitError` AND provider SDKs (e.g.
    esperanto for Gemini) that wrap a 429 / ``RESOURCE_EXHAUSTED`` as a plain
    ``RuntimeError`` carrying the quota message in its text. Without this, a
    Gemini free-tier per-minute trip is misread as a hard failure and the chain
    fails over to local immediately instead of pacing + backoff-retrying on the
    same provider (Track M interim — Track M formalizes per-provider handling)."""
    if isinstance(exc, RateLimitError):
        return True
    msg = str(exc).lower()
    return any(
        marker in msg
        for marker in (
            "resource_exhausted",
            "rate limit",
            "ratelimit",
            " 429",
            "error code: 429",
            "exceeded your current quota",
            "quota exceeded",
        )
    )


# ---------------------------------------------------------------------------
# Result + telemetry records
# ---------------------------------------------------------------------------


class AttemptOutcome(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED_OPEN_CIRCUIT = "skipped_open_circuit"
    # Our local rate limiter could not grant a slot within max_wait. We fail
    # over WITHOUT a breaker failure: a saturated throttle is not a provider
    # fault. Distinct from FAILED so telemetry never confuses the two.
    SKIPPED_RATE_LIMITED = "skipped_rate_limited"


@dataclass
class AttemptRecord:
    """One entry per candidate the executor considered (telemetry for J-Q7)."""

    provider: str
    model_id: Optional[str]
    is_local: bool
    outcome: AttemptOutcome
    error: Optional[str] = None


@dataclass
class FailoverResult(Generic[T]):
    """Successful execution result.

    ``served_provider`` / ``served_model_id`` identify the provider that
    actually produced ``value`` (the single served-provider telemetry field per
    document stage — feeds J-Q7 provenance). ``attempts`` is the ordered trail
    including any failed/skipped candidates before the winner.
    """

    value: T
    served_provider: str
    served_model_id: Optional[str]
    attempts: List[AttemptRecord] = field(default_factory=list)

    @property
    def was_failover(self) -> bool:
        """True when the served provider was not the head of the route."""
        return any(
            a.outcome != AttemptOutcome.SUCCESS for a in self.attempts
        )

    @property
    def fallback_from(self) -> List[str]:
        """Providers attempted (failed/skipped) before the served one."""
        return [
            a.provider
            for a in self.attempts
            if a.outcome != AttemptOutcome.SUCCESS
        ]


class AllProvidersFailedError(Exception):
    """Raised when every candidate in the route was exhausted.

    Carries the per-attempt ``attempts`` records so the caller can log/telemeter
    exactly what was tried and why each failed.
    """

    def __init__(self, attempts: List[AttemptRecord]) -> None:
        self.attempts = attempts
        summary = ", ".join(
            f"{a.provider}:{a.outcome.value}" for a in attempts
        ) or "<no candidates>"
        super().__init__(f"all providers failed [{summary}]")


class FailoverExecutor:
    """Walks a resolved route with circuit-breaking, rate-limiting and backoff.

    Constructed with a :class:`CircuitBreakerRegistry` and a
    :class:`ProviderRateLimiter` (both process singletons via DI). The
    eligibility/rate-limit predicates are injectable so J.4 can supply real
    SDK-error mappings without changing this executor.
    """

    def __init__(
        self,
        circuit_breakers: CircuitBreakerRegistry,
        rate_limiter: ProviderRateLimiter,
        *,
        is_failover_eligible: Callable[
            [Exception], bool
        ] = default_is_failover_eligible,
        is_rate_limit: Callable[[Exception], bool] = default_is_rate_limit,
        backoff_max_retries: int = 3,
        backoff_base_delay: float = 1.0,
        backoff_max_delay: float = 30.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._breakers = circuit_breakers
        self._rate_limiter = rate_limiter
        self._is_failover_eligible = is_failover_eligible
        self._is_rate_limit = is_rate_limit
        self._backoff_max_retries = backoff_max_retries
        self._backoff_base_delay = backoff_base_delay
        self._backoff_max_delay = backoff_max_delay
        self._sleep = sleep

    async def execute_with_failover(
        self,
        route: ResolvedRoute,
        call: Callable[[ModelCandidate], Awaitable[T]],
    ) -> FailoverResult[T]:
        """Execute ``call`` against the route's candidates with failover.

        Returns a :class:`FailoverResult` on the first success. Raises
        :class:`AllProvidersFailedError` only when all candidates are exhausted;
        a non-failover-eligible exception propagates immediately.
        """
        attempts: List[AttemptRecord] = []

        for candidate in route.ordered_candidates:
            breaker = await self._breakers.get(candidate.provider)
            if not await breaker.allow():
                logger.info(
                    f"failover: skipping {candidate.provider} (circuit open)"
                )
                attempts.append(
                    AttemptRecord(
                        provider=candidate.provider,
                        model_id=candidate.model_id,
                        is_local=candidate.is_local,
                        outcome=AttemptOutcome.SKIPPED_OPEN_CIRCUIT,
                    )
                )
                continue

            try:
                value = await self._invoke(candidate, call)
            except RateLimitTimeout as exc:
                # OUR throttle saturated — not the provider being unhealthy.
                # Fail over to the next candidate (ultimately local, which is
                # exempt) WITHOUT a breaker failure: penalizing the breaker here
                # would wrongly mark a healthy provider as down (J-Q4).
                attempts.append(
                    AttemptRecord(
                        provider=candidate.provider,
                        model_id=candidate.model_id,
                        is_local=candidate.is_local,
                        outcome=AttemptOutcome.SKIPPED_RATE_LIMITED,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                logger.warning(
                    f"failover: {candidate.provider} rate-limit saturated "
                    "(local throttle); advancing without breaker penalty"
                )
                continue
            except Exception as exc:  # noqa: BLE001 — classify below
                if not self._is_failover_eligible(exc):
                    # Real bug / non-transient — do not mask as failover (AC5).
                    # The breaker is NOT tripped: this is not a provider fault.
                    raise
                await breaker.record_failure()
                attempts.append(
                    AttemptRecord(
                        provider=candidate.provider,
                        model_id=candidate.model_id,
                        is_local=candidate.is_local,
                        outcome=AttemptOutcome.FAILED,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
                logger.warning(
                    f"failover: {candidate.provider} failed "
                    f"({type(exc).__name__}); advancing"
                )
                continue

            await breaker.record_success()
            attempts.append(
                AttemptRecord(
                    provider=candidate.provider,
                    model_id=candidate.model_id,
                    is_local=candidate.is_local,
                    outcome=AttemptOutcome.SUCCESS,
                )
            )
            return FailoverResult(
                value=value,
                served_provider=candidate.provider,
                served_model_id=candidate.model_id,
                attempts=attempts,
            )

        raise AllProvidersFailedError(attempts)

    async def _invoke(
        self,
        candidate: ModelCandidate,
        call: Callable[[ModelCandidate], Awaitable[T]],
    ) -> T:
        """Invoke ``call`` for one candidate.

        Cloud candidates are rate-limited and wrapped in bounded backoff so a
        429 is retried on the same provider before being treated as a failure.
        A rate-limit slot is acquired BEFORE every actual request — both the
        first call and each backoff retry — so each request to the endpoint
        counts against the rpm cap (a sustained-429 stage would otherwise issue
        several requests against one consumed slot, under-counting fair-use).

        Local candidates skip both — no fair-use concern.

        If ``acquire`` cannot grant a slot within ``max_wait`` it raises
        :class:`RateLimitTimeout`; that is not a 429 so ``backoff_retry`` does
        not retry it — it propagates to ``execute_with_failover``, which fails
        over to the next candidate without penalizing the breaker.
        """
        if candidate.is_local:
            return await call(candidate)

        async def _do() -> T:
            # Re-acquire per request (initial + each retry) so every endpoint
            # hit is counted against the fair-use cap.
            await self._rate_limiter.acquire(
                candidate.provider, is_local=candidate.is_local
            )
            return await call(candidate)

        return await backoff_retry(
            _do,
            is_rate_limit=self._is_rate_limit,
            base_delay=self._backoff_base_delay,
            factor=2.0,
            max_delay=self._backoff_max_delay,
            max_retries=self._backoff_max_retries,
            sleep=self._sleep,
        )
