"""Privacy-aware model routing for the LLM pipeline stages (Track J).

J.1 introduces the ordered provider-chain resolver: given an :class:`LLMTask`
and a :class:`PrivacyMode`, :func:`resolve_route` returns the ordered list of
candidate ``model`` rows to attempt. J.2 adds the EXECUTION layer: a
:class:`FailoverExecutor` walks the chain on provider errors with a per-provider
:class:`ProviderCircuitBreaker` (skip dead providers fast) and a fair-use
:class:`ProviderRateLimiter` (stay under cloud quotas).

Guardrail (§1.2 of the J plan): :class:`LLMTask` has EXACTLY three members
(ENTITY_EXTRACTION, SUMMARIZATION, CHAT). Embeddings and parsing are out of
cloud routing and MUST NOT gain a member here — the I.G 768-dim embedding pin
depends on embeddings staying local + fixed.
"""

from app_main.services.model_routing.circuit_breaker import (
    CircuitBreakerRegistry,
    CircuitState,
    ProviderCircuitBreaker,
)
from app_main.services.model_routing.failover_executor import (
    AllProvidersFailedError,
    AttemptOutcome,
    AttemptRecord,
    FailoverEligibleError,
    FailoverExecutor,
    FailoverResult,
    ProviderAuthError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    RateLimitError,
)
from app_main.services.model_routing.privacy_resolver import (
    resolve_privacy_mode,
)
from app_main.services.model_routing.rate_limiter import (
    ProviderRateLimiter,
    RateLimitTimeout,
    backoff_retry,
)
from app_main.services.model_routing.route_resolver import (
    LLMTask,
    ModelCandidate,
    PrivacyMode,
    ResolvedRoute,
    RouteResolver,
    is_provider_healthy,
    resolve_route,
)

__all__ = [
    # J.1 — resolver
    "LLMTask",
    "PrivacyMode",
    "ModelCandidate",
    "ResolvedRoute",
    "RouteResolver",
    "resolve_route",
    "is_provider_healthy",
    # J.3 — privacy resolver
    "resolve_privacy_mode",
    # J.2 — circuit breaker
    "CircuitState",
    "ProviderCircuitBreaker",
    "CircuitBreakerRegistry",
    # J.2 — rate limiter
    "ProviderRateLimiter",
    "RateLimitTimeout",
    "backoff_retry",
    # J.2 — failover executor
    "FailoverExecutor",
    "FailoverResult",
    "AttemptRecord",
    "AttemptOutcome",
    "AllProvidersFailedError",
    "FailoverEligibleError",
    "RateLimitError",
    "ProviderUnavailableError",
    "ProviderTimeoutError",
    "ProviderAuthError",
]
