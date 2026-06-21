"""Privacy-aware model routing for the LLM pipeline stages (Track J).

J.1 introduces the ordered provider-chain resolver: given an :class:`LLMTask`
and a :class:`PrivacyMode`, :func:`resolve_route` returns the ordered list of
candidate ``model`` rows to attempt. Failover EXECUTION (walking the chain on
provider errors) lands in J.2; this package only produces the route plan.

Guardrail (§1.2 of the J plan): :class:`LLMTask` has EXACTLY three members
(ENTITY_EXTRACTION, SUMMARIZATION, CHAT). Embeddings and parsing are out of
cloud routing and MUST NOT gain a member here — the I.G 768-dim embedding pin
depends on embeddings staying local + fixed.
"""

from app_main.services.model_routing.route_resolver import (
    LLMTask,
    ModelCandidate,
    PrivacyMode,
    ResolvedRoute,
    RouteResolver,
    resolve_route,
)

__all__ = [
    "LLMTask",
    "PrivacyMode",
    "ModelCandidate",
    "ResolvedRoute",
    "RouteResolver",
    "resolve_route",
]
