"""Per-candidate LLM dispatch for the failover executor (Track J.4).

:func:`call_candidate` is the ``call`` handed to
:meth:`FailoverExecutor.execute_with_failover`. It builds the esperanto
``LanguageModel`` for a single :class:`ModelCandidate` (via
:meth:`ModelManager.get_model_from_config`, the same seam B.8 used) and
dispatches one ``achat_complete``.

The candidate may carry a resolved :class:`~shared.models.llm.Model` row
(``candidate.model``) — the common DB-backed path. When it does not (a
synthesized default-chain entry with only a ``model_id`` name), we construct a
minimal :class:`Model` from the candidate so a provider with no DB row (e.g. the
hard-coded NIM default before the startup seed runs) is still callable.

Provider locality + base_url threading happens inside ``ModelFactory`` (a
``provider="nvidia"`` row is remapped onto esperanto's ``openai-compatible``
LanguageModel + the NIM base_url). This module is provider-agnostic.

Errors raised by ``achat_complete`` (esperanto ``RuntimeError`` HTTP wrappers,
httpx transport errors) propagate to the executor, which classifies them via
:func:`app_main.services.model_routing.error_mapping.is_failover_eligible`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from loguru import logger
from shared.models.llm import Model

from app_main.services.model_routing.route_resolver import ModelCandidate


@dataclass
class CandidateResult:
    """The text + served-provider metadata produced by one candidate call.

    ``served_provider`` / ``served_model_id`` mirror what
    :class:`FailoverResult` records, but are also attached here so a caller that
    only sees the ``value`` (e.g. the extraction ``_caller``) can stamp
    provenance with the provider that actually answered.
    """

    text: str
    served_provider: str
    served_model_id: Optional[str]


def _model_for_candidate(candidate: ModelCandidate) -> Model:
    """Return the :class:`Model` row to build the esperanto LanguageModel from.

    Prefers the candidate's resolved DB row; synthesizes a minimal language
    ``Model`` from the candidate's provider/model_id when no row exists so the
    hard-coded default chain (which has no DB row until the J.4 seed runs) still
    dispatches.
    """
    if candidate.model is not None:
        return candidate.model
    return Model(
        name=candidate.model_id or "",
        provider=candidate.provider,
        type="language",
        is_local=candidate.is_local,
    )


def _extract_text(response: Any) -> str:
    """Pull the assistant text out of an esperanto ``ChatCompletion``.

    Defensive: returns "" rather than raising if the response shape changes, so
    the extractor's tolerant JSON parser sees empty output instead of an
    exception (matches the B.8 ``make_default_llm_caller`` behavior).
    """
    try:
        return response.choices[0].message.content or ""
    except (AttributeError, IndexError) as e:
        logger.error(
            f"Unexpected esperanto ChatCompletion shape: {e}; response={response!r}"
        )
        return ""


async def call_candidate(
    candidate: ModelCandidate,
    system: str,
    user: str,
    **kwargs: Any,
) -> CandidateResult:
    """Build the LanguageModel for ``candidate`` and run one chat completion.

    Args:
        candidate: The route candidate to dispatch against (provider + model).
        system: System prompt.
        user: User prompt.
        **kwargs: Reserved for per-call overrides (temperature, max_tokens);
            forwarded to ``get_model_from_config`` so the cache key reflects them.

    Returns:
        A :class:`CandidateResult` carrying the response text + the served
        provider/model (for J-Q7 provenance).

    Raises:
        Any error from ``achat_complete`` (esperanto ``RuntimeError`` /
        ``httpx`` transport errors) — propagated for the executor to classify.
    """
    from esperanto import LanguageModel
    from llm_manager import get_model_manager

    mm = get_model_manager()
    model_record = _model_for_candidate(candidate)

    instance = mm.get_model_from_config(model_record, **kwargs)
    if not isinstance(instance, LanguageModel):
        # A non-language model row reached the router — a config error, not a
        # transient fault. Raise TypeError so it propagates (non-failover) and
        # surfaces the misconfiguration instead of silently failing over.
        raise TypeError(
            f"Candidate {candidate.provider}/{candidate.model_id} is not a "
            f"LanguageModel (got {type(instance).__name__})."
        )

    response = await instance.achat_complete(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )
    return CandidateResult(
        text=_extract_text(response),
        served_provider=candidate.provider,
        served_model_id=candidate.model_id,
    )
