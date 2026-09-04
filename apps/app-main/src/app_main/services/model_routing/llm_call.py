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


# Esperanto's provider-agnostic structured-output config value (FU-J4-4). The
# base ``LanguageModel`` exposes a ``structured`` field; each provider translates
# it: openai/openai-compatible (NVIDIA NIM) -> ``response_format={"type":
# "json_object"}``, ollama -> ``format="json"``. Providers that don't honor it
# (e.g. anthropic) simply ignore the field, so threading it universally degrades
# gracefully — no capability gate needed.
_STRUCTURED_JSON = {"type": "json_object"}

# Conservative fraction of an esperanto chars→token estimate; mirrors the
# context packer's ceil(chars/4). Used only by the M.4 oversized-prompt guard.
_CHARS_PER_TOKEN = 4


def _ollama_num_ctx_for(candidate: ModelCandidate) -> Optional[int]:
    """Track M.4 guard: the ``num_ctx`` to request from a local Ollama runtime.

    Ollama defaults ``num_ctx`` to ~2-4K regardless of the model's theoretical
    max, so a prompt packed for a big primary that fails over to local llama
    would be silently truncated by the runtime, not the model. Passing the
    candidate's (deliberately conservative — see ``seed_chain_models``)
    ``context_window`` as ``num_ctx`` makes the runtime allocate the window the
    packer sized against. Returns ``None`` for non-Ollama candidates or when the
    model row carries no context_window (esperanto/runtime default applies).
    """
    if (candidate.provider or "").lower() != "ollama":
        return None
    model = candidate.model
    if model is None or model.context_window is None:
        return None
    return int(model.context_window)


async def call_candidate(
    candidate: ModelCandidate,
    system: str,
    user: str,
    *,
    json_mode: bool = False,
    **kwargs: Any,
) -> CandidateResult:
    """Build the LanguageModel for ``candidate`` and run one chat completion.

    Args:
        candidate: The route candidate to dispatch against (provider + model).
        system: System prompt.
        user: User prompt.
        json_mode: When ``True`` (the EXTRACTION path, FU-J4-4), request
            structured JSON output from the provider so the model stops emitting
            malformed/partial JSON that the extractor's parser drops. Threaded
            into esperanto's provider-agnostic ``structured`` config field, which
            each provider translates (NIM -> ``response_format``, ollama ->
            ``format=json``). Chat/summarization callers leave this ``False`` so
            their prose output is unaffected.
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

    if json_mode:
        # Pass via the esperanto-native ``structured`` config field. It reaches
        # the model through ``get_model_from_config`` -> ``ModelFactory`` ->
        # ``AIFactory.create_language(config=...)`` and also keys the model cache,
        # so a json-mode build never collides with a prose build of the same id.
        kwargs.setdefault("structured", _STRUCTURED_JSON)

    # M.4 oversized-chunk GUARD. A prompt packed for a big primary that fails
    # over to a small-context candidate (local llama) would overflow. Two
    # defences:
    #   1. For local Ollama, request num_ctx = the candidate's (conservative)
    #      context_window so the runtime allocates the window the packer sized
    #      against (Ollama's ~2-4K default would otherwise truncate silently).
    #   2. Log a WARNING when the estimated prompt tokens exceed the candidate's
    #      context_window so an overflow on failover is observable rather than a
    #      silent quality drop. (The packer re-splits for the HEAD; this catches
    #      the per-call-failover case where a head-sized pack reaches a smaller
    #      fallback — the interim V1 guard, Decision M-D3 (b).)
    # PC.6: defence 1 above DOES NOT WORK and the `setdefault` is gone. esperanto
    # filters it out before its Ollama provider ever sees it —
    # `providers/llm/base.py::get_completion_kwargs` returns only
    # max_tokens/temperature/top_p/streaming, and `num_ctx` appears zero times in
    # the whole package. Measured through the real factory:
    #     in  config={"num_ctx": 16384, "temperature": 0.1, "max_tokens": 8}
    #     out {"options": {"num_predict": 8, "temperature": 0.1, "top_p": 0.9}}
    # So every app-main LLM call runs at Ollama's default context regardless of
    # what the packer sized against, and the line that claimed otherwise made the
    # truncation look handled. On THIS path the context can only be set by baking
    # `PARAMETER num_ctx` into the model — `check_ollama_context` reports when a
    # routed model's baked window is smaller than the one the config promises.
    #
    # `shared/model_routing._call_ollama` is unaffected: it posts to Ollama
    # directly and does send `num_ctx`.
    num_ctx = _ollama_num_ctx_for(candidate)
    cand_ctx = model_record.context_window
    if cand_ctx is not None:
        est_prompt_tokens = (len(system) + len(user)) // _CHARS_PER_TOKEN
        if est_prompt_tokens > cand_ctx:
            logger.warning(
                "M.4 guard: prompt ~{est} tokens exceeds {prov}/{name} "
                "context_window {ctx}; the pack was sized for a larger primary "
                "and reached a smaller fallback. The runtime window is NOT "
                "requested per call on this path (esperanto drops num_ctx); it "
                "is whatever the model bakes in, so {num_ctx} tokens is what the "
                "packer assumed, not what the model will allocate.",
                est=est_prompt_tokens,
                prov=candidate.provider,
                name=candidate.model_id,
                ctx=cand_ctx,
                num_ctx=num_ctx,
            )

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
