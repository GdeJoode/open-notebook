from esperanto import LanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from llm_manager import ModelManager
from loguru import logger


def _get_model_manager() -> ModelManager:
    """Get or create the ModelManager singleton."""
    return ModelManager()


def _token_count(content: str) -> int:
    """Count tokens in content."""
    from shared.utils import token_count

    return token_count(content)


# default_type (the chat-graph caller's string) -> routing LLMTask. Chat and
# large_context both route through the CHAT chain; transformation routes through
# SUMMARIZATION (it is a text-transformation task). Anything else falls back to
# CHAT (the universal default) — there is deliberately no embedding/parsing task.
_DEFAULT_TYPE_TO_TASK = {
    "chat": "CHAT",
    "large_context": "CHAT",
    "transformation": "SUMMARIZATION",
}


async def provision_langchain_model(
    content, model_id, default_type, **kwargs
) -> BaseChatModel:
    """Return a LangChain chat model for the requested stage, route-resolved.

    Track J.4: the chat/transformation stages now resolve a privacy-aware
    ORDERED route (like extraction/summarization) instead of a single default
    model. Failover here is **construction-time only** (J-D4): we walk the
    route's candidates and return the LangChain model of the FIRST candidate
    whose esperanto ``LanguageModel`` *constructs* successfully. Once a model is
    returned, this function is done — it never observes the stream. A provider
    that succeeds at construction but then errors at runtime (e.g. mid-stream,
    or on the first token) surfaces that error to the chat caller; it is NOT
    re-issued on the next candidate, because by then we've already handed back a
    single model and emitting on a second provider would double-emit tokens.
    Runtime/mid-stream chat failover is explicitly out of scope for V1 (J-D4).

    Track J.6 (FU-J4-1): when a candidate constructs we emit one
    ``routing.served`` telemetry event for the served provider/model, closing the
    chat telemetry gap J.4 left open. ``was_failover`` is True when an earlier
    candidate failed to construct and we fell back to a later one; ``fallback_from``
    names the providers that failed construction. This is construction-time
    provenance only — a runtime/mid-stream chat error is not a failover here
    (J-D4) and is therefore not recorded as one.

    A large content (> 105k tokens) forces the large-context head model id when
    one is configured (preserving the pre-J behavior), but still resolves
    through the CHAT route so the local fallback + privacy mode still apply.

    Args:
        content: The prompt/messages (used only for the token-count heuristic).
        model_id: Optional explicit model id override (head of the route).
        default_type: The caller's stage string ("chat" | "large_context" |
            "transformation" | ...), mapped to a routing :class:`LLMTask`.
        **kwargs: Forwarded to ``get_model_from_config`` (e.g. ``max_tokens``).
    """
    from app_main.dependencies import get_route_resolver
    from app_main.services.model_routing.route_resolver import LLMTask, PrivacyMode

    tokens = _token_count(str(content))

    # Large content: prefer the configured large_context model id as the route
    # head when no explicit override was passed.
    head_model_id = model_id
    if tokens > 105_000 and not head_model_id:
        logger.debug(
            f"Using large context model because the content has {tokens} tokens"
        )
        defaults = _get_model_manager().get_defaults()
        head_model_id = getattr(defaults, "large_context_model", None) if defaults else None

    task_name = _DEFAULT_TYPE_TO_TASK.get(default_type, "CHAT")
    task = LLMTask[task_name]

    # CHAT/transformation are not yet privacy-flag-plumbed at the request layer
    # (J.3 wired the document path); chat resolves under the global/cloud default
    # for now. Privacy-per-conversation is a future enhancement (out of J scope).
    resolver = get_route_resolver()
    route = await resolver.resolve(task, PrivacyMode.CLOUD, model_id=head_model_id)
    if not route.ordered_candidates:
        raise RuntimeError(
            f"No model candidates resolved for {task} — configure a provider "
            "chain or default model."
        )

    manager = _get_model_manager()
    last_error: Exception | None = None
    # Providers whose construction failed before the served one — the chat
    # equivalent of the executor's fallback trail (FU-J4-1 telemetry).
    fallback_from: list[str] = []

    # Stream-start failover: build the first candidate that constructs cleanly.
    for candidate in route.ordered_candidates:
        from shared.models.llm import Model

        model_record = candidate.model or Model(
            name=candidate.model_id or "",
            provider=candidate.provider,
            type="language",
            is_local=candidate.is_local,
        )
        try:
            model = manager.get_model_from_config(model_record, **kwargs)
            assert isinstance(model, LanguageModel), (
                f"Model is not a LanguageModel: {model}"
            )
            logger.debug(
                f"chat route: using {candidate.provider}/{candidate.model_id}"
            )
            await _record_chat_routing_event(
                task=task,
                served_provider=candidate.provider,
                served_model_id=candidate.model_id,
                fallback_from=fallback_from,
            )
            return model.to_langchain()
        except Exception as e:  # noqa: BLE001 — try the next candidate at start
            last_error = e
            fallback_from.append(candidate.provider)
            logger.warning(
                f"chat route: {candidate.provider}/{candidate.model_id} failed "
                f"to construct ({e}); trying next candidate"
            )

    raise RuntimeError(
        f"All chat route candidates failed to construct for {task}: {last_error}"
    )


async def _record_chat_routing_event(
    *,
    task,
    served_provider: str,
    served_model_id,
    fallback_from: list[str],
) -> None:
    """Emit the chat ``routing.served`` telemetry event (FU-J4-1).

    Mirrors the extraction/summarization callers' provenance write so the J.5
    health endpoint's recent-fallback counts and the per-notebook routing
    summary see chat the same way they see the other two stages. Best-effort:
    ``record_routing_event`` already swallows its own failures, but we guard the
    call too so a telemetry hiccup can never break chat model construction.
    """
    from app_main.services.model_routing.telemetry import record_routing_event

    try:
        await record_routing_event(
            task=task,
            served_provider=served_provider,
            served_model_id=served_model_id,
            was_failover=bool(fallback_from),
            fallback_from=list(fallback_from),
        )
    except Exception as e:  # noqa: BLE001 — telemetry never fails the chat path
        logger.warning(f"chat routing telemetry write failed (ignored): {e}")
