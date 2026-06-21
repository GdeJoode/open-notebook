"""Privacy-aware ordered provider-chain resolver (Track J.1).

Generalizes B.8's ``resolve_default_model_id`` from "one model id" to an
ORDERED route: :func:`resolve_route` returns the list of candidate ``model``
rows to attempt for a given task + privacy mode. No failover execution here —
that is J.2; this phase produces the plan only.

Resolution pipeline (per the J.1 plan §3):
  1. Load the ``model_route`` row for the task; fall back to a hard-coded
     default chain (Anthropic -> OpenAI -> local) when no row exists, so an
     empty ``model_route`` table never breaks resolution.
  2. CLOUD mode: walk ``provider_chain``, then append the local model as the
     FINAL fallback candidate. PRIVATE mode: use ``private_chain`` (or derive
     it by filtering ``provider_chain`` to local providers) — local only,
     never a cloud provider.
  3. Drop providers whose ``api_key_env_var`` is unset (J-Q6) and entries /
     model rows that are disabled.
  4. Honor B.8 head-of-chain precedence: the per-task default model
     (``default_extraction_model`` etc.) is the model id for its provider's
     entry; an explicit per-call ``model_id`` override still wins.

Guardrail: :class:`LLMTask` has EXACTLY three members. Adding EMBEDDING /
PARSING here would route embeddings through the cloud chain and break the
I.G 768-dim pin (§1.2).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from loguru import logger

from llm_manager.providers.registry import get_provider_config
from shared.models.llm import Model, ModelRoute


class LLMTask(str, Enum):
    """The three LLM pipeline stages that participate in cloud routing.

    EXACTLY these three — embeddings and parsing are deliberately absent
    (§1.2 guardrail). The ``value`` matches the ``model_route.task`` column and
    is also used to pick the ``DefaultModels`` head-of-chain field via
    :data:`_TASK_DEFAULT_FIELD`.
    """

    ENTITY_EXTRACTION = "entity_extraction"
    SUMMARIZATION = "summarization"
    CHAT = "chat"


class PrivacyMode(str, Enum):
    """Resolved privacy mode for a routing decision.

    CLOUD: prefer cloud providers, fall back to local as the final candidate.
    PRIVATE: local providers only — a cloud provider must never appear.
    """

    CLOUD = "cloud"
    PRIVATE = "private"


# Maps each task to its DefaultModels head-of-chain field. The resolver uses
# this to honor the B.8 per-function default as the model id for the cloud
# entries (and as the local fallback model id).
_TASK_DEFAULT_FIELD = {
    LLMTask.ENTITY_EXTRACTION: "default_extraction_model",
    LLMTask.SUMMARIZATION: "default_transformation_model",
    LLMTask.CHAT: "default_chat_model",
}

# Hard-coded default provider chain per task (J-Q3: Anthropic -> OpenAI ->
# local). Used only when no ``model_route`` row exists for the task so an empty
# table never breaks resolution. The model NAMES below are sensible
# placeholders; the resolver prefers the configured ``model`` row id from
# DefaultModels for each provider when one exists.
#
# TODO(J-Q3): the exact Anthropic Claude model id is a J-Q3 decision-point.
# The ``claude-api`` skill/reference was not available in this implementation
# sandbox, so "claude-sonnet-4-5" is a placeholder. J.4 seeds the actual
# Anthropic ``model`` row + final id; this default chain only references it by
# name when no DB-backed default model id is configured.
_DEFAULT_CLOUD_PROVIDERS = ["anthropic", "openai"]
_DEFAULT_LOCAL_PROVIDER = "ollama"
_DEFAULT_PROVIDER_MODEL_NAME = {
    "anthropic": "claude-sonnet-4-5",  # TODO(J-Q3): confirm via claude-api ref
    "openai": "gpt-4o",
    "ollama": "llama3.1:8b",
}


@dataclass(frozen=True)
class ModelCandidate:
    """One ordered candidate in a resolved route.

    ``model`` is the resolved :class:`Model` row when one exists in the DB;
    ``model_id`` is always populated (it is the head of the B.8 precedence and
    the shim's return value). ``is_local`` mirrors the model/provider locality
    and is the falsifiable privacy signal the J.1 ACs assert on.
    """

    provider: str
    model_id: Optional[str]
    is_local: bool
    model: Optional[Model] = None


@dataclass
class ResolvedRoute:
    """The ordered plan returned by :func:`resolve_route`.

    ``ordered_candidates`` is what J.2's executor will walk head-to-tail. J.1
    only constructs it; the B.8 shim consumes ``ordered_candidates[0]``.
    """

    task: LLMTask
    mode: PrivacyMode
    ordered_candidates: List[ModelCandidate] = field(default_factory=list)


def _provider_key_present(provider: str) -> bool:
    """True iff the provider is local OR its API key env var is set (J-Q6).

    Reuses the existing registry check (``bool(os.environ.get(api_key_env_var))``)
    so the "is this provider configured" semantics match ``providers.py``. Local
    providers (ollama/llamacpp) never need a key.
    """
    cfg = get_provider_config(provider)
    if cfg is None:
        return False
    if cfg.is_local:
        return True
    return bool(os.environ.get(cfg.api_key_env_var))


def _provider_is_local(provider: str) -> bool:
    cfg = get_provider_config(provider)
    return bool(cfg and cfg.is_local)


class RouteResolver:
    """Resolves an :class:`LLMTask` + :class:`PrivacyMode` into a route.

    Constructed with a model-route repo and a model repo so it can be unit
    tested with fakes (the J.1 ACs inject a fake ``ModelRouteRepository`` and
    drive provider availability via ``monkeypatch.setenv/delenv``). Production
    wiring is the :func:`get_route_resolver` DI factory.
    """

    def __init__(self, model_route_repo, model_repo):
        self._route_repo = model_route_repo
        self._model_repo = model_repo

    async def resolve(
        self,
        task: LLMTask,
        mode: PrivacyMode,
        model_id: Optional[str] = None,
    ) -> ResolvedRoute:
        """Build the ordered route for ``task`` under ``mode``.

        Args:
            task: Which LLM stage is resolving a model.
            mode: CLOUD (cloud-first, local-last) or PRIVATE (local-only).
            model_id: Explicit per-call override. When set it becomes the
                model id of the HEAD candidate (B.8 precedence), winning over
                the per-task default.
        """
        route_row = await self._route_repo.get_by_task(task.value)
        default_model_id = await self._resolve_default_model_id(task, model_id)

        if mode == PrivacyMode.PRIVATE:
            candidates = await self._resolve_private(
                task, route_row, default_model_id
            )
        else:
            candidates = await self._resolve_cloud(
                task, route_row, default_model_id
            )

        if mode == PrivacyMode.PRIVATE:
            # Hard invariant: a private route can never contain a cloud
            # provider. Assert here so a bug surfaces loudly rather than
            # leaking a document to the cloud.
            for c in candidates:
                assert c.is_local, (
                    f"PRIVATE route for {task.value} produced a non-local "
                    f"candidate ({c.provider}) — privacy violation"
                )

        return ResolvedRoute(task=task, mode=mode, ordered_candidates=candidates)

    async def _resolve_default_model_id(
        self, task: LLMTask, model_id: Optional[str]
    ) -> Optional[str]:
        """B.8 head-of-chain precedence: explicit override -> per-task default
        -> default_chat_model. Returns ``None`` when nothing is configured."""
        if model_id:
            return model_id

        from llm_manager import get_model_manager

        from app_main.dependencies import get_default_models_repo

        mm = get_model_manager()
        defaults = mm.get_defaults()
        if not defaults or not defaults.default_chat_model:
            defaults = await get_default_models_repo().get()
            mm.set_defaults(defaults)
        if not defaults:
            return None

        field_name = _TASK_DEFAULT_FIELD[task]
        return getattr(defaults, field_name, None) or defaults.default_chat_model

    async def _entries_for(
        self, route_row: Optional[ModelRoute], local_only: bool
    ) -> List[dict]:
        """Return the raw chain entries to consider, in order.

        Picks ``private_chain`` vs ``provider_chain`` (deriving the private
        chain from the cloud chain's local providers when the private chain is
        empty), or the hard-coded default providers when no row exists.
        """
        if route_row is None:
            providers = (
                [_DEFAULT_LOCAL_PROVIDER]
                if local_only
                else _DEFAULT_CLOUD_PROVIDERS + [_DEFAULT_LOCAL_PROVIDER]
            )
            return [
                {"provider": p, "model_id": None, "enabled": True}
                for p in providers
            ]

        if local_only:
            chain = route_row.private_chain or [
                e
                for e in route_row.provider_chain
                if _provider_is_local(e.get("provider", ""))
            ]
            return list(chain)

        return list(route_row.provider_chain)

    async def _resolve_candidate(
        self,
        provider: str,
        entry_model_id: Optional[str],
        default_model_id: Optional[str],
    ) -> Optional[ModelCandidate]:
        """Build a :class:`ModelCandidate` for a single chain entry.

        Drops the entry (returns ``None``) when the provider's key is missing
        or the resolved model row is disabled.
        """
        if not _provider_key_present(provider):
            return None

        is_local = _provider_is_local(provider)
        # B.8 head-of-chain: a chain entry may pin its own model id; otherwise
        # use the per-task default model id resolved above.
        candidate_model_id = entry_model_id or default_model_id

        model_row: Optional[Model] = None
        if candidate_model_id:
            try:
                model_row = await self._model_repo.get(candidate_model_id)
            except Exception as e:  # noqa: BLE001 — degrade, don't crash routing
                logger.warning(
                    f"route_resolver: failed to load model {candidate_model_id!r} "
                    f"for provider {provider}: {e}"
                )
            if model_row is not None:
                if model_row.is_enabled is False:
                    return None
                is_local = bool(model_row.is_local)

        if candidate_model_id is None and model_row is None:
            # No configured model id and no DB row — synthesize a placeholder id
            # from the default-chain name so the candidate is still actionable.
            candidate_model_id = _DEFAULT_PROVIDER_MODEL_NAME.get(provider)

        return ModelCandidate(
            provider=provider,
            model_id=candidate_model_id,
            is_local=is_local,
            model=model_row,
        )

    async def _resolve_cloud(
        self,
        task: LLMTask,
        route_row: Optional[ModelRoute],
        default_model_id: Optional[str],
    ) -> List[ModelCandidate]:
        candidates: List[ModelCandidate] = []
        seen_providers: set[str] = set()

        for entry in await self._entries_for(route_row, local_only=False):
            if entry.get("enabled") is False:
                continue
            provider = entry.get("provider", "")
            candidate = await self._resolve_candidate(
                provider, entry.get("model_id"), default_model_id
            )
            if candidate is not None:
                candidates.append(candidate)
                seen_providers.add(provider)

        # Append the local model as the FINAL fallback if not already present
        # (J-Q3: ... -> local; J-Q5: no-cloud -> single local entry).
        if not any(c.is_local for c in candidates):
            local = await self._resolve_candidate(
                _DEFAULT_LOCAL_PROVIDER, None, default_model_id
            )
            if local is not None:
                candidates.append(local)

        return candidates

    async def _resolve_private(
        self,
        task: LLMTask,
        route_row: Optional[ModelRoute],
        default_model_id: Optional[str],
    ) -> List[ModelCandidate]:
        candidates: List[ModelCandidate] = []

        for entry in await self._entries_for(route_row, local_only=True):
            if entry.get("enabled") is False:
                continue
            provider = entry.get("provider", "")
            if not _provider_is_local(provider):
                # Defensive: never let a cloud provider into a private chain,
                # even if a misconfigured private_chain row lists one.
                continue
            candidate = await self._resolve_candidate(
                provider, entry.get("model_id"), default_model_id
            )
            if candidate is not None:
                candidates.append(candidate)

        if not candidates:
            local = await self._resolve_candidate(
                _DEFAULT_LOCAL_PROVIDER, None, default_model_id
            )
            if local is not None:
                candidates.append(local)

        return candidates


async def resolve_route(
    task: LLMTask,
    mode: PrivacyMode,
    model_id: Optional[str] = None,
) -> ResolvedRoute:
    """Module-level convenience wrapper around :class:`RouteResolver`.

    Builds the resolver from the DI factories so call sites that don't already
    hold a resolver (the B.8 shim, ad-hoc callers) get the same behavior. Tests
    construct :class:`RouteResolver` directly with fakes.
    """
    from app_main.dependencies import get_route_resolver

    resolver = get_route_resolver()
    return await resolver.resolve(task, mode, model_id=model_id)
