"""Unit tests for the Track J.1 privacy-aware route resolver.

Covers the 6 fast (no-docker) acceptance criteria from the J.1 plan §3:

1. CLOUD + both cloud keys -> [anthropic, openai, <local>], final is local.
2. CLOUD with ANTHROPIC_API_KEY unset -> [openai, <local>] (J-Q6 drop).
3. PRIVATE -> only local candidates for all three tasks (no cloud provider).
4. CLOUD with no cloud key -> single-entry [<local>] (J-Q5 no-cloud -> local).
5. LLMTask has exactly three members (§1.2 guardrail).
6. resolve_default_model_id shim still returns a single id (B.8 preserved) —
   exercised here against the resolver head; the full B.8 suite is the
   authoritative regression gate.

The resolver is driven directly with fakes for the model_route + model repos
and ``monkeypatch.setenv/delenv`` for provider keys, so no DB or live providers
are touched.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import pytest
from app_main.services.model_routing.route_resolver import (
    LLMTask,
    PrivacyMode,
    ResolvedRoute,
    RouteResolver,
)
from shared.models.llm import Model, ModelRoute

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRouteRepo:
    """In-memory ModelRouteRepository stand-in keyed by task string."""

    def __init__(self, rows: Optional[Dict[str, ModelRoute]] = None):
        self._rows = rows or {}

    async def get_by_task(self, task: str) -> Optional[ModelRoute]:
        return self._rows.get(task)


class FakeModelRepo:
    """In-memory model repo; returns configured Model rows by id."""

    def __init__(self, models: Optional[Dict[str, Model]] = None):
        self._models = models or {}

    async def get(self, model_id: str) -> Optional[Model]:
        return self._models.get(model_id)


@pytest.fixture
def fake_defaults(monkeypatch):
    """Inject a DefaultModels singleton via the model manager so the resolver's
    head-of-chain precedence has something to resolve."""
    from shared.models import DefaultModels

    defaults = DefaultModels(
        default_chat_model="model:chat",
        default_extraction_model="model:extract",
        default_transformation_model="model:summ",
    )

    class _MM:
        def get_defaults(self):
            return defaults

        def set_defaults(self, d):
            pass

    monkeypatch.setattr(
        "llm_manager.get_model_manager", lambda: _MM(), raising=True
    )
    return defaults


def _resolver(models: Optional[Dict[str, Model]] = None, rows=None) -> RouteResolver:
    return RouteResolver(
        model_route_repo=FakeRouteRepo(rows),
        model_repo=FakeModelRepo(models or {}),
    )


def _set_cloud_keys(monkeypatch, anthropic=True, openai=True):
    if anthropic:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    else:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    if openai:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
    else:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _providers(route: ResolvedRoute) -> List[str]:
    return [c.provider for c in route.ordered_candidates]


# ---------------------------------------------------------------------------
# AC#1 — CLOUD + NIM key -> [nvidia, local] (J.4 repointed the default chain
# from the anthropic/openai placeholder to the real cloud provider, NVIDIA NIM)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cloud_nvidia_key_orders_nvidia_local(monkeypatch, fake_defaults):
    monkeypatch.setenv("NVIDIA_API_KEY", "nvapi-test")
    resolver = _resolver()

    route = await resolver.resolve(LLMTask.ENTITY_EXTRACTION, PrivacyMode.CLOUD)

    assert _providers(route) == ["nvidia", "ollama"]
    # Final candidate is the local fallback.
    assert route.ordered_candidates[-1].is_local is True
    assert route.mode == PrivacyMode.CLOUD


# ---------------------------------------------------------------------------
# AC#2 — CLOUD, NIM key unset -> [local] (J-Q6 drop of the keyless provider)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cloud_missing_nvidia_key_drops_nvidia(monkeypatch, fake_defaults):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    resolver = _resolver()

    route = await resolver.resolve(LLMTask.ENTITY_EXTRACTION, PrivacyMode.CLOUD)

    assert _providers(route) == ["ollama"]
    assert route.ordered_candidates[-1].is_local is True


# ---------------------------------------------------------------------------
# AC#3 — PRIVATE -> local only, no cloud provider for any task
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_private_returns_local_only_for_all_tasks(monkeypatch, fake_defaults):
    # Even with cloud keys present, PRIVATE must never surface a cloud provider.
    _set_cloud_keys(monkeypatch, anthropic=True, openai=True)
    resolver = _resolver()

    for task in LLMTask:
        route = await resolver.resolve(task, PrivacyMode.PRIVATE)
        assert route.ordered_candidates, f"empty private route for {task}"
        assert all(c.is_local for c in route.ordered_candidates), task
        assert all(c.provider == "ollama" for c in route.ordered_candidates), task


@pytest.mark.asyncio
async def test_private_drops_cloud_entry_in_configured_chain(
    monkeypatch, fake_defaults
):
    """A misconfigured private_chain listing a cloud provider is filtered out."""
    _set_cloud_keys(monkeypatch, anthropic=True, openai=True)
    rows = {
        "chat": ModelRoute(
            task="chat",
            provider_chain=[
                {"provider": "anthropic", "model_id": "model:a", "enabled": True}
            ],
            private_chain=[
                {"provider": "openai", "model_id": "model:b", "enabled": True},
                {"provider": "ollama", "model_id": "model:l", "enabled": True},
            ],
        )
    }
    resolver = _resolver(rows=rows)

    route = await resolver.resolve(LLMTask.CHAT, PrivacyMode.PRIVATE)
    assert _providers(route) == ["ollama"]


# ---------------------------------------------------------------------------
# AC#4 — CLOUD with no cloud key -> single-entry [local]
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cloud_no_keys_returns_single_local(monkeypatch, fake_defaults):
    _set_cloud_keys(monkeypatch, anthropic=False, openai=False)
    resolver = _resolver()

    route = await resolver.resolve(LLMTask.CHAT, PrivacyMode.CLOUD)

    assert _providers(route) == ["ollama"]
    assert len(route.ordered_candidates) == 1
    assert route.ordered_candidates[0].is_local is True


# ---------------------------------------------------------------------------
# AC#5 — LLMTask has exactly three members (§1.2 guardrail)
# ---------------------------------------------------------------------------


def test_llm_task_has_exactly_three_members():
    assert {m.name for m in LLMTask} == {
        "ENTITY_EXTRACTION",
        "SUMMARIZATION",
        "CHAT",
    }
    # Belt-and-braces: no embedding/parsing member can ever route through here.
    assert not hasattr(LLMTask, "EMBEDDING")
    assert not hasattr(LLMTask, "PARSING")


# ---------------------------------------------------------------------------
# AC#6 — head-of-chain precedence + shim single-id behavior
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_head_candidate_honors_per_task_default(monkeypatch, fake_defaults):
    """The head candidate's model_id is the per-task default (B.8 precedence)."""
    _set_cloud_keys(monkeypatch, anthropic=True, openai=True)
    resolver = _resolver()

    route = await resolver.resolve(LLMTask.ENTITY_EXTRACTION, PrivacyMode.CLOUD)
    # default_extraction_model="model:extract" is the head id for the chain.
    assert route.ordered_candidates[0].model_id == "model:extract"


@pytest.mark.asyncio
async def test_explicit_model_id_override_wins(monkeypatch, fake_defaults):
    """An explicit per-call model_id overrides the per-task default (B.8)."""
    _set_cloud_keys(monkeypatch, anthropic=True, openai=True)
    resolver = _resolver()

    route = await resolver.resolve(
        LLMTask.ENTITY_EXTRACTION, PrivacyMode.CLOUD, model_id="model:override"
    )
    assert route.ordered_candidates[0].model_id == "model:override"


@pytest.mark.asyncio
async def test_disabled_model_row_drops_candidate(monkeypatch, fake_defaults):
    """An entry whose resolved model row is disabled is dropped."""
    _set_cloud_keys(monkeypatch, anthropic=True, openai=True)
    rows = {
        "chat": ModelRoute(
            task="chat",
            provider_chain=[
                {"provider": "anthropic", "model_id": "model:dead", "enabled": True},
                {"provider": "openai", "model_id": "model:live", "enabled": True},
            ],
        )
    }
    models = {
        "model:dead": Model(
            id="model:dead",
            name="dead",
            provider="anthropic",
            type="language",
            is_enabled=False,
        ),
        "model:live": Model(
            id="model:live",
            name="live",
            provider="openai",
            type="language",
            is_enabled=True,
        ),
    }
    resolver = _resolver(models=models, rows=rows)

    route = await resolver.resolve(LLMTask.CHAT, PrivacyMode.CLOUD)
    assert "anthropic" not in _providers(route)
    assert "openai" in _providers(route)


@pytest.mark.asyncio
async def test_disabled_chain_entry_is_skipped(monkeypatch, fake_defaults):
    """A chain entry flagged enabled=False is skipped regardless of key."""
    _set_cloud_keys(monkeypatch, anthropic=True, openai=True)
    rows = {
        "chat": ModelRoute(
            task="chat",
            provider_chain=[
                {"provider": "anthropic", "model_id": "model:a", "enabled": False},
                {"provider": "openai", "model_id": "model:b", "enabled": True},
            ],
        )
    }
    resolver = _resolver(rows=rows)

    route = await resolver.resolve(LLMTask.CHAT, PrivacyMode.CLOUD)
    assert "anthropic" not in _providers(route)


@pytest.mark.asyncio
async def test_resolve_default_model_id_shim_returns_single_id(
    monkeypatch, fake_defaults
):
    """AC#6: the B.8 shim still returns ONE id (the route head).

    Wires the resolver through DI so the shim exercises the real
    resolve_route -> get_route_resolver path with fakes.
    """
    import app_main.dependencies as deps
    from app_main.services.entity_extraction_service import (
        resolve_default_model_id,
    )

    _set_cloud_keys(monkeypatch, anthropic=True, openai=True)
    monkeypatch.setattr(deps, "get_route_resolver", lambda: _resolver(), raising=True)

    result = await resolve_default_model_id(
        "default_extraction_model", "model:x"
    )
    assert result == "model:x"
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_configured_chain_overrides_default_chain(monkeypatch, fake_defaults):
    """A model_route row replaces the hard-coded default provider order."""
    _set_cloud_keys(monkeypatch, anthropic=True, openai=True)
    rows = {
        "chat": ModelRoute(
            task="chat",
            provider_chain=[
                {"provider": "openai", "model_id": "model:b", "enabled": True},
                {"provider": "anthropic", "model_id": "model:a", "enabled": True},
            ],
        )
    }
    resolver = _resolver(rows=rows)

    route = await resolver.resolve(LLMTask.CHAT, PrivacyMode.CLOUD)
    # Configured order is openai-first; local still appended last.
    assert _providers(route) == ["openai", "anthropic", "ollama"]
