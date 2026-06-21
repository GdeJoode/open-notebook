"""Routed-extraction failover integration tests (Track J.4).

Exercises the rebuilt ``make_default_llm_caller`` end-to-end through the J.2
failover executor + J.4 error mapping, with the esperanto model factory mocked
(no live NIM key). Covers:

  1. failover-to-next: cloud raises a failover-eligible error -> next candidate
     serves; served provenance + was_failover reflect the fallback.
  2. fail-to-local: every cloud candidate fails -> the local last-resort serves.
  3. private-never-cloud: a PRIVATE route never constructs a cloud LanguageModel
     (the factory is only ever called with a local provider) — AC6.

All provider behavior is injected by monkeypatching
``ModelManager.get_model_from_config`` (the single esperanto-construction seam
``call_candidate`` crosses), so no SDK/network is touched.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app_main.services.entity_extraction_service import make_default_llm_caller
from app_main.services.model_routing.route_resolver import (
    LLMTask,
    ModelCandidate,
    PrivacyMode,
    ResolvedRoute,
)


def _candidate(provider: str, model_id: str, is_local: bool) -> ModelCandidate:
    from shared.models.llm import Model

    return ModelCandidate(
        provider=provider,
        model_id=model_id,
        is_local=is_local,
        model=Model(id=model_id, name=model_id, provider=provider, type="language", is_local=is_local),
    )


class _FakeLM:
    """Minimal esperanto-LanguageModel stand-in.

    ``behavior`` is either an Exception instance (raised) or a string (returned
    as the completion text). Records every call so a test can assert it was (or
    was not) constructed/invoked.
    """

    def __init__(self, behavior):
        self._behavior = behavior

    async def achat_complete(self, messages):
        if isinstance(self._behavior, Exception):
            raise self._behavior
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content=self._behavior))]
        return resp


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Fresh circuit-breaker + rate-limiter registries per test so breaker state
    from one scenario never leaks into the next."""
    from app_main import dependencies

    dependencies.get_circuit_breaker_registry.cache_clear()
    dependencies.get_rate_limiter_registry.cache_clear()
    yield
    dependencies.get_circuit_breaker_registry.cache_clear()
    dependencies.get_rate_limiter_registry.cache_clear()


def _patch_factory(monkeypatch, behavior_by_provider):
    """Patch ModelManager.get_model_from_config to return a _FakeLM per provider.

    Returns a list that records the providers the factory was asked to build —
    the private-never-cloud test asserts on it (AC6).
    """
    built_providers = []

    def _fake_get_model_from_config(self, model, **kwargs):
        built_providers.append(model.provider)
        behavior = behavior_by_provider[model.provider]
        return _FakeLM(behavior)

    from llm_manager.manager import ModelManager

    monkeypatch.setattr(
        ModelManager, "get_model_from_config", _fake_get_model_from_config
    )
    # call_candidate isinstance-checks against esperanto.LanguageModel.
    import esperanto

    monkeypatch.setattr(esperanto, "LanguageModel", _FakeLM)
    return built_providers


def _patch_route(monkeypatch, route: ResolvedRoute):
    """Force resolve_route (via the resolver returned by get_route_resolver) to
    yield ``route`` regardless of DB/env."""
    resolver = MagicMock()
    resolver.resolve = AsyncMock(return_value=route)
    from app_main import dependencies

    monkeypatch.setattr(dependencies, "get_route_resolver", lambda: resolver)
    return resolver


@pytest.mark.asyncio
async def test_failover_to_next_cloud_provider(monkeypatch):
    """Cloud candidate fails with a 503 -> the next candidate serves; provenance
    records the failover."""
    route = ResolvedRoute(
        task=LLMTask.ENTITY_EXTRACTION,
        mode=PrivacyMode.CLOUD,
        ordered_candidates=[
            _candidate("nvidia", "model:nim", is_local=False),
            _candidate("openai", "model:oai", is_local=False),
        ],
    )
    _patch_route(monkeypatch, route)
    _patch_factory(
        monkeypatch,
        {
            "nvidia": RuntimeError(
                "OpenAI-compatible endpoint error: HTTP 503: service unavailable"
            ),
            "openai": "EXTRACTED",
        },
    )

    caller = await make_default_llm_caller(
        default_field="default_extraction_model", privacy_mode=PrivacyMode.CLOUD
    )
    text = await caller("sys", "usr", "default")

    assert text == "EXTRACTED"
    assert caller.served_provider == "openai"
    assert caller.was_failover is True
    assert caller.fallback_from == ["nvidia"]


@pytest.mark.asyncio
async def test_fail_to_local_last_resort(monkeypatch):
    """Both cloud candidates fail (5xx) -> the local last-resort serves."""
    route = ResolvedRoute(
        task=LLMTask.ENTITY_EXTRACTION,
        mode=PrivacyMode.CLOUD,
        ordered_candidates=[
            _candidate("nvidia", "model:nim", is_local=False),
            _candidate("openai", "model:oai", is_local=False),
            _candidate("ollama", "model:local", is_local=True),
        ],
    )
    _patch_route(monkeypatch, route)
    _patch_factory(
        monkeypatch,
        {
            "nvidia": RuntimeError("OpenAI-compatible endpoint error: HTTP 500: boom"),
            "openai": RuntimeError("OpenAI API error: HTTP 502: bad gateway"),
            "ollama": "LOCAL_SUMMARY",
        },
    )

    caller = await make_default_llm_caller(
        default_field="default_extraction_model", privacy_mode=PrivacyMode.CLOUD
    )
    text = await caller("sys", "usr", "default")

    assert text == "LOCAL_SUMMARY"
    assert caller.served_provider == "ollama"
    assert caller.was_failover is True
    assert set(caller.fallback_from) == {"nvidia", "openai"}


@pytest.mark.asyncio
async def test_private_never_constructs_cloud_model(monkeypatch):
    """AC6: a PRIVATE route only ever constructs a LOCAL LanguageModel.

    The factory-build spy must never see a cloud provider, even though cloud
    behavior is registered. The PRIVATE route (local-only) is the guarantee."""
    route = ResolvedRoute(
        task=LLMTask.ENTITY_EXTRACTION,
        mode=PrivacyMode.PRIVATE,
        ordered_candidates=[
            _candidate("ollama", "model:local", is_local=True),
        ],
    )
    _patch_route(monkeypatch, route)
    built = _patch_factory(
        monkeypatch,
        {
            "ollama": "PRIVATE_RESULT",
            # Registered but must never be built — a cloud build here is the bug.
            "nvidia": "LEAKED_TO_CLOUD",
            "openai": "LEAKED_TO_CLOUD",
        },
    )

    caller = await make_default_llm_caller(
        default_field="default_extraction_model", privacy_mode=PrivacyMode.PRIVATE
    )
    text = await caller("sys", "usr", "default")

    assert text == "PRIVATE_RESULT"
    assert caller.served_provider == "ollama"
    # The load-bearing assertion: no cloud LanguageModel was ever constructed.
    assert built == ["ollama"]
    assert "nvidia" not in built
    assert "openai" not in built
