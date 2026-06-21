"""Structured-JSON-output tests for the extraction LLM path (FU-J4-4).

The live NIM run + local-model tests showed frequent ``Failed to parse LLM
response`` errors: models emitted malformed/partial JSON and the extractor
dropped whole batches. The fix requests structured JSON output from the model
for the EXTRACTION path only, via esperanto's provider-agnostic ``structured``
config field (openai-compatible/NIM -> ``response_format``, ollama ->
``format=json``).

These tests spy the esperanto build seam (``ModelManager.get_model_from_config``)
to assert:

  1. extraction requests JSON mode -> ``structured={"type": "json_object"}`` is
     threaded into the model build for every candidate (AC1);
  2. chat/summarization callers do NOT request JSON mode (AC2);
  3. a provider that doesn't honor ``structured`` does not crash the run (AC3) —
     esperanto's base ``LanguageModel`` accepts the field and providers ignore
     it, so no capability gate is needed.

The build seam is mocked, so no SDK/network is touched.
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
        model=Model(
            id=model_id,
            name=model_id,
            provider=provider,
            type="language",
            is_local=is_local,
        ),
    )


class _FakeLM:
    """Minimal esperanto-LanguageModel stand-in returning fixed text."""

    def __init__(self, text):
        self._text = text

    async def achat_complete(self, messages):
        resp = MagicMock()
        resp.choices = [MagicMock(message=MagicMock(content=self._text))]
        return resp


@pytest.fixture(autouse=True)
def _reset_singletons():
    """Fresh circuit-breaker + rate-limiter registries per test."""
    from app_main import dependencies

    dependencies.get_circuit_breaker_registry.cache_clear()
    dependencies.get_rate_limiter_registry.cache_clear()
    yield
    dependencies.get_circuit_breaker_registry.cache_clear()
    dependencies.get_rate_limiter_registry.cache_clear()


def _patch_factory(monkeypatch):
    """Spy ``ModelManager.get_model_from_config``; record the kwargs per call.

    Returns the recorded-kwargs list so a test can assert whether
    ``structured`` was (or was not) threaded into the build.
    """
    recorded_kwargs = []

    def _fake_get_model_from_config(self, model, **kwargs):
        recorded_kwargs.append(kwargs)
        return _FakeLM("{}")

    from llm_manager.manager import ModelManager

    monkeypatch.setattr(
        ModelManager, "get_model_from_config", _fake_get_model_from_config
    )
    import esperanto

    monkeypatch.setattr(esperanto, "LanguageModel", _FakeLM)
    return recorded_kwargs


def _patch_route(monkeypatch, route: ResolvedRoute):
    resolver = MagicMock()
    resolver.resolve = AsyncMock(return_value=route)
    from app_main import dependencies

    monkeypatch.setattr(dependencies, "get_route_resolver", lambda: resolver)
    return resolver


@pytest.mark.asyncio
async def test_extraction_requests_json_object_response_format(monkeypatch):
    """AC1: the extraction caller threads ``structured={"type":"json_object"}``
    into the esperanto build — the provider-agnostic value esperanto translates
    to ``response_format`` (openai-compatible/NIM) or ``format=json`` (ollama)."""
    route = ResolvedRoute(
        task=LLMTask.ENTITY_EXTRACTION,
        mode=PrivacyMode.CLOUD,
        ordered_candidates=[_candidate("nvidia", "model:nim", is_local=False)],
    )
    _patch_route(monkeypatch, route)
    recorded = _patch_factory(monkeypatch)

    caller = await make_default_llm_caller(
        default_field="default_extraction_model", privacy_mode=PrivacyMode.CLOUD
    )
    await caller("sys", "usr", "default")

    assert len(recorded) == 1
    assert recorded[0].get("structured") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_extraction_json_mode_applies_to_every_candidate(monkeypatch):
    """AC1: JSON mode is requested for EACH candidate the route tries, including
    a local ollama fallback (which esperanto maps to ``format=json``)."""
    # Cloud head fails over to local last-resort; both builds must request JSON.
    route = ResolvedRoute(
        task=LLMTask.ENTITY_EXTRACTION,
        mode=PrivacyMode.CLOUD,
        ordered_candidates=[
            _candidate("nvidia", "model:nim", is_local=False),
            _candidate("ollama", "model:local", is_local=True),
        ],
    )
    _patch_route(monkeypatch, route)

    recorded = []

    def _fake(self, model, **kwargs):
        recorded.append((model.provider, kwargs))
        if model.provider == "nvidia":
            # Failover-eligible 5xx -> next candidate.
            raise RuntimeError(
                "OpenAI-compatible endpoint error: HTTP 503: service unavailable"
            )
        return _FakeLM("{}")

    from llm_manager.manager import ModelManager

    monkeypatch.setattr(ModelManager, "get_model_from_config", _fake)
    import esperanto

    monkeypatch.setattr(esperanto, "LanguageModel", _FakeLM)

    caller = await make_default_llm_caller(
        default_field="default_extraction_model", privacy_mode=PrivacyMode.CLOUD
    )
    await caller("sys", "usr", "default")

    assert [p for p, _ in recorded] == ["nvidia", "ollama"]
    for _provider, kwargs in recorded:
        assert kwargs.get("structured") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_chat_caller_does_not_request_json_mode(monkeypatch):
    """AC2: a CHAT-task caller must NOT request structured output (prose path)."""
    route = ResolvedRoute(
        task=LLMTask.CHAT,
        mode=PrivacyMode.CLOUD,
        ordered_candidates=[_candidate("openai", "model:chat", is_local=False)],
    )
    _patch_route(monkeypatch, route)
    recorded = _patch_factory(monkeypatch)

    caller = await make_default_llm_caller(
        default_field="default_chat_model", privacy_mode=PrivacyMode.CLOUD
    )
    await caller("sys", "usr", "default")

    assert len(recorded) == 1
    assert "structured" not in recorded[0]


@pytest.mark.asyncio
async def test_summarization_caller_does_not_request_json_mode(monkeypatch):
    """AC2: a SUMMARIZATION-task caller must NOT request structured output."""
    route = ResolvedRoute(
        task=LLMTask.SUMMARIZATION,
        mode=PrivacyMode.CLOUD,
        ordered_candidates=[_candidate("openai", "model:sum", is_local=False)],
    )
    _patch_route(monkeypatch, route)
    recorded = _patch_factory(monkeypatch)

    caller = await make_default_llm_caller(
        default_field="default_transformation_model",
        privacy_mode=PrivacyMode.CLOUD,
    )
    await caller("sys", "usr", "default")

    assert len(recorded) == 1
    assert "structured" not in recorded[0]


@pytest.mark.asyncio
async def test_explicit_json_mode_override_off_for_extraction(monkeypatch):
    """An explicit ``json_mode=False`` overrides the per-task default."""
    route = ResolvedRoute(
        task=LLMTask.ENTITY_EXTRACTION,
        mode=PrivacyMode.CLOUD,
        ordered_candidates=[_candidate("nvidia", "model:nim", is_local=False)],
    )
    _patch_route(monkeypatch, route)
    recorded = _patch_factory(monkeypatch)

    caller = await make_default_llm_caller(
        default_field="default_extraction_model",
        privacy_mode=PrivacyMode.CLOUD,
        json_mode=False,
    )
    await caller("sys", "usr", "default")

    assert len(recorded) == 1
    assert "structured" not in recorded[0]


@pytest.mark.asyncio
async def test_json_mode_degrades_gracefully_for_unsupporting_provider(monkeypatch):
    """AC3: a provider that doesn't honor ``structured`` must not crash the run.

    Esperanto's base ``LanguageModel`` accepts ``structured`` and providers that
    don't implement it (e.g. anthropic) ignore the field — so threading it
    universally is safe. We simulate that here: the build accepts the kwarg and
    the call still succeeds, returning the model's text.
    """
    route = ResolvedRoute(
        task=LLMTask.ENTITY_EXTRACTION,
        mode=PrivacyMode.CLOUD,
        ordered_candidates=[
            _candidate("anthropic", "model:claude", is_local=False)
        ],
    )
    _patch_route(monkeypatch, route)

    def _fake(self, model, **kwargs):
        # Provider silently ignores ``structured`` (no crash), like anthropic.
        assert kwargs.get("structured") == {"type": "json_object"}
        return _FakeLM('{"entities": []}')

    from llm_manager.manager import ModelManager

    monkeypatch.setattr(ModelManager, "get_model_from_config", _fake)
    import esperanto

    monkeypatch.setattr(esperanto, "LanguageModel", _FakeLM)

    caller = await make_default_llm_caller(
        default_field="default_extraction_model", privacy_mode=PrivacyMode.CLOUD
    )
    text = await caller("sys", "usr", "default")

    assert text == '{"entities": []}'
    assert caller.served_provider == "anthropic"
