"""Routed-summarization tests (Track J.4).

Two guarantees:

  1. routed cloud summary: when the app injects the J.4
     :class:`FailoverSummarizationModel`, the summarization strategy produces a
     summary via the routed (cloud) provider — proving the summarization path no
     longer hard-pins ollama through ``LLMConfig`` / ``AIFactory``.
  2. back-compat: a bare ``SummarizationWorkflow(config)`` with NO injected model
     still runs the legacy env-driven ``AIFactory.create_language`` ollama path,
     so CLI/tests don't break.

SDKs are mocked at the model-factory seam; no live keys.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from summarization.config import LLMConfig, NaiveConfig, SummarizationConfig
from summarization.models.result import ChunkInput
from summarization.naive.strategy import NaiveLLMStrategy

from app_main.services.model_routing.route_resolver import (
    LLMTask,
    ModelCandidate,
    PrivacyMode,
    ResolvedRoute,
)


def _chunks():
    return [ChunkInput(text="Hello world. This is a document.", chunk_id="c:1", order=0)]


def _candidate(provider, model_id, is_local):
    from shared.models.llm import Model

    return ModelCandidate(
        provider=provider,
        model_id=model_id,
        is_local=is_local,
        model=Model(id=model_id, name=model_id, provider=provider, type="language", is_local=is_local),
    )


@pytest.fixture(autouse=True)
def _reset_singletons():
    from app_main import dependencies

    dependencies.get_circuit_breaker_registry.cache_clear()
    dependencies.get_rate_limiter_registry.cache_clear()
    yield
    dependencies.get_circuit_breaker_registry.cache_clear()
    dependencies.get_rate_limiter_registry.cache_clear()


class _FakeResponse:
    """esperanto-ChatCompletion-shaped: exposes both ``.content`` (read by the
    summarization strategies directly) and ``.choices[0].message.content`` (read
    by ``call_candidate`` on the routed path)."""

    def __init__(self, text):
        self.content = text
        self.choices = [MagicMock(message=MagicMock(content=text))]


class _FakeLM:
    def __init__(self, text):
        self._text = text

    async def achat_complete(self, messages):
        return _FakeResponse(self._text)


@pytest.mark.asyncio
async def test_routed_cloud_summary(monkeypatch):
    """An injected FailoverSummarizationModel produces a summary via the routed
    cloud provider (NIM), proving summarization is no longer ollama-pinned."""
    from app_main.services.model_routing.summarization_model import (
        FailoverSummarizationModel,
    )

    route = ResolvedRoute(
        task=LLMTask.SUMMARIZATION,
        mode=PrivacyMode.CLOUD,
        ordered_candidates=[
            _candidate("nvidia", "model:nim", is_local=False),
            _candidate("ollama", "model:local", is_local=True),
        ],
    )
    resolver = MagicMock()
    resolver.resolve = AsyncMock(return_value=route)
    from app_main import dependencies

    monkeypatch.setattr(dependencies, "get_route_resolver", lambda: resolver)

    built = []

    def _fake_get_model_from_config(self, model, **kwargs):
        built.append(model.provider)
        return _FakeLM("CLOUD SUMMARY")

    from llm_manager.manager import ModelManager

    monkeypatch.setattr(
        ModelManager, "get_model_from_config", _fake_get_model_from_config
    )
    import esperanto

    monkeypatch.setattr(esperanto, "LanguageModel", _FakeLM)

    injected = FailoverSummarizationModel(mode=PrivacyMode.CLOUD)
    config = SummarizationConfig(
        strategy="naive",
        language_model=injected,
        naive=NaiveConfig(max_input_length=10000),
        llm=LLMConfig(provider="ollama", model_name="llama3.1:8b", base_url="http://x"),
    )
    strategy = NaiveLLMStrategy(config)
    result = await strategy.summarize(_chunks())

    assert result.document_summary == "CLOUD SUMMARY"
    # The cloud provider served the summary (not ollama).
    assert injected.served_provider == "nvidia"
    assert built == ["nvidia"]


@pytest.mark.asyncio
async def test_backcompat_no_injection_uses_env_ollama(monkeypatch):
    """No injected model -> the strategy builds via AIFactory.create_language
    from the env-driven LLMConfig (legacy ollama path). Back-compat guard."""
    created = {}

    def _fake_create_language(provider, model_name, config=None):
        created["provider"] = provider
        created["model_name"] = model_name
        created["base_url"] = (config or {}).get("base_url")
        return _FakeLM("LOCAL SUMMARY")

    monkeypatch.setattr(
        "summarization.naive.strategy.AIFactory.create_language",
        _fake_create_language,
    )

    config = SummarizationConfig(
        strategy="naive",
        # No language_model injected -> env-driven path.
        naive=NaiveConfig(max_input_length=10000),
        llm=LLMConfig(
            provider="ollama",
            model_name="llama3.1:8b",
            base_url="http://localhost:11434",
        ),
    )
    strategy = NaiveLLMStrategy(config)
    result = await strategy.summarize(_chunks())

    assert result.document_summary == "LOCAL SUMMARY"
    # The legacy AIFactory path was used with the ollama config.
    assert created["provider"] == "ollama"
    assert created["base_url"] == "http://localhost:11434"
