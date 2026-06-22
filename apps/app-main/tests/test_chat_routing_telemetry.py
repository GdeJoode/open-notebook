"""Chat-path routing telemetry tests (Track J.6, FU-J4-1).

J.4 left the chat path (``graphs/utils.py::provision_langchain_model``) without a
``routing.served`` telemetry write — extraction and summarization stamped their
served provider, chat did not (FU-J4-1). J.6 closes that gap. These tests pin:

  1. A clean first-candidate construction records ONE ``routing.served`` event
     with ``was_failover=False`` and an empty ``fallback_from``.
  2. A first-candidate construction failure that falls back to the next records
     ``was_failover=True`` with the failed provider in ``fallback_from``.
  3. A telemetry write failure never breaks chat model construction (best-effort).

The esperanto factory + ``record_routing_event`` are both mocked — no SDK, no DB.
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from app_main.services.model_routing.route_resolver import (
    LLMTask,
    ModelCandidate,
    PrivacyMode,
    ResolvedRoute,
)


def _load_graph_utils():
    """Load ``app_main.graphs.utils`` without triggering the package ``__init__``.

    The ``app_main.graphs`` package ``__init__`` eagerly imports sibling graph
    modules (ask/chat/...) that depend on the optional ``ai_prompter`` package,
    which is not installed in the unit-test environment. ``utils`` itself has no
    such dependency, so we load it directly from its file path.
    """
    if "app_main.graphs.utils" in sys.modules:
        return sys.modules["app_main.graphs.utils"]
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "app_main"
        / "graphs"
        / "utils.py"
    )
    spec = importlib.util.spec_from_file_location("app_main.graphs.utils", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["app_main.graphs.utils"] = module
    spec.loader.exec_module(module)
    return module


graph_utils = _load_graph_utils()


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


class _FakeLangchainModel:
    """A stand-in for what ``to_langchain()`` returns."""


class _FakeLM:
    """esperanto-LanguageModel stand-in that records construction order."""

    def to_langchain(self):
        return _FakeLangchainModel()


def _patch_route(monkeypatch, route: ResolvedRoute):
    resolver = MagicMock()
    resolver.resolve = AsyncMock(return_value=route)
    monkeypatch.setattr(graph_utils, "get_route_resolver", lambda: resolver, raising=False)
    # provision_langchain_model imports get_route_resolver from app_main.dependencies
    from app_main import dependencies

    monkeypatch.setattr(dependencies, "get_route_resolver", lambda: resolver)
    return resolver


def _patch_factory(monkeypatch, behavior_by_provider):
    """Patch ModelManager.get_model_from_config; raise or return _FakeLM per provider."""
    built: list[str] = []

    def _fake(self, model, **kwargs):
        built.append(model.provider)
        behavior = behavior_by_provider[model.provider]
        if isinstance(behavior, Exception):
            raise behavior
        return _FakeLM()

    from llm_manager.manager import ModelManager

    monkeypatch.setattr(ModelManager, "get_model_from_config", _fake)

    # The function isinstance-checks against esperanto.LanguageModel; align it.
    import esperanto

    monkeypatch.setattr(esperanto, "LanguageModel", _FakeLM)
    monkeypatch.setattr(graph_utils, "LanguageModel", _FakeLM)
    return built


def _spy_telemetry(monkeypatch):
    """Capture record_routing_event kwargs without writing anything."""
    events: list[dict] = []

    async def _fake_record(**kwargs):
        events.append(kwargs)

    import app_main.services.model_routing.telemetry as telemetry_mod

    monkeypatch.setattr(telemetry_mod, "record_routing_event", _fake_record)
    return events


@pytest.mark.asyncio
async def test_chat_records_served_provider_no_failover(monkeypatch):
    """First candidate constructs cleanly -> one event, was_failover False."""
    route = ResolvedRoute(
        task=LLMTask.CHAT,
        mode=PrivacyMode.CLOUD,
        ordered_candidates=[
            _candidate("nvidia", "model:nim", is_local=False),
            _candidate("ollama", "model:local", is_local=True),
        ],
    )
    _patch_route(monkeypatch, route)
    built = _patch_factory(monkeypatch, {"nvidia": "ok", "ollama": "ok"})
    events = _spy_telemetry(monkeypatch)

    model = await graph_utils.provision_langchain_model("hi", None, "chat")

    assert isinstance(model, _FakeLangchainModel)
    assert built == ["nvidia"]  # second candidate never built
    assert len(events) == 1
    assert events[0]["served_provider"] == "nvidia"
    assert events[0]["served_model_id"] == "model:nim"
    assert events[0]["was_failover"] is False
    assert events[0]["fallback_from"] == []
    assert events[0]["task"] == LLMTask.CHAT


@pytest.mark.asyncio
async def test_chat_records_failover_on_construction_failure(monkeypatch):
    """First candidate fails to construct -> fall back; event carries the trail."""
    route = ResolvedRoute(
        task=LLMTask.CHAT,
        mode=PrivacyMode.CLOUD,
        ordered_candidates=[
            _candidate("nvidia", "model:nim", is_local=False),
            _candidate("ollama", "model:local", is_local=True),
        ],
    )
    _patch_route(monkeypatch, route)
    built = _patch_factory(
        monkeypatch,
        {"nvidia": RuntimeError("construction blew up"), "ollama": "ok"},
    )
    events = _spy_telemetry(monkeypatch)

    model = await graph_utils.provision_langchain_model("hi", None, "chat")

    assert isinstance(model, _FakeLangchainModel)
    assert built == ["nvidia", "ollama"]
    assert len(events) == 1
    assert events[0]["served_provider"] == "ollama"
    assert events[0]["was_failover"] is True
    assert events[0]["fallback_from"] == ["nvidia"]


@pytest.mark.asyncio
async def test_chat_telemetry_failure_does_not_break_construction(monkeypatch):
    """A telemetry write failure never fails chat model construction."""
    route = ResolvedRoute(
        task=LLMTask.CHAT,
        mode=PrivacyMode.CLOUD,
        ordered_candidates=[_candidate("ollama", "model:local", is_local=True)],
    )
    _patch_route(monkeypatch, route)
    _patch_factory(monkeypatch, {"ollama": "ok"})

    async def _boom(**kwargs):
        raise RuntimeError("telemetry sink down")

    import app_main.services.model_routing.telemetry as telemetry_mod

    # A raising sink (simulating record_routing_event's own swallow being
    # bypassed) must NOT propagate through chat model construction — the helper
    # has its own guard.
    monkeypatch.setattr(telemetry_mod, "record_routing_event", _boom)

    model = await graph_utils.provision_langchain_model("hi", None, "chat")
    assert isinstance(model, _FakeLangchainModel)
