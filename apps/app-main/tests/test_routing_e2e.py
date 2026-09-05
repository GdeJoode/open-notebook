"""End-to-end routing guarantee tests (Track J.6).

These exercise the THREE headline Track-J behaviors through the real stack —
the real :class:`RouteResolver`, the real failover executor, the real
error-mapping predicate, and the real telemetry write path — with ONLY the
esperanto model factory mocked (no live NIM, no network, no SDK). Unlike the
per-phase unit tests (which inject a pre-built ``ResolvedRoute``), here the
route is *resolved* from a fake ``model_route`` repo + env keys so the
resolution → failover → telemetry chain is proven together.

Scenarios (the J.6 plan §"Phase J.6" E2E list):

  1. forced-outage failover: [nvidia -> ollama], nvidia forced down -> the
     document's extraction completes on the local provider and telemetry records
     the fallback event.
  2. private-never-cloud: a PRIVATE route, even with a cloud key present and a
     cloud entry in the persisted chain, never constructs a cloud LanguageModel
     for any stage.
  3. no-cloud->local: with the cloud key unset and global mode CLOUD, resolution
     yields a single-entry local route and the run completes on local.

Mocking strategy mirrors ``test_routed_extraction.py``: patch
``ModelManager.get_model_from_config`` (the single esperanto-construction seam)
and force ``esperanto.LanguageModel`` to the fake so ``call_candidate``'s
isinstance check passes.
"""

from typing import Dict, List, Optional
from unittest.mock import MagicMock

import pytest
from app_main.services.entity_extraction_service import make_default_llm_caller
from app_main.services.model_routing.route_resolver import (
    LLMTask,
    PrivacyMode,
    RouteResolver,
)
from shared.models.llm import Model, ModelRoute


# --------------------------------------------------------------------------- #
# Fakes (resolver repos + esperanto factory)
# --------------------------------------------------------------------------- #
class _FakeRouteRepo:
    def __init__(self, rows: Optional[Dict[str, ModelRoute]] = None):
        self._rows = rows or {}

    async def get_by_task(self, task: str) -> Optional[ModelRoute]:
        return self._rows.get(task)


class _FakeModelRepo:
    def __init__(self, models: Optional[Dict[str, Model]] = None):
        self._models = models or {}

    async def get(self, model_id: str) -> Optional[Model]:
        return self._models.get(model_id)


class _FakeLM:
    """esperanto-LanguageModel stand-in: raise or return completion text."""

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
    """Fresh breaker + rate-limiter registries per test (state isolation)."""
    from app_main import dependencies

    dependencies.get_circuit_breaker_registry.cache_clear()
    dependencies.get_rate_limiter_registry.cache_clear()
    yield
    dependencies.get_circuit_breaker_registry.cache_clear()
    dependencies.get_rate_limiter_registry.cache_clear()


@pytest.fixture(autouse=True)
def _quiet_metrics(monkeypatch):
    """Disable the real metrics DB write; the telemetry events are spied instead."""
    monkeypatch.setenv("OPEN_NOTEBOOK_DISABLE_METRICS", "1")


def _patch_factory(monkeypatch, behavior_by_provider):
    """Patch the esperanto-construction seam; record providers actually built."""
    built: List[str] = []

    def _fake_get_model_from_config(self, model, **kwargs):
        built.append(model.provider)
        return _FakeLM(behavior_by_provider[model.provider])

    from llm_manager.manager import ModelManager

    monkeypatch.setattr(
        ModelManager, "get_model_from_config", _fake_get_model_from_config
    )
    import esperanto

    monkeypatch.setattr(esperanto, "LanguageModel", _FakeLM)
    return built


def _install_resolver(monkeypatch, *, rows=None, models=None):
    """Wire a REAL RouteResolver (fake repos) behind get_route_resolver."""
    resolver = RouteResolver(
        model_route_repo=_FakeRouteRepo(rows),
        model_repo=_FakeModelRepo(models or {}),
    )
    from app_main import dependencies

    monkeypatch.setattr(dependencies, "get_route_resolver", lambda: resolver)
    return resolver


def _spy_telemetry(monkeypatch):
    """Capture the routing.served events the routed caller emits."""
    events: List[dict] = []

    async def _fake_record(**kwargs):
        events.append(kwargs)

    # The caller does ``from ...telemetry import record_routing_event`` inside
    # __call__, which resolves the module attribute at call time — patching the
    # module attribute is sufficient.
    import app_main.services.model_routing.telemetry as telemetry_mod

    monkeypatch.setattr(telemetry_mod, "record_routing_event", _fake_record)
    return events


def _cloud_route_row(task: str) -> ModelRoute:
    """A persisted [nvidia -> ollama] chain for ``task`` (cloud head, local tail)."""
    return ModelRoute(
        task=task,
        provider_chain=[
            {"provider": "nvidia", "model_id": "model:nim", "enabled": True},
            {"provider": "ollama", "model_id": "model:local", "enabled": True},
        ],
        private_chain=[{"provider": "ollama", "model_id": "model:local", "enabled": True}],
    )


def _allow_cloud(monkeypatch) -> None:
    """Treat the cloud provider as NOT retired for this test.

    PC.6 declared NVIDIA unavailable in `model_routing.yaml` and the resolver now
    honours it, so a cloud candidate no longer appears at all. These tests are
    about FAILOVER — cloud outage degrading to local — and need something to fail
    over from. Steering the retirement rather than the key keeps them testing
    failover instead of silently testing the retirement.
    """
    monkeypatch.setattr(
        "app_main.services.model_routing.route_resolver._provider_is_retired",
        lambda provider: False,
    )


def _models() -> Dict[str, Model]:
    return {
        "model:nim": Model(
            id="model:nim", name="mistral-medium", provider="nvidia",
            type="language", is_local=False,
        ),
        "model:local": Model(
            id="model:local", name="llama3.1:8b", provider="ollama",
            type="language", is_local=True,
        ),
    }


# --------------------------------------------------------------------------- #
# Scenario 1 — forced-outage failover
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_e2e_forced_outage_fails_over_to_local(monkeypatch):
    """nvidia forced down -> extraction completes on local; telemetry records it."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nim-key")  # cloud configured
    _allow_cloud(monkeypatch)
    _install_resolver(
        monkeypatch,
        rows={"entity_extraction": _cloud_route_row("entity_extraction")},
        models=_models(),
    )
    built = _patch_factory(
        monkeypatch,
        {
            # A failover-eligible 503 from the esperanto OpenAI-compatible wrapper.
            "nvidia": RuntimeError(
                "OpenAI-compatible endpoint error: HTTP 503: service unavailable"
            ),
            "ollama": "LOCAL_EXTRACTION",
        },
    )
    events = _spy_telemetry(monkeypatch)

    caller = await make_default_llm_caller(
        default_field="default_extraction_model",
        privacy_mode=PrivacyMode.CLOUD,
        source_id="source:doc1",
        notebook_id="notebook:nb1",
    )
    text = await caller("sys", "usr", "default")

    assert text == "LOCAL_EXTRACTION"
    assert caller.served_provider == "ollama"
    assert caller.was_failover is True
    assert caller.fallback_from == ["nvidia"]
    # nvidia WAS attempted (constructed) then failed; ollama served.
    assert built == ["nvidia", "ollama"]
    # Telemetry recorded the fallback for the document.
    assert len(events) == 1
    assert events[0]["served_provider"] == "ollama"
    assert events[0]["was_failover"] is True
    assert events[0]["fallback_from"] == ["nvidia"]
    assert events[0]["source_id"] == "source:doc1"
    assert events[0]["notebook_id"] == "notebook:nb1"


# --------------------------------------------------------------------------- #
# Scenario 2 — private-never-cloud
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_e2e_private_never_constructs_cloud(monkeypatch):
    """A PRIVATE run never builds a cloud model, even with cloud key + cloud chain."""
    monkeypatch.setenv("NVIDIA_API_KEY", "nim-key")  # cloud IS available...
    _allow_cloud(monkeypatch)
    _install_resolver(
        monkeypatch,
        # ...and the persisted chain even lists nvidia first.
        rows={"entity_extraction": _cloud_route_row("entity_extraction")},
        models=_models(),
    )
    built = _patch_factory(
        monkeypatch,
        {
            "ollama": "PRIVATE_RESULT",
            # Registered but constructing this is the bug we are guarding against.
            "nvidia": "LEAKED_TO_CLOUD",
        },
    )

    caller = await make_default_llm_caller(
        default_field="default_extraction_model",
        privacy_mode=PrivacyMode.PRIVATE,
        source_id="source:secret",
    )
    text = await caller("sys", "usr", "default")

    assert text == "PRIVATE_RESULT"
    assert caller.served_provider == "ollama"
    # The load-bearing guarantee: no cloud LanguageModel was EVER constructed.
    assert "nvidia" not in built
    assert built == ["ollama"]


# --------------------------------------------------------------------------- #
# Scenario 3 — no-cloud -> local
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_e2e_no_cloud_key_resolves_single_local(monkeypatch):
    """With the cloud key unset and mode CLOUD, resolution drops to local-only."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)  # cloud NOT configured
    resolver = _install_resolver(
        monkeypatch,
        rows={"entity_extraction": _cloud_route_row("entity_extraction")},
        models=_models(),
    )

    # Prove resolution itself drops the keyless cloud provider (J-Q5/J-Q6).
    route = await resolver.resolve(
        LLMTask.ENTITY_EXTRACTION, PrivacyMode.CLOUD, model_id=None
    )
    providers = [c.provider for c in route.ordered_candidates]
    assert "nvidia" not in providers
    assert providers == ["ollama"]  # single-entry local route
    assert route.ordered_candidates[0].is_local is True

    built = _patch_factory(monkeypatch, {"ollama": "LOCAL_ONLY"})
    caller = await make_default_llm_caller(
        default_field="default_extraction_model",
        privacy_mode=PrivacyMode.CLOUD,
    )
    text = await caller("sys", "usr", "default")

    assert text == "LOCAL_ONLY"
    assert caller.served_provider == "ollama"
    assert caller.was_failover is False
    assert built == ["ollama"]
