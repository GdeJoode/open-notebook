"""
Shared test fixtures for app-main.

All repositories are AsyncMock — no database required.
"""

import inspect
import os
import re
from datetime import datetime
from typing import Any, Dict, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from shared.models import (
    ChatMessage,
    ChatSession,
    Chunk,
    ContentSettings,
    DefaultModels,
    Model,
    Note,
    Notebook,
    Source,
    SourceEmbedding,
    SourceInsight,
    Transformation,
)
from shared.models.podcast import EpisodeProfile, PodcastEpisode, SpeakerProfile
from shared.models.transformation import DefaultPrompts

# Re-export the testcontainers-backed SurrealDB fixture so app-main integration
# tests (e.g. test_relation_merge.py) can request ``live_surrealdb`` by name,
# mirroring packages/surrealdb-service/tests/conftest.py. Auto-skips when Docker
# is unreachable (see surrealdb_service.testing.fixtures).
from surrealdb_service.testing import live_surrealdb, live_surrealdb_rocksdb  # noqa: F401

_DISABLE_METRICS_KEY = "OPEN_NOTEBOOK_DISABLE_METRICS"

# The registry's database-first reads as shipped, captured at import — before
# the autouse `_ontology_registry_reads_files` patches them. A test that reads
# ontologies from a real SurrealDB restores these; capturing them inside a
# fixture would capture the patch instead (OE.5 review M2).
from ontology_manager.registry import OntologyRegistry as _OntologyRegistry  # noqa: E402

REGISTRY_LOAD_FROM_DB = _OntologyRegistry._load_from_db
REGISTRY_LIST_ONTOLOGIES = _OntologyRegistry.list_ontologies


@pytest.fixture(autouse=True)
def _testsleutel_beheerder(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Elk verzoek in een test krijgt een sleutel met recht ``beheerder`` (F2-01).

    Sinds F2-01 vraagt elke route een sleutel. De meeste tests gaan over de
    route zelf, niet over de sleutel, en sturen geen header. Deze fixture
    vervangt daarom alleen in tests ``authenticate_request`` door een vaste
    sleutel ``beheerder``. Tests van de sleutelcontrole zelf dragen de markering
    ``echte_sleutels`` en lopen door de echte controle.
    """
    from app_main.api import auth

    # De drempel voor mislukte sleutels is per IP; alle tests delen "testclient".
    auth._mislukt.clear()
    if request.node.get_closest_marker("echte_sleutels"):
        return
    # Een test die 401 of 403 verwacht, toetst de sleutelcontrole en moet die
    # dus echt doorlopen: zonder markering faalt hij hier, in plaats van stil
    # met de testsleutel te slagen of om de verkeerde reden te falen. Herkend
    # aan een vergelijking (`== 401`, `!= 403`) in de testfunctie zelf; een
    # verwachting in een hulpfunctie ziet deze controle niet.
    try:
        bron = inspect.getsource(request.node.function)
    except (AttributeError, OSError, TypeError):
        bron = ""
    if re.search(r"(?:==|!=)\s*40[13](?!\d)", bron):
        pytest.fail(
            f"{request.node.nodeid} verwacht 401 of 403 maar draagt de markering "
            "'echte_sleutels' niet (F2-01)."
        )
    from app_main.api import agent_auth
    from shared.models.agents import Right

    ctx = agent_auth.AgentKeyContext(
        key_id="agent_keys:test",
        agent_id="test",
        permission=Right.BEHEERDER,
        person="Testpersoon",
    )

    async def _vaste_sleutel(_request):
        return ctx

    monkeypatch.setattr(agent_auth, "authenticate_request", _vaste_sleutel)


@pytest.fixture(autouse=True)
def _disable_metrics_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Suppress ``record_metric`` writes in unit tests.

    The shared ``shared.services.metrics.record_metric`` helper is
    failure-tolerant but it still tries to reach SurrealDB unless the
    env flag is set. In unit tests we never want that — both for speed
    and because most tests are AsyncMock-only and shouldn't take a
    DB dependency. Tests that explicitly want to assert on the metric
    write monkeypatch ``shared.services.metrics.record_metric`` directly.
    """
    monkeypatch.setenv(_DISABLE_METRICS_KEY, "1")


@pytest.fixture(autouse=True)
def _ontology_registry_reads_files(monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests read the shipped YAML ontologies, never a database's.

    PC.4: typing loads every ontology through the registry, which asks the
    database first — through the GLOBAL pool, i.e. whatever the env default
    points at (a developer's server), and an unreachable one hangs the pool
    rather than failing. Same reasoning as the metrics switch above: a unit test
    takes no DB dependency. No app-main test exercises the registry's database
    path; the ontology-manager package tests it.
    """
    from ontology_manager.registry import OntologyRegistry

    async def _no_db_version(self, name, version=None):
        return None

    async def _file_names(self):
        return sorted(p.stem for p in self._ontology_dir.glob("*.yaml"))

    monkeypatch.setattr(OntologyRegistry, "_load_from_db", _no_db_version)
    monkeypatch.setattr(OntologyRegistry, "list_ontologies", _file_names)


# ---------------------------------------------------------------------------
# Timestamp helpers
# ---------------------------------------------------------------------------
_NOW = datetime(2025, 6, 15, 12, 0, 0)


# ---------------------------------------------------------------------------
# Model factories
# ---------------------------------------------------------------------------

def make_notebook(**overrides: Any) -> Notebook:
    defaults: Dict[str, Any] = {
        "id": "notebook:test1",
        "name": "Test Notebook",
        "description": "A test notebook",
        "archived": False,
        "created": _NOW,
        "updated": _NOW,
    }
    defaults.update(overrides)
    return Notebook(**defaults)


def make_source(**overrides: Any) -> Source:
    defaults: Dict[str, Any] = {
        "id": "source:test1",
        "title": "Test Source",
        "full_text": "Some content here.",
        "topics": ["test"],
        "created": _NOW,
        "updated": _NOW,
    }
    defaults.update(overrides)
    return Source(**defaults)


def make_note(**overrides: Any) -> Note:
    defaults: Dict[str, Any] = {
        "id": "note:test1",
        "title": "Test Note",
        "content": "Note content here.",
        "note_type": "human",
        "created": _NOW,
        "updated": _NOW,
    }
    defaults.update(overrides)
    return Note(**defaults)


def make_model(**overrides: Any) -> Model:
    defaults: Dict[str, Any] = {
        "id": "model:test1",
        "name": "gpt-4o-mini",
        "provider": "openai",
        "type": "language",
        "created": _NOW,
        "updated": _NOW,
    }
    defaults.update(overrides)
    return Model(**defaults)


def make_chat_session(**overrides: Any) -> ChatSession:
    defaults: Dict[str, Any] = {
        "id": "chat_session:test1",
        "title": "Test Session",
        "created": _NOW,
        "updated": _NOW,
    }
    defaults.update(overrides)
    return ChatSession(**defaults)


def make_chat_message(**overrides: Any) -> ChatMessage:
    defaults: Dict[str, Any] = {
        "id": "chat_message:test1",
        "session": "chat_session:test1",
        "role": "user",
        "content": "Hello!",
        "created": _NOW,
        "updated": _NOW,
    }
    defaults.update(overrides)
    return ChatMessage(**defaults)


def make_transformation(**overrides: Any) -> Transformation:
    defaults: Dict[str, Any] = {
        "id": "transformation:test1",
        "name": "summarize",
        "title": "Summarize",
        "description": "Summarize the text",
        "prompt": "Please summarize: {text}",
        "apply_default": False,
        "created": _NOW,
        "updated": _NOW,
    }
    defaults.update(overrides)
    return Transformation(**defaults)


def make_settings(**overrides: Any) -> ContentSettings:
    return ContentSettings(**overrides)


def make_default_models(**overrides: Any) -> DefaultModels:
    return DefaultModels(**overrides)


def make_default_prompts(**overrides: Any) -> DefaultPrompts:
    defaults: Dict[str, Any] = {
        "transformation_instructions": "Default instructions",
    }
    defaults.update(overrides)
    return DefaultPrompts(**defaults)


def make_insight(**overrides: Any) -> SourceInsight:
    defaults: Dict[str, Any] = {
        "id": "source_insight:test1",
        "source": "source:test1",
        "insight_type": "summary",
        "content": "This is an insight.",
        "created": _NOW,
        "updated": _NOW,
    }
    defaults.update(overrides)
    return SourceInsight(**defaults)


def make_episode(**overrides: Any) -> PodcastEpisode:
    defaults: Dict[str, Any] = {
        "id": "podcast_episode:test1",
        "name": "Test Episode",
        "episode_profile": {"name": "default"},
        "speaker_profile": {"name": "default"},
        "briefing": "Test briefing",
        "content": "Episode content",
        "created": _NOW,
        "updated": _NOW,
    }
    defaults.update(overrides)
    return PodcastEpisode(**defaults)


def make_episode_profile(**overrides: Any) -> EpisodeProfile:
    defaults: Dict[str, Any] = {
        "id": "episode_profile:test1",
        "name": "Test Profile",
        "speaker_config": "default",
        "outline_provider": "openai",
        "outline_model": "gpt-4o",
        "transcript_provider": "openai",
        "transcript_model": "gpt-4o",
        "default_briefing": "Test briefing template",
        "num_segments": 5,
        "created": _NOW,
        "updated": _NOW,
    }
    defaults.update(overrides)
    return EpisodeProfile(**defaults)


def make_speaker_profile(**overrides: Any) -> SpeakerProfile:
    defaults: Dict[str, Any] = {
        "id": "speaker_profile:test1",
        "name": "Test Speaker Profile",
        "tts_provider": "openai",
        "tts_model": "tts-1",
        "speakers": [
            {
                "name": "Host",
                "voice_id": "alloy",
                "backstory": "A podcast host",
                "personality": "friendly",
            }
        ],
        "created": _NOW,
        "updated": _NOW,
    }
    defaults.update(overrides)
    return SpeakerProfile(**defaults)


# ---------------------------------------------------------------------------
# Mock repository fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def notebook_repo():
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=None)
    repo.get_all = AsyncMock(return_value=[])
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    repo.delete = AsyncMock(return_value=True)
    repo.get_sources = AsyncMock(return_value=[])
    repo.get_notes = AsyncMock(return_value=[])
    repo.get_chat_sessions = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def source_repo():
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=None)
    repo.get_all = AsyncMock(return_value=[])
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    repo.delete = AsyncMock(return_value=True)
    repo.get_chunks = AsyncMock(return_value=[])
    repo.get_insights = AsyncMock(return_value=[])
    repo.get_embeddings = AsyncMock(return_value=[])
    repo.get_embedding_count = AsyncMock(return_value=0)
    repo.delete_embeddings = AsyncMock(return_value=0)
    repo.add_to_notebook = AsyncMock(return_value=True)
    repo.add_insight = AsyncMock()
    repo.add_embedding = AsyncMock()
    return repo


@pytest.fixture
def note_repo():
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=None)
    repo.get_all = AsyncMock(return_value=[])
    repo.create = AsyncMock()
    repo.create_with_embedding = AsyncMock()
    repo.update = AsyncMock()
    repo.delete = AsyncMock(return_value=True)
    repo.add_to_notebook = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def chat_session_repo():
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=None)
    repo.create = AsyncMock()
    repo.delete = AsyncMock(return_value=True)
    repo.relate_to_notebook = AsyncMock()
    repo.relate_to_source = AsyncMock()
    return repo


@pytest.fixture
def chat_message_repo():
    repo = AsyncMock()
    repo.get_session_messages = AsyncMock(return_value=[])
    repo.create = AsyncMock()
    return repo


@pytest.fixture
def search_repo():
    repo = AsyncMock()
    repo.text_search = AsyncMock(return_value=[])
    repo.hybrid_search = AsyncMock(return_value=[])
    return repo


@pytest.fixture
def model_repo():
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=None)
    repo.get_all = AsyncMock(return_value=[])
    repo.get_by_type = AsyncMock(return_value=[])
    repo.create = AsyncMock()
    repo.delete = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def defaults_repo():
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=make_default_models())
    repo.update = AsyncMock()
    return repo


@pytest.fixture
def transformation_repo():
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=None)
    repo.get_all = AsyncMock(return_value=[])
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    repo.delete = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def default_prompts_repo():
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=make_default_prompts())
    repo.update = AsyncMock()
    return repo


@pytest.fixture
def settings_repo():
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=make_settings())
    repo.update = AsyncMock()
    repo.patch = AsyncMock()
    return repo


@pytest.fixture
def insight_repo():
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=None)
    repo.get_source = AsyncMock(return_value=None)
    repo.delete = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def chunk_repo():
    return AsyncMock()


@pytest.fixture
def embedding_repo():
    return AsyncMock()


@pytest.fixture
def episode_repo():
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=None)
    repo.get_all = AsyncMock(return_value=[])
    repo.create = AsyncMock()
    repo.delete = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def episode_profile_repo():
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=None)
    repo.get_all = AsyncMock(return_value=[])
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    repo.delete = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def speaker_profile_repo():
    repo = AsyncMock()
    repo.get = AsyncMock(return_value=None)
    repo.get_all = AsyncMock(return_value=[])
    repo.create = AsyncMock()
    repo.update = AsyncMock()
    repo.delete = AsyncMock(return_value=True)
    return repo


@pytest.fixture
def model_manager():
    mm = MagicMock()
    mm.get_defaults = MagicMock(return_value=make_default_models())
    mm.set_defaults = MagicMock()
    mm.clear_cache = MagicMock()
    return mm


@pytest.fixture
def git_flows(tmp_path):
    """Een Git-repository met één flow in ``services/langflow/flows`` (F2-04).

    Geeft ``map``, ``head``, ``hash`` (SHA-256 van de flow in HEAD) en een
    ``register`` op die map. ``commit(naam, inhoud)`` legt een flow vast en
    werkt ``head`` bij. De flow is synthetisch.
    """
    import hashlib
    import json
    import subprocess
    from types import SimpleNamespace

    from app_main.services.flow_register import FlowRegister

    root = tmp_path / "repo"
    map_ = root / "services" / "langflow" / "flows"
    map_.mkdir(parents=True)

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), "-c", "user.name=Test", "-c", "user.email=test@example.org",
             *args],
            capture_output=True, text=True, check=True,
        ).stdout.strip()

    git("init", "-q")
    ns = SimpleNamespace(map=map_, git=git)

    def commit(naam: str, inhoud: dict) -> str:
        pad = map_ / f"{naam}.json"
        pad.write_text(json.dumps(inhoud), encoding="utf-8")
        git("add", "-A")
        git("commit", "-q", "-m", f"flow {naam}")
        ns.head = git("rev-parse", "HEAD")
        return hashlib.sha256(pad.read_bytes()).hexdigest()

    ns.commit = commit
    ns.hash = commit("inname_pdf", {"id": "inname-pdf", "name": "Inname PDF",
                                    "data": {"nodes": [], "edges": []}})
    ns.register = FlowRegister(map_)
    return ns
