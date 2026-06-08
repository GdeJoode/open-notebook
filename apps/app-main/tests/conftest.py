"""
Shared test fixtures for app-main.

All repositories are AsyncMock — no database required.
"""

import os
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


_DISABLE_METRICS_KEY = "OPEN_NOTEBOOK_DISABLE_METRICS"


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
