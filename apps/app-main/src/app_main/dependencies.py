"""
FastAPI dependency injection for app-main.

Provides repository and service instances via FastAPI's Depends() mechanism.
"""

from functools import lru_cache

from esperanto import EmbeddingModel
from llm_manager import ModelManager, get_model_manager
from surrealdb_service.repositories import (
    ChatMessageRepository,
    ChatSessionRepository,
    ChunkRepository,
    ContentSettingsRepository,
    DefaultModelsRepository,
    DefaultPromptsRepository,
    EpisodeProfileRepository,
    ModelRepository,
    NoteRepository,
    NotebookRepository,
    PodcastEpisodeRepository,
    SearchRepository,
    SourceEmbeddingRepository,
    SourceInsightRepository,
    SourceRepository,
    SpeakerProfileRepository,
    TransformationRepository,
)
from surrealdb_service.connection import execute_query

from app_main.services.notebook_service import NotebookService
from app_main.services.source_service import SourceService
from app_main.services.note_service import NoteService
from app_main.services.chat_service import ChatService
from app_main.services.search_service import SearchService
from app_main.services.model_service import ModelService
from app_main.services.transformation_service import TransformationService
from app_main.services.podcast_service import PodcastService
from app_main.services.settings_service import SettingsService
from app_main.services.insight_service import InsightService


# --- Repository providers ---

def get_notebook_repo() -> NotebookRepository:
    return NotebookRepository()


def get_source_repo() -> SourceRepository:
    return SourceRepository()


def get_note_repo() -> NoteRepository:
    return NoteRepository()


def get_chat_session_repo() -> ChatSessionRepository:
    return ChatSessionRepository()


def get_chat_message_repo() -> ChatMessageRepository:
    return ChatMessageRepository()


def get_chunk_repo() -> ChunkRepository:
    return ChunkRepository()


def get_insight_repo() -> SourceInsightRepository:
    return SourceInsightRepository()


def get_embedding_repo() -> SourceEmbeddingRepository:
    return SourceEmbeddingRepository()


def get_search_repo() -> SearchRepository:
    return SearchRepository()


def get_model_repo() -> ModelRepository:
    return ModelRepository()


def get_default_models_repo() -> DefaultModelsRepository:
    return DefaultModelsRepository()


def get_transformation_repo() -> TransformationRepository:
    return TransformationRepository()


def get_default_prompts_repo() -> DefaultPromptsRepository:
    return DefaultPromptsRepository()


def get_settings_repo() -> ContentSettingsRepository:
    return ContentSettingsRepository()


def get_speaker_profile_repo() -> SpeakerProfileRepository:
    return SpeakerProfileRepository()


def get_episode_profile_repo() -> EpisodeProfileRepository:
    return EpisodeProfileRepository()


def get_podcast_episode_repo() -> PodcastEpisodeRepository:
    return PodcastEpisodeRepository()


# --- Service providers ---

def get_notebook_service() -> NotebookService:
    return NotebookService(
        notebook_repo=get_notebook_repo(),
        source_repo=get_source_repo(),
        note_repo=get_note_repo(),
    )


def get_source_service() -> SourceService:
    return SourceService(
        source_repo=get_source_repo(),
        chunk_repo=get_chunk_repo(),
        insight_repo=get_insight_repo(),
        embedding_repo=get_embedding_repo(),
    )


def get_note_service() -> NoteService:
    return NoteService(
        note_repo=get_note_repo(),
        notebook_repo=get_notebook_repo(),
    )


def get_chat_service() -> ChatService:
    return ChatService(
        session_repo=get_chat_session_repo(),
        message_repo=get_chat_message_repo(),
    )


def get_search_service() -> SearchService:
    return SearchService(
        search_repo=get_search_repo(),
        model_manager=get_model_manager(),
    )


def get_model_service() -> ModelService:
    return ModelService(
        model_repo=get_model_repo(),
        defaults_repo=get_default_models_repo(),
        model_manager=get_model_manager(),
    )


def get_transformation_service() -> TransformationService:
    return TransformationService(
        transformation_repo=get_transformation_repo(),
        default_prompts_repo=get_default_prompts_repo(),
    )


def get_podcast_service() -> PodcastService:
    return PodcastService(
        episode_repo=get_podcast_episode_repo(),
        episode_profile_repo=get_episode_profile_repo(),
        speaker_profile_repo=get_speaker_profile_repo(),
    )


def get_settings_service() -> SettingsService:
    return SettingsService(
        settings_repo=get_settings_repo(),
    )


def get_insight_service() -> InsightService:
    return InsightService(
        insight_repo=get_insight_repo(),
        note_repo=get_note_repo(),
    )


# --- Embedding service factory (for handler / non-DI use) ---

async def get_embedding_service():
    """Create an EmbeddingService instance for handler use.

    Resolves the default embedding model from the DB via ModelManager,
    then creates the service with a SourceRepository.
    """
    from embeddings.service import EmbeddingService

    mm = get_model_manager()

    # Ensure defaults are loaded
    defaults = mm.get_defaults()
    if not defaults or not defaults.default_embedding_model:
        defaults_repo = get_default_models_repo()
        defaults = await defaults_repo.get()
        mm.set_defaults(defaults)

    model_id = defaults.default_embedding_model
    if not model_id:
        raise ValueError("No embedding model configured.")

    model_repo = get_model_repo()
    model = await model_repo.get(model_id)
    if not model:
        raise ValueError(f"Embedding model '{model_id}' not found in database.")

    embedding_model = mm.get_model_from_config(model)
    if not isinstance(embedding_model, EmbeddingModel):
        raise TypeError(
            f"Model '{model_id}' is not an EmbeddingModel, got {type(embedding_model)}"
        )

    source_repo = get_source_repo()
    return EmbeddingService(
        source_repo=source_repo,
        embedding_model=embedding_model,
    )
