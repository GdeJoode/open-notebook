"""
FastAPI dependency injection for app-main.

Provides repository and service instances via FastAPI's Depends() mechanism.
"""

from functools import lru_cache

from esperanto import EmbeddingModel
from llm_manager import ModelManager, get_model_manager
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories import (
    ChatMessageRepository,
    ChatSessionRepository,
    ChunkRepository,
    ContentSettingsRepository,
    DefaultModelsRepository,
    DefaultPromptsRepository,
    EntityRepository,
    EpisodeProfileRepository,
    ModelRepository,
    ModelRouteRepository,
    NotebookRepository,
    NoteRepository,
    PodcastEpisodeRepository,
    ReferenceEntityRepository,
    SearchRepository,
    SourceEmbeddingRepository,
    SourceInsightRepository,
    SourceRepository,
    SpeakerProfileRepository,
    SummaryRepository,
    TransformationRepository,
)
from surrealdb_service.repositories.notebook_event import (
    NotebookEventRepository,
)
from surrealdb_service.repositories.notebook_schema import (
    NotebookSchemaRepository,
    Pass1ResultRepository,
)

from app_main.services.chat_service import ChatService
from app_main.services.chunking.chunk_mutator import ChunkMutator
from app_main.services.cites_materialization_service import (
    CitesMaterializationService,
)
from app_main.services.context_service import ContextService
from app_main.services.entity_extraction_service import EntityExtractionService
from app_main.services.graph.doc_graph_builder import DocGraphBuilder
from app_main.services.hybrid_retrieval_service import HybridRetrievalService
from app_main.services.insight_service import InsightService
from app_main.services.jsonl_export_service import JsonlExportService
from app_main.services.kg_retrieval_service import KGRetrievalService
from app_main.services.knowledge_graph_service import KnowledgeGraphService
from app_main.services.mentions_projection_service import (
    MentionsProjectionService,
)
from app_main.services.model_service import ModelService
from app_main.services.networkx_export_service import NetworkxExportService
from app_main.services.note_auto_link_service import NoteAutoLinkService
from app_main.services.note_service import NoteService
from app_main.services.notebook_merge_service import NotebookMergeService
from app_main.services.notebook_service import NotebookService
from app_main.services.obsidian_export_service import ObsidianExportService
from app_main.services.ontology_service import OntologyService
from app_main.services.podcast_service import PodcastService
from app_main.services.preprocessing_service import PreprocessingService
from app_main.services.reextract_service import ReextractService
from app_main.services.schema_edit_service import SchemaEditService
from app_main.services.search_service import SearchService
from app_main.services.settings_service import SettingsService
from app_main.services.source_embedding_orchestrator import SourceEmbeddingOrchestrator
from app_main.services.source_extractor import SourceExtractor
from app_main.services.source_processor import SourceProcessor
from app_main.services.source_service import SourceService
from app_main.services.source_summarization_orchestrator import (
    SourceSummarizationOrchestrator,
)
from app_main.services.summarization_service import SummarizationService
from app_main.services.transformation_service import TransformationService

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


def get_chunk_mutator() -> ChunkMutator:
    return ChunkMutator()


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


def get_model_route_repo() -> ModelRouteRepository:
    """Provider for the per-task model_route rows (Track J.1)."""
    return ModelRouteRepository()


def get_route_resolver():
    """Provider for the privacy-aware route resolver (Track J.1).

    Returns a :class:`RouteResolver` wired with the model_route + model repos.
    Constructed per-call (the repos are cheap, stateless handles) — matches the
    repo-factory convention above rather than caching a singleton.
    """
    from app_main.services.model_routing.route_resolver import RouteResolver

    return RouteResolver(
        model_route_repo=get_model_route_repo(),
        model_repo=get_model_repo(),
    )


@lru_cache(maxsize=1)
def get_circuit_breaker_registry():
    """Process-singleton per-provider circuit-breaker registry (Track J.2).

    Cached so every failover execution in this worker shares the same breaker
    state. In-memory / per-process (J-D3 multi-worker caveat documented on
    :class:`CircuitBreakerRegistry`).
    """
    from app_main.services.model_routing.circuit_breaker import (
        CircuitBreakerRegistry,
    )

    return CircuitBreakerRegistry()


@lru_cache(maxsize=1)
def get_rate_limiter_registry():
    """Process-singleton per-provider fair-use rate limiter (Track J.2).

    Conservative cloud default (~20 rpm) so the NIM endpoint is not overloaded;
    local providers are effectively unthrottled. Cached so the sliding windows
    are shared across all routed calls in this worker.
    """
    from app_main.services.model_routing.rate_limiter import (
        ProviderRateLimiter,
        resolve_per_provider_rpm,
    )

    # Track M.2: per-PROVIDER request-rate caps (the per-MODEL context window is
    # a separate axis handled by the context packer). Caps pace each cloud
    # provider under its quota so batch extraction does not burst into 429 /
    # RESOURCE_EXHAUSTED — the limiter blocks/waits for a slot, and combined with
    # ``default_is_rate_limit`` a transient 429 backoff-retries on the SAME
    # provider before failing over (pace, not instant-skip; Decision M-D1).
    #
    #   google ~ 6/min   (Gemini free tier ~10; stay conservatively under it)
    #   nvidia ~ 30/min  (NIM fair-use ~40; headroom under the ceiling)
    #   ollama  unlimited (local GPU has no fair-use concern)
    #
    # Env-overridable (Decision M-D2: DI/env map, no registry schema change):
    # GOOGLE_RPM / NVIDIA_RPM / OLLAMA_RPM.
    return ProviderRateLimiter(per_provider_rpm=resolve_per_provider_rpm())


def get_failover_executor():
    """Provider for the per-document failover executor (Track J.2).

    Wired with the process-singleton circuit-breaker + rate-limiter registries.
    Constructed per-call (cheap; holds references to the shared singletons) with
    the default failover-eligibility whitelist; J.4 supplies the concrete
    error-mapping predicates.
    """
    from app_main.services.model_routing.failover_executor import FailoverExecutor

    return FailoverExecutor(
        circuit_breakers=get_circuit_breaker_registry(),
        rate_limiter=get_rate_limiter_registry(),
    )


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


def get_entity_repo() -> EntityRepository:
    return EntityRepository()


def get_reference_entity_repo() -> ReferenceEntityRepository:
    return ReferenceEntityRepository()


# --- Track Q.4: triage pipeline + queue ------------------------------------


def get_triage_queue_repo():
    """Provider for the Q.4 triage review-queue repository."""
    from app_main.services.triage.triage_queue import TriageQueueRepository

    return TriageQueueRepository()


def get_status_change_log_repo():
    """Provider for the Q.3 status-change audit log repository."""
    from app_main.services.triage.status_change_log import (
        StatusChangeLogRepository,
    )

    return StatusChangeLogRepository()


def get_triage_pipeline():
    """Construct the Q.4 :class:`TriagePipeline` with its FROZEN collaborators.

    Wires the reuse-first orchestrator: K.5 candidate dedup (B1 match conflicts),
    K.3 recanonicalization (B2 collision merges), the Q.2 signals service, the Q.3
    status-assignment service, the Q.4 queue repo, and the Q.4 backbone check —
    all sharing ONE ``EntityRepository`` so the batch loaders hit the same config.
    Imports are local so a process that never extracts (e.g. a CLI tool) does not
    pay the triage import cost.
    """
    from app_main.services.entity_resolution import (
        CandidateDedupService,
        RecanonicalizationService,
    )
    from app_main.services.triage.backbone_check import BackboneCheckService
    from app_main.services.triage.signals_service import TriageSignalsService
    from app_main.services.triage.status_assignment_service import (
        StatusAssignmentService,
    )
    from app_main.services.triage.triage_config import load_triage_config
    from app_main.services.triage.triage_pipeline import TriagePipeline

    config = load_triage_config()
    entity_repo = EntityRepository()
    return TriagePipeline(
        entity_repository=entity_repo,
        signals_service=TriageSignalsService(entity_repo, config=config),
        status_service=StatusAssignmentService(config, entity_repo),
        queue_repository=get_triage_queue_repo(),
        backbone_service=BackboneCheckService(entity_repo, config=config),
        candidate_service=CandidateDedupService(entity_repo=entity_repo),
        recanonicalization_service=RecanonicalizationService(),
        config=config,
    )


def get_summary_repo() -> SummaryRepository:
    return SummaryRepository()


def get_notebook_schema_repo() -> NotebookSchemaRepository:
    """FastAPI provider for the per-notebook schema row.

    Lifted from ``api/routers/schemas.py`` (B.3a attempt-2) once a second
    consumer appeared: B.3b's edit-ops endpoints + B.3c's soft-nudge
    toggle both need the same repo, so centralising the factory here
    avoids drift between routers.
    """
    return NotebookSchemaRepository()


def get_pass1_result_repo() -> Pass1ResultRepository:
    """FastAPI provider for the pass-1 results table.

    Lifted from ``api/routers/schemas.py`` for the same reason as
    ``get_notebook_schema_repo`` — the orchestrator side
    (``entity_extraction_service``) already instantiates the repo
    directly, and the B.3a router was the second non-orchestrator
    consumer.
    """
    return Pass1ResultRepository()


def get_notebook_event_repo() -> NotebookEventRepository:
    """FastAPI provider for the notebook_event stream (Phase B.3b).

    Introduced alongside the schema edit-ops layer — B.3b's
    ``SchemaEditService`` emits one ``schema_changed`` row per op via
    this repo. B.3c (soft-nudge banner) and B.3d (re-extract prompt)
    consume the same stream, and Track G5's webhook fan-out joins the
    party later, so the factory lives here rather than inline.
    """
    return NotebookEventRepository()


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


def get_kg_retrieval_service() -> KGRetrievalService:
    return KGRetrievalService(
        entity_repo=get_entity_repo(),
        source_repo=get_source_repo(),
    )


def get_hybrid_retrieval_service() -> HybridRetrievalService:
    return HybridRetrievalService(
        source_service=get_source_service(),
        kg_service=get_kg_retrieval_service(),
    )


def get_context_service() -> ContextService:
    return ContextService(
        source_repo=get_source_repo(),
        insight_repo=get_insight_repo(),
        notebook_repo=get_notebook_repo(),
        note_repo=get_note_repo(),
        chunk_repo=get_chunk_repo(),
    )


def get_source_extractor() -> SourceExtractor:
    return SourceExtractor()


def get_doc_graph_builder() -> DocGraphBuilder:
    return DocGraphBuilder()


def get_source_processor() -> SourceProcessor:
    return SourceProcessor(
        source_repo=get_source_repo(),
        chunk_repo=get_chunk_repo(),
        settings_repo=get_settings_repo(),
        extractor=get_source_extractor(),
        graph_builder=get_doc_graph_builder(),
    )


def get_source_embedding_orchestrator() -> SourceEmbeddingOrchestrator:
    return SourceEmbeddingOrchestrator(source_repo=get_source_repo())


def get_source_summarization_orchestrator() -> SourceSummarizationOrchestrator:
    return SourceSummarizationOrchestrator(
        source_repo=get_source_repo(),
        transformation_repo=get_transformation_repo(),
    )


def get_note_service() -> NoteService:
    return NoteService(
        note_repo=get_note_repo(),
        notebook_repo=get_notebook_repo(),
    )


async def get_note_auto_link_service() -> NoteAutoLinkService:
    """Build the Y.2 auto-link orchestrator.

    Async because it resolves the embedding model from the DB via
    ``get_embedding_service`` (the same path the embedding handlers use). The
    embed step keeps this in app-main; surrealdb-service stays embedding-free.
    """
    embedding_service = await get_embedding_service()
    return NoteAutoLinkService(
        note_repo=get_note_repo(),
        embedding_service=embedding_service,
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


def get_ontology_service() -> OntologyService:
    return OntologyService()


def get_knowledge_graph_service() -> KnowledgeGraphService:
    return KnowledgeGraphService(
        entity_repo=get_entity_repo(),
    )


def get_mentions_projection_service() -> MentionsProjectionService:
    return MentionsProjectionService(
        entity_repo=get_entity_repo(),
        source_repo=get_source_repo(),
    )


def get_cites_materialization_service() -> CitesMaterializationService:
    return CitesMaterializationService(
        source_repo=get_source_repo(),
    )


def get_summarization_service() -> SummarizationService:
    return SummarizationService(
        summary_repo=get_summary_repo(),
    )


def get_entity_extraction_service() -> EntityExtractionService:
    return EntityExtractionService(source_repo=get_source_repo())


def get_schema_edit_service() -> SchemaEditService:
    """FastAPI provider for the B.3b schema edit-ops service.

    Wires the notebook_schema + notebook_event repos. The router under
    ``api/routers/schemas.py`` is the sole HTTP consumer; tests
    instantiate ``SchemaEditService`` directly with AsyncMock repos.
    """
    return SchemaEditService(
        schema_repo=get_notebook_schema_repo(),
        event_repo=get_notebook_event_repo(),
    )


def get_reextract_service() -> ReextractService:
    """FastAPI provider for the B.3d schema-change re-extract orchestrator.

    Wires the global ``CommandService`` singleton (used by every other
    job-submitting endpoint) and the ``JobRepository`` it owns. The
    repo is read for dedup so an idempotent re-call (user double-
    clicks ``[Re-extract all]``) does not stack duplicate jobs.

    ``CommandService`` is exposed as a class with ``@staticmethod``
    members; the protocol the service uses is structural, so we wrap
    it in a lightweight adapter to expose the staticmethod via an
    instance — keeps tests free of the import-time singleton.
    """
    from app_main.services.command_service import CommandService, _repository

    class _CommandServiceAdapter:
        async def submit_command_job(
            self,
            module_name: str,
            command_name: str,
            command_args: dict,
            context: dict | None = None,
        ) -> str:
            return await CommandService.submit_command_job(
                module_name, command_name, command_args, context
            )

    return ReextractService(
        command_service=_CommandServiceAdapter(),
        job_repo=_repository,
    )


def get_notebook_merge_service() -> NotebookMergeService:
    """FastAPI provider for the B.6 cross-notebook merge service.

    Wires the shared ``EntityRepository`` so entity writes hit the
    canonical migration-39+44 schema via ``upsert_entity``. Relation
    writes are issued directly through ``execute_query`` from within
    the service (no dedicated relation repo today — see B.6 service
    docstring on relation-write rationale).
    """
    return NotebookMergeService(entity_repository=get_entity_repo())


def get_networkx_export_service() -> NetworkxExportService:
    """FastAPI provider for the D.3 NetworkX export service.

    Both ``list_entities_for_notebook`` and ``list_relations_for_notebook``
    live on ``EntityRepository`` (see D.0), so we wire the same repo
    instance into both slots. The split keeps the service's constructor
    forward-compatible with a future relation-repo extraction.
    """
    entity_repo = get_entity_repo()
    return NetworkxExportService(
        entity_repository=entity_repo,
        relation_repository=entity_repo,
        # U.5 document layer: the source repo answers the notebook→source
        # scoping + cites/title seams. Off unless the filter opts in.
        source_repository=get_source_repo(),
    )


def get_jsonl_export_service() -> JsonlExportService:
    """FastAPI provider for the D.2 JSONL streaming export service.

    Both ``list_entities_for_notebook`` and ``list_relations_for_notebook``
    live on ``EntityRepository`` (see D.0), so we wire the same repo into
    both slots — mirrors :func:`get_networkx_export_service` and
    :func:`get_obsidian_export_service`. No settings dep because the JSONL
    surface is a single zip download, never a filesystem-write target.
    """
    entity_repo = get_entity_repo()
    return JsonlExportService(
        entity_repository=entity_repo,
        relation_repository=entity_repo,
        # U.5 document layer: off unless the filter opts in.
        source_repository=get_source_repo(),
    )


def get_obsidian_export_service() -> ObsidianExportService:
    """FastAPI provider for the D.1a Obsidian vault export service.

    Mirrors :func:`get_networkx_export_service` -- the entity repo
    answers both ``list_entities_for_notebook`` and
    ``list_relations_for_notebook`` per D.0. The settings service is
    wired in now so D.1b (direct-write-to-vault mode) can read the
    operator's ``vault_path`` config without another DI churn.
    """
    entity_repo = get_entity_repo()
    return ObsidianExportService(
        entity_repository=entity_repo,
        relation_repository=entity_repo,
        settings_service=get_settings_service(),
    )


def get_preprocessing_service() -> PreprocessingService:
    return PreprocessingService(
        chunk_repo=get_chunk_repo(),
        language_model=None,  # resolved at call-time; see run endpoint
    )


async def get_preprocessing_service_with_llm() -> PreprocessingService:
    """Create a PreprocessingService with the default transformation model.

    Similar to ``get_embedding_service`` — resolves the configured language
    model from the database via ModelManager.
    """
    from esperanto import LanguageModel as _LM
    from fastapi import HTTPException

    mm = get_model_manager()

    defaults = mm.get_defaults()
    if not defaults or not defaults.default_transformation_model:
        defaults_repo = get_default_models_repo()
        defaults = await defaults_repo.get()
        mm.set_defaults(defaults)

    model_id = getattr(defaults, "default_transformation_model", None)
    if not model_id:
        # Fallback: try the chat model
        model_id = getattr(defaults, "default_chat_model", None)
    if not model_id:
        raise HTTPException(
            status_code=422,
            detail="No transformation or chat model configured. "
            "Please set a default model in Settings → Models.",
        )

    model_repo = get_model_repo()
    model = await model_repo.get(model_id)
    if not model:
        raise HTTPException(
            status_code=422,
            detail=f"Model '{model_id}' not found in database.",
        )

    language_model = mm.get_model_from_config(model)
    if not isinstance(language_model, _LM):
        raise HTTPException(
            status_code=500,
            detail=f"Model '{model_id}' is not a LanguageModel.",
        )

    # Read classification_max_chars from settings
    from app_main.services.preprocessing_service import DEFAULT_MAX_INPUT_CHARS

    settings_repo = get_settings_repo()
    settings = await settings_repo.get()
    max_chars = getattr(settings, "classification_max_chars", None) or DEFAULT_MAX_INPUT_CHARS

    return PreprocessingService(
        chunk_repo=get_chunk_repo(),
        language_model=language_model,
        max_input_chars=max_chars,
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
