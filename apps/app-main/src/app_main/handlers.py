"""
Job handler registration.

Registers async handler functions with the global HandlerRegistry.
Import this module at app startup so handlers are available to the worker.

Each handler receives a payload dict and returns a result dict.
The old surreal_commands @command() pattern is replaced by
@registry.register(JobType.X).
"""

import time
from typing import Any, Dict, List, Optional

from loguru import logger
from pydantic import BaseModel

from shared.types.enums import JobType

from app_main.services.command_service import get_registry

registry = get_registry()


# ---------------------------------------------------------------------------
# Payload schemas for validation
# ---------------------------------------------------------------------------


class DocumentParsePayload(BaseModel):
    source_id: str
    content_state: Dict[str, Any]
    notebook_ids: List[str] = []
    processing_overrides: Optional[Dict[str, Any]] = None


class InsightExtractPayload(BaseModel):
    source_id: str
    command_name: str = "run_summaries"
    transformation_ids: Optional[List[str]] = None


class EntityExtractPayload(BaseModel):
    source_id: str
    ontology_name: str = "general"
    extractor_type: str = "llm"
    langextract_model_id: Optional[str] = None
    langextract_model_url: Optional[str] = None
    langextract_temperature: Optional[float] = None
    langextract_use_schema_constraints: Optional[bool] = None
    langextract_fence_output: Optional[bool] = None
    # B.1f back-compat kill-switch. Ops can submit a job with
    # ``multi_schema_enabled=False`` to force the legacy single-schema
    # path if the orchestrator misbehaves in production. Defaults
    # ``True`` so the multi-schema path is the new normal.
    multi_schema_enabled: bool = True


class EmbeddingPayload(BaseModel):
    command_name: str = "embed_single_item"
    source_id: Optional[str] = None
    item_id: Optional[str] = None
    item_type: Optional[str] = None
    mode: str = "existing"
    include_sources: bool = True
    include_notes: bool = True
    include_insights: bool = True


class NoteAutoLinkPayload(BaseModel):
    """Payload for the Track Y.3 background auto-link job.

    Enqueued after a note is embedded. ``k`` / ``min_similarity`` are optional
    so the chained enqueue can omit them and the orchestrator applies the
    conservative Y.2 defaults.
    """

    command_name: str = "auto_link_note"
    note_id: str
    k: Optional[int] = None
    min_similarity: Optional[float] = None


class ExportObsidianPayload(BaseModel):
    """Payload schema for the auto-pipeline ``EXPORT_OBSIDIAN`` job (D.1b).

    ``filter`` is a free-form dict matching
    :class:`shared.models.export.ExportFilter`. We validate it inside
    the handler by constructing ``ExportFilter(**filter)`` so the
    handler still rejects bad knobs at the job-execution boundary
    (Pydantic validation moves from the router to the handler when
    the job is submitted asynchronously).
    """

    notebook_id: str
    filter: Dict[str, Any] = {}


# ---------------------------------------------------------------------------
# DOCUMENT_PARSE — process_source
# ---------------------------------------------------------------------------


@registry.register(JobType.DOCUMENT_PARSE)
async def handle_process_source(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract content from a source, then auto-enqueue vector embedding.

    Track R.0 forward-fix: ``SourceProcessor.process_source`` deliberately only
    extracts + persists chunks; before R.0 nothing wired the (fully working)
    embedding step into ingest, so every freshly ingested source had chunks but
    NO ``source_embedding`` rows until someone hit ``POST /sources/{id}/run-embed``
    by hand. We close that gap HERE, at the job boundary, by enqueuing the
    existing ``embed_source`` job once chunks exist — async + idempotent,
    mirroring exactly how ``run-embed`` and the orchestrator already enqueue.

    Seam choice (see commit message): the handler — not ``SourceProcessor`` — is
    the right place because (a) it keeps the domain orchestrator free of a
    job-queue dependency, (b) every ingest path (async + sync upload, reprocess)
    funnels through this single ``DOCUMENT_PARSE`` handler, and (c) ``embed_source``
    is already idempotent (it deletes existing embeddings first), so a re-run of
    the parse job re-enqueues a safe re-embed.
    """
    validated = DocumentParsePayload(**payload)
    start_time = time.time()

    try:
        from app_main.dependencies import get_source_processor

        logger.info(f"Starting source extraction for source: {validated.source_id}")

        service = get_source_processor()
        result = await service.process_source(
            source_id=validated.source_id,
            content_state=validated.content_state,
            notebook_ids=validated.notebook_ids,
            processing_overrides=validated.processing_overrides,
        )

        # R.0 forward-fix: chain embedding off a successful extraction that
        # produced chunks. Best-effort enqueue — a queue hiccup must not fail
        # the (already-persisted) extraction; the operator can still run
        # ``run-embed`` or the backfill script. No chunks => nothing to embed.
        embed_command_id: Optional[str] = None
        if result.get("chunk_count", 0) > 0:
            try:
                from app_main.services.command_service import CommandService

                embed_command_id = await CommandService.submit_command_job(
                    "open_notebook",
                    "embed_source",
                    {"source_id": result["source_id"]},
                )
                logger.info(
                    f"Auto-enqueued embed_source job {embed_command_id} for "
                    f"source {result['source_id']}"
                )
            except Exception as embed_err:  # noqa: BLE001 — never fail ingest
                logger.error(
                    f"Failed to auto-enqueue embedding for source "
                    f"{result['source_id']} (extraction succeeded; run-embed "
                    f"or the backfill script can recover): {embed_err}"
                )

        processing_time = time.time() - start_time
        logger.info(
            f"Successfully extracted source {validated.source_id} in {processing_time:.2f}s"
        )

        return {
            "success": True,
            "source_id": result["source_id"],
            "chunk_count": result["chunk_count"],
            "embed_command_id": embed_command_id,
            "processing_time": processing_time,
        }

    except Exception as e:
        logger.error(f"Source extraction failed: {e}")
        raise


# ---------------------------------------------------------------------------
# BATCH_PROCESS — generate_podcast
# ---------------------------------------------------------------------------


@registry.register(JobType.BATCH_PROCESS)
async def handle_generate_podcast(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a podcast episode.

    Temporarily disabled — podcast pipeline has not been migrated to the
    workspace yet.  Returns a clean error so callers get a structured
    response rather than an import crash.
    """
    logger.warning(
        "Podcast generation requested but is temporarily disabled "
        "(pending workspace migration)"
    )
    return {
        "success": False,
        "error": "Podcast generation temporarily disabled",
    }


# ---------------------------------------------------------------------------
# INSIGHT_EXTRACT — run_summaries / analyze_data
# ---------------------------------------------------------------------------


@registry.register(JobType.INSIGHT_EXTRACT)
async def handle_insight_extract(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run summarization / insight extraction on a source."""
    validated = InsightExtractPayload(**payload)
    start_time = time.time()

    try:
        from app_main.dependencies import get_source_summarization_orchestrator

        logger.info(f"Starting summaries for source: {validated.source_id}")
        service = get_source_summarization_orchestrator()
        result = await service.run_summaries(
            source_id=validated.source_id,
            transformation_ids=validated.transformation_ids,
        )

        processing_time = time.time() - start_time
        logger.info(
            f"Summaries completed for source {validated.source_id} in {processing_time:.2f}s"
        )
        return {
            "success": True,
            **result,
            "processing_time": processing_time,
        }

    except Exception as e:
        logger.error(f"Insight extraction failed for source {validated.source_id}: {e}")
        raise


# ---------------------------------------------------------------------------
# ENTITY_EXTRACT — run_entities (ontology-guided extraction)
# ---------------------------------------------------------------------------


@registry.register(JobType.ENTITY_EXTRACT)
async def handle_entity_extract(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run ontology-guided entity extraction on a source.

    Phase B.1f: resolves the source's owning notebook before invoking
    extraction so the multi-schema orchestrator can run when the source
    belongs to a notebook with a configured schema. Sources unlinked
    from any notebook (CLI uploads, orphans) fall through to the legacy
    single-schema path because ``notebook_id`` stays ``None``.

    Raising :class:`SchemaReviewPendingError` is the contract for "user
    must review pending extensions before this source can be extracted".
    The exception bubbles to the worker which translates it into the
    ``PAUSED_FOR_REVIEW`` job state (see ``packages/job-queue/worker.py``);
    callers checking job status see this state and surface a 409 in the
    UI.
    """
    validated = EntityExtractPayload(**payload)
    start_time = time.time()

    try:
        from app_main.dependencies import get_entity_extraction_service, get_source_repo
        from app_main.services.entity_extraction_service import (
            SchemaReviewPendingError,
        )

        logger.info(f"Starting entity extraction for source: {validated.source_id}")
        service = get_entity_extraction_service()
        # Collect langextract config overrides from validated payload
        config_overrides = {}
        for key in (
            "langextract_model_id",
            "langextract_model_url",
            "langextract_temperature",
            "langextract_use_schema_constraints",
            "langextract_fence_output",
        ):
            value = getattr(validated, key, None)
            if value is not None:
                config_overrides[key] = value

        # Resolve the owning notebook so the multi-schema orchestrator
        # can load NotebookSchema. ``None`` is fine (legacy path).
        notebook_id: Optional[str] = None
        try:
            notebook_id = await get_source_repo().get_notebook_id(
                validated.source_id
            )
        except Exception as e:
            # Edge resolution is best-effort. If it fails, log and fall
            # through to single-schema rather than aborting the whole
            # extraction.
            logger.warning(
                f"Could not resolve notebook for source "
                f"{validated.source_id}: {e}; defaulting to single-schema"
            )

        result = await service.run_extraction(
            source_id=validated.source_id,
            ontology_name=validated.ontology_name,
            extractor_type=validated.extractor_type,
            config_overrides=config_overrides if config_overrides else None,
            notebook_id=notebook_id,
            multi_schema_enabled=validated.multi_schema_enabled,
        )

        processing_time = time.time() - start_time
        logger.info(
            f"Entity extraction completed for source {validated.source_id} "
            f"in {processing_time:.2f}s"
        )
        return {
            "success": True,
            **result,
            "processing_time": processing_time,
        }

    except SchemaReviewPendingError as e:
        # Reraise so the worker / queue machinery parks the job in
        # PAUSED_FOR_REVIEW (see worker.py B.1f branch). The exception
        # carries the notebook_id / pending_count for downstream UI use.
        logger.warning(
            f"Entity extraction paused for review: notebook={e.notebook_id} "
            f"source={e.source_id} pending={e.pending_count}"
        )
        raise

    except Exception as e:
        logger.error(f"Entity extraction failed for source {validated.source_id}: {e}")
        raise


# ---------------------------------------------------------------------------
# EMBEDDING_GENERATE — embed_single_item + embed_source + rebuild_embeddings
# ---------------------------------------------------------------------------


@registry.register(JobType.EMBEDDING_GENERATE)
async def handle_embedding(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch embedding jobs.

    The payload's `command_name` distinguishes between embed_single_item,
    embed_source, and rebuild_embeddings.
    """
    validated = EmbeddingPayload(**payload)

    if validated.command_name == "rebuild_embeddings":
        return await _handle_rebuild_embeddings(payload)
    elif validated.command_name == "embed_source":
        return await _handle_embed_source(payload)
    else:
        return await _handle_embed_single_item(payload)


async def _handle_embed_source(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Embed all chunks for a source."""
    start_time = time.time()
    source_id = payload["source_id"]

    from app_main.dependencies import get_source_embedding_orchestrator

    logger.info(f"Starting embedding for source: {source_id}")
    service = get_source_embedding_orchestrator()
    result = await service.embed_source(source_id)

    processing_time = time.time() - start_time
    logger.info(
        f"Embedding completed for source {source_id} in {processing_time:.2f}s"
    )
    return {
        "success": True,
        **result,
        "processing_time": processing_time,
    }


async def _handle_embed_single_item(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Embed a single source, note, or insight."""
    start_time = time.time()
    item_id = payload["item_id"]
    item_type = payload["item_type"]

    from app_main.dependencies import get_embedding_service

    service = await get_embedding_service()

    if item_type == "source":
        result = await service.embed_source(item_id)
    elif item_type == "note":
        result = await service.embed_note(item_id)
    elif item_type == "insight":
        result = await service.embed_insight(item_id)
    else:
        raise ValueError(f"Invalid item_type: {item_type}")

    # Track Y.3: chain background auto-link off a successful note embed. A note
    # can only be ranked by similarity once it HAS an embedding, so this is the
    # trigger point — the embed job is the gate, the auto-link job is the
    # follow-up. Best-effort enqueue: a queue hiccup here must not fail the
    # (already-persisted) embed; the on-demand endpoint / a re-run can recover.
    # We only chain when something was actually embedded (a no-content note
    # produces 0 embeddings and has nothing to link).
    auto_link_command_id: Optional[str] = None
    if item_type == "note" and result.embeddings_created > 0:
        try:
            from app_main.services.command_service import CommandService

            auto_link_command_id = await CommandService.submit_command_job(
                "open_notebook",
                "auto_link_note",
                {"note_id": item_id},
            )
            logger.info(
                f"Auto-enqueued auto_link_note job {auto_link_command_id} "
                f"for note {item_id}"
            )
        except Exception as link_err:  # noqa: BLE001 — never fail the embed
            logger.error(
                f"Failed to auto-enqueue auto-link for note {item_id} "
                f"(embedding succeeded; the on-demand endpoint can recover): "
                f"{link_err}"
            )

    processing_time = time.time() - start_time
    return {
        "success": True,
        "item_id": item_id,
        "item_type": item_type,
        "chunks_created": result.embeddings_created,
        "auto_link_command_id": auto_link_command_id,
        "processing_time": processing_time,
    }


async def _handle_rebuild_embeddings(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild embeddings for sources, notes, and/or insights."""
    start_time = time.time()

    from app_main.dependencies import get_embedding_service

    service = await get_embedding_service()

    mode = payload.get("mode", "existing")
    include_sources = payload.get("include_sources", True)
    include_notes = payload.get("include_notes", True)
    include_insights = payload.get("include_insights", True)

    result = await service.rebuild_embeddings(
        include_sources=include_sources,
        include_notes=include_notes,
        include_insights=include_insights,
        mode=mode,
    )

    processing_time = time.time() - start_time
    processed = result.sources_processed + result.notes_processed + result.insights_processed
    total = processed + len(result.errors)

    return {
        "success": True,
        "total_items": total,
        "processed_items": processed,
        "failed_items": len(result.errors),
        "sources_processed": result.sources_processed,
        "notes_processed": result.notes_processed,
        "insights_processed": result.insights_processed,
        "processing_time": processing_time,
    }


# ---------------------------------------------------------------------------
# NOTE_AUTO_LINK — background auto-link, chained after a note is embedded (Y.3)
# ---------------------------------------------------------------------------


@registry.register(JobType.NOTE_AUTO_LINK)
async def handle_note_auto_link(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Background auto-link handler for ``JobType.NOTE_AUTO_LINK`` (Track Y.3).

    Runs the Y.2 orchestrator (``NoteAutoLinkService.auto_link``) for a note
    that already has an embedding (this job is chained off a successful embed,
    see ``_handle_embed_single_item``). It ranks the note's most-related notes
    by cosine and writes idempotent ``related_note`` edges above the
    conservative threshold.

    Isolation: this is a SEPARATE job from the embed. The note and its
    embedding are already persisted by the time this runs, so a failure here
    cannot corrupt the note — it fails this job only (the worker records it),
    and the operator / on-demand endpoint can re-run. The handler raises on a
    hard failure (so the worker marks the job FAILED, not silently swallowed),
    but the ordinary no-embedding / not-found cases are reported via the
    orchestrator's ``status`` and are NOT errors.

    Idempotent: ``auto_link`` clears-before-relates each pair, so re-running
    this job yields the identical edge set (Y.1/Y.2 idempotency).
    """
    start_time = time.time()
    validated = NoteAutoLinkPayload(**payload)

    from app_main.dependencies import get_note_auto_link_service

    service = await get_note_auto_link_service()
    result = await service.auto_link(
        validated.note_id,
        k=validated.k,
        min_similarity=validated.min_similarity,
    )

    processing_time = time.time() - start_time
    logger.info(
        f"Auto-link job for note {validated.note_id} finished "
        f"(status={result.status}, created={result.created}) "
        f"in {processing_time:.2f}s"
    )
    return {
        "success": True,
        **result.to_dict(),
        "processing_time": processing_time,
    }


# ---------------------------------------------------------------------------
# EXPORT_OBSIDIAN — Obsidian vault export (auto-pipeline entry point, D.1b)
# ---------------------------------------------------------------------------


@registry.register(JobType.EXPORT_OBSIDIAN)
async def handle_export_obsidian(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Auto-pipeline handler for ``JobType.EXPORT_OBSIDIAN`` (Phase D.1b).

    Always runs in ``mode="vault_path"`` -- the async job pathway is
    the auto-pipeline write-to-disk surface. The user-initiated zip
    export stays sync on the ``POST /export-obsidian`` route and never
    hits this handler.

    Payload contract::

        {"notebook_id": "notebook:abc", "filter": {ExportFilter.model_dump()}}

    The handler returns a flattened summary of the
    :class:`shared.models.export.ExportReport` so the job result is a
    plain dict (job-queue contract).

    Raises ``VaultPathNotConfigured`` if Settings is unset -- the
    worker logs and parks the job as FAILED; the operator fixes the
    config and resubmits.
    """
    validated = ExportObsidianPayload(**payload)
    start_time = time.time()

    try:
        from shared.models.export import ExportFilter, ObsidianExportRequest

        from app_main.dependencies import get_obsidian_export_service

        logger.info(
            f"Starting Obsidian vault export for notebook: "
            f"{validated.notebook_id}"
        )
        service = get_obsidian_export_service()
        request = ObsidianExportRequest(
            mode="vault_path",
            filter=ExportFilter(**validated.filter),
        )
        artifact = await service.export(validated.notebook_id, request)

        processing_time = time.time() - start_time
        logger.info(
            f"Obsidian vault export completed for notebook "
            f"{validated.notebook_id} in {processing_time:.2f}s"
        )
        report = artifact.report.model_dump(mode="json")
        return {
            "success": True,
            # Flatten report into the result so job consumers can read
            # counts without an extra nested object. ``mode`` is fixed
            # to "vault_path" for this handler.
            "mode": "vault_path",
            **report,
            "processing_time": processing_time,
        }

    except Exception as e:
        logger.error(
            f"Obsidian vault export failed for notebook "
            f"{validated.notebook_id}: {e}"
        )
        raise
