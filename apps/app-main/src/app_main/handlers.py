"""
Job handler registration.

Registers async handler functions with the global HandlerRegistry.
Import this module at app startup so handlers are available to the worker.

Each handler receives a payload dict and returns a result dict.
The old surreal_commands @command() pattern is replaced by
@registry.register(JobType.X).
"""

import time
from typing import Any, Dict

from loguru import logger

from shared.types.enums import JobType

from app_main.services.command_service import get_registry

registry = get_registry()


# ---------------------------------------------------------------------------
# DOCUMENT_PARSE — process_source
# ---------------------------------------------------------------------------


@registry.register(JobType.DOCUMENT_PARSE)
async def handle_process_source(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Process a source through the ingestion pipeline."""
    start_time = time.time()
    source_id = payload["source_id"]

    try:
        from app_main.dependencies import get_source_processing_service

        logger.info(f"Starting source processing for source: {source_id}")

        service = get_source_processing_service()
        result = await service.process_source(
            source_id=source_id,
            content_state=payload["content_state"],
            apply_transformations=bool(payload.get("transformations")),
            embed=payload.get("embed", False),
            notebook_ids=payload.get("notebook_ids", []),
            transformation_ids=payload.get("transformations", []) or None,
        )

        processing_time = time.time() - start_time
        logger.info(
            f"Successfully processed source {source_id} in {processing_time:.2f}s"
        )

        return {
            "success": True,
            "source_id": result["source_id"],
            "embedded_chunks": result["embedded_chunks"],
            "insights_created": result["insights_created"],
            "processing_time": processing_time,
        }

    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Source processing failed: {e}")
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
# EMBEDDING_GENERATE — embed_single_item + rebuild_embeddings
# ---------------------------------------------------------------------------


@registry.register(JobType.EMBEDDING_GENERATE)
async def handle_embedding(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dispatch embedding jobs.

    The payload's `command_name` distinguishes between embed_single_item
    and rebuild_embeddings.
    """
    command_name = payload.get("command_name", "embed_single_item")

    if command_name == "rebuild_embeddings":
        return await _handle_rebuild_embeddings(payload)
    else:
        return await _handle_embed_single_item(payload)


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

    processing_time = time.time() - start_time
    return {
        "success": True,
        "item_id": item_id,
        "item_type": item_type,
        "chunks_created": result.embeddings_created,
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
