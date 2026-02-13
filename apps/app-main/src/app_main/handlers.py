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
        from open_notebook.domain.notebook import Source
        from open_notebook.domain.transformation import Transformation

        logger.info(f"Starting source processing for source: {source_id}")

        content_state = payload["content_state"]
        notebook_ids = payload.get("notebook_ids", [])
        transformation_ids = payload.get("transformations", [])
        embed = payload.get("embed", False)

        # Load transformations
        transformations = []
        for trans_id in transformation_ids:
            transformation = await Transformation.get(trans_id)
            if not transformation:
                raise ValueError(f"Transformation '{trans_id}' not found")
            transformations.append(transformation)

        # Verify source exists
        source = await Source.get(source_id)
        if not source:
            raise ValueError(f"Source '{source_id}' not found")

        # Import and run source graph
        from open_notebook.graphs.source import source_graph

        result = await source_graph.ainvoke(
            {
                "content_state": content_state,
                "notebook_ids": notebook_ids,
                "apply_transformations": transformations,
                "embed": embed,
                "source_id": source_id,
            }
        )

        processed_source = result["source"]
        embedded_chunks = (
            await processed_source.get_embedded_chunks() if embed else 0
        )
        insights_list = await processed_source.get_insights()

        processing_time = time.time() - start_time
        logger.info(
            f"Successfully processed source {source_id} in {processing_time:.2f}s"
        )

        return {
            "success": True,
            "source_id": str(processed_source.id),
            "embedded_chunks": embedded_chunks,
            "insights_created": len(insights_list),
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
    """Generate a podcast episode."""
    start_time = time.time()

    try:
        from pathlib import Path

        from pydantic import BaseModel

        from open_notebook.config import DATA_FOLDER
        from open_notebook.database.repository import ensure_record_id, repo_query
        from open_notebook.domain.podcast import (
            EpisodeProfile,
            PodcastEpisode,
            SpeakerProfile,
        )

        def full_model_dump(model):
            if isinstance(model, BaseModel):
                return model.model_dump()
            elif isinstance(model, dict):
                return {k: full_model_dump(v) for k, v in model.items()}
            elif isinstance(model, list):
                return [full_model_dump(item) for item in model]
            return model

        episode_name = payload["episode_name"]
        logger.info(f"Starting podcast generation for episode: {episode_name}")

        # Load profiles
        episode_profile = await EpisodeProfile.get_by_name(payload["episode_profile"])
        if not episode_profile:
            raise ValueError(f"Episode profile '{payload['episode_profile']}' not found")

        speaker_profile = await SpeakerProfile.get_by_name(
            episode_profile.speaker_config
        )
        if not speaker_profile:
            raise ValueError(
                f"Speaker profile '{episode_profile.speaker_config}' not found"
            )

        # Load all profiles for podcast-creator config
        episode_profiles = await repo_query("SELECT * FROM episode_profile")
        speaker_profiles = await repo_query("SELECT * FROM speaker_profile")
        episode_profiles_dict = {p["name"]: p for p in episode_profiles}
        speaker_profiles_dict = {p["name"]: p for p in speaker_profiles}

        # Generate briefing
        briefing = episode_profile.default_briefing
        if payload.get("briefing_suffix"):
            briefing += f"\n\nAdditional instructions: {payload['briefing_suffix']}"

        # Create episode record
        episode = PodcastEpisode(
            name=episode_name,
            episode_profile=full_model_dump(episode_profile.model_dump()),
            speaker_profile=full_model_dump(speaker_profile.model_dump()),
            briefing=briefing,
            content=payload["content"],
            audio_file=None,
            transcript=None,
            outline=None,
        )
        await episode.save()

        # Configure and run podcast-creator
        from podcast_creator import configure, create_podcast

        configure("speakers_config", {"profiles": speaker_profiles_dict})
        configure("episode_config", {"profiles": episode_profiles_dict})

        output_dir = Path(f"{DATA_FOLDER}/podcasts/episodes/{episode_name}")
        output_dir.mkdir(parents=True, exist_ok=True)

        result = await create_podcast(
            content=payload["content"],
            briefing=briefing,
            episode_name=episode_name,
            output_dir=str(output_dir),
            speaker_config=speaker_profile.name,
            episode_profile=episode_profile.name,
        )

        episode.audio_file = (
            str(result.get("final_output_file_path")) if result else None
        )
        episode.transcript = {
            "transcript": full_model_dump(result["transcript"]) if result else None
        }
        episode.outline = full_model_dump(result["outline"]) if result else None
        await episode.save()

        processing_time = time.time() - start_time
        logger.info(f"Generated podcast {episode.id} in {processing_time:.2f}s")

        return {
            "success": True,
            "episode_id": str(episode.id),
            "audio_file_path": str(result.get("final_output_file_path")) if result else None,
            "processing_time": processing_time,
        }

    except Exception as e:
        logger.error(f"Podcast generation failed: {e}")
        raise


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

    try:
        from open_notebook.database.repository import ensure_record_id, repo_query
        from open_notebook.domain.models import model_manager
        from open_notebook.domain.notebook import Note, Source, SourceInsight

        EMBEDDING_MODEL = await model_manager.get_embedding_model()
        if not EMBEDDING_MODEL:
            raise ValueError("No embedding model configured.")

        chunks_created = 0

        if item_type == "source":
            source = await Source.get(item_id)
            if not source:
                raise ValueError(f"Source '{item_id}' not found")
            await source.vectorize()

            chunks_result = await repo_query(
                "SELECT VALUE count() FROM source_embedding WHERE source = $source_id GROUP ALL",
                {"source_id": ensure_record_id(item_id)},
            )
            if chunks_result and isinstance(chunks_result[0], dict):
                chunks_created = chunks_result[0].get("count", 0)
            elif chunks_result and isinstance(chunks_result[0], int):
                chunks_created = chunks_result[0]

        elif item_type == "note":
            note = await Note.get(item_id)
            if not note:
                raise ValueError(f"Note '{item_id}' not found")
            await note.save()

        elif item_type == "insight":
            insight = await SourceInsight.get(item_id)
            if not insight:
                raise ValueError(f"Insight '{item_id}' not found")
            embedding = (await EMBEDDING_MODEL.aembed([insight.content]))[0]
            await repo_query(
                "UPDATE $insight_id SET embedding = $embedding",
                {"insight_id": ensure_record_id(item_id), "embedding": embedding},
            )
        else:
            raise ValueError(f"Invalid item_type: {item_type}")

        processing_time = time.time() - start_time
        return {
            "success": True,
            "item_id": item_id,
            "item_type": item_type,
            "chunks_created": chunks_created,
            "processing_time": processing_time,
        }

    except Exception as e:
        logger.error(f"Embedding failed for {item_type} {item_id}: {e}")
        raise


async def _handle_rebuild_embeddings(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild embeddings for sources, notes, and/or insights."""
    start_time = time.time()

    try:
        from open_notebook.database.repository import ensure_record_id, repo_query
        from open_notebook.domain.models import model_manager
        from open_notebook.domain.notebook import Note, Source, SourceInsight

        mode = payload.get("mode", "existing")
        include_sources = payload.get("include_sources", True)
        include_notes = payload.get("include_notes", True)
        include_insights = payload.get("include_insights", True)

        EMBEDDING_MODEL = await model_manager.get_embedding_model()
        if not EMBEDDING_MODEL:
            raise ValueError("No embedding model configured.")

        # Collect items
        items: Dict[str, list] = {"sources": [], "notes": [], "insights": []}

        if include_sources:
            if mode == "existing":
                result = await repo_query(
                    "RETURN array::distinct("
                    "SELECT VALUE source.id FROM source_embedding "
                    "WHERE embedding != none AND array::len(embedding) > 0)"
                )
                items["sources"] = [str(i) for i in result] if result else []
            else:
                result = await repo_query(
                    "SELECT id FROM source WHERE full_text != none"
                )
                items["sources"] = [str(i["id"]) for i in result] if result else []

        if include_notes:
            if mode == "existing":
                result = await repo_query(
                    "SELECT id FROM note WHERE embedding != none AND array::len(embedding) > 0"
                )
            else:
                result = await repo_query(
                    "SELECT id FROM note WHERE content != none"
                )
            items["notes"] = [str(i["id"]) for i in result] if result else []

        if include_insights:
            if mode == "existing":
                result = await repo_query(
                    "SELECT id FROM source_insight WHERE embedding != none AND array::len(embedding) > 0"
                )
            else:
                result = await repo_query("SELECT id FROM source_insight")
            items["insights"] = [str(i["id"]) for i in result] if result else []

        total = sum(len(v) for v in items.values())
        if total == 0:
            return {
                "success": True,
                "total_items": 0,
                "processed_items": 0,
                "failed_items": 0,
                "processing_time": time.time() - start_time,
            }

        sources_ok = notes_ok = insights_ok = failed = 0

        for sid in items["sources"]:
            try:
                source = await Source.get(sid)
                if source:
                    await source.vectorize()
                    sources_ok += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Failed to re-embed source {sid}: {e}")
                failed += 1

        for nid in items["notes"]:
            try:
                note = await Note.get(nid)
                if note:
                    await note.save()
                    notes_ok += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Failed to re-embed note {nid}: {e}")
                failed += 1

        for iid in items["insights"]:
            try:
                insight = await SourceInsight.get(iid)
                if insight:
                    emb = (await EMBEDDING_MODEL.aembed([insight.content]))[0]
                    await repo_query(
                        "UPDATE $iid SET embedding = $embedding",
                        {"iid": ensure_record_id(iid), "embedding": emb},
                    )
                    insights_ok += 1
                else:
                    failed += 1
            except Exception as e:
                logger.error(f"Failed to re-embed insight {iid}: {e}")
                failed += 1

        processing_time = time.time() - start_time
        processed = sources_ok + notes_ok + insights_ok

        logger.info(
            f"Rebuild complete: {processed}/{total} items, "
            f"{failed} failed, {processing_time:.2f}s"
        )

        return {
            "success": True,
            "total_items": total,
            "processed_items": processed,
            "failed_items": failed,
            "sources_processed": sources_ok,
            "notes_processed": notes_ok,
            "insights_processed": insights_ok,
            "processing_time": processing_time,
        }

    except Exception as e:
        logger.error(f"Rebuild embeddings failed: {e}")
        raise
