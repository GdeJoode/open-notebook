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

# Track PL.4: the auto-chain wiring + the stage-status write live in ONE place —
# the declarative ``SOURCE_PIPELINE`` + the ``advance_source`` driver. The
# handlers below are THIN: they do their stage's work, write their produced
# stage, then call ``advance_source`` to dispatch the next stage. No handler
# holds an ad-hoc next-stage enqueue any more (PL.1–PL.3 had them scattered).
from app_main.services.source_pipeline import (
    advance_source,
)
from app_main.services.source_pipeline import (
    set_processing_stage as _set_processing_stage,
)

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


class ReferenceExtractPayload(BaseModel):
    """Payload for the Track V.5 reference-extraction post-ingest job.

    Enqueued (best-effort, config-gated) after a source is ingested. ``source_id``
    is the TRIGGERING source (carried for job↔source linkage + logging); the pass
    itself is whole-corpus because U.3's ``materialize`` regenerates the entire
    ``cites`` projection. ``resolve_external`` is opt-in and defaults OFF so the
    auto-chained pass stays fully offline.
    """

    command_name: str = "extract_references"
    source_id: Optional[str] = None
    resolve_external: bool = False


class LibrarianAuditPayload(BaseModel):
    """Payload for the Track F.5 opt-in periodic librarian audit job.

    Enqueued per opt-in notebook by ``LibrarianService.enqueue_due``; the consumer
    re-runs the F.1 quality audit for ``notebook_id``.
    """

    command_name: str = "run_librarian_audit"
    notebook_id: str


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

    PL.4: the next-stage enqueue is no longer inline here — after writing the
    ``ingested`` stage the handler calls ``advance_source``, which dispatches the
    EMBED stage per the ``SOURCE_PIPELINE`` definition. The ``chunk_count > 0``
    guard stays at this seam (a zero-chunk parse has nothing to embed, so we do
    not advance).
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

        # PL.2: the parse + chunk persist is done -> stage = ingested. Best-effort
        # status write (never fails the already-persisted extraction).
        await _set_processing_stage(result["source_id"], "ingested")

        # PL.4: advance the pipeline off the now-``ingested`` source -> the EMBED
        # stage (best-effort enqueue lives in ``advance_source``). The chunk-count
        # guard stays here: no chunks => nothing to embed, so do not advance.
        embed_command_id: Optional[str] = None
        if result.get("chunk_count", 0) > 0:
            advance = await advance_source(result["source_id"])
            embed_command_id = advance.enqueued.get("embed")

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
        # PL.2: a hard parse failure -> stage = failed (best-effort; never masks
        # the original error). Resumable: a re-run retries the parse.
        await _set_processing_stage(validated.source_id, "failed")
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

        # PL.2: extraction (incl. best-effort triage) succeeded -> extracted.
        await _set_processing_stage(validated.source_id, "extracted")

        # PL.4: advance the pipeline off the now-``extracted`` source -> the GRAPH
        # stage. ``advance_source`` runs the source-scoped ``mentions`` refresh
        # INLINE (the cheap, no-LLM projection — not worth a job) and advances
        # ``graphed`` -> ``complete``. Best-effort: a refresh failure leaves the
        # stage at ``extracted`` (a re-run / global regenerate recovers). The
        # gate is NOT reached here — a parked extraction raises
        # SchemaReviewPendingError below before this success path.
        await advance_source(validated.source_id)

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
        # PL.2: the schema-review gate parked extraction -> the source stage is
        # awaiting_schema_review (visible to the UI; no entities were written).
        # Set BEFORE the reraise so the stage is persisted regardless of how the
        # worker translates the exception. Then reraise so the worker parks the
        # job in PAUSED_FOR_REVIEW (see worker.py B.1f branch); the exception
        # carries notebook_id / pending_count for downstream UI use.
        await _set_processing_stage(
            validated.source_id, "awaiting_schema_review"
        )
        logger.warning(
            f"Entity extraction paused for review: notebook={e.notebook_id} "
            f"source={e.source_id} pending={e.pending_count}"
        )
        raise

    except Exception as e:
        # PL.2: a hard failure in extraction -> stage = failed (resumable: the
        # operator / a re-run can retry from here). Status write is best-effort
        # and must not mask the original error, so it precedes the reraise.
        await _set_processing_stage(validated.source_id, "failed")
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
    try:
        result = await service.embed_source(source_id)
    except Exception as embed_err:
        # PL.2: a hard embed failure -> stage = failed (best-effort; never masks
        # the original error). Resumable: a re-run / run-embed retries.
        await _set_processing_stage(source_id, "failed")
        logger.error(f"Embedding failed for source {source_id}: {embed_err}")
        raise

    # PL.2: the per-chunk + aggregate embed is done -> stage = embedded.
    await _set_processing_stage(source_id, "embedded")

    # PL.4: advance the pipeline off the now-``embedded`` source. From EMBED the
    # ``SOURCE_PIPELINE`` fans out to TWO stages: the spine successor EXTRACT
    # (``run_entities`` — the KG chain; honours the schema-review gate) and the
    # PARALLEL INSIGHTS branch (``run_summaries``, behind the per-notebook
    # ``auto_insights`` toggle, default ON — does not gate the KG chain or
    # ``processing_stage``). ``advance_source`` applies the PL.3 chunk-count guard
    # (a zero-chunk embed chains neither — nothing to graph / summarize) and the
    # best-effort posture (a queue hiccup / toggle-read error never fails the
    # already-persisted embed). SOURCES ONLY — the note path is
    # _handle_embed_single_item, which chains NOTE_AUTO_LINK, not extraction.
    embedded_chunks = int(result.get("embedded_chunks", 0) or 0)
    advance = await advance_source(source_id, embedded_chunks=embedded_chunks)
    extract_command_id: Optional[str] = advance.enqueued.get("extract")
    insight_command_id: Optional[str] = advance.enqueued.get("insights")

    processing_time = time.time() - start_time
    logger.info(
        f"Embedding completed for source {source_id} in {processing_time:.2f}s"
    )
    return {
        "success": True,
        **result,
        "extract_command_id": extract_command_id,
        "insight_command_id": insight_command_id,
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
    """Background auto-link handler for ``JobType.NOTE_AUTO_LINK`` (Track Y.3 +
    NS.2).

    Runs the orchestrator (``NoteAutoLinkService.auto_link``) for a note that
    already has an embedding (this job is chained off a successful embed, see
    ``_handle_embed_single_item``). Over that one embedding it runs two passes:
    it ranks the note's most-related **notes** by cosine and writes idempotent
    ``related_note`` edges, and ranks the most-related **sources** and writes
    idempotent ``note_about`` edges — both above the conservative threshold. The
    returned ``result.to_dict()`` carries both counter families, so the source
    links carry through this trigger automatically (NS.2).

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
# REFERENCE_EXTRACT — extract_references, chained (gated) after ingest (V.5)
# ---------------------------------------------------------------------------


@registry.register(JobType.REFERENCE_EXTRACT)
async def handle_reference_extract(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Track V.5 handler: extract corpus references and materialize ``cites``.

    Runs the whole-corpus reference pass (``ReferenceExtractionService`` — extract
    each source's references from its persisted chunks, then feed U.3's existing
    ``materialize`` entry point). The pass is a SEPARATE job from ``DOCUMENT_PARSE``
    so a reference/matching failure stays isolated from the (already-persisted)
    source + chunks: this handler raises on a hard failure (the worker records the
    job FAILED), but the ingest that enqueued it has long since succeeded.

    Whole-corpus, not per-source: U.3's ``materialize`` is a clear-before-relate
    regenerator of the entire ``cites`` table, so feeding it one source's refs
    would wipe every other source's edges. The pass therefore (re)assembles the
    full corpus map and materializes once — idempotent (re-run yields the same
    edge set) and non-destructive (canonical ``source`` rows untouched).
    """
    validated = ReferenceExtractPayload(**payload)
    start_time = time.time()

    try:
        from app_main.dependencies import get_reference_extraction_service

        logger.info(
            f"Starting reference extraction pass "
            f"(triggered by source={validated.source_id}, "
            f"resolve_external={validated.resolve_external})"
        )
        service = get_reference_extraction_service(
            resolve_external=validated.resolve_external
        )
        summary = await service.materialize_corpus(
            resolve_external=validated.resolve_external
        )

        processing_time = time.time() - start_time
        logger.info(
            f"Reference extraction pass completed in {processing_time:.2f}s "
            f"(refs={summary.refs_extracted}, "
            f"cites_edges={summary.edges_materialized})"
        )
        return {
            "success": True,
            "sources_scanned": summary.sources_scanned,
            "sources_with_references": summary.sources_with_references,
            "refs_extracted": summary.refs_extracted,
            "edges_materialized": summary.edges_materialized,
            "external_resolved": summary.external_resolved,
            "processing_time": processing_time,
        }


    except Exception as e:
        logger.error(
            f"Reference extraction pass failed "
            f"(triggered by source={validated.source_id}): {e}"
        )
        raise


# ---------------------------------------------------------------------------
# LIBRARIAN_AUDIT — opt-in periodic quality audit per notebook (F.5)
# ---------------------------------------------------------------------------


@registry.register(JobType.LIBRARIAN_AUDIT)
async def handle_librarian_audit(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Track F.5 handler: re-run the F.1 quality audit for one notebook.

    Enqueued per opt-in notebook by ``LibrarianService.enqueue_due``. Runs the
    always-on (LLM-free) checks and writes a fresh ``audit_findings`` snapshot.
    One notebook's audit is isolated from the others (each is its own job), so a
    failure here fails only this notebook's job, not the librarian sweep.
    """
    validated = LibrarianAuditPayload(**payload)
    start_time = time.time()

    from app_main.services.audit.audit_service import AuditService

    logger.info(f"Librarian audit starting for notebook {validated.notebook_id}")
    result = await AuditService().run_all(validated.notebook_id)
    processing_time = time.time() - start_time
    logger.info(
        f"Librarian audit for {validated.notebook_id} complete: "
        f"run={result.run_id} findings={len(result.findings)} "
        f"in {processing_time:.2f}s"
    )
    return {
        "success": True,
        "notebook_id": validated.notebook_id,
        "run_id": result.run_id,
        "findings": len(result.findings),
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


# ---------------------------------------------------------------------------
# EXPORT_OKF — Open Knowledge Format bundle export (large-notebook path, OKF.2)
# ---------------------------------------------------------------------------


class ExportOkfPayload(BaseModel):
    """Payload schema for the deferred ``JobType.EXPORT_OKF`` job (OKF.2).

    ``filter`` is a free-form dict matching
    :class:`shared.models.export.ExportFilter`; it is validated inside the
    handler by constructing ``ExportFilter(**filter)`` so a bad knob is
    rejected at the job-execution boundary.
    """

    notebook_id: str
    filter: Dict[str, Any] = {}


@registry.register(JobType.EXPORT_OKF)
async def handle_export_okf(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Deferred handler for a large-notebook OKF bundle export (Phase OKF.2).

    The user-initiated route runs the export inline for normal notebooks and
    enqueues this job only when the surviving-entity count exceeds the inline
    threshold. The handler rebuilds the bundle and, when a server-side
    ``vault_path`` is configured, persists the zip under
    ``<vault_path>/okf_exports/`` so the operator can retrieve it; the job
    result carries the flattened :class:`ExportReport` (incl. the OKF-D3
    omitted-field ledger) for the pollable job-status surface.
    """
    validated = ExportOkfPayload(**payload)
    start_time = time.time()

    from shared.models.export import ExportFilter, ObsidianExportRequest

    from app_main.dependencies import get_okf_export_service, get_settings_service

    try:
        logger.info(
            f"Starting OKF bundle export for notebook: {validated.notebook_id}"
        )
        service = get_okf_export_service()
        request = ObsidianExportRequest(
            mode="zip", filter=ExportFilter(**validated.filter)
        )
        artifact = await service.export(validated.notebook_id, request)

        bundle_persisted = False
        try:
            settings = await get_settings_service().get()
            vault_path_raw = getattr(settings, "vault_path", None)
            if vault_path_raw and artifact.zip_bytes:
                _persist_okf_zip(
                    vault_path_raw, validated.notebook_id, artifact.zip_bytes
                )
                bundle_persisted = True
            else:
                logger.warning(
                    "OKF export job {nb}: no vault_path configured; the zip is "
                    "not persisted server-side (report-only job result).",
                    nb=validated.notebook_id,
                )
        except Exception as persist_exc:  # noqa: BLE001 - persistence is best-effort
            logger.error(
                "OKF export job {nb}: failed to persist zip: {err}",
                nb=validated.notebook_id,
                err=persist_exc,
            )

        processing_time = time.time() - start_time
        report = artifact.report.model_dump(mode="json")
        return {
            "success": True,
            **report,
            "bundle_persisted": bundle_persisted,
            "processing_time": processing_time,
        }

    except Exception as e:
        logger.error(
            f"OKF bundle export failed for notebook {validated.notebook_id}: {e}"
        )
        raise


def _persist_okf_zip(vault_path_raw: str, notebook_id: str, zip_bytes: bytes) -> str:
    """Atomically write an OKF export zip under ``<vault_path>/okf_exports/``.

    The target path comes from :func:`okf_artifact_path` — the single scheme the
    download endpoint reads back, so the two never drift. Uses the same
    tmp-write-then-``os.replace`` atomicity as the Obsidian vault writer so a
    half-written archive is impossible. Returns the final path. Raises
    ``ValueError`` if ``vault_path`` is not an absolute existing writable dir.
    """
    import os
    from pathlib import Path

    from app_main.services.okf_export_service import okf_artifact_path

    vault_path = Path(vault_path_raw)
    if not os.access(vault_path, os.W_OK):
        raise ValueError(f"vault_path is not writable: {vault_path!r}")

    final_path = okf_artifact_path(vault_path_raw, notebook_id)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = final_path.with_name(f"{final_path.stem}.zip.tmp.{os.getpid()}")
    with open(tmp_path, "wb") as fh:
        fh.write(zip_bytes)
    os.replace(tmp_path, final_path)
    return str(final_path)
