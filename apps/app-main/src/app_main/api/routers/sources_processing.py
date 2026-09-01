"""Sources processing sub-router — pipeline triggering endpoints."""

from typing import Any, Dict, Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from loguru import logger
from pydantic import BaseModel, Field

from app_main.api.schemas import SourceJudgeContradictionsResponse
from app_main.dependencies import (
    get_contradiction_judge_service,
    get_source_service,
)
from app_main.services.contradiction_judge_service import (
    ContradictionJudgeService,
    MAX_K,
)
from app_main.services.source_service import SourceService
from surrealdb_service.repositories.base import _validate_record_id

router = APIRouter()


def _validate_source_id(source_id: str) -> str:
    """Reject a malformed/unsafe source id at the route boundary (Z.3).

    The judge persists through Z.1's ``relate_verdict``, which strict-validates
    ids before any interpolation. Validating here too means a bad id (a
    ``;``-bearing injection payload, the wrong table, garbage) returns a clean
    422 instead of falling through to the service/DB. Mirrors the Y.2 auto-link
    discipline: validate at the boundary, not only in the repo.
    """
    try:
        validated = _validate_record_id(str(source_id))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid source id: {exc}")
    if not validated.startswith("source:"):
        raise HTTPException(
            status_code=422, detail="id must be a source record (source:...)"
        )
    return validated


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ReprocessRequest(BaseModel):
    """Per-document Docling pipeline configuration for reprocessing.

    All fields default to maximum quality settings. Any field set to None
    uses the global default from ContentSettings.
    """
    # Parser engine override (Phase A.1b, Q-A-6)
    # None means "use global ContentSettings.parser_engine".
    parser_engine: Optional[Literal["simple", "docling", "mineru", "markitdown", "auto"]] = Field(
        None,
        description="Override the global parser engine for this run (simple | docling | mineru | auto)",
    )

    # Docling confidence threshold override (Phase A.1c)
    # Only used when parser_engine resolves to "auto". None means "use
    # ContentSettings.docling_min_confidence" (default 0.95).
    docling_min_confidence: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Override the auto-mode confidence threshold for this run",
    )

    # OCR
    docling_ocr_engine: Optional[str] = Field("easyocr", description="OCR engine: easyocr, rapidocr, tesseract")
    docling_ocr_languages: Optional[list[str]] = Field(["en", "nl"], description="OCR languages")

    # Table extraction
    docling_table_mode: Optional[str] = Field("accurate", description="Table structure mode: accurate or fast")

    # VLM pipeline
    docling_pipeline: Optional[str] = Field("vlm", description="Pipeline: vlm (max) or standard")
    docling_vlm_model: Optional[str] = Field("granite-docling-258m", description="VLM model")

    # Image extraction
    docling_auto_export_images: Optional[bool] = Field(True, description="Export extracted images")
    docling_image_scale: Optional[float] = Field(2.0, ge=1.0, le=4.0, description="Image scale factor (1.0-4.0)")
    docling_generate_page_images: Optional[bool] = Field(False, description="Render a full-page image per page")

    # Enrichment (Phase I.D-2)
    # Default None -> keep the DoclingConfig default (off), preserving pre-I.D-2
    # behavior (AC3). An explicit true/false from the UI overrides it.
    docling_do_code_enrichment: Optional[bool] = Field(None, description="Detect and enrich code blocks")
    docling_do_formula_enrichment: Optional[bool] = Field(None, description="Detect and extract math formulas")
    # None -> follow the VLM toggle (preserves pre-I.D-2 coupling).
    docling_do_picture_classification: Optional[bool] = Field(None, description="Classify pictures independent of the VLM toggle")

    # Chunking
    docling_chunking_enabled: Optional[bool] = Field(True, description="Enable chunking")
    docling_chunking_method: Optional[str] = Field("hybrid", description="Chunking method: hybrid or hierarchical")
    docling_chunking_max_tokens: Optional[int] = Field(512, description="Max tokens per chunk")


class RunEntitiesRequest(BaseModel):
    ontology_name: str = "general"
    extractor_type: str = "llm"  # "llm" or "langextract"
    # LangExtract-specific options (forwarded when extractor_type == "langextract")
    langextract_model_id: Optional[str] = None
    langextract_model_url: Optional[str] = None
    langextract_temperature: Optional[float] = None
    langextract_use_schema_constraints: Optional[bool] = None
    langextract_fence_output: Optional[bool] = None
    # B.1f back-compat kill-switch — defaults true so multi-schema is the
    # new normal; ops sends ``false`` to roll back to the legacy path.
    multi_schema_enabled: bool = True


class RunFilteringRequest(BaseModel):
    dedup_enabled: bool = True
    dedup_similarity_threshold: float = 0.85
    fuzzy_dedup_enabled: bool = False
    fuzzy_similarity_threshold: float = 0.85
    embedding_dedup_enabled: bool = False
    embedding_similarity_threshold: float = 0.90
    edge_prediction_enabled: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/{source_id}/run-summaries")
async def run_summaries(
    source_id: str,
    source_svc: SourceService = Depends(get_source_service),
):
    """Trigger summarization / insight extraction for a source."""
    try:
        source = await source_svc.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        from app_main.services.command_service import CommandService

        command_id = await CommandService.submit_command_job(
            "open_notebook",
            "run_summaries",
            {"source_id": str(source.id)},
        )
        return {"command_id": command_id, "status": "queued"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering summaries for source {source_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error triggering summaries: {str(e)}",
        )


@router.post("/{source_id}/run-entities")
async def run_entities(
    source_id: str,
    body: RunEntitiesRequest = RunEntitiesRequest(),
    source_svc: SourceService = Depends(get_source_service),
):
    """Trigger ontology-guided entity extraction for a source.

    Phase B.1f: pre-checks the owning notebook's schema-review state.
    When ``review_required=True`` and no accepted extensions exist yet,
    returns 409 Conflict (rather than queueing a job that will pause
    immediately). The handler still catches the same condition as a
    defence-in-depth in case the notebook state changes between the
    pre-check and the worker pulling the job.
    """
    try:
        source = await source_svc.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        from app_main.dependencies import get_source_repo
        from app_main.services.command_service import CommandService
        from surrealdb_service.repositories import NotebookSchemaRepository

        # Pre-check: synchronous 409 when the owning notebook has the
        # review-pending toggle set. The handler also enforces this on
        # the worker side; doing it here saves a roundtrip when the UI
        # would just have to wait for the same answer.
        if body.extractor_type == "llm" and body.multi_schema_enabled:
            try:
                notebook_id = await get_source_repo().get_notebook_id(
                    str(source.id)
                )
            except Exception as e:
                logger.warning(
                    f"run-entities: could not resolve notebook for "
                    f"{source.id}: {e}"
                )
                notebook_id = None

            if notebook_id is not None:
                nb_schema_repo = NotebookSchemaRepository()
                nb_schema = await nb_schema_repo.get_by_notebook(notebook_id)
                if (
                    nb_schema is not None
                    and nb_schema.review_required
                    and not any(
                        # N.4d.3: a re-parent moves an EXISTING type and says
                        # nothing about the pending extensions awaiting review,
                        # so it must not lift the gate. Mirrors
                        # `entity_extraction_service._counts_toward_review`.
                        ext.get("op") != "reparent"
                        for ext in nb_schema.accepted_extensions
                    )
                ):
                    raise HTTPException(
                        status_code=409,
                        detail={
                            "code": "schema_review_pending",
                            "message": (
                                "Notebook has pending schema-extension "
                                "review. Resolve pending extensions before "
                                "running entity extraction."
                            ),
                            "notebook_id": notebook_id,
                            "pending_count": len(nb_schema.pending_extensions),
                        },
                    )

        command_payload = {
            "source_id": str(source.id),
            "ontology_name": body.ontology_name,
            "extractor_type": body.extractor_type,
            "multi_schema_enabled": body.multi_schema_enabled,
        }
        # Forward langextract options if provided
        for field_name in (
            "langextract_model_id",
            "langextract_model_url",
            "langextract_temperature",
            "langextract_use_schema_constraints",
            "langextract_fence_output",
        ):
            val = getattr(body, field_name, None)
            if val is not None:
                command_payload[field_name] = val

        command_id = await CommandService.submit_command_job(
            "open_notebook",
            "run_entities",
            command_payload,
        )
        return {"command_id": command_id, "status": "queued"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering entities for source {source_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error triggering entities: {str(e)}",
        )


@router.get("/{source_id}/extraction-result")
async def get_extraction_result(
    source_id: str,
    source_svc: SourceService = Depends(get_source_service),
):
    """Get full extraction result (entities, relations, metadata) for a source."""
    try:
        source = await source_svc.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        from surrealdb_service.connection import execute_query

        rows = await execute_query(
            "SELECT * FROM extraction_result WHERE source_id = $source_id LIMIT 1",
            {"source_id": str(source.id)},
        )
        if not rows:
            return {
                "entities": [],
                "relations": [],
                "metadata": {},
                "entity_count": 0,
                "relation_count": 0,
            }
        row = rows[0]
        return {
            "entities": row.get("entities", []),
            "relations": row.get("relations", []),
            "metadata": row.get("metadata", {}),
            "entity_count": row.get("entity_count", 0),
            "relation_count": row.get("relation_count", 0),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error fetching extraction result for source {source_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching extraction result: {str(e)}",
        )


@router.post("/{source_id}/run-filtering")
async def run_filtering(
    source_id: str,
    body: RunFilteringRequest = RunFilteringRequest(),
    source_svc: SourceService = Depends(get_source_service),
):
    """Run entity filtering/deduplication on existing extraction results.

    Fetches the raw extraction_result, runs the FilteringWorkflow with
    the specified config, persists filtered entities to the KG tables,
    and returns filtering statistics.
    """
    try:
        source = await source_svc.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        from entity_filtering.config import (
            EmbeddingDedupConfig,
            FilteringConfig,
            FuzzyDedupConfig,
        )

        from app_main.dependencies import get_entity_extraction_service

        service = get_entity_extraction_service()

        f_config = FilteringConfig(
            dedup_enabled=body.dedup_enabled,
            dedup_similarity_threshold=body.dedup_similarity_threshold,
            fuzzy_dedup=FuzzyDedupConfig(
                enabled=body.fuzzy_dedup_enabled,
                similarity_threshold=body.fuzzy_similarity_threshold,
            ),
            embedding_dedup=EmbeddingDedupConfig(
                enabled=body.embedding_dedup_enabled,
                similarity_threshold=body.embedding_similarity_threshold,
            ),
            edge_prediction_enabled=body.edge_prediction_enabled,
        )

        result = await service.run_filtering_only(
            source_id=str(source.id),
            filtering_config=f_config,
        )

        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running filtering for source {source_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error running filtering: {str(e)}",
        )


@router.post("/{source_id}/run-embed")
async def run_embed(
    source_id: str,
    source_svc: SourceService = Depends(get_source_service),
):
    """Trigger embedding generation for a source."""
    try:
        source = await source_svc.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        from app_main.services.command_service import CommandService

        command_id = await CommandService.submit_command_job(
            "open_notebook",
            "embed_source",
            {"source_id": str(source.id)},
        )
        return {"command_id": command_id, "status": "queued"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error triggering embedding for source {source_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error triggering embedding: {str(e)}",
        )


@router.post("/{source_id}/reprocess")
async def reprocess_source(
    source_id: str,
    body: ReprocessRequest = ReprocessRequest(),
    source_svc: SourceService = Depends(get_source_service),
):
    """Re-run Docling ingestion with per-document pipeline overrides.

    Accepts configurable options for OCR, table extraction, VLM, images,
    and chunking. All defaults are set to maximum quality.
    """
    try:
        source = await source_svc.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Build content_state from existing source asset
        content_state: Dict[str, Any] = {}
        if source.asset and source.asset.file_path:
            content_state = {
                "file_path": source.asset.file_path,
                "delete_source": False,
            }
        elif source.asset and source.asset.url:
            content_state = {"url": source.asset.url}
        else:
            raise HTTPException(
                status_code=400,
                detail="Source has no file or URL to reprocess",
            )

        # Build processing_overrides from the request body
        overrides = {
            k: v
            for k, v in body.model_dump().items()
            if v is not None
        }

        # Get notebook associations
        from surrealdb_service.connection import execute_query, ensure_record_id

        references = await execute_query(
            "SELECT VALUE out FROM reference WHERE in = $source_id",
            {"source_id": ensure_record_id(source_id)},
        )
        notebook_ids = [str(ref) for ref in references] if references else []

        from app_main.services.command_service import CommandService

        command_args = {
            "source_id": str(source.id),
            "content_state": content_state,
            "notebook_ids": notebook_ids,
            "transformations": [],
            "embed": True,
            "processing_overrides": overrides,
        }

        command_id = await CommandService.submit_command_job(
            "open_notebook",
            "process_source",
            command_args,
        )

        logger.info(
            f"Submitted reprocess command: {command_id} "
            f"for source {source_id} with overrides: {list(overrides.keys())}"
        )

        # Update source with new command ID
        await execute_query(
            "UPDATE $source_id SET command = $command_id",
            {
                "source_id": ensure_record_id(source_id),
                "command_id": ensure_record_id(command_id),
            },
        )

        return {
            "command_id": command_id,
            "status": "queued",
            "overrides_applied": list(overrides.keys()),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reprocessing source {source_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error reprocessing: {str(e)}",
        )


@router.get("/{source_id}/processing-logs")
async def get_processing_logs(
    source_id: str,
    source_svc: SourceService = Depends(get_source_service),
):
    """Get historical processing logs for a source (JSON array)."""
    source = await source_svc.get(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    from app_main.services.log_stream import get_log_stream

    log_stream = get_log_stream()

    # Try in-memory buffer first
    entries = log_stream.get_entries(str(source.id))
    if not entries:
        # Fall back to persisted JSONL file
        entries = log_stream.get_persisted_logs(str(source.id))

    return [entry.to_dict() for entry in entries]


@router.get("/{source_id}/logs")
async def stream_source_logs(
    source_id: str,
    after: int = 0,
    source_svc: SourceService = Depends(get_source_service),
):
    """Stream processing logs for a source via SSE.

    Pass ``?after=N`` to skip the first N buffered entries (e.g. on
    reconnection when the client already received them).
    """
    source = await source_svc.get(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    from app_main.services.log_stream import get_log_stream

    log_stream = get_log_stream()

    async def event_generator():
        async for entry in log_stream.subscribe(str(source.id), after=after):
            yield entry.to_sse()

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post(
    "/{source_id}/judge-contradictions",
    response_model=SourceJudgeContradictionsResponse,
)
async def judge_contradictions(
    source_id: str,
    k: int = Query(
        5,
        ge=1,
        le=MAX_K,
        description=(
            "Max related sources to judge against (top-k from the Track R "
            "related substrate — bounds LLM cost to O(top-k), not O(corpus))"
        ),
    ),
    min_confidence: float = Query(
        0.7,
        ge=0.0,
        le=1.0,
        description=(
            "Precision gate: only contradicts/reinforces verdicts at/above this "
            "confidence are persisted as a source_verdict edge"
        ),
    ),
    source_svc: SourceService = Depends(get_source_service),
    judge_svc: ContradictionJudgeService = Depends(
        get_contradiction_judge_service
    ),
):
    """Judge a source against its related sources for contradictions (Track Z.3).

    The on-demand trigger for the contradiction judge. Pulls the source's top-``k``
    related sources from the Track R substrate (already topically related —
    candidate pairs, NOT O(n²)), asks an LLM to judge each pair as
    contradicts/reinforces/neutral, and persists a ``source_verdict`` edge ONLY
    for confident contradicts/reinforces (``confidence >= min_confidence``).
    Precision-first: a false contradiction is worse than a missed one, so
    neutral/below-threshold verdicts are judged but written nowhere. Idempotent
    (Z.1 clear-before-relate) — re-running yields the same edge set.

    Validation happens at the boundary: the source id is strict-validated, and
    ``k`` / ``min_confidence`` are bounded by ``Query`` constraints — a bad value
    is a 422, not a 500. A missing source is a 404.
    """
    # Route-layer id validation (defense-in-depth; the repo also validates).
    _validate_source_id(source_id)

    existing = await source_svc.get(source_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Source not found")

    result = await judge_svc.judge_source(
        source_id, k=k, min_confidence=min_confidence
    )
    logger.info(
        "Judged source {} -> {} edges (judged={}, status={})",
        source_id,
        result.edges_written,
        result.judged,
        result.status,
    )
    return SourceJudgeContradictionsResponse(**result.to_dict())
