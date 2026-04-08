"""Sources CRUD sub-router — basic CRUD, list, status, insights, and chunks."""

from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from loguru import logger

from app_main.api.schemas import (
    AssetModel,
    ChunkCreate,
    ChunkUpdate,
    CreateSourceInsightRequest,
    SourceInsightResponse,
    SourceListResponse,
    SourceResponse,
    SourceStatusResponse,
    SourceUpdate,
)
from app_main.config import UPLOADS_FOLDER
from app_main.dependencies import (
    get_chunk_repo,
    get_notebook_service,
    get_source_service,
    get_transformation_service,
)
from surrealdb_service.repositories.source import ChunkRepository
from app_main.exceptions import InvalidInputError
from app_main.services.notebook_service import NotebookService
from app_main.services.source_service import SourceService
from app_main.services.transformation_service import TransformationService

from app_main.api.routers.sources_files import _is_source_file_available

router = APIRouter()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/", response_model=List[SourceListResponse])
async def get_sources(
    notebook_id: Optional[str] = Query(
        None, description="Filter by notebook ID"
    ),
    limit: int = Query(
        50, ge=1, le=100, description="Number of sources to return (1-100)"
    ),
    offset: int = Query(0, ge=0, description="Number of sources to skip"),
    sort_by: str = Query(
        "updated", description="Field to sort by (created or updated)"
    ),
    sort_order: str = Query("desc", description="Sort order (asc or desc)"),
    source_svc: SourceService = Depends(get_source_service),
    notebook_svc: NotebookService = Depends(get_notebook_service),
):
    """Get sources with pagination and sorting support."""
    try:
        if sort_by not in ["created", "updated"]:
            raise HTTPException(
                status_code=400,
                detail="sort_by must be 'created' or 'updated'",
            )
        if sort_order.lower() not in ["asc", "desc"]:
            raise HTTPException(
                status_code=400,
                detail="sort_order must be 'asc' or 'desc'",
            )

        if notebook_id:
            notebook = await notebook_svc.get(notebook_id)
            if not notebook:
                raise HTTPException(
                    status_code=404, detail="Notebook not found"
                )

        from surrealdb_service.repositories import SourceRepository

        source_repo = SourceRepository()
        result = await source_repo.list_with_metadata(
            notebook_id=notebook_id,
            order_by=sort_by,
            order_dir=sort_order.upper(),
            limit=limit,
            offset=offset,
        )

        # Batch fetch command statuses in a single query
        command_ids = []
        for row in result:
            command = row.get("command")
            if command:
                command_ids.append(str(command))

        command_statuses = await source_repo.batch_get_command_status(
            command_ids
        )

        response_list = []
        for row in result:
            command = row.get("command")
            command_id = str(command) if command else None
            status = None
            processing_info = None

            if command_id and command_id in command_statuses:
                job = command_statuses[command_id]
                status = job.get("status")
                result_data = job.get("result")
                execution_metadata = (
                    result_data.get("execution_metadata", {})
                    if isinstance(result_data, dict)
                    else {}
                )
                processing_info = {
                    "started_at": execution_metadata.get("started_at")
                    or job.get("started_at"),
                    "completed_at": execution_metadata.get("completed_at")
                    or job.get("completed_at"),
                    "error": job.get("error_message"),
                }
            elif command_id:
                status = "unknown"

            response_list.append(
                SourceListResponse(
                    id=row["id"],
                    title=row.get("title"),
                    topics=row.get("topics") or [],
                    asset=AssetModel(
                        file_path=(
                            row["asset"].get("file_path")
                            if row.get("asset")
                            else None
                        ),
                        url=(
                            row["asset"].get("url")
                            if row.get("asset")
                            else None
                        ),
                    )
                    if row.get("asset")
                    else None,
                    embedded=row.get("embedded", False),
                    embedded_chunks=0,
                    insights_count=row.get("insights_count", 0),
                    created=str(row["created"]),
                    updated=str(row["updated"]),
                    command_id=command_id,
                    status=status,
                    processing_info=processing_info,
                )
            )

        return response_list
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching sources: {e}")
        raise HTTPException(
            status_code=500, detail=f"Error fetching sources: {str(e)}"
        )


@router.get("/{source_id}", response_model=SourceResponse)
async def get_source(
    source_id: str,
    source_svc: SourceService = Depends(get_source_service),
):
    """Get a specific source by ID."""
    try:
        source = await source_svc.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Get status information if command exists
        status = None
        processing_info = None
        if hasattr(source, "command") and source.command:
            try:
                from app_main.services.command_service import CommandService

                status_data = await CommandService.get_command_status(
                    str(source.command)
                )
                status = status_data.get("status")
                result_data = status_data.get("result")
                execution_metadata = (
                    result_data.get("execution_metadata", {})
                    if isinstance(result_data, dict)
                    else {}
                )
                processing_info = {
                    "started_at": execution_metadata.get("started_at"),
                    "completed_at": execution_metadata.get("completed_at"),
                    "error": status_data.get("error_message"),
                }
            except Exception as e:
                logger.warning(
                    f"Failed to get status for source {source_id}: {e}"
                )
                status = "unknown"

        embedded_chunks = await source_svc.get_embedding_count(source_id)

        # Get associated notebooks + counts
        from surrealdb_service.connection import execute_query, ensure_record_id

        notebooks_query = await execute_query(
            "SELECT VALUE out FROM reference WHERE in = $source_id",
            {"source_id": ensure_record_id(source.id or source_id)},
        )
        notebook_ids = (
            [str(nb_id) for nb_id in notebooks_query]
            if notebooks_query
            else []
        )

        # Get insights count
        insights_rows = await execute_query(
            "SELECT VALUE count() FROM source_insight "
            "WHERE source = $source_id GROUP ALL",
            {"source_id": ensure_record_id(source.id or source_id)},
        )
        insights_count = insights_rows[0] if insights_rows else 0

        # Get extraction result counts
        extraction_rows = await execute_query(
            "SELECT entity_count, relation_count FROM extraction_result "
            "WHERE source_id = $source_id LIMIT 1",
            {"source_id": str(source.id or source_id)},
        )
        entity_count = 0
        relation_count = 0
        if extraction_rows:
            entity_count = extraction_rows[0].get("entity_count", 0) or 0
            relation_count = extraction_rows[0].get("relation_count", 0) or 0

        return SourceResponse(
            id=source.id or "",
            title=source.title,
            topics=source.topics or [],
            asset=AssetModel(
                file_path=(
                    source.asset.file_path if source.asset else None
                ),
                url=source.asset.url if source.asset else None,
            )
            if source.asset
            else None,
            full_text=source.full_text,
            embedded=embedded_chunks > 0,
            embedded_chunks=embedded_chunks,
            insights_count=insights_count,
            entity_count=entity_count,
            relation_count=relation_count,
            file_available=_is_source_file_available(source.asset),
            created=str(source.created),
            updated=str(source.updated),
            command_id=(
                str(source.command) if hasattr(source, "command") and source.command else None
            ),
            status=status,
            processing_info=processing_info,
            notebooks=notebook_ids,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching source {source_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching source: {str(e)}",
        )


@router.put("/{source_id}", response_model=SourceResponse)
async def update_source(
    source_id: str,
    source_update: SourceUpdate,
    source_svc: SourceService = Depends(get_source_service),
):
    """Update a source."""
    try:
        source = await source_svc.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        update_data = source_update.model_dump(exclude_none=True)
        if update_data:
            source = await source_svc.update(source_id, update_data)

        embedded_chunks = await source_svc.get_embedding_count(source_id)

        return SourceResponse(
            id=source.id or "",
            title=source.title,
            topics=source.topics or [],
            asset=AssetModel(
                file_path=(
                    source.asset.file_path if source.asset else None
                ),
                url=source.asset.url if source.asset else None,
            )
            if source.asset
            else None,
            full_text=source.full_text,
            embedded=embedded_chunks > 0,
            embedded_chunks=embedded_chunks,
            created=str(source.created),
            updated=str(source.updated),
        )
    except HTTPException:
        raise
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating source {source_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error updating source: {str(e)}",
        )


@router.delete("/{source_id}")
async def delete_source(
    source_id: str,
    source_svc: SourceService = Depends(get_source_service),
):
    """Delete a source."""
    try:
        source = await source_svc.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        await source_svc.delete(source_id)
        return {"message": "Source deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting source {source_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting source: {str(e)}",
        )


@router.get("/{source_id}/status", response_model=SourceStatusResponse)
async def get_source_status(
    source_id: str,
    source_svc: SourceService = Depends(get_source_service),
):
    """Get processing status for a source."""
    try:
        source = await source_svc.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        if not hasattr(source, "command") or not source.command:
            return SourceStatusResponse(
                status=None,
                message="Legacy source (completed before async processing)",
                processing_info=None,
                command_id=None,
            )

        try:
            from app_main.services.command_service import CommandService

            status_data = await CommandService.get_command_status(
                str(source.command)
            )
            status = status_data.get("status", "unknown")

            if status == "completed":
                message = "Source processing completed successfully"
            elif status == "failed":
                message = "Source processing failed"
            elif status == "running":
                message = "Source processing in progress"
            elif status == "queued":
                message = "Source processing queued"
            else:
                message = f"Source processing status: {status}"

            result_data = status_data.get("result")
            processing_info = (
                result_data
                if isinstance(result_data, dict)
                else None
            )

            return SourceStatusResponse(
                status=status,
                message=message,
                processing_info=processing_info,
                command_id=str(source.command),
            )

        except Exception as e:
            logger.warning(
                f"Failed to get status for source {source_id}: {e}"
            )
            return SourceStatusResponse(
                status="unknown",
                message="Failed to retrieve processing status",
                processing_info=None,
                command_id=str(source.command),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching status for source {source_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching source status: {str(e)}",
        )


@router.get(
    "/{source_id}/insights",
    response_model=List[SourceInsightResponse],
)
async def get_source_insights(
    source_id: str,
    source_svc: SourceService = Depends(get_source_service),
):
    """Get all insights for a specific source."""
    try:
        source = await source_svc.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        insights = await source_svc.get_insights(source_id)
        return [
            SourceInsightResponse(
                id=insight.id or "",
                source_id=source_id,
                insight_type=insight.insight_type,
                content=insight.content,
                created=str(insight.created),
                updated=str(insight.updated),
            )
            for insight in insights
        ]
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error fetching insights for source {source_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching insights: {str(e)}",
        )


@router.post(
    "/{source_id}/insights",
    response_model=SourceInsightResponse,
)
async def create_source_insight(
    source_id: str,
    request: CreateSourceInsightRequest,
    source_svc: SourceService = Depends(get_source_service),
    transformation_svc: TransformationService = Depends(
        get_transformation_service
    ),
):
    """Create a new insight for a source by running a transformation."""
    try:
        source = await source_svc.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        transformation = await transformation_svc.get(
            request.transformation_id
        )
        if not transformation:
            raise HTTPException(
                status_code=404, detail="Transformation not found"
            )

        # Run transformation graph
        from app_main.graphs.transformation import graph as transform_graph

        await transform_graph.ainvoke(
            input=dict(source=source, transformation=transformation)
        )

        # Get the newly created insight
        insights = await source_svc.get_insights(source_id)
        if insights:
            newest = insights[-1]
            return SourceInsightResponse(
                id=newest.id or "",
                source_id=source_id,
                insight_type=newest.insight_type,
                content=newest.content,
                created=str(newest.created),
                updated=str(newest.updated),
            )
        else:
            raise HTTPException(
                status_code=500, detail="Failed to create insight"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error creating insight for source {source_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error creating insight: {str(e)}",
        )


@router.get("/{source_id}/chunks")
async def get_source_chunks(
    source_id: str,
    source_svc: SourceService = Depends(get_source_service),
):
    """Get chunks with bounding box positions for a source."""
    try:
        source = await source_svc.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        chunks = await source_svc.get_chunks(source_id)

        chunks_data = []
        for chunk in chunks:
            chunks_data.append(
                {
                    "id": chunk.id,
                    "text": chunk.text,
                    "order": chunk.order,
                    "physical_page": getattr(chunk, "physical_page", None),
                    "printed_page": getattr(chunk, "printed_page", None),
                    "chapter": getattr(chunk, "chapter", None),
                    "paragraph_number": getattr(
                        chunk, "paragraph_number", None
                    ),
                    "element_type": getattr(chunk, "element_type", None),
                    "positions": getattr(chunk, "positions", None),
                    "metadata": getattr(chunk, "metadata", None),
                    "is_content": getattr(chunk, "is_content", True),
                }
            )

        has_spatial_data = any(
            cd.get("positions") and len(cd["positions"]) > 0
            for cd in chunks_data
        )

        return {
            "chunks": chunks_data,
            "total_chunks": len(chunks_data),
            "has_spatial_data": has_spatial_data,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error fetching chunks for source {source_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching chunks: {str(e)}",
        )


@router.patch("/{source_id}/chunks/{chunk_id}")
async def update_chunk(
    source_id: str,
    chunk_id: str,
    body: ChunkUpdate,
    chunk_repo: ChunkRepository = Depends(get_chunk_repo),
    source_svc: SourceService = Depends(get_source_service),
):
    """Update a chunk's text, element type, positions, or content flag."""
    try:
        source = await source_svc.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        chunk = await chunk_repo.get(chunk_id)
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")
        if str(chunk.source) != str(source_id) and str(chunk.source) != str(source.id):
            raise HTTPException(status_code=403, detail="Chunk does not belong to this source")

        update_data = body.model_dump(exclude_unset=True)
        if not update_data:
            return chunk.model_dump()

        updated = await chunk_repo.update(chunk_id, update_data)
        return updated.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating chunk {chunk_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating chunk: {str(e)}")


@router.delete("/{source_id}/chunks/{chunk_id}")
async def delete_chunk(
    source_id: str,
    chunk_id: str,
    chunk_repo: ChunkRepository = Depends(get_chunk_repo),
    source_svc: SourceService = Depends(get_source_service),
):
    """Delete a single chunk."""
    try:
        source = await source_svc.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        chunk = await chunk_repo.get(chunk_id)
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found")
        if str(chunk.source) != str(source_id) and str(chunk.source) != str(source.id):
            raise HTTPException(status_code=403, detail="Chunk does not belong to this source")

        await chunk_repo.delete(chunk_id)
        return {"message": "Chunk deleted"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting chunk {chunk_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting chunk: {str(e)}")


@router.post("/{source_id}/chunks")
async def create_chunk(
    source_id: str,
    body: ChunkCreate,
    chunk_repo: ChunkRepository = Depends(get_chunk_repo),
    source_svc: SourceService = Depends(get_source_service),
):
    """Create a new manual chunk for a source."""
    try:
        source = await source_svc.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Compute next order value
        from surrealdb_service.connection import ensure_record_id
        existing = await chunk_repo.get_by_source(str(source.id))
        max_order = max((c.order for c in existing), default=-1)

        chunk_data = {
            "source": ensure_record_id(str(source.id)),
            "text": body.text,
            "element_type": body.element_type,
            "physical_page": body.physical_page,
            "printed_page": body.physical_page + 1,
            "positions": body.positions,
            "is_content": body.is_content,
            "chapter": body.chapter,
            "order": max_order + 1,
            "chunk_type": "manual",
            "source_element_count": 1,
            "section_path": [],
            "section_level": 0,
            "metadata": {**body.metadata, "manual": True},
        }

        created = await chunk_repo.create(chunk_data)
        return created.model_dump()

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating chunk for source {source_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating chunk: {str(e)}")
