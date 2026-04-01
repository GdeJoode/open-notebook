"""Sources router - full CRUD for sources including file upload and processing."""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse, Response, StreamingResponse
from loguru import logger

from app_main.api.schemas import (
    AssetModel,
    CreateSourceInsightRequest,
    SourceCreate,
    SourceInsightResponse,
    SourceListResponse,
    SourceResponse,
    SourceStatusResponse,
    SourceUpdate,
)
from pydantic import BaseModel

from app_main.config import UPLOADS_FOLDER
from app_main.dependencies import (
    get_notebook_service,
    get_source_service,
    get_transformation_service,
)
from app_main.exceptions import InvalidInputError
from app_main.services.notebook_service import NotebookService
from app_main.services.source_service import SourceService
from app_main.services.transformation_service import TransformationService

router = APIRouter(prefix="/sources", tags=["sources"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_unique_filename(original_filename: str, upload_folder: str) -> str:
    """Generate unique filename (append counter if file exists)."""
    file_path = Path(upload_folder)
    file_path.mkdir(parents=True, exist_ok=True)

    stem = Path(original_filename).stem
    suffix = Path(original_filename).suffix

    counter = 0
    while True:
        if counter == 0:
            new_filename = original_filename
        else:
            new_filename = f"{stem} ({counter}){suffix}"

        full_path = file_path / new_filename
        if not full_path.exists():
            return str(full_path)
        counter += 1


async def save_uploaded_file(upload_file: UploadFile) -> str:
    """Save uploaded file to uploads folder and return file path."""
    if not upload_file.filename:
        raise ValueError("No filename provided")

    file_path = generate_unique_filename(upload_file.filename, UPLOADS_FOLDER)

    try:
        with open(file_path, "wb") as f:
            content = await upload_file.read()
            f.write(content)
        logger.info(f"Saved uploaded file to: {file_path}")
        return file_path
    except Exception as e:
        logger.error(f"Failed to save uploaded file: {e}")
        if os.path.exists(file_path):
            os.unlink(file_path)
        raise


def parse_source_form_data(
    type: str = Form(...),
    notebook_id: Optional[str] = Form(None),
    notebooks: Optional[str] = Form(None),
    url: Optional[str] = Form(None),
    content: Optional[str] = Form(None),
    title: Optional[str] = Form(None),
    transformations: Optional[str] = Form(None),
    embed: str = Form("false"),
    delete_source: str = Form("false"),
    async_processing: str = Form("false"),
    processing_overrides: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
) -> tuple[SourceCreate, Optional[UploadFile]]:
    """Parse form data into SourceCreate model and return upload file."""
    import json

    def str_to_bool(value: str) -> bool:
        return value.lower() in ("true", "1", "yes", "on")

    embed_bool = str_to_bool(embed)
    delete_source_bool = str_to_bool(delete_source)
    async_processing_bool = str_to_bool(async_processing)

    notebooks_list = None
    if notebooks:
        try:
            notebooks_list = json.loads(notebooks)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in notebooks field")

    transformations_list: list = []
    if transformations:
        try:
            transformations_list = json.loads(transformations)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in transformations field")

    overrides_dict = None
    if processing_overrides:
        try:
            overrides_dict = json.loads(processing_overrides)
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON in processing_overrides field")

    source_data = SourceCreate(
        type=type,
        notebook_id=notebook_id,
        notebooks=notebooks_list,
        url=url,
        content=content,
        title=title,
        file_path=None,
        transformations=transformations_list,
        embed=embed_bool,
        delete_source=delete_source_bool,
        async_processing=async_processing_bool,
        processing_overrides=overrides_dict,
    )

    return source_data, file


def _is_source_file_available(asset) -> Optional[bool]:
    """Check if a source's file is available on disk."""
    if not asset or not asset.file_path:
        return None

    file_path = asset.file_path
    safe_root = os.path.realpath(UPLOADS_FOLDER)
    resolved_path = os.path.realpath(file_path)

    if not resolved_path.startswith(safe_root):
        return False

    return os.path.exists(resolved_path)


async def _resolve_source_file(
    source_svc: SourceService, source_id: str,
) -> tuple[str, str]:
    """Resolve and validate a source's file path for download."""
    source = await source_svc.get(source_id)
    if not source:
        raise HTTPException(status_code=404, detail="Source not found")

    file_path = source.asset.file_path if source.asset else None
    if not file_path:
        raise HTTPException(
            status_code=404, detail="Source has no file to download"
        )

    safe_root = os.path.realpath(UPLOADS_FOLDER)
    resolved_path = os.path.realpath(file_path)

    if not resolved_path.startswith(safe_root):
        logger.warning(
            f"Blocked download outside uploads directory for source "
            f"{source_id}: {resolved_path}"
        )
        raise HTTPException(status_code=403, detail="Access to file denied")

    if not os.path.exists(resolved_path):
        raise HTTPException(
            status_code=404, detail="File not found on server"
        )

    filename = os.path.basename(resolved_path)
    return resolved_path, filename


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("", response_model=List[SourceListResponse])
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


@router.post("", response_model=SourceResponse)
async def create_source(
    form_data: tuple[SourceCreate, Optional[UploadFile]] = Depends(
        parse_source_form_data
    ),
    source_svc: SourceService = Depends(get_source_service),
    notebook_svc: NotebookService = Depends(get_notebook_service),
    transformation_svc: TransformationService = Depends(
        get_transformation_service
    ),
):
    """Create a new source with support for both JSON and multipart form data."""
    source_data, upload_file = form_data
    file_path = None

    try:
        # Verify all specified notebooks exist
        for nb_id in source_data.notebooks or []:
            notebook = await notebook_svc.get(nb_id)
            if not notebook:
                raise HTTPException(
                    status_code=404,
                    detail=f"Notebook {nb_id} not found",
                )

        # Handle file upload
        if upload_file and source_data.type == "upload":
            try:
                file_path = await save_uploaded_file(upload_file)
            except Exception as e:
                logger.error(f"File upload failed: {e}")
                raise HTTPException(
                    status_code=400,
                    detail=f"File upload failed: {str(e)}",
                )

        # Prepare content_state
        content_state: Dict[str, Any] = {}

        if source_data.type == "link":
            if not source_data.url:
                raise HTTPException(
                    status_code=400,
                    detail="URL is required for link type",
                )
            content_state["url"] = source_data.url
        elif source_data.type == "upload":
            final_file_path = file_path or source_data.file_path
            if not final_file_path:
                raise HTTPException(
                    status_code=400,
                    detail="File upload or file_path is required for upload type",
                )
            content_state["file_path"] = final_file_path
            content_state["delete_source"] = source_data.delete_source
        elif source_data.type == "text":
            if not source_data.content:
                raise HTTPException(
                    status_code=400,
                    detail="Content is required for text type",
                )
            content_state["content"] = source_data.content
        else:
            raise HTTPException(
                status_code=400,
                detail="Invalid source type. Must be link, upload, or text",
            )

        # Validate transformations exist
        transformation_ids = source_data.transformations or []
        for trans_id in transformation_ids:
            transformation = await transformation_svc.get(trans_id)
            if not transformation:
                raise HTTPException(
                    status_code=404,
                    detail=f"Transformation {trans_id} not found",
                )

        if source_data.async_processing:
            # ASYNC PATH
            logger.info("Using async processing path")

            source = await source_svc.create(
                {
                    "title": source_data.title or "Processing...",
                    "topics": [],
                }
            )

            # Add source to notebooks immediately
            for nb_id in source_data.notebooks or []:
                await source_svc.add_to_notebook(source.id, nb_id)

            try:
                from app_main.services.command_service import CommandService

                command_args = {
                    "source_id": str(source.id),
                    "content_state": content_state,
                    "notebook_ids": source_data.notebooks,
                    "processing_overrides": source_data.processing_overrides,
                }

                command_id = await CommandService.submit_command_job(
                    "open_notebook",
                    "process_source",
                    command_args,
                )

                logger.info(
                    f"Submitted async extraction command: {command_id}"
                )

                # Update source with command reference
                from surrealdb_service.connection import (
                    execute_query,
                    ensure_record_id,
                )

                await execute_query(
                    "UPDATE $source_id SET command = $command_id",
                    {
                        "source_id": ensure_record_id(source.id),
                        "command_id": ensure_record_id(command_id),
                    },
                )

                return SourceResponse(
                    id=source.id or "",
                    title=source.title,
                    topics=source.topics or [],
                    asset=None,
                    full_text=None,
                    embedded=False,
                    embedded_chunks=0,
                    created=str(source.created),
                    updated=str(source.updated),
                    command_id=command_id,
                    status="new",
                    processing_info={"async": True, "queued": True},
                )

            except Exception as e:
                logger.error(
                    f"Failed to submit async processing command: {e}"
                )
                try:
                    await source_svc.delete(source.id)
                except Exception:
                    pass
                if file_path and upload_file:
                    try:
                        os.unlink(file_path)
                    except Exception:
                        pass
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to queue processing: {str(e)}",
                )

        else:
            # SYNC PATH
            logger.info("Using sync processing path")

            try:
                source = await source_svc.create(
                    {
                        "title": source_data.title or "Processing...",
                        "topics": [],
                    }
                )

                for nb_id in source_data.notebooks or []:
                    await source_svc.add_to_notebook(source.id, nb_id)

                from app_main.services.command_service import CommandService

                command_args = {
                    "source_id": str(source.id),
                    "content_state": content_state,
                    "notebook_ids": source_data.notebooks,
                    "processing_overrides": source_data.processing_overrides,
                }

                command_id = await CommandService.submit_command_job(
                    "open_notebook",
                    "process_source",
                    command_args,
                )

                # Wait for completion with timeout
                import time

                start_time = time.time()
                timeout = 300  # 5 minutes

                while time.time() - start_time < timeout:
                    status_data = await CommandService.get_command_status(
                        command_id
                    )
                    cmd_status = status_data.get("status", "unknown")
                    if cmd_status in ["completed", "failed"]:
                        break
                    await asyncio.sleep(1)

                if cmd_status == "failed":
                    error_msg = status_data.get("error_message", "Unknown error")
                    logger.error(f"Sync processing failed: {error_msg}")
                    try:
                        await source_svc.delete(source.id)
                    except Exception:
                        pass
                    if file_path and upload_file:
                        try:
                            os.unlink(file_path)
                        except Exception:
                            pass
                    raise HTTPException(
                        status_code=500,
                        detail=f"Processing failed: {error_msg}",
                    )

                # Re-fetch the processed source
                processed_source = await source_svc.get(source.id)
                if not processed_source:
                    raise HTTPException(
                        status_code=500,
                        detail="Processed source not found",
                    )

                embedded_chunks = await source_svc.get_embedding_count(
                    source.id
                )

                return SourceResponse(
                    id=processed_source.id or "",
                    title=processed_source.title,
                    topics=processed_source.topics or [],
                    asset=AssetModel(
                        file_path=(
                            processed_source.asset.file_path
                            if processed_source.asset
                            else None
                        ),
                        url=(
                            processed_source.asset.url
                            if processed_source.asset
                            else None
                        ),
                    )
                    if processed_source.asset
                    else None,
                    full_text=processed_source.full_text,
                    embedded=embedded_chunks > 0,
                    embedded_chunks=embedded_chunks,
                    created=str(processed_source.created),
                    updated=str(processed_source.updated),
                )

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Sync processing failed: {e}")
                if file_path and upload_file:
                    try:
                        os.unlink(file_path)
                    except Exception:
                        pass
                raise

    except HTTPException:
        if file_path and upload_file:
            try:
                os.unlink(file_path)
            except Exception:
                pass
        raise
    except InvalidInputError as e:
        if file_path and upload_file:
            try:
                os.unlink(file_path)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating source: {e}")
        if file_path and upload_file:
            try:
                os.unlink(file_path)
            except Exception:
                pass
        raise HTTPException(
            status_code=500, detail=f"Error creating source: {str(e)}"
        )


@router.post("/json", response_model=SourceResponse)
async def create_source_json(
    source_data: SourceCreate,
    source_svc: SourceService = Depends(get_source_service),
    notebook_svc: NotebookService = Depends(get_notebook_service),
    transformation_svc: TransformationService = Depends(
        get_transformation_service
    ),
):
    """Create a new source using JSON payload (legacy endpoint)."""
    form_data = (source_data, None)
    return await create_source(
        form_data, source_svc, notebook_svc, transformation_svc
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


@router.head("/{source_id}/download")
async def check_source_file(
    source_id: str,
    source_svc: SourceService = Depends(get_source_service),
):
    """Check if a source has a downloadable file."""
    try:
        await _resolve_source_file(source_svc, source_id)
        return Response(status_code=200)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error checking file for source {source_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to verify file"
        )


@router.get("/{source_id}/download")
async def download_source_file(
    source_id: str,
    source_svc: SourceService = Depends(get_source_service),
):
    """Download the original file associated with an uploaded source."""
    try:
        resolved_path, filename = await _resolve_source_file(
            source_svc, source_id
        )
        return FileResponse(
            path=resolved_path,
            filename=filename,
            media_type="application/octet-stream",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading file for source {source_id}: {e}")
        raise HTTPException(
            status_code=500, detail="Failed to download source file"
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


@router.post("/{source_id}/retry", response_model=SourceResponse)
async def retry_source_processing(
    source_id: str,
    source_svc: SourceService = Depends(get_source_service),
):
    """Retry processing for a failed or stuck source."""
    try:
        source = await source_svc.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Check if source already has a running command
        if hasattr(source, "command") and source.command:
            try:
                from app_main.services.command_service import CommandService

                status_data = await CommandService.get_command_status(
                    str(source.command)
                )
                cmd_status = status_data.get("status")
                if cmd_status in ["running", "queued"]:
                    raise HTTPException(
                        status_code=400,
                        detail="Source is already processing. Cannot retry.",
                    )
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(
                    f"Failed to check current status for source "
                    f"{source_id}: {e}"
                )

        # Get notebooks this source belongs to
        from surrealdb_service.connection import execute_query, ensure_record_id

        references = await execute_query(
            "SELECT VALUE out FROM reference WHERE in = $source_id",
            {"source_id": ensure_record_id(source_id)},
        )
        notebook_ids = (
            [str(ref) for ref in references] if references else []
        )

        if not notebook_ids:
            raise HTTPException(
                status_code=400,
                detail="Source is not associated with any notebooks",
            )

        # Prepare content_state based on source asset
        content_state: Dict[str, Any] = {}
        if source.asset:
            if source.asset.file_path:
                content_state = {
                    "file_path": source.asset.file_path,
                    "delete_source": False,
                }
            elif source.asset.url:
                content_state = {"url": source.asset.url}
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Source asset has no file_path or url",
                )
        else:
            if source.full_text:
                content_state = {"content": source.full_text}
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot determine source content for retry",
                )

        try:
            from app_main.services.command_service import CommandService

            command_args = {
                "source_id": str(source.id),
                "content_state": content_state,
                "notebook_ids": notebook_ids,
                "transformations": [],
                "embed": True,
            }

            command_id = await CommandService.submit_command_job(
                "open_notebook",
                "process_source",
                command_args,
            )

            logger.info(
                f"Submitted retry processing command: {command_id} "
                f"for source {source_id}"
            )

            # Update source with new command ID
            await execute_query(
                "UPDATE $source_id SET command = $command_id",
                {
                    "source_id": ensure_record_id(source_id),
                    "command_id": ensure_record_id(f"command:{command_id}"),
                },
            )

            embedded_chunks = await source_svc.get_embedding_count(
                source_id
            )

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
                command_id=command_id,
                status="queued",
                processing_info={"retry": True, "queued": True},
            )

        except HTTPException:
            raise
        except Exception as e:
            logger.error(
                f"Failed to submit retry processing command "
                f"for source {source_id}: {e}"
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to queue retry processing: {str(e)}",
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error retrying source processing for {source_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error retrying source processing: {str(e)}",
        )


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


class RunEntitiesRequest(BaseModel):
    ontology_name: str = "general"
    extractor_type: str = "llm"  # "llm" or "langextract"
    # LangExtract-specific options (forwarded when extractor_type == "langextract")
    langextract_model_id: Optional[str] = None
    langextract_model_url: Optional[str] = None
    langextract_temperature: Optional[float] = None
    langextract_use_schema_constraints: Optional[bool] = None
    langextract_fence_output: Optional[bool] = None


@router.post("/{source_id}/run-entities")
async def run_entities(
    source_id: str,
    body: RunEntitiesRequest = RunEntitiesRequest(),
    source_svc: SourceService = Depends(get_source_service),
):
    """Trigger ontology-guided entity extraction for a source."""
    try:
        source = await source_svc.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        from app_main.services.command_service import CommandService

        command_payload = {
            "source_id": str(source.id),
            "ontology_name": body.ontology_name,
            "extractor_type": body.extractor_type,
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


@router.get("/{source_id}/pdf")
@router.head("/{source_id}/pdf")
async def get_source_pdf(
    source_id: str,
    source_svc: SourceService = Depends(get_source_service),
):
    """Return the original PDF file for chunk visualization.

    Supports both GET and HEAD methods for browser/PDF.js compatibility.
    """
    try:
        source = await source_svc.get(source_id)
        if (
            not source
            or not source.asset
            or not source.asset.file_path
        ):
            raise HTTPException(
                status_code=404, detail="PDF file not found"
            )

        file_path = source.asset.file_path

        # Security check
        safe_root = os.path.realpath(UPLOADS_FOLDER)
        resolved_path = os.path.realpath(file_path)

        if not resolved_path.startswith(safe_root):
            logger.warning(
                f"Blocked PDF access outside uploads directory "
                f"for source {source_id}: {resolved_path}"
            )
            raise HTTPException(
                status_code=403, detail="Access to file denied"
            )

        if not os.path.exists(resolved_path):
            raise HTTPException(
                status_code=404, detail="PDF file has been deleted"
            )

        if not resolved_path.lower().endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail="Source file is not a PDF document",
            )

        return FileResponse(
            path=resolved_path,
            media_type="application/pdf",
            headers={
                "Content-Disposition": "inline",
                "Accept-Ranges": "bytes",
                "Access-Control-Expose-Headers": (
                    "Content-Length, Content-Range, Accept-Ranges"
                ),
            },
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error serving PDF for source {source_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error serving PDF: {str(e)}",
        )


@router.get("/{source_id}/images/{filename}")
async def get_source_image(
    source_id: str,
    filename: str,
    source_svc: SourceService = Depends(get_source_service),
):
    """Serve an extracted image file from the ingestion output directory."""
    try:
        source = await source_svc.get(source_id)
        if not source:
            raise HTTPException(status_code=404, detail="Source not found")

        # Security: only allow simple filenames (no path traversal)
        if "/" in filename or "\\" in filename or ".." in filename:
            raise HTTPException(
                status_code=400, detail="Invalid filename"
            )

        output_dir = getattr(source, "output_directory", None)

        # Fallback: derive from source file path + default ingestion output
        if not output_dir and source.asset and source.asset.file_path:
            stem = Path(source.asset.file_path).stem
            candidate = Path("ingestion_output") / stem
            if candidate.exists():
                output_dir = str(candidate)

        if not output_dir:
            raise HTTPException(
                status_code=404,
                detail="No output directory for this source",
            )

        # Images are at: {output_directory}/output/extracted_info/images/{filename}
        image_path = (
            Path(output_dir) / "output" / "extracted_info" / "images" / filename
        )

        resolved = os.path.realpath(str(image_path))

        # Security: ensure the resolved path is under the output directory
        safe_root = os.path.realpath(str(output_dir))
        if not resolved.startswith(safe_root):
            raise HTTPException(
                status_code=403, detail="Access to file denied"
            )

        if not os.path.exists(resolved):
            raise HTTPException(
                status_code=404, detail="Image file not found"
            )

        # Determine media type from extension
        ext = Path(filename).suffix.lower().lstrip(".")
        media_types = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "gif": "image/gif",
            "webp": "image/webp",
            "svg": "image/svg+xml",
            "bmp": "image/bmp",
            "tiff": "image/tiff",
            "tif": "image/tiff",
        }
        media_type = media_types.get(ext, "application/octet-stream")

        return FileResponse(
            path=resolved,
            media_type=media_type,
            headers={"Cache-Control": "public, max-age=86400"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error serving image {filename} for source {source_id}: {e}"
        )
        raise HTTPException(
            status_code=500,
            detail=f"Error serving image: {str(e)}",
        )
