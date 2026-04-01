"""Sources upload sub-router — upload, creation, and retry endpoints."""

import asyncio
import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from loguru import logger

from app_main.api.schemas import (
    AssetModel,
    SourceCreate,
    SourceResponse,
)
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

router = APIRouter()


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


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/", response_model=SourceResponse)
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
