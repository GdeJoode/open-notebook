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
    Request,
    Response,
    UploadFile,
)
from loguru import logger

from app_main.api.rate_limit import limiter
from app_main.api.schemas import (
    AssetModel,
    SourceCreate,
    SourceResponse,
)
from app_main.config import (
    UPLOADS_FOLDER,
    get_max_file_size_mb,
    get_max_page_count,
    get_rate_limit_rpm,
)
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
    """Generate a unique path inside ``upload_folder`` (append a counter if taken).

    ``original_filename`` is the UNTRUSTED multipart filename. Reduce it to its
    basename (``Path(...).name``) before joining, so path separators / ``..`` can
    NOT escape the uploads folder — a filename like ``../../etc/cron.d/x`` would
    otherwise resolve outside it (arbitrary file write / RCE-class primitive,
    reachable via both the UI upload route and the G.3b agent upload endpoints).
    A name that reduces to empty / ``.`` / ``..`` falls back to ``upload``. As
    defense-in-depth the returned path is asserted to stay within the (resolved)
    uploads folder, so a future regression can't silently reintroduce the escape.
    """
    file_path = Path(upload_folder)
    file_path.mkdir(parents=True, exist_ok=True)
    folder_resolved = file_path.resolve()

    safe_name = Path(original_filename).name
    if not safe_name or safe_name in (".", ".."):
        safe_name = "upload"

    stem = Path(safe_name).stem
    suffix = Path(safe_name).suffix

    counter = 0
    while True:
        if counter == 0:
            new_filename = safe_name
        else:
            new_filename = f"{stem} ({counter}){suffix}"

        full_path = file_path / new_filename
        if not full_path.exists():
            resolved = full_path.resolve()
            if folder_resolved != resolved and folder_resolved not in resolved.parents:
                raise HTTPException(
                    status_code=400, detail="Invalid upload filename"
                )
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


async def enforce_upload_guards(upload_file: UploadFile) -> None:
    """Reject oversized or over-paged uploads before they hit disk/processing.

    Two guards, in order:

    1. **Size** (HTTP 413): rejects files larger than ``MAX_FILE_SIZE_MB``.
       ``UploadFile.size`` is populated by Starlette from the multipart
       part length, so this is checked without reading the body.
    2. **Page count** (HTTP 422): for PDFs only, rejects documents with more
       than ``MAX_PAGE_COUNT`` pages — the OOM vector this phase guards
       against (large scanned PDFs). Page count is read with ``pypdfium2``,
       the same library ``/page-count`` uses, streaming from the spooled
       upload file so the body is never loaded into memory in full.

    The page-count guard degrades gracefully: if ``pypdfium2`` is unavailable
    or the file is not a parseable PDF, it is skipped rather than failing the
    upload (the size guard still applies). This mirrors ``/page-count``'s
    own tolerance for a missing pdfium install.
    """
    max_size_mb = get_max_file_size_mb()
    size_bytes = upload_file.size
    if size_bytes is not None and size_bytes > max_size_mb * 1024 * 1024:
        actual_mb = size_bytes / (1024 * 1024)
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {actual_mb:.1f}MB exceeds the "
                f"{max_size_mb}MB limit (MAX_FILE_SIZE_MB)."
            ),
        )

    filename = upload_file.filename or ""
    if not filename.lower().endswith(".pdf"):
        return

    try:
        import pypdfium2 as pdfium
    except ImportError:
        logger.warning(
            "pypdfium2 unavailable — skipping page-count upload guard"
        )
        return

    # Read the page count straight from the spooled upload file instead of
    # pulling the whole body into memory. Starlette spools the multipart part
    # to a SpooledTemporaryFile (on disk once past ~1MB), and pypdfium2 reads
    # it in blocks through the buffer interface — so a near-limit PDF (e.g.
    # 499MB) never materialises as a single in-memory ``bytes`` object, which
    # is the OOM vector this guard exists to prevent.
    await upload_file.seek(0)
    try:
        pdf = pdfium.PdfDocument(upload_file.file)
        try:
            page_count = len(pdf)
        finally:
            pdf.close()
    except Exception as e:  # noqa: BLE001 — non-PDF / corrupt: skip guard
        logger.warning(
            f"Could not read page count for upload '{filename}' "
            f"({e}); skipping page-count guard"
        )
        await upload_file.seek(0)
        return
    await upload_file.seek(0)  # rewind so save_uploaded_file re-reads from 0

    max_pages = get_max_page_count()
    if page_count > max_pages:
        raise HTTPException(
            status_code=422,
            detail=(
                f"PDF has too many pages: {page_count} exceeds the "
                f"{max_pages}-page limit (MAX_PAGE_COUNT)."
            ),
        )


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
    private: str = Form("false"),
    file: Optional[UploadFile] = File(None),
) -> tuple[SourceCreate, Optional[UploadFile]]:
    """Parse form data into SourceCreate model and return upload file."""
    import json

    def str_to_bool(value: str) -> bool:
        return value.lower() in ("true", "1", "yes", "on")

    embed_bool = str_to_bool(embed)
    delete_source_bool = str_to_bool(delete_source)
    async_processing_bool = str_to_bool(async_processing)
    # Track J.3: the layered privacy flag. ``str_to_bool`` mirrors the other
    # boolean form fields; the resolved bool rides on SourceCreate.private and
    # is threaded onto the persisted source + the process_source command args.
    private_bool = str_to_bool(private)

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
        private=private_bool,
    )

    return source_data, file


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/", response_model=SourceResponse)
@limiter.limit(lambda: f"{get_rate_limit_rpm()}/minute")
async def create_source(
    request: Request,
    response: Response,
    form_data: tuple[SourceCreate, Optional[UploadFile]] = Depends(
        parse_source_form_data
    ),
    source_svc: SourceService = Depends(get_source_service),
    notebook_svc: NotebookService = Depends(get_notebook_service),
    transformation_svc: TransformationService = Depends(
        get_transformation_service
    ),
):
    """Create a new source with support for both JSON and multipart form data.

    The slowapi ``@limiter.limit`` decorator enforces per-IP rate limiting
    (``RATE_LIMIT_RPM``). It requires ``request`` to extract the remote
    address and ``response`` to attach rate-limit headers on success (with
    ``headers_enabled=True``); the ``Retry-After`` header on a tripped limit
    is added by the slowapi exception handler.
    """
    source_data, upload_file = form_data
    return await _create_source_impl(
        source_data, upload_file, source_svc, notebook_svc, transformation_svc
    )


async def _create_source_impl(
    source_data: SourceCreate,
    upload_file: Optional[UploadFile],
    source_svc: SourceService,
    notebook_svc: NotebookService,
    transformation_svc: TransformationService,
):
    """Shared create-source logic for the multipart and JSON endpoints.

    Deliberately **not** decorated with ``@limiter.limit``: the two route
    wrappers own rate-limit accounting and hand slowapi the ``request`` /
    ``response`` objects. Invoking a decorated route function directly (as the
    JSON endpoint used to) re-runs slowapi's success-path header injection
    against a missing ``response`` keyword and raises a 500 — delegating to
    this undecorated impl avoids that and the double rate-limit pass.
    """
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
            # Preflight guards (size + page count) reject oversized or
            # over-paged uploads before anything is written to disk.
            await enforce_upload_guards(upload_file)
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

        # Track J.3: carry the per-document privacy flag through to the worker.
        # The privacy resolver (run during extraction) reads source.private; we
        # also stash it on content_state so the process_source command has it
        # without a re-fetch. Additive — does not touch the I.H1 upload guards.
        content_state["private"] = source_data.private

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
                    "private": source_data.private,
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
                    "private": source_data.private,
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
                    private=bool(getattr(source, "private", False)),
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
            #
            # Track PL.5 — decouple from the shared worker. The sync contract is
            # "the caller gets a fully-processed source back synchronously". We
            # honour that contract WITHOUT occupying the shared single-worker for
            # the whole parse: instead of enqueuing ``process_source`` onto the
            # shared queue and polling its status for up to 300s (which both
            # blocked the request AND starved every other queued job behind the
            # one busy worker slot), we run the SAME DOCUMENT_PARSE handler
            # in-request via the registry. The heavy parse therefore executes in
            # this request's own coroutine — the shared worker stays free to
            # drain other jobs concurrently. The handler still calls
            # ``advance_source`` internally, so the lightweight EMBED→EXTRACT→…
            # auto-chain is enqueued as normal background jobs (correct: those
            # are the steps that *should* run on the worker after the parse).
            #
            # What remains by design: a *sync* caller still blocks for the
            # duration of its own parse (that is the contract — callers who do
            # not want to block use ``async_processing=true``, the async path
            # above). The starvation of the shared queue is gone.
            logger.info("Using sync processing path (in-request, off the shared worker)")

            try:
                source = await source_svc.create(
                    {
                        "title": source_data.title or "Processing...",
                        "topics": [],
                        "private": source_data.private,
                    }
                )

                for nb_id in source_data.notebooks or []:
                    await source_svc.add_to_notebook(source.id, nb_id)

                from app_main.services.command_service import get_registry
                from shared.types.enums import JobType

                payload = {
                    "module_name": "open_notebook",
                    "command_name": "process_source",
                    "source_id": str(source.id),
                    "content_state": content_state,
                    "notebook_ids": source_data.notebooks,
                    "processing_overrides": source_data.processing_overrides,
                    "private": source_data.private,
                }

                try:
                    # Run the DOCUMENT_PARSE handler directly (same funnel as the
                    # worker would use), in this request's coroutine. No shared
                    # queue, no poll. The handler's internal advance_source still
                    # enqueues the downstream EMBED chain as background jobs.
                    await get_registry().execute(JobType.DOCUMENT_PARSE, payload)
                except Exception as proc_err:
                    logger.error(f"Sync processing failed: {proc_err}")
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
                        detail=f"Processing failed: {proc_err}",
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
                    private=bool(getattr(processed_source, "private", False)),
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
@limiter.limit(lambda: f"{get_rate_limit_rpm()}/minute")
async def create_source_json(
    request: Request,
    response: Response,
    source_data: SourceCreate,
    source_svc: SourceService = Depends(get_source_service),
    notebook_svc: NotebookService = Depends(get_notebook_service),
    transformation_svc: TransformationService = Depends(
        get_transformation_service
    ),
):
    """Create a new source using JSON payload (legacy endpoint)."""
    return await _create_source_impl(
        source_data, None, source_svc, notebook_svc, transformation_svc
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
                private=bool(getattr(source, "private", False)),
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
