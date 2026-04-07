"""Sources files sub-router — file download, PDF serving, and image endpoints."""

import io
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import FileResponse, Response
from loguru import logger

from app_main.config import UPLOADS_FOLDER
from app_main.dependencies import get_source_service
from app_main.services.source_service import SourceService

router = APIRouter()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


@router.get("/{source_id}/page-preview")
async def get_page_preview(
    source_id: str,
    page: int = Query(1, ge=1, description="1-based page number"),
    dpi: int = Query(150, ge=72, le=300, description="Render DPI"),
    source_svc: SourceService = Depends(get_source_service),
):
    """Render a single PDF page as a PNG image for the document viewer.

    Uses pypdfium2 (already installed as a Docling dependency) to
    rasterize pages server-side, avoiding pdf.js complexity on the
    frontend.
    """
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="pypdfium2 not installed — cannot render page previews",
        )

    try:
        source = await source_svc.get(source_id)
        if not source or not source.asset or not source.asset.file_path:
            raise HTTPException(status_code=404, detail="PDF file not found")

        file_path = source.asset.file_path
        safe_root = os.path.realpath(UPLOADS_FOLDER)
        resolved_path = os.path.realpath(file_path)

        if not resolved_path.startswith(safe_root):
            raise HTTPException(status_code=403, detail="Access to file denied")
        if not os.path.exists(resolved_path):
            raise HTTPException(status_code=404, detail="PDF file has been deleted")
        if not resolved_path.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail="Source file is not a PDF")

        pdf = pdfium.PdfDocument(resolved_path)
        if page > len(pdf):
            raise HTTPException(
                status_code=404,
                detail=f"Page {page} not found (document has {len(pdf)} pages)",
            )

        pdf_page = pdf[page - 1]  # 0-indexed
        scale = dpi / 72  # PDF points are 1/72 inch
        bitmap = pdf_page.render(scale=scale)
        pil_image = bitmap.to_pil()

        buf = io.BytesIO()
        pil_image.save(buf, format="PNG", optimize=True)
        buf.seek(0)

        return Response(
            content=buf.getvalue(),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error rendering page preview for source {source_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error rendering page: {str(e)}")


@router.get("/{source_id}/page-count")
async def get_page_count(
    source_id: str,
    source_svc: SourceService = Depends(get_source_service),
):
    """Return the number of pages and page dimensions for a PDF source."""
    try:
        import pypdfium2 as pdfium
    except ImportError:
        raise HTTPException(status_code=501, detail="pypdfium2 not installed")

    try:
        source = await source_svc.get(source_id)
        if not source or not source.asset or not source.asset.file_path:
            raise HTTPException(status_code=404, detail="PDF file not found")

        file_path = source.asset.file_path
        safe_root = os.path.realpath(UPLOADS_FOLDER)
        resolved_path = os.path.realpath(file_path)

        if not resolved_path.startswith(safe_root):
            raise HTTPException(status_code=403, detail="Access to file denied")
        if not os.path.exists(resolved_path):
            raise HTTPException(status_code=404, detail="PDF file has been deleted")

        pdf = pdfium.PdfDocument(resolved_path)
        pages = []
        for i in range(len(pdf)):
            p = pdf[i]
            pages.append({
                "page_number": i + 1,
                "width": p.get_width(),
                "height": p.get_height(),
            })

        return {"page_count": len(pdf), "pages": pages}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting page count for source {source_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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
