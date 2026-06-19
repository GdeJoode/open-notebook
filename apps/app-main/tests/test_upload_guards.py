"""Tests for upload guards (Track I, Phase I.H1).

Covers the size (413) and page-count (422) preflight guards added to the
source-upload endpoint, plus the two pass-through cases (a valid small PDF,
and a non-PDF / no-file payload). Every service the endpoint touches is
mocked so the guards are exercised in isolation — the guards must reject
*before* any DB or job-queue work happens, and these tests assert that by
asserting the source service is never asked to create anything on rejection.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from app_main.api.rate_limit import limiter
from app_main.api.routers.sources_upload import router
from app_main.dependencies import (
    get_notebook_service,
    get_source_service,
    get_transformation_service,
)
from app_main.services.notebook_service import NotebookService
from app_main.services.source_service import SourceService
from app_main.services.transformation_service import TransformationService
from fastapi import FastAPI
from fastapi.testclient import TestClient

_FIXTURES = Path(__file__).parent / "fixtures"
_CLEAN_PDF = _FIXTURES / "synthetic_clean_text.pdf"


def _make_client(source_svc: AsyncMock) -> TestClient:
    """Wire a minimal app exposing the upload router with mocked services.

    `app.state.limiter` is set because the upload endpoints carry the
    `@limiter.limit` decorator; without it slowapi raises at request time.
    The slowapi middleware itself is intentionally *not* added here so the
    rate limit never trips during guard tests (a generous limit is set via
    env in the fixture, but skipping the middleware keeps these tests
    independent of the limiter entirely).
    """
    app = FastAPI()
    app.state.limiter = limiter
    app.include_router(router, prefix="/sources")

    notebook_svc = AsyncMock(spec=NotebookService)
    transformation_svc = AsyncMock(spec=TransformationService)

    app.dependency_overrides[get_source_service] = lambda: source_svc
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    app.dependency_overrides[get_transformation_service] = (
        lambda: transformation_svc
    )
    return TestClient(app)


@pytest.fixture(autouse=True)
def _generous_rate_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the rate limit out of the way for guard tests."""
    monkeypatch.setenv("RATE_LIMIT_RPM", "100000")


def test_oversize_file_rejected_with_413(monkeypatch: pytest.MonkeyPatch):
    """A file larger than MAX_FILE_SIZE_MB is rejected with 413."""
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "1")
    source_svc = AsyncMock(spec=SourceService)
    client = _make_client(source_svc)

    big = b"x" * (2 * 1024 * 1024)  # 2MB > 1MB limit
    resp = client.post(
        "/sources/",
        data={"type": "upload"},
        files={"file": ("big.bin", big, "application/octet-stream")},
    )

    assert resp.status_code == 413
    detail = resp.json()["detail"]
    assert "limit" in detail.lower()
    assert "MAX_FILE_SIZE_MB" in detail
    # Guard rejected before any source creation happened.
    source_svc.create.assert_not_called()


def test_oversize_pages_rejected_with_422(monkeypatch: pytest.MonkeyPatch):
    """A PDF with more pages than MAX_PAGE_COUNT is rejected with 422."""
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "500")
    monkeypatch.setenv("MAX_PAGE_COUNT", "10")
    source_svc = AsyncMock(spec=SourceService)
    client = _make_client(source_svc)

    # Mock pypdfium so we don't need a genuinely 11-page fixture.
    fake_pdf = list(range(11))  # len() == 11 > 10 limit
    with patch("pypdfium2.PdfDocument", return_value=fake_pdf):
        resp = client.post(
            "/sources/",
            data={"type": "upload"},
            files={"file": ("doc.pdf", b"%PDF-1.4 fake", "application/pdf")},
        )

    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert "MAX_PAGE_COUNT" in detail
    assert "11" in detail
    source_svc.create.assert_not_called()


def test_valid_pdf_passes_guards(monkeypatch: pytest.MonkeyPatch):
    """A small in-limit PDF passes both guards and reaches processing.

    We stop the request at `save_uploaded_file` (patched) and assert the
    guards let it through — i.e. the request proceeds past the preflight to
    the upload-handling branch. A failure here would surface as a 413/422.
    """
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "500")
    monkeypatch.setenv("MAX_PAGE_COUNT", "500")
    source_svc = AsyncMock(spec=SourceService)
    client = _make_client(source_svc)

    pdf_bytes = _CLEAN_PDF.read_bytes()

    # Patch save_uploaded_file so we don't touch disk, and short-circuit the
    # downstream command submission with a controlled failure that proves we
    # got *past* the guards (a 500, not a 413/422).
    with patch(
        "app_main.api.routers.sources_upload.save_uploaded_file",
        new=AsyncMock(side_effect=RuntimeError("stop-after-guards")),
    ):
        resp = client.post(
            "/sources/",
            data={"type": "upload"},
            files={"file": ("clean.pdf", pdf_bytes, "application/pdf")},
        )

    # The guards passed; failure (if any) comes from the patched save step,
    # which the endpoint maps to a 400 "File upload failed" — never 413/422.
    assert resp.status_code not in (413, 422)
    assert resp.status_code == 400
    assert "File upload failed" in resp.json()["detail"]


def test_non_pdf_skips_page_guard(monkeypatch: pytest.MonkeyPatch):
    """A non-PDF upload skips the page-count guard (size guard still applies).

    Mirrors the "missing pdf" scenario: when the upload is not a PDF the
    page-count guard must not run (no pypdfium call) and must not reject;
    only the size guard is relevant. We assert pypdfium is never invoked.
    """
    monkeypatch.setenv("MAX_FILE_SIZE_MB", "500")
    monkeypatch.setenv("MAX_PAGE_COUNT", "1")  # would reject any PDF
    source_svc = AsyncMock(spec=SourceService)
    client = _make_client(source_svc)

    with patch("pypdfium2.PdfDocument") as mock_pdf, patch(
        "app_main.api.routers.sources_upload.save_uploaded_file",
        new=AsyncMock(side_effect=RuntimeError("stop-after-guards")),
    ):
        resp = client.post(
            "/sources/",
            data={"type": "upload"},
            files={"file": ("notes.txt", b"plain text body", "text/plain")},
        )

    # Page guard skipped for non-PDF → pypdfium never touched, not a 422.
    mock_pdf.assert_not_called()
    assert resp.status_code != 422
    assert resp.status_code == 400  # stopped at the patched save step
