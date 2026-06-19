"""Tests for the I.D-4 image-list endpoint ``GET /{source_id}/images``.

The ingestion pipeline writes extracted images to
``{output_directory}/output/extracted_info/images/`` but does not persist their
filenames on chunk records, so this endpoint enumerates that directory. These
tests pin the contract the inspect-workspace ImageGallery depends on:

1. Returns the image filenames present on disk (sorted), excluding the
   ``*_description.txt`` sidecars the exporter also writes.
2. Returns an empty list (not 404) when a source has no output directory or no
   images, so the gallery can render a clean empty state.
3. Returns 404 only when the source itself does not exist.

The SourceService is mocked; only the filesystem walk is exercised for real
(against a tmp_path), mirroring the directory layout in sources_files.py.
"""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.sources_files import router
from app_main.dependencies import get_source_service
from app_main.services.source_service import SourceService


def _make_app(source_svc) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api/sources")
    app.dependency_overrides[get_source_service] = lambda: source_svc
    return TestClient(app)


def _svc_returning(source) -> AsyncMock:
    svc = AsyncMock(spec=SourceService)
    svc.get = AsyncMock(return_value=source)
    return svc


def _source_with_output_dir(output_dir: str) -> MagicMock:
    """A source-like object exposing ``output_directory`` (Source itself has no
    such field; the endpoint reads it via getattr)."""
    source = MagicMock()
    source.output_directory = output_dir
    source.asset = None
    return source


def _images_dir(output_dir: Path) -> Path:
    d = output_dir / "output" / "extracted_info" / "images"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_lists_image_filenames_sorted_excluding_sidecars(tmp_path: Path):
    images = _images_dir(tmp_path)
    (images / "image_002_page1.png").write_bytes(b"x")
    (images / "image_001_page1.png").write_bytes(b"x")
    (images / "image_010_page2.jpg").write_bytes(b"x")
    # Sidecar description files must be excluded.
    (images / "image_001_description.txt").write_text("desc")

    client = _make_app(_svc_returning(_source_with_output_dir(str(tmp_path))))
    resp = client.get("/api/sources/source:abc/images")

    assert resp.status_code == 200
    assert resp.json() == {
        "images": [
            "image_001_page1.png",
            "image_002_page1.png",
            "image_010_page2.jpg",
        ]
    }


def test_empty_list_when_no_images_dir(tmp_path: Path):
    client = _make_app(_svc_returning(_source_with_output_dir(str(tmp_path))))
    resp = client.get("/api/sources/source:abc/images")

    assert resp.status_code == 200
    assert resp.json() == {"images": []}


def test_empty_list_when_no_output_directory():
    source = MagicMock()
    source.output_directory = None
    source.asset = None
    client = _make_app(_svc_returning(source))

    resp = client.get("/api/sources/source:abc/images")

    assert resp.status_code == 200
    assert resp.json() == {"images": []}


def test_404_when_source_missing():
    client = _make_app(_svc_returning(None))
    resp = client.get("/api/sources/source:missing/images")

    assert resp.status_code == 404
