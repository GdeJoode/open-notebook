"""
API integration: Router → real services → mock repos → real filesystem.

Unlike existing router tests (which mock the entire service layer), these
test the full chain: HTTP → router → real service → mock repo → real
filesystem → HTTP response.
"""

import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import AsyncMock

import pytest
from httpx import ASGITransport, AsyncClient
from fastapi import FastAPI

from file_manager.api.routers import source_folders as sf_module
from file_manager.api.routers.source_folders import router
from file_manager.services.duplicate_detector import (
    DuplicateCheckResult,
    DuplicateDetector,
    DuplicateMatch,
)
from file_manager.services.pipeline_cache import PipelineCacheService
from file_manager.services.source_folder import SourceFolderService

from tests.integration.conftest import make_source_folder

pytestmark = pytest.mark.integration


@pytest.fixture
def api_app(
    source_folder_service: SourceFolderService,
    pipeline_cache_service: PipelineCacheService,
    mock_source_folder_repo: AsyncMock,
) -> FastAPI:
    """FastAPI app with real services wired through monkeypatching."""
    app = FastAPI()
    app.include_router(router, prefix="/source-folders")

    # Build a real DuplicateDetector with the mock repo
    detector = DuplicateDetector(repo=mock_source_folder_repo)

    # Monkeypatch the module-level singleton getters
    sf_module._folder_service = source_folder_service
    sf_module._cache_service = pipeline_cache_service
    sf_module._duplicate_detector = detector

    yield app

    # Reset singletons after test
    sf_module._folder_service = None
    sf_module._cache_service = None
    sf_module._duplicate_detector = None


@pytest.fixture
async def client(api_app: FastAPI) -> AsyncClient:
    """Async HTTP client for the test app."""
    transport = ASGITransport(app=api_app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestSourceFoldersAPI:
    """Full-chain API tests: HTTP → service → repo → filesystem."""

    async def test_create_folder_creates_directories(
        self, client: AsyncClient
    ):
        """POST /source-folders → 200, real directories exist on disk."""
        resp = await client.post(
            "/source-folders",
            json={"title": "API Test Document", "source_type": "document"},
        )
        assert resp.status_code == 200
        data = resp.json()

        assert data["title"] == "API Test Document"
        assert data["source_id"]
        assert data["folder_path"]

        folder_path = Path(data["folder_path"])
        assert folder_path.is_dir()
        assert (folder_path / "original").is_dir()
        assert (folder_path / "ingestion").is_dir()
        assert (folder_path / "pipeline_cache").is_dir()

    async def test_store_original_via_api(
        self, client: AsyncClient, integration_temp_dir: Path
    ):
        """POST /{id}/original → file appears in original/ dir."""
        # Create folder
        resp = await client.post(
            "/source-folders",
            json={"title": "Original API Test"},
        )
        folder_data = resp.json()
        folder_id = folder_data["id"]

        # Create source file
        source_file = integration_temp_dir / "api_test.pdf"
        source_file.write_text("api test pdf content")

        # Store original
        resp = await client.post(
            f"/source-folders/{folder_id}/original",
            json={
                "source_file_path": str(source_file),
                "copy_file": True,
            },
        )
        assert resp.status_code == 200

        # Verify file exists in original/
        folder_path = Path(folder_data["folder_path"])
        original_files = list((folder_path / "original").iterdir())
        assert len(original_files) == 1
        assert "api_test.pdf" in original_files[0].name

    async def test_save_and_get_cache_via_api(self, client: AsyncClient):
        """POST + GET /{id}/cache/step → content round-trips."""
        # Create folder
        resp = await client.post(
            "/source-folders",
            json={"title": "Cache API Test"},
        )
        folder_id = resp.json()["id"]

        # Save cache
        test_data = {"entities": ["Alice", "Bob"], "count": 2}
        resp = await client.post(
            f"/source-folders/{folder_id}/cache/entity_extraction",
            json={"data": test_data, "file_format": "json"},
        )
        assert resp.status_code == 200
        save_data = resp.json()
        assert save_data["pipeline_step"] == "entity_extraction"
        assert save_data["version"] == 1

        # Get cache
        resp = await client.get(
            f"/source-folders/{folder_id}/cache/entity_extraction"
        )
        assert resp.status_code == 200
        get_data = resp.json()
        assert get_data["pipeline_step"] == "entity_extraction"

        content = json.loads(get_data["content"])
        assert content["entities"] == ["Alice", "Bob"]
        assert content["count"] == 2

    async def test_list_cache_entries_via_api(self, client: AsyncClient):
        """Save two steps → GET /{id}/cache → 2 entries."""
        # Create folder
        resp = await client.post(
            "/source-folders",
            json={"title": "List Cache Test"},
        )
        folder_id = resp.json()["id"]

        # Save two different pipeline steps
        await client.post(
            f"/source-folders/{folder_id}/cache/parse",
            json={"data": {"pages": 5}},
        )
        await client.post(
            f"/source-folders/{folder_id}/cache/extract",
            json={"data": {"entities": []}},
        )

        # List
        resp = await client.get(f"/source-folders/{folder_id}/cache")
        assert resp.status_code == 200
        entries = resp.json()
        assert len(entries) == 2

        steps = {e["pipeline_step"] for e in entries}
        assert "parse" in steps
        assert "extract" in steps

    async def test_check_duplicate_via_api(
        self,
        client: AsyncClient,
        mock_source_folder_repo: AsyncMock,
    ):
        """Pre-seed repo → POST /check-duplicate → match detected."""
        # Pre-seed a folder with original_filename for duplicate matching
        folder = make_source_folder(
            original_filename="research_paper.pdf",
            file_size_bytes=50000,
            mime_type="application/pdf",
        )
        mock_source_folder_repo.search_by_filename = AsyncMock(
            return_value=[folder]
        )

        resp = await client.post(
            "/source-folders/check-duplicate",
            json={
                "filename": "research_paper.pdf",
                "file_size_bytes": 50000,
                "mime_type": "application/pdf",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["match_type"] in ("definite", "probable")
        assert len(data["matches"]) >= 1

    async def test_folder_contents_via_api(
        self, client: AsyncClient, integration_temp_dir: Path
    ):
        """GET /{id}/contents → structured listing."""
        # Create folder and add content
        resp = await client.post(
            "/source-folders",
            json={"title": "Contents API Test"},
        )
        folder_data = resp.json()
        folder_id = folder_data["id"]

        # Store a file
        source_file = integration_temp_dir / "contents_api.txt"
        source_file.write_text("contents test")
        await client.post(
            f"/source-folders/{folder_id}/original",
            json={"source_file_path": str(source_file), "copy_file": True},
        )

        # Get contents
        resp = await client.get(f"/source-folders/{folder_id}/contents")
        assert resp.status_code == 200
        contents = resp.json()
        assert "original" in contents
        assert len(contents["original"]) >= 1

    async def test_delete_folder_via_api(self, client: AsyncClient):
        """DELETE /{id} → directory removed from disk."""
        # Create folder
        resp = await client.post(
            "/source-folders",
            json={"title": "Delete API Test"},
        )
        folder_data = resp.json()
        folder_id = folder_data["id"]
        folder_path = Path(folder_data["folder_path"])

        assert folder_path.exists()

        # Delete
        resp = await client.delete(
            f"/source-folders/{folder_id}",
            params={"delete_physical": True},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True

        assert not folder_path.exists()
