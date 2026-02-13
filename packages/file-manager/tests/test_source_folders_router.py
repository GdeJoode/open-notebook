"""Tests for source_folders router."""

import json
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from file_manager.api.routers import source_folders as sf_module
from file_manager.services.duplicate_detector import DuplicateCheckResult, DuplicateMatch
from shared.models.file_tracking import PipelineCacheEntry, SourceFolder


def make_source_folder(**overrides) -> SourceFolder:
    defaults = {
        "id": "source_folder:test_id",
        "source_id": "Ab3kX9q2",
        "title": "Test Document",
        "slug": "Test_Document",
        "folder_path": "/tmp/sources/Test_Document_Ab3kX9q2",
        "source_type": "document",
        "total_cache_entries": 0,
        "total_folder_size_bytes": 0,
        "metadata": {},
        "created": datetime(2025, 1, 1),
        "updated": datetime(2025, 1, 1),
    }
    defaults.update(overrides)
    return SourceFolder(**defaults)


def make_cache_entry(**overrides) -> PipelineCacheEntry:
    defaults = {
        "id": "pipeline_cache:test_id",
        "source_folder_id": "source_folder:test_id",
        "source_id": "Ab3kX9q2",
        "pipeline_step": "docling_parse",
        "version": 1,
        "is_current": True,
        "file_path": "/tmp/cache/Ab3kX9q2_docling_parse.json",
        "file_format": "json",
        "file_size_bytes": 512,
        "metadata": {},
        "created": datetime(2025, 1, 1),
        "updated": datetime(2025, 1, 1),
    }
    defaults.update(overrides)
    return PipelineCacheEntry(**defaults)


@pytest.fixture
def mock_folder_service():
    return AsyncMock()


@pytest.fixture
def mock_cache_service():
    return AsyncMock()


@pytest.fixture
def mock_duplicate_detector():
    return AsyncMock()


@pytest.fixture
def app(mock_folder_service, mock_cache_service, mock_duplicate_detector, monkeypatch):
    monkeypatch.setattr(sf_module, "_get_folder_service", lambda: mock_folder_service)
    monkeypatch.setattr(sf_module, "_get_cache_service", lambda: mock_cache_service)
    monkeypatch.setattr(sf_module, "_get_duplicate_detector", lambda: mock_duplicate_detector)
    app = FastAPI()
    app.include_router(sf_module.router, prefix="/source-folders")
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestSourceFolderCRUD:
    def test_create_source_folder(self, client, mock_folder_service):
        folder = make_source_folder()
        mock_folder_service.create_folder.return_value = folder

        response = client.post("/source-folders", json={
            "title": "Test Document",
            "source_type": "document",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["source_id"] == "Ab3kX9q2"
        assert data["title"] == "Test Document"

    def test_create_source_folder_with_metadata(self, client, mock_folder_service):
        folder = make_source_folder(metadata={"tags": ["test"]})
        mock_folder_service.create_folder.return_value = folder

        response = client.post("/source-folders", json={
            "title": "Test Doc",
            "source_type": "document",
            "original_filename": "doc.pdf",
            "mime_type": "application/pdf",
            "file_size_bytes": 1024,
            "metadata": {"tags": ["test"]},
        })
        assert response.status_code == 200

    def test_list_source_folders(self, client, mock_folder_service):
        folders = [
            make_source_folder(id=f"source_folder:{i}", source_id=f"id_{i}")
            for i in range(3)
        ]
        mock_folder_service.list_folders.return_value = folders

        response = client.get("/source-folders")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["folders"]) == 3

    def test_list_source_folders_pagination(self, client, mock_folder_service):
        mock_folder_service.list_folders.return_value = []

        response = client.get("/source-folders?limit=10&offset=5")
        assert response.status_code == 200
        mock_folder_service.list_folders.assert_called_once_with(limit=10, offset=5)

    def test_get_source_folder(self, client, mock_folder_service):
        folder = make_source_folder()
        mock_folder_service.get_folder.return_value = folder

        response = client.get("/source-folders/source_folder:test_id")
        assert response.status_code == 200
        assert response.json()["source_id"] == "Ab3kX9q2"

    def test_get_source_folder_not_found(self, client, mock_folder_service):
        mock_folder_service.get_folder.return_value = None

        response = client.get("/source-folders/source_folder:nonexistent")
        assert response.status_code == 404

    def test_get_folder_by_source_id(self, client, mock_folder_service):
        folder = make_source_folder()
        mock_folder_service.get_folder_by_source_id.return_value = folder

        response = client.get("/source-folders/by-source-id/Ab3kX9q2")
        assert response.status_code == 200
        assert response.json()["title"] == "Test Document"

    def test_get_folder_by_source_id_not_found(self, client, mock_folder_service):
        mock_folder_service.get_folder_by_source_id.return_value = None

        response = client.get("/source-folders/by-source-id/nonexist")
        assert response.status_code == 404

    def test_get_folder_contents(self, client, mock_folder_service):
        mock_folder_service.get_folder_contents.return_value = {
            "original": [{"name": "Ab3kX9q2_doc.pdf", "size_bytes": 1024}],
            "pipeline_cache": [],
        }

        response = client.get("/source-folders/source_folder:test_id/contents")
        assert response.status_code == 200
        data = response.json()
        assert "original" in data

    def test_store_original(self, client, mock_folder_service):
        folder = make_source_folder(
            original_filename="doc.pdf",
            original_file_path="/tmp/sources/Test_Ab3kX9q2/original/Ab3kX9q2_doc.pdf",
        )
        mock_folder_service.store_original.return_value = folder

        response = client.post(
            "/source-folders/source_folder:test_id/original",
            json={"source_file_path": "/tmp/input/doc.pdf", "copy_file": True},
        )
        assert response.status_code == 200
        assert response.json()["original_filename"] == "doc.pdf"

    def test_store_original_file_not_found(self, client, mock_folder_service):
        mock_folder_service.store_original.side_effect = FileNotFoundError("Not found")

        response = client.post(
            "/source-folders/source_folder:test_id/original",
            json={"source_file_path": "/nonexistent/file.pdf", "copy_file": True},
        )
        assert response.status_code == 400

    def test_link_to_source(self, client, mock_folder_service):
        folder = make_source_folder(source_record_id="source:abc")
        mock_folder_service.link_to_source.return_value = folder

        response = client.post(
            "/source-folders/source_folder:test_id/link-source",
            json={"source_record_id": "source:abc"},
        )
        assert response.status_code == 200
        assert response.json()["source_record_id"] == "source:abc"

    def test_delete_source_folder(self, client, mock_folder_service):
        mock_folder_service.delete_folder.return_value = True

        response = client.delete("/source-folders/source_folder:test_id")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_delete_source_folder_not_found(self, client, mock_folder_service):
        mock_folder_service.delete_folder.return_value = False

        response = client.delete("/source-folders/source_folder:nonexistent")
        assert response.status_code == 404


class TestDuplicateCheck:
    def test_check_duplicate_none(self, client, mock_duplicate_detector):
        mock_duplicate_detector.check_duplicate.return_value = DuplicateCheckResult(
            match_type="none",
            message="No duplicates detected.",
            matches=[],
        )

        response = client.post(
            "/source-folders/check-duplicate",
            json={"filename": "unique.pdf"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["match_type"] == "none"
        assert len(data["matches"]) == 0

    def test_check_duplicate_definite(self, client, mock_duplicate_detector):
        folder = make_source_folder(original_filename="report.pdf")
        mock_duplicate_detector.check_duplicate.return_value = DuplicateCheckResult(
            match_type="definite",
            message="Definite duplicate",
            matches=[
                DuplicateMatch(
                    source_folder=folder,
                    similarity_score=1.0,
                    matched_criteria=["filename", "file_size", "mime_type"],
                )
            ],
        )

        response = client.post(
            "/source-folders/check-duplicate",
            json={
                "filename": "report.pdf",
                "file_size_bytes": 1024,
                "mime_type": "application/pdf",
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["match_type"] == "definite"
        assert len(data["matches"]) == 1
        assert data["matches"][0]["source_id"] == "Ab3kX9q2"


class TestCacheEndpoints:
    def test_save_pipeline_cache(self, client, mock_cache_service):
        entry = make_cache_entry()
        mock_cache_service.save_output.return_value = entry

        response = client.post(
            "/source-folders/source_folder:test_id/cache/docling_parse",
            json={"data": {"parsed": True}, "file_format": "json"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["pipeline_step"] == "docling_parse"
        assert data["version"] == 1

    def test_save_pipeline_cache_not_found(self, client, mock_cache_service):
        mock_cache_service.save_output.side_effect = ValueError("Not found")

        response = client.post(
            "/source-folders/source_folder:bad_id/cache/step",
            json={"data": "test"},
        )
        assert response.status_code == 404

    def test_get_pipeline_cache(self, client, mock_cache_service):
        mock_cache_service.get_output.return_value = '{"parsed": true}'
        entry = make_cache_entry()
        mock_cache_service.get_cache_entry.return_value = entry

        response = client.get(
            "/source-folders/source_folder:test_id/cache/docling_parse"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["pipeline_step"] == "docling_parse"
        assert data["content"] == '{"parsed": true}'

    def test_get_pipeline_cache_not_found(self, client, mock_cache_service):
        mock_cache_service.get_output.return_value = None

        response = client.get(
            "/source-folders/source_folder:test_id/cache/nonexistent"
        )
        assert response.status_code == 404

    def test_list_cache_entries(self, client, mock_cache_service):
        entries = [
            make_cache_entry(id=f"pipeline_cache:{i}", pipeline_step=f"step_{i}")
            for i in range(3)
        ]
        mock_cache_service.list_cache_entries.return_value = entries

        response = client.get("/source-folders/source_folder:test_id/cache")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3

    def test_get_cache_version_history(self, client, mock_cache_service):
        entries = [
            make_cache_entry(version=2, is_current=True),
            make_cache_entry(id="pipeline_cache:old", version=1, is_current=False),
        ]
        mock_cache_service.get_version_history.return_value = entries

        response = client.get(
            "/source-folders/source_folder:test_id/cache/docling_parse/history"
        )
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["version"] == 2
        assert data[1]["version"] == 1
