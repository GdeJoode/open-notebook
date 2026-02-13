"""Tests for knowledge_bases router."""

from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from file_manager.api.routers import knowledge_bases as kb_module
from shared.models.file_tracking import KnowledgeBase, TrackedFile
from shared.types.enums import FileStatus, StorageLocation


def make_kb(**overrides) -> KnowledgeBase:
    defaults = {
        "id": "knowledge_base:test_id", "name": "Test KB", "slug": "test-kb",
        "description": "A test knowledge base", "base_path": "/tmp/knowledgebases/test-kb",
        "sources_path": "/tmp/knowledgebases/test-kb/sources",
        "exports_path": "/tmp/knowledgebases/test-kb/exports",
        "media_path": "/tmp/knowledgebases/test-kb/media",
        "file_count": 0, "total_size_bytes": 0,
        "created": datetime(2025, 1, 1), "updated": datetime(2025, 1, 1),
    }
    defaults.update(overrides)
    return KnowledgeBase(**defaults)


def make_tracked_file(**overrides) -> TrackedFile:
    defaults = {
        "id": "tracked_file:test_id", "filename": "document.pdf",
        "file_path": "/tmp/test/document.pdf", "file_hash": "abc123hash",
        "file_size_bytes": 1024, "mime_type": "application/pdf",
        "status": FileStatus.PENDING.value, "storage_location": StorageLocation.TEMP.value,
        "created": datetime(2025, 1, 1), "updated": datetime(2025, 1, 1),
    }
    defaults.update(overrides)
    return TrackedFile(**defaults)


@pytest.fixture
def app(mock_kb_repo, mock_file_repo, monkeypatch):
    """Create test app with mocked repositories."""
    monkeypatch.setattr(kb_module, "_get_kb_repo", lambda: mock_kb_repo)
    monkeypatch.setattr(kb_module, "_get_file_repo", lambda: mock_file_repo)
    monkeypatch.setenv("STORAGE_ROOT", "/tmp/test-storage")
    app = FastAPI()
    app.include_router(kb_module.router, prefix="/knowledge-bases")
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestKnowledgeBaseCRUD:
    """Tests for knowledge base CRUD operations."""

    def test_list_knowledge_bases(self, client, mock_kb_repo):
        kb1 = make_kb(id="knowledge_base:1", name="KB One", slug="kb-one")
        kb2 = make_kb(id="knowledge_base:2", name="KB Two", slug="kb-two")
        mock_kb_repo.get_all.return_value = [kb1, kb2]

        response = client.get("/knowledge-bases")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert data["knowledge_bases"][0]["name"] == "KB One"

    def test_create_knowledge_base(self, client, mock_kb_repo, temp_dir, monkeypatch):
        monkeypatch.setenv("STORAGE_ROOT", str(temp_dir))
        created = make_kb(id="knowledge_base:new", name="New KB", slug="new-kb")
        mock_kb_repo.get_by_slug.return_value = None
        mock_kb_repo.create.return_value = created

        response = client.post("/knowledge-bases", json={"name": "New KB"})
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "New KB"
        assert data["slug"] == "new-kb"
        mock_kb_repo.create.assert_called_once()

    def test_create_knowledge_base_duplicate_slug(self, client, mock_kb_repo):
        mock_kb_repo.get_by_slug.return_value = make_kb()

        response = client.post("/knowledge-bases", json={"name": "Test KB"})
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    def test_get_default_kb(self, client, mock_kb_repo):
        default_kb = make_kb(slug="_default", name="Default")
        mock_kb_repo.get_or_create_default.return_value = default_kb

        response = client.get("/knowledge-bases/default")
        assert response.status_code == 200
        assert response.json()["slug"] == "_default"

    def test_get_kb_by_id(self, client, mock_kb_repo):
        kb = make_kb(id="knowledge_base:abc")
        mock_kb_repo.get.return_value = kb

        response = client.get("/knowledge-bases/knowledge_base:abc")
        assert response.status_code == 200
        assert response.json()["id"] == "knowledge_base:abc"

    def test_get_kb_not_found(self, client, mock_kb_repo):
        mock_kb_repo.get.return_value = None
        mock_kb_repo.get_by_slug.return_value = None

        response = client.get("/knowledge-bases/knowledge_base:nope")
        assert response.status_code == 404

    def test_get_kb_by_slug(self, client, mock_kb_repo):
        kb = make_kb(slug="my-kb")
        mock_kb_repo.get_by_slug.return_value = kb

        response = client.get("/knowledge-bases/by-slug/my-kb")
        assert response.status_code == 200
        assert response.json()["slug"] == "my-kb"

    def test_update_kb(self, client, mock_kb_repo):
        kb = make_kb(id="knowledge_base:upd")
        updated = make_kb(id="knowledge_base:upd", name="Updated KB")
        mock_kb_repo.get.return_value = kb
        mock_kb_repo.update.return_value = updated

        response = client.patch(
            "/knowledge-bases/knowledge_base:upd",
            json={"name": "Updated KB"},
        )
        assert response.status_code == 200
        assert response.json()["name"] == "Updated KB"

    def test_delete_kb(self, client, mock_kb_repo):
        kb = make_kb(id="knowledge_base:del", slug="not-default")
        mock_kb_repo.get.return_value = kb

        response = client.delete("/knowledge-bases/knowledge_base:del")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_kb_repo.delete.assert_called_once()

    def test_delete_default_kb_forbidden(self, client, mock_kb_repo):
        kb = make_kb(id="knowledge_base:def", slug="_default")
        mock_kb_repo.get.return_value = kb

        response = client.delete("/knowledge-bases/knowledge_base:def")
        assert response.status_code == 400
        assert "default" in response.json()["detail"].lower()

    def test_delete_kb_not_found(self, client, mock_kb_repo):
        mock_kb_repo.get.return_value = None

        response = client.delete("/knowledge-bases/knowledge_base:nope")
        assert response.status_code == 404

    def test_delete_kb_with_files(self, client, mock_kb_repo, temp_dir):
        kb_dir = temp_dir / "kb_del"
        kb_dir.mkdir()
        (kb_dir / "file.txt").write_text("content")

        kb = make_kb(id="knowledge_base:del2", slug="deletable", base_path=str(kb_dir))
        mock_kb_repo.get.return_value = kb

        response = client.delete("/knowledge-bases/knowledge_base:del2?delete_files=true")
        assert response.status_code == 200
        data = response.json()
        assert data["files_removed"] == 1
        assert not kb_dir.exists()


class TestKnowledgeBaseFiles:
    """Tests for KB file operations."""

    def test_list_kb_files(self, client, mock_kb_repo, mock_file_repo):
        kb = make_kb(id="knowledge_base:f1")
        tf = make_tracked_file(id="tracked_file:1", filename="notes.pdf")
        mock_kb_repo.get.return_value = kb
        mock_file_repo.get_by_knowledge_base.return_value = [tf]

        response = client.get("/knowledge-bases/knowledge_base:f1/files")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["files"][0]["filename"] == "notes.pdf"

    def test_sync_kb_stats(self, client, mock_kb_repo, temp_dir):
        kb_dir = temp_dir / "sync_kb"
        kb_dir.mkdir()
        (kb_dir / "a.txt").write_text("data")

        kb = make_kb(id="knowledge_base:sync", base_path=str(kb_dir))
        updated = make_kb(
            id="knowledge_base:sync",
            base_path=str(kb_dir),
            file_count=1,
            total_size_bytes=4,
        )
        mock_kb_repo.get.return_value = kb
        mock_kb_repo.update.return_value = updated

        response = client.post("/knowledge-bases/knowledge_base:sync/sync-stats")
        assert response.status_code == 200
        assert response.json()["file_count"] == 1
