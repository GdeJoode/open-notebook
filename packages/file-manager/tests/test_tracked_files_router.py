"""Tests for tracked_files router."""

from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from file_manager.api.routers import tracked_files as tf_module
from shared.models.file_tracking import KnowledgeBase, Project, TrackedFile
from shared.types.enums import FileStatus, StorageLocation


def make_project(**overrides) -> Project:
    defaults = {
        "id": "project:test_id", "name": "Test Project", "slug": "test-project",
        "description": "A test project", "base_path": "/tmp/projects/test-project",
        "input_path": "/tmp/projects/test-project/input",
        "output_path": "/tmp/projects/test-project/output",
        "exports_path": "/tmp/projects/test-project/exports",
        "file_count": 0, "total_size_bytes": 0, "is_processing": False,
        "created": datetime(2025, 1, 1), "updated": datetime(2025, 1, 1),
    }
    defaults.update(overrides)
    return Project(**defaults)


def make_kb(**overrides) -> KnowledgeBase:
    defaults = {
        "id": "knowledge_base:test_id", "name": "Test KB", "slug": "test-kb",
        "description": "A test KB", "base_path": "/tmp/knowledgebases/test-kb",
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
def app(mock_file_repo, mock_project_repo, mock_kb_repo, monkeypatch):
    """Create test app with mocked repositories."""
    monkeypatch.setattr(tf_module, "_get_file_repo", lambda: mock_file_repo)
    monkeypatch.setattr(tf_module, "_get_project_repo", lambda: mock_project_repo)
    monkeypatch.setattr(tf_module, "_get_kb_repo", lambda: mock_kb_repo)
    monkeypatch.setenv("STORAGE_ROOT", "/tmp/test-storage")
    app = FastAPI()
    app.include_router(tf_module.router, prefix="/tracked-files")
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestTrackedFileListing:
    """Tests for listing tracked files."""

    def test_list_all_files(self, client, mock_file_repo):
        f1 = make_tracked_file(id="tracked_file:1", filename="a.pdf")
        f2 = make_tracked_file(id="tracked_file:2", filename="b.pdf")
        mock_file_repo.get_all.return_value = [f1, f2]

        response = client.get("/tracked-files")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2

    def test_list_by_status(self, client, mock_file_repo):
        f1 = make_tracked_file(status=FileStatus.PENDING.value)
        mock_file_repo.get_by_status.return_value = [f1]

        response = client.get("/tracked-files?status=pending")
        assert response.status_code == 200
        assert response.json()["total"] == 1

    def test_list_by_status_invalid(self, client, mock_file_repo):
        response = client.get("/tracked-files?status=bogus")
        assert response.status_code == 400

    def test_list_by_project(self, client, mock_file_repo):
        mock_file_repo.get_by_project.return_value = [make_tracked_file()]

        response = client.get("/tracked-files?project_id=project:123")
        assert response.status_code == 200
        mock_file_repo.get_by_project.assert_called_once_with("project:123")

    def test_list_by_kb(self, client, mock_file_repo):
        mock_file_repo.get_by_knowledge_base.return_value = [make_tracked_file()]

        response = client.get("/tracked-files?kb_id=knowledge_base:456")
        assert response.status_code == 200
        mock_file_repo.get_by_knowledge_base.assert_called_once_with("knowledge_base:456")


class TestTrackedFileCreate:
    """Tests for creating tracked files."""

    def test_create_tracked_file(self, client, mock_file_repo, temp_dir):
        # Create a real file on disk
        test_file = temp_dir / "test.pdf"
        test_file.write_bytes(b"%PDF-1.4 content")

        mock_file_repo.get_by_path.return_value = None
        mock_file_repo.get_by_hash.return_value = None
        created = make_tracked_file(
            filename="test.pdf",
            file_path=str(test_file),
        )
        mock_file_repo.create.return_value = created

        response = client.post(
            "/tracked-files",
            json={"filename": "test.pdf", "file_path": str(test_file)},
        )
        assert response.status_code == 200
        assert response.json()["filename"] == "test.pdf"
        mock_file_repo.create.assert_called_once()

    def test_create_duplicate_path(self, client, mock_file_repo, temp_dir):
        test_file = temp_dir / "dup.pdf"
        test_file.write_bytes(b"%PDF-1.4 dup")

        existing = make_tracked_file(id="tracked_file:existing")
        mock_file_repo.get_by_path.return_value = existing

        response = client.post(
            "/tracked-files",
            json={"filename": "dup.pdf", "file_path": str(test_file)},
        )
        assert response.status_code == 400
        assert "already tracked" in response.json()["detail"]

    def test_create_duplicate_hash(self, client, mock_file_repo, temp_dir):
        test_file = temp_dir / "dup_hash.pdf"
        test_file.write_bytes(b"%PDF-1.4 hash dup")

        mock_file_repo.get_by_path.return_value = None
        mock_file_repo.get_by_hash.return_value = make_tracked_file(id="tracked_file:hash_dup")

        response = client.post(
            "/tracked-files",
            json={"filename": "dup_hash.pdf", "file_path": str(test_file)},
        )
        assert response.status_code == 400
        assert "Duplicate file" in response.json()["detail"]

    def test_create_file_not_exists(self, client, mock_file_repo):
        response = client.post(
            "/tracked-files",
            json={"filename": "ghost.pdf", "file_path": "/nonexistent/ghost.pdf"},
        )
        assert response.status_code == 400
        assert "does not exist" in response.json()["detail"]

    def test_upload_file(self, client, mock_file_repo, temp_dir, monkeypatch):
        monkeypatch.setenv("STORAGE_ROOT", str(temp_dir))

        mock_file_repo.get_by_hash.return_value = None
        created = make_tracked_file(filename="upload.txt")
        mock_file_repo.create.return_value = created

        response = client.post(
            "/tracked-files/upload",
            files={"file": ("upload.txt", b"file content", "text/plain")},
        )
        assert response.status_code == 200
        assert response.json()["filename"] == "upload.txt"


class TestTrackedFileLookup:
    """Tests for looking up tracked files."""

    def test_get_by_id(self, client, mock_file_repo):
        tf = make_tracked_file(id="tracked_file:abc")
        mock_file_repo.get.return_value = tf

        response = client.get("/tracked-files/tracked_file:abc")
        assert response.status_code == 200
        assert response.json()["id"] == "tracked_file:abc"

    def test_get_by_id_not_found(self, client, mock_file_repo):
        mock_file_repo.get.return_value = None

        response = client.get("/tracked-files/tracked_file:nope")
        assert response.status_code == 404

    def test_get_by_path(self, client, mock_file_repo):
        tf = make_tracked_file(file_path="/some/path/doc.pdf")
        mock_file_repo.get_by_path.return_value = tf

        response = client.get("/tracked-files/by-path?path=/some/path/doc.pdf")
        assert response.status_code == 200
        assert response.json()["file_path"] == "/some/path/doc.pdf"

    def test_get_by_path_not_found(self, client, mock_file_repo):
        mock_file_repo.get_by_path.return_value = None

        response = client.get("/tracked-files/by-path?path=/no/such/file")
        assert response.status_code == 404

    def test_get_by_hash(self, client, mock_file_repo):
        tf = make_tracked_file(file_hash="deadbeef")
        mock_file_repo.get_by_hash.return_value = tf

        response = client.get("/tracked-files/by-hash/deadbeef")
        assert response.status_code == 200
        assert response.json()["file_hash"] == "deadbeef"

    def test_get_by_hash_not_found(self, client, mock_file_repo):
        mock_file_repo.get_by_hash.return_value = None

        response = client.get("/tracked-files/by-hash/no_such_hash")
        assert response.status_code == 404


class TestTrackedFileStats:
    """Tests for file statistics."""

    def test_get_stats(self, client, mock_file_repo):
        mock_file_repo.get_stats.return_value = {
            "total_count": 5,
            "total_size": 10240,
            "by_status": {"pending": {"count": 3, "size": 6000}},
        }

        response = client.get("/tracked-files/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] == 5
        assert data["total_size"] == 10240

    def test_get_pending(self, client, mock_file_repo):
        pending = [
            make_tracked_file(id="tracked_file:p1", status=FileStatus.PENDING.value),
            make_tracked_file(id="tracked_file:p2", status=FileStatus.PENDING.value),
        ]
        mock_file_repo.get_pending_files.return_value = pending

        response = client.get("/tracked-files/pending")
        assert response.status_code == 200
        assert response.json()["total"] == 2


class TestTrackedFileOperations:
    """Tests for file operations (status, link, move)."""

    def test_update_status(self, client, mock_file_repo):
        tf = make_tracked_file(id="tracked_file:s1")
        updated = make_tracked_file(id="tracked_file:s1", status=FileStatus.INGESTED.value)
        mock_file_repo.get.return_value = tf
        mock_file_repo.update_status.return_value = updated

        response = client.patch(
            "/tracked-files/tracked_file:s1/status",
            json={"status": "ingested"},
        )
        assert response.status_code == 200
        assert response.json()["status"] == "ingested"

    def test_update_status_invalid(self, client, mock_file_repo):
        tf = make_tracked_file(id="tracked_file:s2")
        mock_file_repo.get.return_value = tf

        response = client.patch(
            "/tracked-files/tracked_file:s2/status",
            json={"status": "banana"},
        )
        assert response.status_code == 400

    def test_update_status_not_found(self, client, mock_file_repo):
        mock_file_repo.get.return_value = None

        response = client.patch(
            "/tracked-files/tracked_file:nope/status",
            json={"status": "ingested"},
        )
        assert response.status_code == 404

    def test_link_to_source(self, client, mock_file_repo):
        tf = make_tracked_file(id="tracked_file:l1")
        linked = make_tracked_file(id="tracked_file:l1", source_id="source:abc")
        mock_file_repo.get.return_value = tf
        mock_file_repo.link_to_source.return_value = linked

        response = client.post(
            "/tracked-files/tracked_file:l1/link-source?source_id=source:abc"
        )
        assert response.status_code == 200
        assert response.json()["source_id"] == "source:abc"

    def test_link_to_transcription(self, client, mock_file_repo):
        tf = make_tracked_file(id="tracked_file:l2")
        linked = make_tracked_file(id="tracked_file:l2", transcription_id="trans:xyz")
        mock_file_repo.get.return_value = tf
        mock_file_repo.link_to_transcription.return_value = linked

        response = client.post(
            "/tracked-files/tracked_file:l2/link-transcription?transcription_id=trans:xyz"
        )
        assert response.status_code == 200
        assert response.json()["transcription_id"] == "trans:xyz"

    def test_move_to_project(self, client, mock_file_repo, mock_project_repo, temp_dir):
        src_file = temp_dir / "moveable.pdf"
        src_file.write_bytes(b"%PDF-1.4 move me")

        tf = make_tracked_file(
            id="tracked_file:m1",
            filename="moveable.pdf",
            file_path=str(src_file),
            file_size_bytes=16,
        )
        project = make_project(
            id="project:target",
            input_path=str(temp_dir / "project_input"),
        )
        moved = make_tracked_file(
            id="tracked_file:m1",
            project_id="project:target",
            storage_location=StorageLocation.PROJECT.value,
        )

        mock_file_repo.get.return_value = tf
        mock_project_repo.get.return_value = project
        mock_file_repo.move_to_project.return_value = moved
        mock_file_repo.update.return_value = moved

        response = client.post(
            "/tracked-files/tracked_file:m1/move",
            json={"target_type": "project", "target_id": "project:target"},
        )
        assert response.status_code == 200
        mock_file_repo.move_to_project.assert_called_once()

    def test_move_invalid_target_type(self, client, mock_file_repo):
        tf = make_tracked_file(id="tracked_file:m_bad")
        mock_file_repo.get.return_value = tf

        response = client.post(
            "/tracked-files/tracked_file:m_bad/move",
            json={"target_type": "invalid", "target_id": "xxx"},
        )
        assert response.status_code == 400


class TestTrackedFileDelete:
    """Tests for deleting tracked files."""

    def test_delete_file_record_only(self, client, mock_file_repo):
        tf = make_tracked_file(id="tracked_file:d1", file_path="/some/file.pdf")
        mock_file_repo.get.return_value = tf

        response = client.delete("/tracked-files/tracked_file:d1")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["physical_deleted"] is False
        mock_file_repo.delete.assert_called_once_with("tracked_file:d1")

    def test_delete_file_with_physical(self, client, mock_file_repo, temp_dir):
        physical_file = temp_dir / "deleteme.pdf"
        physical_file.write_bytes(b"%PDF-1.4 delete me")

        tf = make_tracked_file(
            id="tracked_file:d2",
            file_path=str(physical_file),
            file_size_bytes=18,
        )
        mock_file_repo.get.return_value = tf

        response = client.delete("/tracked-files/tracked_file:d2?delete_physical=true")
        assert response.status_code == 200
        data = response.json()
        assert data["physical_deleted"] is True
        assert not physical_file.exists()

    def test_delete_file_not_found(self, client, mock_file_repo):
        mock_file_repo.get.return_value = None

        response = client.delete("/tracked-files/tracked_file:nope")
        assert response.status_code == 404
