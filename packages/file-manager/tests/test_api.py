"""Tests for API routers."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from file_manager.api.routers import files, storage
from file_manager.config import FileManagerConfig, set_config


@pytest.fixture
def app(test_config):
    """Create test FastAPI app."""
    app = FastAPI()
    app.include_router(files.router, prefix="/files", tags=["files"])
    app.include_router(storage.router, prefix="/storage", tags=["storage"])
    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


class TestFilesRouter:
    """Tests for files router."""

    def test_list_uploads_empty(self, client, test_config):
        """Test listing uploads when empty."""
        # Setup storage first
        response = client.post("/storage/setup")
        assert response.status_code == 200

        response = client.get("/files/list/uploads")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0
        assert data["files"] == []

    def test_list_unknown_directory(self, client, test_config):
        """Test listing unknown directory."""
        response = client.get("/files/list/unknown")
        assert response.status_code == 400

    def test_file_info_not_found(self, client, test_config):
        """Test getting info for non-existent file."""
        response = client.get("/files/info", params={"path": "/nonexistent/file.txt"})
        assert response.status_code == 404

    def test_read_file_not_found(self, client, test_config):
        """Test reading non-existent file."""
        response = client.get("/files/read", params={"path": "/nonexistent/file.txt"})
        assert response.status_code == 404

    def test_write_and_read_file(self, client, test_config, temp_dir):
        """Test writing and reading a file."""
        file_path = str(temp_dir / "test_write.txt")

        # Write
        response = client.post(
            "/files/write",
            json={"path": file_path, "content": "Test content"},
        )
        assert response.status_code == 200
        assert response.json()["success"] is True

        # Read
        response = client.get("/files/read", params={"path": file_path})
        assert response.status_code == 200
        assert response.json()["content"] == "Test content"

    def test_delete_file(self, client, test_config, sample_file):
        """Test deleting a file."""
        response = client.delete("/files/delete", params={"path": str(sample_file)})
        assert response.status_code == 200
        assert response.json()["deleted"] is True

    def test_delete_file_not_found(self, client, test_config):
        """Test deleting non-existent file."""
        response = client.delete("/files/delete", params={"path": "/nonexistent/file.txt"})
        assert response.status_code == 404

    def test_validate_file_valid(self, client, test_config):
        """Test validating a valid file extension."""
        response = client.post("/files/validate", params={"path": "document.pdf"})
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["mime_type"] == "application/pdf"

    def test_validate_file_invalid(self, client, test_config):
        """Test validating an invalid file extension."""
        response = client.post("/files/validate", params={"path": "script.exe"})
        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is False
        assert data["error"] is not None


class TestStorageRouter:
    """Tests for storage router."""

    def test_setup_storage(self, client, test_config):
        """Test storage setup."""
        response = client.post("/storage/setup")
        assert response.status_code == 200
        assert "initialized" in response.json()["status"].lower()

    def test_get_config(self, client, test_config):
        """Test getting storage config."""
        response = client.get("/storage/config")
        assert response.status_code == 200
        data = response.json()
        assert "upload_dir" in data
        assert "max_file_size_mb" in data
        assert "allowed_extensions" in data

    def test_get_storage_stats(self, client, test_config):
        """Test getting storage statistics."""
        # Setup first
        client.post("/storage/setup")

        response = client.get("/storage/stats")
        assert response.status_code == 200
        data = response.json()
        assert "uploads" in data
        assert "input" in data
        assert "output" in data

    def test_cleanup_temp(self, client, test_config, temp_dir):
        """Test cleaning up temp files."""
        # Setup first
        client.post("/storage/setup")

        # Create a temp file
        temp_path = temp_dir / "temp" / "temp_file.txt"
        temp_path.write_text("temp content")

        response = client.post("/storage/cleanup-temp")
        assert response.status_code == 200
        assert response.json()["deleted_count"] == 1

    def test_create_document_directory(self, client, test_config):
        """Test creating document directory."""
        # Setup first
        client.post("/storage/setup")

        response = client.post(
            "/storage/document-directory",
            params={"document_name": "test_doc", "add_timestamp": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert "root" in data
        assert "images" in data
        assert "tables" in data

    def test_organize_file_invalid_extension(self, client, test_config, temp_dir):
        """Test organizing file with invalid extension."""
        # Setup first
        client.post("/storage/setup")

        # Create a file with invalid extension
        invalid_file = temp_dir / "script.exe"
        invalid_file.write_text("fake exe")

        response = client.post(
            "/storage/organize",
            json={"file_path": str(invalid_file)},
        )
        assert response.status_code == 400
