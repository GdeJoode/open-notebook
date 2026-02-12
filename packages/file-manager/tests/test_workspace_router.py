"""Tests for workspace router."""

import os

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from file_manager.api.routers import workspace


@pytest.fixture
def app(temp_dir, monkeypatch):
    """Create test app with workspace router and temp STORAGE_ROOT."""
    monkeypatch.setenv("STORAGE_ROOT", str(temp_dir / "workspace"))
    app = FastAPI()
    app.include_router(workspace.router, prefix="/workspace")
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestWorkspaceConfig:
    """Tests for GET /workspace/config."""

    def test_get_config_returns_paths(self, client, temp_dir):
        response = client.get("/workspace/config")
        assert response.status_code == 200
        data = response.json()
        assert "storage_root" in data
        assert "projects_path" in data
        assert "knowledgebases_path" in data
        assert "temp_path" in data

    def test_config_uses_env_variable(self, client, temp_dir):
        response = client.get("/workspace/config")
        data = response.json()
        assert str(temp_dir / "workspace") == data["storage_root"]

    def test_config_creates_directories(self, client, temp_dir):
        response = client.get("/workspace/config")
        data = response.json()
        assert os.path.isdir(data["projects_path"])
        assert os.path.isdir(data["knowledgebases_path"])
        assert os.path.isdir(data["temp_path"])


class TestWorkspaceStats:
    """Tests for GET /workspace/stats."""

    def test_stats_empty_workspace(self, client):
        response = client.get("/workspace/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["projects_count"] == 0
        assert data["knowledgebases_count"] == 0
        assert data["temp_files_count"] == 0
        assert data["total_size_bytes"] == 0

    def test_stats_with_files(self, client, temp_dir):
        # Initialize workspace first
        client.post("/workspace/initialize")

        # Create a temp file inside the workspace temp directory
        workspace_temp = temp_dir / "workspace" / "temp"
        (workspace_temp / "testfile.txt").write_text("hello")

        response = client.get("/workspace/stats")
        data = response.json()
        assert data["temp_files_count"] == 1
        assert data["total_size_bytes"] > 0


class TestWorkspaceOperations:
    """Tests for workspace operation endpoints."""

    def test_initialize_workspace(self, client, temp_dir):
        response = client.post("/workspace/initialize")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["storage_root"] == str(temp_dir / "workspace")
        assert len(data["created_directories"]) == 3

    def test_list_directories(self, client):
        # Initialize first so directories exist
        client.post("/workspace/initialize")

        response = client.get("/workspace/directories")
        assert response.status_code == 200
        data = response.json()
        assert len(data) == 3
        names = {d["name"] for d in data}
        assert names == {"projects", "knowledgebases", "temp"}
        for d in data:
            assert d["exists"] is True

    def test_cleanup_temp(self, client, temp_dir):
        # Initialize workspace
        client.post("/workspace/initialize")

        # Create temp files
        workspace_temp = temp_dir / "workspace" / "temp"
        (workspace_temp / "a.txt").write_text("data")
        (workspace_temp / "b.txt").write_text("more data")

        response = client.delete("/workspace/temp/cleanup")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["deleted_count"] == 2
        assert data["deleted_size_bytes"] > 0

    def test_set_custom_directory_create(self, client, temp_dir):
        new_dir = str(temp_dir / "workspace" / "custom_dir")
        response = client.post(
            "/workspace/directory",
            json={"path": new_dir, "create_if_missing": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert data["exists"] is True
        assert os.path.isdir(new_dir)

    def test_set_custom_directory_existing(self, client, temp_dir):
        existing_dir = temp_dir / "workspace" / "existing"
        existing_dir.mkdir(parents=True)

        response = client.post(
            "/workspace/directory",
            json={"path": str(existing_dir), "create_if_missing": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "selected"
        assert data["exists"] is True

    def test_set_custom_directory_no_create(self, client, temp_dir):
        nonexistent = str(temp_dir / "workspace" / "no_such_dir")
        response = client.post(
            "/workspace/directory",
            json={"path": nonexistent, "create_if_missing": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "error"
        assert data["exists"] is False
