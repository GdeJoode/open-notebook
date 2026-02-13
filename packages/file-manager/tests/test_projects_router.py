"""Tests for projects router."""

from datetime import datetime

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from file_manager.api.routers import projects as projects_module
from shared.models.file_tracking import Project, TrackedFile
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
def app(mock_project_repo, mock_file_repo, monkeypatch):
    """Create test app with mocked repositories."""
    monkeypatch.setattr(projects_module, "_get_project_repo", lambda: mock_project_repo)
    monkeypatch.setattr(projects_module, "_get_file_repo", lambda: mock_file_repo)
    monkeypatch.setenv("STORAGE_ROOT", "/tmp/test-storage")
    app = FastAPI()
    app.include_router(projects_module.router, prefix="/projects")
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


class TestProjectCRUD:
    """Tests for project CRUD operations."""

    def test_list_projects(self, client, mock_project_repo):
        p1 = make_project(id="project:1", name="Project One", slug="project-one")
        p2 = make_project(id="project:2", name="Project Two", slug="project-two")
        mock_project_repo.get_active_projects.return_value = [p1, p2]

        response = client.get("/projects")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert data["projects"][0]["name"] == "Project One"

    def test_list_projects_include_empty(self, client, mock_project_repo):
        mock_project_repo.get_all.return_value = [make_project()]

        response = client.get("/projects?include_empty=true")
        assert response.status_code == 200
        mock_project_repo.get_all.assert_called_once()

    def test_create_project(self, client, mock_project_repo, temp_dir, monkeypatch):
        monkeypatch.setenv("STORAGE_ROOT", str(temp_dir))
        created = make_project(id="project:new", name="New Project", slug="new-project")
        mock_project_repo.get_by_slug.return_value = None
        mock_project_repo.create.return_value = created

        response = client.post("/projects", json={"name": "New Project"})
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "New Project"
        assert data["slug"] == "new-project"
        mock_project_repo.create.assert_called_once()

    def test_create_project_duplicate_slug(self, client, mock_project_repo):
        mock_project_repo.get_by_slug.return_value = make_project()

        response = client.post("/projects", json={"name": "Test Project"})
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    def test_get_project_by_id(self, client, mock_project_repo):
        project = make_project(id="project:abc")
        mock_project_repo.get.return_value = project

        response = client.get("/projects/project:abc")
        assert response.status_code == 200
        assert response.json()["id"] == "project:abc"

    def test_get_project_not_found(self, client, mock_project_repo):
        mock_project_repo.get.return_value = None
        mock_project_repo.get_by_slug.return_value = None

        response = client.get("/projects/project:nonexistent")
        assert response.status_code == 404

    def test_get_project_by_slug(self, client, mock_project_repo):
        project = make_project(slug="my-project")
        mock_project_repo.get_by_slug.return_value = project

        response = client.get("/projects/by-slug/my-project")
        assert response.status_code == 200
        assert response.json()["slug"] == "my-project"

    def test_update_project(self, client, mock_project_repo):
        project = make_project(id="project:upd")
        updated = make_project(id="project:upd", name="Updated Name")
        mock_project_repo.get.return_value = project
        mock_project_repo.update.return_value = updated

        response = client.patch("/projects/project:upd", json={"name": "Updated Name"})
        assert response.status_code == 200
        assert response.json()["name"] == "Updated Name"

    def test_delete_project(self, client, mock_project_repo):
        project = make_project(id="project:del", base_path="/nonexistent/path")
        mock_project_repo.get.return_value = project

        response = client.delete("/projects/project:del")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        mock_project_repo.delete.assert_called_once_with("project:del")

    def test_delete_project_not_found(self, client, mock_project_repo):
        mock_project_repo.get.return_value = None

        response = client.delete("/projects/project:nope")
        assert response.status_code == 404

    def test_delete_project_with_files(self, client, mock_project_repo, temp_dir):
        # Create actual directory with files for deletion
        project_dir = temp_dir / "proj"
        project_dir.mkdir()
        (project_dir / "file.txt").write_text("content")

        project = make_project(id="project:del2", base_path=str(project_dir))
        mock_project_repo.get.return_value = project

        response = client.delete("/projects/project:del2?delete_files=true")
        assert response.status_code == 200
        data = response.json()
        assert data["files_removed"] == 1
        assert not project_dir.exists()


class TestProjectFiles:
    """Tests for project file operations."""

    def test_list_project_files(self, client, mock_project_repo, mock_file_repo):
        project = make_project(id="project:f1")
        tf = make_tracked_file(id="tracked_file:1", filename="doc.pdf")
        mock_project_repo.get.return_value = project
        mock_file_repo.get_by_project.return_value = [tf]

        response = client.get("/projects/project:f1/files")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["files"][0]["filename"] == "doc.pdf"

    def test_start_processing(self, client, mock_project_repo):
        project = make_project(id="project:proc")
        mock_project_repo.get.return_value = project
        mock_project_repo.update.return_value = make_project(
            id="project:proc", is_processing=True
        )

        response = client.post("/projects/project:proc/processing/start")
        assert response.status_code == 200
        assert response.json()["is_processing"] is True

    def test_complete_processing(self, client, mock_project_repo):
        project = make_project(id="project:proc", is_processing=True)
        completed = make_project(id="project:proc", is_processing=False)
        mock_project_repo.get.return_value = project
        mock_project_repo.update.return_value = completed

        response = client.post("/projects/project:proc/processing/complete")
        assert response.status_code == 200
        assert response.json()["is_processing"] is False


class TestProjectSync:
    """Tests for project sync operations."""

    def test_sync_stats(self, client, mock_project_repo, temp_dir):
        project_dir = temp_dir / "sync_proj"
        project_dir.mkdir()
        (project_dir / "a.txt").write_text("data")
        (project_dir / "b.txt").write_text("more data")

        project = make_project(id="project:sync", base_path=str(project_dir))
        updated = make_project(
            id="project:sync",
            base_path=str(project_dir),
            file_count=2,
            total_size_bytes=13,
        )
        mock_project_repo.get.return_value = project
        mock_project_repo.update.return_value = updated

        response = client.post("/projects/project:sync/sync-stats")
        assert response.status_code == 200
        data = response.json()
        assert data["file_count"] == 2
