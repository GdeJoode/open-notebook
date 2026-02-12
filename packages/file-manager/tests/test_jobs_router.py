"""Tests for jobs router."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from file_manager.api.routers import jobs as jobs_module
from shared.types.enums import JobStatus, JobType


@pytest.fixture(autouse=True)
def reset_jobs_state(monkeypatch):
    """Ensure clean state: no Redis, empty local jobs dict."""
    monkeypatch.setattr(jobs_module, "_redis_available", False)
    monkeypatch.setattr(jobs_module, "_queue_manager", None)
    jobs_module._local_jobs.clear()


@pytest.fixture
def app():
    app = FastAPI()
    app.include_router(jobs_module.router, prefix="/jobs")
    return app


@pytest.fixture
def client(app):
    return TestClient(app)


def _submit_job(client, **overrides):
    """Helper to submit a job and return the response data."""
    payload = {
        "job_type": "document_parse",
        "priority": "normal",
        "file_path": "/tmp/test.pdf",
    }
    payload.update(overrides)
    response = client.post("/jobs", json=payload)
    assert response.status_code == 200
    return response.json()


class TestJobSubmission:
    """Tests for submitting jobs."""

    def test_submit_job(self, client):
        data = _submit_job(client)
        assert data["id"]
        assert data["job_type"] == "document_parse"
        assert data["priority"] == "normal"
        assert data["status"] == "queued"

    def test_submit_job_with_file_id(self, client):
        data = _submit_job(client, file_id="tracked_file:abc", file_path=None)
        assert data["status"] == "queued"

    def test_submit_batch_job(self, client):
        data = _submit_job(
            client,
            job_type="batch_process",
            file_path=None,
            file_paths=["/tmp/a.pdf", "/tmp/b.pdf"],
        )
        assert data["job_type"] == "batch_process"
        assert data["status"] == "queued"

    def test_submit_job_invalid_type(self, client):
        response = client.post(
            "/jobs",
            json={"job_type": "nonexistent_type", "priority": "normal", "file_path": "/f"},
        )
        assert response.status_code == 400
        assert "Invalid job type" in response.json()["detail"]

    def test_submit_job_invalid_priority(self, client):
        response = client.post(
            "/jobs",
            json={"job_type": "document_parse", "priority": "ultra", "file_path": "/f"},
        )
        assert response.status_code == 400
        assert "Invalid priority" in response.json()["detail"]

    def test_submit_job_no_input(self, client):
        response = client.post(
            "/jobs",
            json={"job_type": "document_parse", "priority": "normal"},
        )
        assert response.status_code == 400
        assert "At least one" in response.json()["detail"]


class TestJobRetrieval:
    """Tests for getting job details."""

    def test_get_job(self, client):
        submitted = _submit_job(client)
        job_id = submitted["id"]

        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["id"] == job_id

    def test_get_job_not_found(self, client):
        response = client.get("/jobs/nonexistent-id")
        assert response.status_code == 404

    def test_list_jobs(self, client):
        _submit_job(client, file_path="/tmp/a.pdf")
        _submit_job(client, file_path="/tmp/b.pdf")

        response = client.get("/jobs")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 2
        assert len(data["jobs"]) == 2

    def test_list_jobs_by_status(self, client):
        _submit_job(client, file_path="/tmp/a.pdf")

        response = client.get("/jobs?status=queued")
        assert response.status_code == 200
        assert response.json()["total"] == 1

        response = client.get("/jobs?status=completed")
        assert response.status_code == 200
        assert response.json()["total"] == 0

    def test_list_jobs_by_type(self, client):
        _submit_job(client, file_path="/tmp/a.pdf", job_type="document_parse")
        _submit_job(client, file_path="/tmp/b.pdf", job_type="audio_transcribe")

        response = client.get("/jobs?job_type=document_parse")
        assert response.status_code == 200
        assert response.json()["total"] == 1


class TestJobManagement:
    """Tests for job management operations."""

    def test_cancel_job(self, client):
        submitted = _submit_job(client)
        job_id = submitted["id"]

        response = client.delete(f"/jobs/{job_id}")
        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"

    def test_cancel_job_not_found(self, client):
        response = client.delete("/jobs/nonexistent-id")
        assert response.status_code == 404

    def test_queue_stats(self, client):
        _submit_job(client, file_path="/tmp/a.pdf")
        _submit_job(client, file_path="/tmp/b.pdf")

        response = client.get("/jobs/stats/queue")
        assert response.status_code == 200
        data = response.json()
        assert data["redis_available"] is False
        assert data["queued"] == 2
        assert data["processing"] == 0

    def test_retry_failed_jobs_no_redis(self, client):
        response = client.post("/jobs/failed/retry")
        assert response.status_code == 503
        assert "Redis not available" in response.json()["detail"]
