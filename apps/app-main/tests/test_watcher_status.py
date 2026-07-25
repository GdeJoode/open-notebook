"""Tests for the G.6 watcher-status endpoint.

Contract test (no DB): the endpoint reflects the env-managed config and derives
recent activity directly from the inbox's _processed/_errors folders.
"""

from __future__ import annotations

from app_main.api.routers import watcher
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _client() -> TestClient:
    app = FastAPI()
    app.include_router(watcher.router, prefix="/api")
    return TestClient(app, raise_server_exceptions=False)


def test_status_reflects_config_and_recent_activity(monkeypatch, tmp_path):
    inbox = tmp_path / "inbox"
    (inbox / "_processed").mkdir(parents=True)
    (inbox / "_errors").mkdir(parents=True)
    (inbox / "_processed" / "done.pdf").write_bytes(b"x")
    (inbox / "_errors" / "bad.mp3").write_bytes(b"x")

    monkeypatch.setenv("FILE_WATCHER_ENABLED", "true")
    monkeypatch.setenv("INBOX_PATHS", str(inbox))
    monkeypatch.setenv("INBOX_DEFAULT_NOTEBOOK_ID", "notebook:d")
    monkeypatch.delenv("INBOX_DEBOUNCE_SECONDS", raising=False)

    resp = _client().get("/api/watcher/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is True
    assert body["paths"] == [str(inbox)]
    assert body["default_notebook_id"] == "notebook:d"
    assert body["debounce_seconds"] == 3  # default
    names = {(e["name"], e["status"]) for e in body["recent"]}
    assert ("done.pdf", "processed") in names
    assert ("bad.mp3", "errored") in names
    # every entry carries an ISO timestamp
    assert all(e["when"] for e in body["recent"])


def test_status_disabled_and_empty(monkeypatch, tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)  # no _processed/_errors yet
    monkeypatch.delenv("FILE_WATCHER_ENABLED", raising=False)
    monkeypatch.setenv("INBOX_PATHS", str(inbox))

    resp = _client().get("/api/watcher/status")
    assert resp.status_code == 200
    body = resp.json()
    assert body["enabled"] is False
    assert body["recent"] == []


def test_status_never_500s_on_bad_path(monkeypatch):
    monkeypatch.setenv("INBOX_PATHS", "/nonexistent/definitely/not/here")
    resp = _client().get("/api/watcher/status")
    assert resp.status_code == 200
    assert resp.json()["recent"] == []
