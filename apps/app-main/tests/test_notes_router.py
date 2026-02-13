"""Tests for the notes router."""

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.notes import router
from app_main.dependencies import get_note_service, get_notebook_service
from app_main.services.note_service import NoteService
from app_main.services.notebook_service import NotebookService
from tests.conftest import make_note, make_notebook


def _make_app(note_svc, notebook_svc=None):
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_note_service] = lambda: note_svc
    if notebook_svc is None:
        notebook_svc = AsyncMock(spec=NotebookService)
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    return TestClient(app)


class TestListNotes:

    def test_list_all_notes(self):
        svc = AsyncMock(spec=NoteService)
        svc.get_all.return_value = [make_note()]
        client = _make_app(svc)

        resp = client.get("/api/notes")

        assert resp.status_code == 200
        assert len(resp.json()) == 1


class TestCreateNote:

    def test_create_note(self):
        svc = AsyncMock(spec=NoteService)
        note = make_note()
        svc.create.return_value = note
        client = _make_app(svc)

        resp = client.post("/api/notes", json={"content": "Hello!"})

        assert resp.status_code == 201
        assert resp.json()["content"] == "Note content here."


class TestGetNote:

    def test_get_note(self):
        svc = AsyncMock(spec=NoteService)
        svc.get.return_value = make_note()
        client = _make_app(svc)

        resp = client.get("/api/notes/note:1")

        assert resp.status_code == 200

    def test_get_note_not_found(self):
        svc = AsyncMock(spec=NoteService)
        svc.get.return_value = None
        client = _make_app(svc)

        resp = client.get("/api/notes/note:missing")

        assert resp.status_code == 404


class TestUpdateNote:

    def test_update_note(self):
        svc = AsyncMock(spec=NoteService)
        svc.get.return_value = make_note()
        svc.update.return_value = make_note(title="Updated")
        client = _make_app(svc)

        resp = client.put("/api/notes/note:1", json={"title": "Updated"})

        assert resp.status_code == 200


class TestDeleteNote:

    def test_delete_note(self):
        svc = AsyncMock(spec=NoteService)
        svc.get.return_value = make_note()
        svc.delete.return_value = True
        client = _make_app(svc)

        resp = client.delete("/api/notes/note:1")

        assert resp.status_code == 204

    def test_delete_note_not_found(self):
        svc = AsyncMock(spec=NoteService)
        svc.get.return_value = None
        client = _make_app(svc)

        resp = client.delete("/api/notes/note:missing")

        assert resp.status_code == 404
