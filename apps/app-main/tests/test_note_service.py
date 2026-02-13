"""Tests for NoteService."""

import pytest

from app_main.services.note_service import NoteService
from tests.conftest import make_note


class TestNoteServiceGet:

    @pytest.mark.asyncio
    async def test_get_delegates(self, note_repo, notebook_repo):
        note = make_note()
        note_repo.get.return_value = note
        service = NoteService(note_repo, notebook_repo)

        result = await service.get("note:test1")

        note_repo.get.assert_called_once_with("note:test1")
        assert result == note

    @pytest.mark.asyncio
    async def test_get_returns_none(self, note_repo, notebook_repo):
        note_repo.get.return_value = None
        service = NoteService(note_repo, notebook_repo)

        assert await service.get("note:missing") is None


class TestNoteServiceGetAll:

    @pytest.mark.asyncio
    async def test_get_all(self, note_repo, notebook_repo):
        note_repo.get_all.return_value = [make_note()]
        service = NoteService(note_repo, notebook_repo)

        result = await service.get_all()

        assert len(result) == 1


class TestNoteServiceCreate:

    @pytest.mark.asyncio
    async def test_create_simple(self, note_repo, notebook_repo):
        note = make_note()
        note_repo.create.return_value = note
        service = NoteService(note_repo, notebook_repo)

        result = await service.create(title="Title", content="Content")

        note_repo.create.assert_called_once()
        assert result == note

    @pytest.mark.asyncio
    async def test_create_with_notebook(self, note_repo, notebook_repo):
        note = make_note()
        note_repo.create.return_value = note
        service = NoteService(note_repo, notebook_repo)

        await service.create(title="T", content="C", notebook_id="notebook:1")

        note_repo.add_to_notebook.assert_called_once_with("note:test1", "notebook:1")

    @pytest.mark.asyncio
    async def test_create_with_embedding(self, note_repo, notebook_repo):
        note = make_note()
        note_repo.create_with_embedding.return_value = note
        service = NoteService(note_repo, notebook_repo)

        await service.create(title="T", content="C", embedding=[0.1, 0.2])

        note_repo.create_with_embedding.assert_called_once()
        note_repo.create.assert_not_called()


class TestNoteServiceUpdate:

    @pytest.mark.asyncio
    async def test_update(self, note_repo, notebook_repo):
        updated = make_note(title="New Title")
        note_repo.update.return_value = updated
        service = NoteService(note_repo, notebook_repo)

        result = await service.update("note:1", {"title": "New Title"})

        assert result.title == "New Title"


class TestNoteServiceDelete:

    @pytest.mark.asyncio
    async def test_delete(self, note_repo, notebook_repo):
        note_repo.delete.return_value = True
        service = NoteService(note_repo, notebook_repo)

        assert await service.delete("note:1") is True


class TestNoteServiceAddToNotebook:

    @pytest.mark.asyncio
    async def test_add_to_notebook(self, note_repo, notebook_repo):
        note_repo.add_to_notebook.return_value = True
        service = NoteService(note_repo, notebook_repo)

        result = await service.add_to_notebook("note:1", "notebook:1")

        note_repo.add_to_notebook.assert_called_once_with("note:1", "notebook:1")
        assert result is True
