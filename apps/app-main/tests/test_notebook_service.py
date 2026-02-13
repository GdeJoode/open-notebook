"""Tests for NotebookService."""

from unittest.mock import AsyncMock, patch

import pytest

from app_main.services.notebook_service import NotebookService
from tests.conftest import make_notebook


class TestNotebookServiceGet:

    @pytest.mark.asyncio
    async def test_get_delegates_to_repo(self, notebook_repo, source_repo, note_repo):
        nb = make_notebook()
        notebook_repo.get.return_value = nb
        service = NotebookService(notebook_repo, source_repo, note_repo)

        result = await service.get("notebook:test1")

        notebook_repo.get.assert_called_once_with("notebook:test1")
        assert result == nb

    @pytest.mark.asyncio
    async def test_get_returns_none_when_missing(self, notebook_repo, source_repo, note_repo):
        notebook_repo.get.return_value = None
        service = NotebookService(notebook_repo, source_repo, note_repo)

        result = await service.get("notebook:missing")

        assert result is None


class TestNotebookServiceCreate:

    @pytest.mark.asyncio
    async def test_create_passes_name_and_description(self, notebook_repo, source_repo, note_repo):
        nb = make_notebook()
        notebook_repo.create.return_value = nb
        service = NotebookService(notebook_repo, source_repo, note_repo)

        result = await service.create("My NB", "Desc")

        notebook_repo.create.assert_called_once_with({"name": "My NB", "description": "Desc"})
        assert result == nb


class TestNotebookServiceUpdate:

    @pytest.mark.asyncio
    async def test_update_delegates_to_repo(self, notebook_repo, source_repo, note_repo):
        nb = make_notebook(name="Updated")
        notebook_repo.update.return_value = nb
        service = NotebookService(notebook_repo, source_repo, note_repo)

        result = await service.update("notebook:test1", {"name": "Updated"})

        notebook_repo.update.assert_called_once_with("notebook:test1", {"name": "Updated"})
        assert result.name == "Updated"


class TestNotebookServiceDelete:

    @pytest.mark.asyncio
    async def test_delete_delegates_to_repo(self, notebook_repo, source_repo, note_repo):
        notebook_repo.delete.return_value = True
        service = NotebookService(notebook_repo, source_repo, note_repo)

        result = await service.delete("notebook:test1")

        notebook_repo.delete.assert_called_once_with("notebook:test1")
        assert result is True


class TestNotebookServiceGetSources:

    @pytest.mark.asyncio
    async def test_get_sources_delegates(self, notebook_repo, source_repo, note_repo):
        notebook_repo.get_sources.return_value = [{"id": "source:1"}]
        service = NotebookService(notebook_repo, source_repo, note_repo)

        result = await service.get_sources("notebook:test1")

        notebook_repo.get_sources.assert_called_once_with("notebook:test1")
        assert len(result) == 1


class TestNotebookServiceGetNotes:

    @pytest.mark.asyncio
    async def test_get_notes_delegates(self, notebook_repo, source_repo, note_repo):
        notebook_repo.get_notes.return_value = [{"id": "note:1"}]
        service = NotebookService(notebook_repo, source_repo, note_repo)

        result = await service.get_notes("notebook:test1")

        assert len(result) == 1


class TestNotebookServiceGetChatSessions:

    @pytest.mark.asyncio
    async def test_get_chat_sessions_delegates(self, notebook_repo, source_repo, note_repo):
        notebook_repo.get_chat_sessions.return_value = [{"id": "session:1"}]
        service = NotebookService(notebook_repo, source_repo, note_repo)

        result = await service.get_chat_sessions("notebook:test1")

        assert len(result) == 1


class TestNotebookServiceAddSource:

    @pytest.mark.asyncio
    async def test_add_source_creates_relation(self, notebook_repo, source_repo, note_repo):
        service = NotebookService(notebook_repo, source_repo, note_repo)

        with patch("app_main.services.notebook_service.execute_query", new_callable=AsyncMock) as mock_eq:
            mock_eq.return_value = []  # No existing reference
            result = await service.add_source("notebook:1", "source:1")

        assert result is True
        # Called twice: once to check, once to create
        assert mock_eq.call_count == 2

    @pytest.mark.asyncio
    async def test_add_source_skips_if_exists(self, notebook_repo, source_repo, note_repo):
        service = NotebookService(notebook_repo, source_repo, note_repo)

        with patch("app_main.services.notebook_service.execute_query", new_callable=AsyncMock) as mock_eq:
            mock_eq.return_value = [{"existing": True}]  # Already exists
            result = await service.add_source("notebook:1", "source:1")

        assert result is True
        mock_eq.assert_called_once()  # Only the check query


class TestNotebookServiceGetAllWithCounts:

    @pytest.mark.asyncio
    async def test_returns_all_notebooks(self, notebook_repo, source_repo, note_repo):
        service = NotebookService(notebook_repo, source_repo, note_repo)

        with patch("app_main.services.notebook_service.execute_query", new_callable=AsyncMock) as mock_eq:
            mock_eq.return_value = [
                {"id": "notebook:1", "name": "NB1", "archived": False, "source_count": 2, "note_count": 1},
            ]
            result = await service.get_all_with_counts()

        assert len(result) == 1
        assert result[0]["source_count"] == 2
