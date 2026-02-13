"""Tests for InsightService."""

import pytest

from app_main.services.insight_service import InsightService
from tests.conftest import make_insight, make_note, make_source


class TestInsightServiceGet:

    @pytest.mark.asyncio
    async def test_get(self, insight_repo, note_repo):
        insight = make_insight()
        insight_repo.get.return_value = insight
        service = InsightService(insight_repo, note_repo)

        result = await service.get("source_insight:test1")

        insight_repo.get.assert_called_once_with("source_insight:test1")
        assert result == insight

    @pytest.mark.asyncio
    async def test_get_returns_none(self, insight_repo, note_repo):
        insight_repo.get.return_value = None
        service = InsightService(insight_repo, note_repo)

        assert await service.get("source_insight:missing") is None


class TestInsightServiceGetSource:

    @pytest.mark.asyncio
    async def test_get_source_for_insight(self, insight_repo, note_repo):
        src = make_source()
        insight_repo.get_source.return_value = src
        service = InsightService(insight_repo, note_repo)

        result = await service.get_source_for_insight("source_insight:1")

        insight_repo.get_source.assert_called_once_with("source_insight:1")
        assert result == src


class TestInsightServiceDelete:

    @pytest.mark.asyncio
    async def test_delete(self, insight_repo, note_repo):
        insight_repo.delete.return_value = True
        service = InsightService(insight_repo, note_repo)

        assert await service.delete("source_insight:1") is True


class TestInsightServiceSaveAsNote:

    @pytest.mark.asyncio
    async def test_save_as_note(self, insight_repo, note_repo):
        insight = make_insight(insight_type="summary", content="Summary text")
        note = make_note(title="Insight: summary", content="Summary text")
        note_repo.create.return_value = note
        service = InsightService(insight_repo, note_repo)

        result = await service.save_as_note(insight)

        note_repo.create.assert_called_once_with({
            "title": "Insight: summary",
            "content": "Summary text",
            "note_type": "ai",
        })
        assert result == note

    @pytest.mark.asyncio
    async def test_save_as_note_with_notebook(self, insight_repo, note_repo):
        insight = make_insight()
        note = make_note()
        note_repo.create.return_value = note
        service = InsightService(insight_repo, note_repo)

        await service.save_as_note(insight, notebook_id="notebook:1")

        note_repo.add_to_notebook.assert_called_once_with("note:test1", "notebook:1")
