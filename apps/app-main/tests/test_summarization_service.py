"""Tests for the summarization service."""

from unittest.mock import AsyncMock

import pytest

from app_main.services.summarization_service import SummarizationService


@pytest.fixture
def summary_repo():
    repo = AsyncMock()
    repo.list_summaries = AsyncMock(return_value=[])
    repo.get_summary = AsyncMock(return_value=None)
    repo.create_summary = AsyncMock(return_value={"id": "summary:new"})
    repo.delete_summary = AsyncMock(return_value=True)
    return repo


class TestListStrategies:

    @pytest.mark.asyncio
    async def test_returns_all_strategies(self, summary_repo):
        svc = SummarizationService(summary_repo=summary_repo)

        result = await svc.list_strategies()

        assert len(result) > 0
        names = [s["name"] for s in result]
        assert "raptor" in names
        assert "treekg" in names
        assert "naive" in names
        # Check implementation flags
        impl_map = {s["name"]: s["implemented"] for s in result}
        assert impl_map["raptor"] is True
        assert impl_map["map_reduce"] is False


class TestListSummaries:

    @pytest.mark.asyncio
    async def test_list_all(self, summary_repo):
        summary_repo.list_summaries.return_value = [
            {"id": "summary:1", "strategy": "raptor"}
        ]
        svc = SummarizationService(summary_repo=summary_repo)

        result = await svc.list_summaries()

        summary_repo.list_summaries.assert_called_once_with(source_id=None)
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_list_by_source(self, summary_repo):
        svc = SummarizationService(summary_repo=summary_repo)

        await svc.list_summaries(source_id="source:1")

        summary_repo.list_summaries.assert_called_once_with(source_id="source:1")


class TestGetSummary:

    @pytest.mark.asyncio
    async def test_get_found(self, summary_repo):
        summary_repo.get_summary.return_value = {"id": "summary:1"}
        svc = SummarizationService(summary_repo=summary_repo)

        result = await svc.get_summary("summary:1")

        assert result is not None
        assert result["id"] == "summary:1"

    @pytest.mark.asyncio
    async def test_get_not_found(self, summary_repo):
        svc = SummarizationService(summary_repo=summary_repo)

        result = await svc.get_summary("summary:missing")

        assert result is None


class TestDeleteSummary:

    @pytest.mark.asyncio
    async def test_delete(self, summary_repo):
        svc = SummarizationService(summary_repo=summary_repo)

        result = await svc.delete_summary("summary:1")

        assert result is True
        summary_repo.delete_summary.assert_called_once_with("summary:1")
