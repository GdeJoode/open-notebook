"""Tests for SourceService."""

import pytest

from app_main.services.source_service import SourceService
from tests.conftest import make_insight, make_source


class TestSourceServiceGet:

    @pytest.mark.asyncio
    async def test_get_delegates_to_repo(self, source_repo, chunk_repo, insight_repo, embedding_repo):
        src = make_source()
        source_repo.get.return_value = src
        service = SourceService(source_repo, chunk_repo, insight_repo, embedding_repo)

        result = await service.get("source:test1")

        source_repo.get.assert_called_once_with("source:test1")
        assert result == src

    @pytest.mark.asyncio
    async def test_get_returns_none(self, source_repo, chunk_repo, insight_repo, embedding_repo):
        source_repo.get.return_value = None
        service = SourceService(source_repo, chunk_repo, insight_repo, embedding_repo)

        result = await service.get("source:missing")

        assert result is None


class TestSourceServiceGetAll:

    @pytest.mark.asyncio
    async def test_get_all(self, source_repo, chunk_repo, insight_repo, embedding_repo):
        source_repo.get_all.return_value = [make_source(), make_source(id="source:2")]
        service = SourceService(source_repo, chunk_repo, insight_repo, embedding_repo)

        result = await service.get_all()

        source_repo.get_all.assert_called_once_with(order_by="updated DESC")
        assert len(result) == 2


class TestSourceServiceCreate:

    @pytest.mark.asyncio
    async def test_create(self, source_repo, chunk_repo, insight_repo, embedding_repo):
        src = make_source()
        source_repo.create.return_value = src
        service = SourceService(source_repo, chunk_repo, insight_repo, embedding_repo)

        result = await service.create({"title": "New"})

        source_repo.create.assert_called_once_with({"title": "New"})
        assert result == src


class TestSourceServiceDelete:

    @pytest.mark.asyncio
    async def test_delete(self, source_repo, chunk_repo, insight_repo, embedding_repo):
        source_repo.delete.return_value = True
        service = SourceService(source_repo, chunk_repo, insight_repo, embedding_repo)

        result = await service.delete("source:test1")

        assert result is True


class TestSourceServiceChunks:

    @pytest.mark.asyncio
    async def test_get_chunks(self, source_repo, chunk_repo, insight_repo, embedding_repo):
        source_repo.get_chunks.return_value = [{"text": "chunk"}]
        service = SourceService(source_repo, chunk_repo, insight_repo, embedding_repo)

        result = await service.get_chunks("source:1")

        source_repo.get_chunks.assert_called_once_with("source:1")
        assert len(result) == 1


class TestSourceServiceInsights:

    @pytest.mark.asyncio
    async def test_get_insights(self, source_repo, chunk_repo, insight_repo, embedding_repo):
        source_repo.get_insights.return_value = [make_insight()]
        service = SourceService(source_repo, chunk_repo, insight_repo, embedding_repo)

        result = await service.get_insights("source:1")

        assert len(result) == 1


class TestSourceServiceEmbeddings:

    @pytest.mark.asyncio
    async def test_get_embedding_count(self, source_repo, chunk_repo, insight_repo, embedding_repo):
        source_repo.get_embedding_count.return_value = 42
        service = SourceService(source_repo, chunk_repo, insight_repo, embedding_repo)

        result = await service.get_embedding_count("source:1")

        assert result == 42

    @pytest.mark.asyncio
    async def test_delete_embeddings(self, source_repo, chunk_repo, insight_repo, embedding_repo):
        source_repo.delete_embeddings.return_value = 5
        service = SourceService(source_repo, chunk_repo, insight_repo, embedding_repo)

        result = await service.delete_embeddings("source:1")

        assert result == 5


class TestSourceServiceAddToNotebook:

    @pytest.mark.asyncio
    async def test_add_to_notebook(self, source_repo, chunk_repo, insight_repo, embedding_repo):
        source_repo.add_to_notebook.return_value = True
        service = SourceService(source_repo, chunk_repo, insight_repo, embedding_repo)

        result = await service.add_to_notebook("source:1", "notebook:1")

        source_repo.add_to_notebook.assert_called_once_with("source:1", "notebook:1")
        assert result is True
