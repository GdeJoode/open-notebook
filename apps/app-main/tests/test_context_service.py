"""Tests for app_main.services.context_service."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app_main.services.context_service import ContextItem, ContextService
from shared.models.classification import ClassificationScope

ALLES = ClassificationScope.ALLES


def _make_source(**overrides):
    defaults = {
        "id": "source:abc",
        "title": "Test Source",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_insight(**overrides):
    defaults = {
        "id": "source_insight:xyz",
        "insight_type": "summary",
        "content": "This is an insight.",
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_chunk(**overrides):
    defaults = {
        "id": "chunk:c1",
        "text": "Chunk text.",
        "physical_page": 3,
        "printed_page": 4,
        "section_path": ["Intro"],
        "element_type": "text",
        "order": 0,
        "is_content": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_service(
    source=None,
    insights=None,
    chunks=None,
    text="Some full text content.",
):
    source_repo = MagicMock()
    source_repo.get_scoped = AsyncMock(return_value=source)
    source_repo.get_text = AsyncMock(return_value=text)

    insight_repo = MagicMock()
    insight_repo.get_by_source = AsyncMock(return_value=insights or [])

    notebook_repo = MagicMock()
    note_repo = MagicMock()

    chunk_repo = None
    if chunks is not None:
        chunk_repo = MagicMock()
        chunk_repo.get_by_source = AsyncMock(return_value=chunks)

    svc = ContextService(
        source_repo=source_repo,
        insight_repo=insight_repo,
        notebook_repo=notebook_repo,
        note_repo=note_repo,
        chunk_repo=chunk_repo,
    )
    return svc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestContextItem:
    def test_auto_token_count(self):
        item = ContextItem(id="x", type="source", content={"text": "hello"})
        assert item.tokens is not None
        assert item.tokens > 0

    def test_explicit_token_count(self):
        item = ContextItem(
            id="x", type="source", content={"text": "hello"}, tokens=42
        )
        assert item.tokens == 42


class TestBuildSourceContext:
    """F2-05 (ADR 0004): alleen de bron, met de tekst uit haar fragmenten,
    binnen het classificatiebereik; geen notities of inzichten."""

    @pytest.mark.asyncio
    async def test_source_not_found(self):
        svc = _make_service(source=None)
        result = await svc.build_source_context("source:missing", scope=ALLES)

        assert result["sources"] == []
        assert result["insights"] == []
        assert result["total_items"] == 0

    @pytest.mark.asyncio
    async def test_source_only(self):
        svc = _make_service(source=_make_source())
        result = await svc.build_source_context("source:abc", scope=ALLES)

        assert len(result["sources"]) == 1
        assert result["sources"][0]["id"] == "source:abc"
        assert result["sources"][0]["full_text"] == "Some full text content."
        assert result["insights"] == []
        assert result["total_items"] == 1
        assert result["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_no_insights_even_when_present(self):
        svc = _make_service(
            source=_make_source(),
            insights=[_make_insight(), _make_insight(id="source_insight:yyy")],
        )
        result = await svc.build_source_context("source:abc", scope=ALLES)

        assert result["insights"] == []
        assert result["notes"] == []
        assert result["total_items"] == 1
        svc._insight_repo.get_by_source.assert_not_called()

    @pytest.mark.asyncio
    async def test_scope_is_passed_and_required(self):
        svc = _make_service(source=_make_source())
        await svc.build_source_context("source:abc", scope=ClassificationScope.EXTERN)
        svc._source_repo.get_scoped.assert_awaited_once_with(
            "source:abc", scope=ClassificationScope.EXTERN
        )
        svc._source_repo.get_text.assert_awaited_once_with(
            "source:abc", scope=ClassificationScope.EXTERN
        )
        with pytest.raises(TypeError):
            await svc.build_source_context("source:abc")

    @pytest.mark.asyncio
    async def test_max_tokens_truncation(self):
        svc = _make_service(source=_make_source(), text="word " * 2000)
        result = await svc.build_source_context("source:abc", max_tokens=50, scope=ALLES)

        assert result["total_items"] == 0

    @pytest.mark.asyncio
    async def test_output_shape(self):
        svc = _make_service(source=_make_source(), insights=[_make_insight()])
        result = await svc.build_source_context("source:abc", scope=ALLES)

        # Verify required keys
        assert "sources" in result
        assert "insights" in result
        assert "notes" in result
        assert "total_tokens" in result
        assert "total_items" in result
        assert "metadata" in result
        assert "source_count" in result["metadata"]
        assert "note_count" in result["metadata"]
        assert "insight_count" in result["metadata"]

    @pytest.mark.asyncio
    async def test_source_id_prefix_added(self):
        """Passing an ID without prefix should still work."""
        svc = _make_service(source=_make_source(id="source:abc"))
        await svc.build_source_context("abc", scope=ALLES)

        # The repo should have been called with the prefixed ID
        svc._source_repo.get_scoped.assert_awaited_once_with("source:abc", scope=ALLES)


class TestBuildSourceChunks:
    """Chunk-level, page-carrying context (Track X.2)."""

    @pytest.mark.asyncio
    async def test_returns_chunks_with_provenance(self):
        svc = _make_service(
            source=_make_source(),
            chunks=[
                _make_chunk(id="chunk:a", physical_page=3, section_path=["A"]),
                _make_chunk(id="chunk:b", physical_page=7, section_path=["B"]),
            ],
        )
        out = await svc.build_source_chunks("source:abc")

        assert [c["chunk_id"] for c in out] == ["chunk:a", "chunk:b"]
        assert [c["physical_page"] for c in out] == [3, 7]
        assert all(c["source"] == "source:abc" for c in out)
        assert out[0]["section_path"] == ["A"]

    @pytest.mark.asyncio
    async def test_skips_noise_chunks(self):
        svc = _make_service(
            source=_make_source(),
            chunks=[
                _make_chunk(id="chunk:keep", is_content=True),
                _make_chunk(id="chunk:noise", is_content=False),
            ],
        )
        out = await svc.build_source_chunks("source:abc")
        assert [c["chunk_id"] for c in out] == ["chunk:keep"]

    @pytest.mark.asyncio
    async def test_max_chunks_cap(self):
        svc = _make_service(
            source=_make_source(),
            chunks=[_make_chunk(id=f"chunk:{i}") for i in range(10)],
        )
        out = await svc.build_source_chunks("source:abc", max_chunks=3)
        assert len(out) == 3

    @pytest.mark.asyncio
    async def test_no_chunk_repo_returns_empty(self):
        # chunk_repo not injected -> graceful empty (caller falls back to text).
        svc = _make_service(source=_make_source())
        out = await svc.build_source_chunks("source:abc")
        assert out == []

    @pytest.mark.asyncio
    async def test_lookup_failure_returns_empty(self):
        svc = _make_service(source=_make_source(), chunks=[])
        svc._chunk_repo.get_by_source = AsyncMock(side_effect=RuntimeError("db"))
        out = await svc.build_source_chunks("source:abc")
        assert out == []

    @pytest.mark.asyncio
    async def test_prefix_added(self):
        svc = _make_service(source=_make_source(), chunks=[_make_chunk()])
        await svc.build_source_chunks("abc")
        svc._chunk_repo.get_by_source.assert_awaited_once_with("source:abc")
