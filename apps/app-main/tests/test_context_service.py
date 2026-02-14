"""Tests for app_main.services.context_service."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app_main.services.context_service import ContextItem, ContextService


def _make_source(**overrides):
    defaults = {
        "id": "source:abc",
        "title": "Test Source",
        "full_text": "Some full text content.",
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


def _make_service(
    source=None,
    insights=None,
):
    source_repo = MagicMock()
    source_repo.get = AsyncMock(return_value=source)

    insight_repo = MagicMock()
    insight_repo.get_by_source = AsyncMock(return_value=insights or [])

    notebook_repo = MagicMock()
    note_repo = MagicMock()

    svc = ContextService(
        source_repo=source_repo,
        insight_repo=insight_repo,
        notebook_repo=notebook_repo,
        note_repo=note_repo,
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
    @pytest.mark.asyncio
    async def test_source_not_found(self):
        svc = _make_service(source=None)
        result = await svc.build_source_context("source:missing")

        assert result["sources"] == []
        assert result["insights"] == []
        assert result["total_items"] == 0

    @pytest.mark.asyncio
    async def test_source_only(self):
        svc = _make_service(source=_make_source())
        result = await svc.build_source_context(
            "source:abc", include_insights=False
        )

        assert len(result["sources"]) == 1
        assert result["sources"][0]["id"] == "source:abc"
        assert result["insights"] == []
        assert result["total_items"] == 1
        assert result["total_tokens"] > 0

    @pytest.mark.asyncio
    async def test_with_insights(self):
        svc = _make_service(
            source=_make_source(),
            insights=[_make_insight(), _make_insight(id="source_insight:yyy")],
        )
        result = await svc.build_source_context("source:abc")

        assert len(result["sources"]) == 1
        assert len(result["insights"]) == 2
        assert result["total_items"] == 3

    @pytest.mark.asyncio
    async def test_deduplication(self):
        dup_insight = _make_insight()
        svc = _make_service(
            source=_make_source(),
            insights=[dup_insight, dup_insight],
        )
        result = await svc.build_source_context("source:abc")

        # Duplicate insight should be removed
        assert len(result["insights"]) == 1
        assert result["total_items"] == 2

    @pytest.mark.asyncio
    async def test_max_tokens_truncation(self):
        svc = _make_service(
            source=_make_source(),
            insights=[
                _make_insight(id=f"source_insight:{i}", content="word " * 500)
                for i in range(10)
            ],
        )
        result = await svc.build_source_context("source:abc", max_tokens=50)

        # Should have fewer items than the 11 we provided
        assert result["total_items"] < 11

    @pytest.mark.asyncio
    async def test_output_shape(self):
        svc = _make_service(source=_make_source(), insights=[_make_insight()])
        result = await svc.build_source_context("source:abc")

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
        result = await svc.build_source_context("abc")

        # The repo should have been called with the prefixed ID
        svc._source_repo.get.assert_awaited_once_with("source:abc")

    @pytest.mark.asyncio
    async def test_priority_ordering(self):
        """Sources (priority 100) should appear before insights (75)."""
        svc = _make_service(
            source=_make_source(),
            insights=[_make_insight()],
        )
        result = await svc.build_source_context("source:abc")

        # Sources listed first
        assert result["sources"][0]["id"] == "source:abc"

    @pytest.mark.asyncio
    async def test_notes_empty_for_source_context(self):
        svc = _make_service(source=_make_source(), insights=[_make_insight()])
        result = await svc.build_source_context("source:abc")
        assert result["notes"] == []
