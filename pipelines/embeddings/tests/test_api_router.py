"""Tests for the embeddings API router."""

from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from embeddings.api.router import create_router
from embeddings.config import EmbeddingConfig
from embeddings.service import EmbeddingResult, EmbeddingService, RebuildResult


def _make_mock_service() -> AsyncMock:
    """Create a mock EmbeddingService with sensible defaults."""
    service = AsyncMock(spec=EmbeddingService)
    service.config = EmbeddingConfig()
    return service


def _make_app(service: AsyncMock) -> FastAPI:
    """Build a minimal FastAPI app with the embeddings router."""
    app = FastAPI()
    router = create_router(lambda: service)
    app.include_router(router)
    return app


class TestEmbedSourceEndpoint:

    def test_embed_source_success(self):
        service = _make_mock_service()
        service.embed_source.return_value = EmbeddingResult(
            id="source:1",
            embeddings_created=5,
            chunks_used=5,
            used_structural_chunks=True,
        )
        client = TestClient(_make_app(service))

        resp = client.post("/embed/source/source:1")

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "source:1"
        assert data["embeddings_created"] == 5
        assert data["used_structural_chunks"] is True
        service.embed_source.assert_called_once_with("source:1")

    def test_embed_source_with_errors(self):
        service = _make_mock_service()
        service.embed_source.return_value = EmbeddingResult(
            id="source:2",
            embeddings_created=0,
            chunks_used=0,
            used_structural_chunks=False,
            errors=["Something went wrong"],
        )
        client = TestClient(_make_app(service))

        resp = client.post("/embed/source/source:2")

        assert resp.status_code == 200
        assert resp.json()["errors"] == ["Something went wrong"]


class TestEmbedNoteEndpoint:

    def test_embed_note_success(self):
        service = _make_mock_service()
        service.embed_note.return_value = EmbeddingResult(
            id="note:1",
            embeddings_created=1,
            chunks_used=1,
            used_structural_chunks=False,
        )
        client = TestClient(_make_app(service))

        resp = client.post("/embed/note/note:1")

        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "note:1"
        assert data["embeddings_created"] == 1
        service.embed_note.assert_called_once_with("note:1")


class TestRebuildEndpoint:

    def test_rebuild_default_params(self):
        service = _make_mock_service()
        service.rebuild_embeddings.return_value = RebuildResult(
            sources_processed=10,
            notes_processed=5,
            insights_processed=3,
            total_embeddings=50,
        )
        client = TestClient(_make_app(service))

        resp = client.post("/rebuild", json={})

        assert resp.status_code == 200
        data = resp.json()
        assert data["sources_processed"] == 10
        assert data["total_embeddings"] == 50
        service.rebuild_embeddings.assert_called_once_with(
            include_sources=True,
            include_notes=True,
            include_insights=True,
            mode="all",
        )

    def test_rebuild_custom_params(self):
        service = _make_mock_service()
        service.rebuild_embeddings.return_value = RebuildResult(
            sources_processed=3, total_embeddings=15,
        )
        client = TestClient(_make_app(service))

        resp = client.post(
            "/rebuild",
            json={
                "include_sources": True,
                "include_notes": False,
                "include_insights": False,
                "mode": "existing",
            },
        )

        assert resp.status_code == 200
        service.rebuild_embeddings.assert_called_once_with(
            include_sources=True,
            include_notes=False,
            include_insights=False,
            mode="existing",
        )

    def test_rebuild_with_errors(self):
        service = _make_mock_service()
        service.rebuild_embeddings.return_value = RebuildResult(
            sources_processed=2,
            total_embeddings=8,
            errors=["Failed source:bad1"],
        )
        client = TestClient(_make_app(service))

        resp = client.post("/rebuild", json={})

        assert resp.status_code == 200
        assert resp.json()["errors"] == ["Failed source:bad1"]


class TestConfigEndpoint:

    def test_get_config_returns_defaults(self):
        service = _make_mock_service()
        service.config = EmbeddingConfig()
        client = TestClient(_make_app(service))

        resp = client.get("/config")

        assert resp.status_code == 200
        data = resp.json()
        assert data["batch_size"] == 10
        assert data["max_concurrent"] == 5
        assert data["retry_attempts"] == 2

    def test_get_config_returns_custom_values(self):
        service = _make_mock_service()
        service.config = EmbeddingConfig(
            batch_size=20, max_concurrent=10, retry_delay=2.5,
        )
        client = TestClient(_make_app(service))

        resp = client.get("/config")

        data = resp.json()
        assert data["batch_size"] == 20
        assert data["max_concurrent"] == 10
        assert data["retry_delay"] == 2.5
