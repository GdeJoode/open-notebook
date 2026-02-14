"""Tests for the summaries router."""

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.summaries import router
from app_main.dependencies import get_summarization_service
from app_main.services.summarization_service import SummarizationService


def _make_app(svc):
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_summarization_service] = lambda: svc
    return TestClient(app)


_STRATEGY = {"name": "raptor", "implemented": True}

_SUMMARY_ITEM = {
    "id": "summary:1",
    "source_id": "source:1",
    "strategy": "raptor",
    "document_summary": "A brief summary.",
    "num_layers": 2,
    "num_input_chunks": 10,
    "node_count": 5,
    "created": "2025-06-15T12:00:00",
}

_SUMMARY_DETAIL = {
    **_SUMMARY_ITEM,
    "summary_nodes": [
        {
            "text": "Layer 0 summary",
            "layer": 0,
            "source_chunk_ids": ["c:1"],
            "children_ids": [],
            "node_id": "n:1",
            "section_title": None,
            "metadata": {},
        }
    ],
    "metadata": {"time_elapsed": 12.5},
}


class TestListStrategies:

    def test_list(self):
        svc = AsyncMock(spec=SummarizationService)
        svc.list_strategies.return_value = [
            _STRATEGY,
            {"name": "naive", "implemented": True},
            {"name": "map_reduce", "implemented": False},
        ]
        client = _make_app(svc)

        resp = client.get("/api/summaries/strategies")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 3
        assert data[0]["name"] == "raptor"


class TestListSummaries:

    def test_list_all(self):
        svc = AsyncMock(spec=SummarizationService)
        svc.list_summaries.return_value = [_SUMMARY_ITEM]
        client = _make_app(svc)

        resp = client.get("/api/summaries")

        assert resp.status_code == 200
        assert len(resp.json()) == 1

    def test_list_by_source(self):
        svc = AsyncMock(spec=SummarizationService)
        svc.list_summaries.return_value = [_SUMMARY_ITEM]
        client = _make_app(svc)

        resp = client.get("/api/summaries?source_id=source:1")

        assert resp.status_code == 200
        svc.list_summaries.assert_called_once_with(source_id="source:1")

    def test_list_empty(self):
        svc = AsyncMock(spec=SummarizationService)
        svc.list_summaries.return_value = []
        client = _make_app(svc)

        resp = client.get("/api/summaries")

        assert resp.status_code == 200
        assert resp.json() == []


class TestGetSummary:

    def test_get_found(self):
        svc = AsyncMock(spec=SummarizationService)
        svc.get_summary.return_value = _SUMMARY_DETAIL
        client = _make_app(svc)

        resp = client.get("/api/summaries/summary:1")

        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy"] == "raptor"
        assert len(data["summary_nodes"]) == 1

    def test_get_not_found(self):
        svc = AsyncMock(spec=SummarizationService)
        svc.get_summary.return_value = None
        client = _make_app(svc)

        resp = client.get("/api/summaries/summary:missing")

        assert resp.status_code == 404


class TestGenerateSummary:

    def test_generate_success(self):
        svc = AsyncMock(spec=SummarizationService)
        svc.generate_summary.return_value = _SUMMARY_DETAIL
        client = _make_app(svc)

        resp = client.post(
            "/api/summaries/generate",
            json={"source_id": "source:1", "strategy": "raptor"},
        )

        assert resp.status_code == 201
        assert resp.json()["strategy"] == "raptor"

    def test_generate_bad_strategy(self):
        svc = AsyncMock(spec=SummarizationService)
        svc.generate_summary.side_effect = ValueError("Unknown strategy: bad")
        client = _make_app(svc)

        resp = client.post(
            "/api/summaries/generate",
            json={"source_id": "source:1", "strategy": "bad"},
        )

        assert resp.status_code == 400


class TestDeleteSummary:

    def test_delete_success(self):
        svc = AsyncMock(spec=SummarizationService)
        svc.delete_summary.return_value = True
        client = _make_app(svc)

        resp = client.delete("/api/summaries/summary:1")

        assert resp.status_code == 204

    def test_delete_not_found(self):
        svc = AsyncMock(spec=SummarizationService)
        svc.delete_summary.return_value = False
        client = _make_app(svc)

        resp = client.delete("/api/summaries/summary:missing")

        assert resp.status_code == 404
