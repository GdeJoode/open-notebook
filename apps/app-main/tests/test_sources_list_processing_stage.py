"""Track UX.1 (Part B): `processing_stage` on the sources LIST endpoint.

The list handler builds ``SourceListResponse`` from the repository's
``list_with_metadata`` rows. UX.1 adds ``processing_stage`` to that schema and
populates it from the source row so list-derived cards read the pipeline spine
without an N-per-card fan-out.

These tests assert at the serialization seam (schema default + handler mapping),
mirroring ``test_schemas_soft_nudge.py`` — a mocked repo, no testcontainer.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.sources_crud import router as sources_router
from app_main.api.schemas import SourceListResponse
from app_main.dependencies import get_notebook_service, get_source_service


def _row(source_id: str, stage: str | None) -> dict:
    row = {
        "id": source_id,
        "title": f"Doc {source_id}",
        "topics": [],
        "asset": None,
        "embedded": True,
        "insights_count": 0,
        "created": "2026-06-01T00:00:00Z",
        "updated": "2026-06-01T00:00:00Z",
        "command": None,
    }
    if stage is not None:
        row["processing_stage"] = stage
    return row


def _make_client(rows: list[dict]) -> TestClient:
    app = FastAPI()
    app.include_router(sources_router, prefix="/sources")
    app.dependency_overrides[get_source_service] = lambda: AsyncMock()
    app.dependency_overrides[get_notebook_service] = lambda: AsyncMock()
    return TestClient(app)


class TestSourceListResponseSchema:
    def test_processing_stage_serializes(self):
        resp = SourceListResponse(
            id="source:s1",
            title="Doc",
            topics=[],
            asset=None,
            embedded=True,
            embedded_chunks=0,
            insights_count=0,
            created="2026-06-01T00:00:00Z",
            updated="2026-06-01T00:00:00Z",
            processing_stage="graphed",
        )
        assert resp.model_dump()["processing_stage"] == "graphed"

    def test_processing_stage_defaults_to_ingested(self):
        resp = SourceListResponse(
            id="source:s1",
            title="Doc",
            topics=[],
            asset=None,
            embedded=True,
            embedded_chunks=0,
            insights_count=0,
            created="2026-06-01T00:00:00Z",
            updated="2026-06-01T00:00:00Z",
        )
        assert resp.model_dump()["processing_stage"] == "ingested"


class TestListEndpointPopulatesStage:
    def test_stage_from_row_is_returned(self):
        rows = [_row("source:s1", "complete"), _row("source:s2", "embedded")]
        fake_repo = MagicMock()
        fake_repo.list_with_metadata = AsyncMock(return_value=rows)
        fake_repo.batch_get_command_status = AsyncMock(return_value={})

        client = _make_client(rows)
        with patch(
            "surrealdb_service.repositories.SourceRepository",
            return_value=fake_repo,
        ):
            resp = client.get("/sources/")

        assert resp.status_code == 200
        body = resp.json()
        by_id = {s["id"]: s for s in body}
        assert by_id["source:s1"]["processing_stage"] == "complete"
        assert by_id["source:s2"]["processing_stage"] == "embedded"

    def test_missing_stage_falls_back_to_ingested(self):
        rows = [_row("source:s3", None)]
        fake_repo = MagicMock()
        fake_repo.list_with_metadata = AsyncMock(return_value=rows)
        fake_repo.batch_get_command_status = AsyncMock(return_value={})

        client = _make_client(rows)
        with patch(
            "surrealdb_service.repositories.SourceRepository",
            return_value=fake_repo,
        ):
            resp = client.get("/sources/")

        assert resp.status_code == 200
        assert resp.json()[0]["processing_stage"] == "ingested"
