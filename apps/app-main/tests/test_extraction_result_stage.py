"""PC.5c: the stored extraction result says its entities are pre-filter.

``_save_result`` stores the RAW extraction (the re-filter cache) beside
``metadata.filtering``, which describes the FILTERED set. Nothing said which is
which. This writes a row through ``_save_result`` into a real SurrealDB and
reads it back through ``GET /sources/{id}/extraction-result``.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace

import pytest
import surrealdb_service.connection as connection_module
from app_main.api.routers import sources_processing
from app_main.dependencies import get_source_service
from app_main.services import entity_extraction_service as ees
from fastapi import FastAPI
from fastapi.testclient import TestClient
from shared.models.extraction import ExtractedEntity, ExtractionResult
from surrealdb_service.config import SurrealDBConfig

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]


async def test_the_row_and_the_route_say_pre_filter(live_surrealdb: SurrealDBConfig, monkeypatch):
    real = connection_module.execute_query

    async def _bound(q, params=None, config=None):
        return await real(q, params, config or live_surrealdb)

    monkeypatch.setattr(ees, "execute_query", _bound)
    monkeypatch.setattr(connection_module, "execute_query", _bound)

    source_id = f"source:{uuid.uuid4().hex[:8]}"
    result = ExtractionResult(
        entities=[ExtractedEntity(text="Economische Zaken", label="organization"),
                  ExtractedEntity(text="ruis", label="topic")],
        metadata={"filtering": {"entities_before": 2, "entities_after": 1}},
    )
    svc = ees.EntityExtractionService.__new__(ees.EntityExtractionService)
    await svc._save_result(source_id, result)

    [row] = await real("SELECT * FROM extraction_result WHERE source_id = $s;",
                       {"s": source_id}, live_surrealdb)
    assert row["metadata"]["entities_stage"] == "pre_filter"
    assert row["metadata"]["filtering"]["entities_after"] == 1  # untouched
    assert len(row["entities"]) == 2  # the raw set, as labelled

    class _Src:
        async def get(self, _id):
            return SimpleNamespace(id=source_id)

    app = FastAPI()
    app.include_router(sources_processing.router, prefix="/api/sources")
    app.dependency_overrides[get_source_service] = lambda: _Src()
    body = TestClient(app).get(f"/api/sources/{source_id}/extraction-result").json()
    assert body["entities_stage"] == "pre_filter"
    assert body["entity_count"] == 2
