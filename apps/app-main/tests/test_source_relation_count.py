"""PC.5c: the source's relation count is what reached the graph.

``_save_result`` stores the RAW extraction (the re-filter cache): its
``relation_count`` counts relations filtering may have dropped. ``GET
/sources/{id}`` rendered that number as the source's relations (the pipeline's
completion tab). It now reads ``metadata.persisted``. The row is written by the
real ``_save_result`` into a real SurrealDB and read by the real route, so the
``metadata.persisted AS persisted`` projection is exercised, not assumed.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import surrealdb_service.connection as connection_module
from app_main.api.routers.sources_crud import router
from app_main.dependencies import get_source_service
from app_main.services import entity_extraction_service as ees
from fastapi import FastAPI
from fastapi.testclient import TestClient
from shared.models.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from surrealdb_service.config import SurrealDBConfig

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]


def _rel(a: str, b: str) -> ExtractedRelation:
    return ExtractedRelation(source_entity=a, target_entity=b, relation_type="related_to")


async def _saved_then_read(live: SurrealDBConfig, monkeypatch, metadata) -> int:
    real = connection_module.execute_query

    async def _bound(q, params=None, config=None):
        return await real(q, params, config or live)

    monkeypatch.setattr(ees, "execute_query", _bound)
    monkeypatch.setattr(connection_module, "execute_query", _bound)

    source_id = f"source:{uuid.uuid4().hex[:8]}"
    result = ExtractionResult(
        entities=[ExtractedEntity(text=t, label="organization") for t in "abcd"],
        relations=[_rel("a", "b"), _rel("b", "c"), _rel("c", "d")],
        metadata=metadata,
    )
    svc = ees.EntityExtractionService.__new__(ees.EntityExtractionService)
    await svc._save_result(source_id, result)

    src = AsyncMock()
    src.get.return_value = SimpleNamespace(
        id=source_id, title="t", topics=[], asset=None, full_text="x",
        created="2026-01-01T00:00:00Z", updated="2026-01-01T00:00:00Z",
        command=None, private=False, processing_stage="complete",
    )
    src.get_embedding_count.return_value = 0
    app = FastAPI()
    app.include_router(router, prefix="/api/sources")
    app.dependency_overrides[get_source_service] = lambda: src
    resp = TestClient(app).get(f"/api/sources/{source_id}")
    assert resp.status_code == 200
    return resp.json()["relation_count"]


async def test_the_count_is_what_persistence_wrote(live_surrealdb, monkeypatch):
    # 3 relations extracted; filtering kept 1, which was created, and 1 merged.
    n = await _saved_then_read(live_surrealdb, monkeypatch, {
        "filtering": {"entities_before": 4, "entities_after": 2},
        "persisted": {"entities_upserted": 2, "relations_created": 1, "relations_merged": 1},
    })
    assert n == 2  # not 3, the raw extraction


async def test_no_persistence_record_is_no_graph_count(live_surrealdb, monkeypatch):
    """Filtering skipped: nothing was written, so the raw 3 must not show."""
    assert await _saved_then_read(live_surrealdb, monkeypatch, {}) == 0
