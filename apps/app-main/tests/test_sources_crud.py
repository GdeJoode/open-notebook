"""Tests for the sources CRUD router.

Focus: the live entity-count fix in ``get_source``. The detail response must
report a count of *active* entities whose ``source_documents`` links back to the
source (the live ``entity`` table) rather than the previously-read, possibly
stale ``extraction_result.entity_count``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import surrealdb_service.connection as conn
from app_main.api.routers.sources_crud import router
from app_main.dependencies import get_source_service


SOURCE_ID = "source:dndibxmjveoxk7tfqfsl"


def _fake_source():
    """A minimal Source-like object covering the fields ``get_source`` reads."""
    return SimpleNamespace(
        id=SOURCE_ID,
        title="Midden-Limburg",
        topics=[],
        asset=None,
        full_text="body",
        created="2026-01-01T00:00:00Z",
        updated="2026-01-02T00:00:00Z",
        command=None,
        private=False,
    )


def _make_app(source_svc):
    app = FastAPI()
    app.include_router(router, prefix="/api/sources")
    app.dependency_overrides[get_source_service] = lambda: source_svc
    return TestClient(app)


def _patch_execute_query(monkeypatch, *, entity_count, extraction_rows):
    """Route the router's per-purpose ``execute_query`` calls to fixtures.

    ``get_source`` issues several queries against the same ``execute_query``
    symbol; dispatch on a distinctive fragment of each query string so the live
    entity-count query and the (stale) extraction_result query return
    independent values.
    """

    async def fake_execute_query(query, params=None, *args, **kwargs):
        q = " ".join(query.split())
        if "FROM reference" in q:
            return []  # no notebooks
        if "FROM source_insight" in q:
            return [0]
        if "source_documents CONTAINS $source_id" in q and "FROM entity" in q:
            # The live count query under test.
            assert params["source_id"] == SOURCE_ID
            return [entity_count] if entity_count is not None else []
        if "FROM extraction_result" in q:
            return extraction_rows
        return []

    monkeypatch.setattr(conn, "execute_query", fake_execute_query)


@pytest.mark.usefixtures("monkeypatch")
class TestGetSourceEntityCount:

    def test_live_entity_count_when_extraction_result_stale(self, monkeypatch):
        """110 live active entities, extraction_result row absent → count = 110.

        Reproduces the Midden-Limburg bug: the denormalized count would have
        reported 0, but the live ``entity`` table has 110 active rows.
        """
        svc = AsyncMock()
        svc.get.return_value = _fake_source()
        svc.get_embedding_count.return_value = 5
        _patch_execute_query(
            monkeypatch, entity_count=110, extraction_rows=[]
        )
        client = _make_app(svc)

        resp = client.get(f"/api/sources/{SOURCE_ID}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["entity_count"] == 110
        # extraction_result absent → relation_count falls back to 0
        assert data["relation_count"] == 0

    def test_live_count_overrides_stale_zero(self, monkeypatch):
        """A stale extraction_result (entity_count=0) is ignored.

        Even if an extraction_result row exists with entity_count 0, the live
        query wins, so the response reflects the entity table.
        """
        svc = AsyncMock()
        svc.get.return_value = _fake_source()
        svc.get_embedding_count.return_value = 0
        _patch_execute_query(
            monkeypatch,
            entity_count=42,
            extraction_rows=[{"relation_count": 7}],
        )
        client = _make_app(svc)

        resp = client.get(f"/api/sources/{SOURCE_ID}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["entity_count"] == 42
        # relation_count still reads from extraction_result.
        assert data["relation_count"] == 7

    def test_no_entities_returns_zero(self, monkeypatch):
        """Empty live count (GROUP ALL returns []) → entity_count 0."""
        svc = AsyncMock()
        svc.get.return_value = _fake_source()
        svc.get_embedding_count.return_value = 0
        _patch_execute_query(
            monkeypatch, entity_count=None, extraction_rows=[]
        )
        client = _make_app(svc)

        resp = client.get(f"/api/sources/{SOURCE_ID}")

        assert resp.status_code == 200
        assert resp.json()["entity_count"] == 0
