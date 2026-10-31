"""Tests for the sources CRUD router.

Focus: ``get_source``. Since F2-05 the counts come from
``SourceRepository.count_related`` (only records within the key's scope); the
original fix stands: the detail response must
report a count of *active* entities whose ``source_documents`` links back to the
source (the live ``entity`` table) rather than the previously-read, possibly
stale ``extraction_result.entity_count``.
"""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.sources_crud import router
from app_main.dependencies import get_source_service


SOURCE_ID = "source:dndibxmjveoxk7tfqfsl"


def _fake_source(processing_stage="complete"):
    """A minimal Source-like object covering the fields ``get_source`` reads."""
    return SimpleNamespace(
        id=SOURCE_ID,
        title="Midden-Limburg",
        topics=[],
        asset=None,
        created="2026-01-01T00:00:00Z",
        updated="2026-01-02T00:00:00Z",
        command=None,
        private=False,
        processing_stage=processing_stage,
    )


def _make_app(source_svc):
    app = FastAPI()
    app.include_router(router, prefix="/api/sources")
    app.dependency_overrides[get_source_service] = lambda: source_svc
    return TestClient(app)


def _svc(source, *, entity_count=0, embedding_count=0, relation_count=0):
    """A SourceService stand-in for ``get_source`` (F2-05: scoped reads).

    The counts come from ``SourceRepository.count_related`` (only records
    within the key's scope); the text from ``get_text`` (the chunks of the
    current version).
    """
    svc = AsyncMock()
    svc.get_scoped.return_value = source
    svc.get_text.return_value = "body"
    svc.source_repo = SimpleNamespace(
        count_related=AsyncMock(return_value={
            "insights_count": 0,
            "embedding_count": embedding_count,
            "entity_count": entity_count or 0,
            "relation_count": relation_count,
        }),
        batch_get_command_status=AsyncMock(return_value={}),
        get_notebook_ids_scoped=AsyncMock(return_value=[]),
    )
    return svc


@pytest.mark.usefixtures("monkeypatch")
class TestGetSourceEntityCount:

    def test_live_entity_count_when_extraction_result_stale(self, monkeypatch):
        """110 live active entities, extraction_result row absent → count = 110.

        Reproduces the Midden-Limburg bug: the denormalized count would have
        reported 0, but the live ``entity`` table has 110 active rows.
        """
        svc = _svc(_fake_source(), entity_count=110, embedding_count=5)
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
        svc = _svc(_fake_source(), entity_count=42, embedding_count=0, relation_count=7)
        client = _make_app(svc)

        resp = client.get(f"/api/sources/{SOURCE_ID}")

        assert resp.status_code == 200
        data = resp.json()
        assert data["entity_count"] == 42
        # relation_count is what persistence wrote (PC.5c), from extraction_result.
        assert data["relation_count"] == 7

    def test_relation_count_is_not_the_raw_extraction(self, monkeypatch):
        """PC.5c: a row without `metadata.persisted` has no graph count.

        `extraction_result.relation_count` is the pre-filter extraction; showing
        it as the source's relations reported relations that never reached the
        graph."""
        svc = _svc(_fake_source(), entity_count=3, embedding_count=0)
        data = _make_app(svc).get(f"/api/sources/{SOURCE_ID}").json()
        assert data["relation_count"] == 0

    def test_no_entities_returns_zero(self, monkeypatch):
        """Empty live count (GROUP ALL returns []) → entity_count 0."""
        svc = _svc(_fake_source(), entity_count=None, embedding_count=0)
        client = _make_app(svc)

        resp = client.get(f"/api/sources/{SOURCE_ID}")

        assert resp.status_code == 200
        assert resp.json()["entity_count"] == 0


@pytest.mark.usefixtures("monkeypatch")
class TestGetSourceProcessingStage:
    """PL.4 AC2: ``GET /sources/{id}`` surfaces the per-source pipeline stage."""

    def test_processing_stage_returned(self, monkeypatch):
        """The source's ``processing_stage`` is echoed in the detail response so
        the UI can show per-document progress."""
        svc = _svc(_fake_source(processing_stage="graphed"), entity_count=3, embedding_count=5)
        client = _make_app(svc)

        resp = client.get(f"/api/sources/{SOURCE_ID}")

        assert resp.status_code == 200
        assert resp.json()["processing_stage"] == "graphed"

    def test_processing_stage_defaults_ingested(self, monkeypatch):
        """A source object without the field (legacy/None) falls back to
        ``ingested`` rather than erroring."""
        src = _fake_source()
        # Simulate a source row whose stage is unset (None).
        src.processing_stage = None
        svc = _svc(src, entity_count=None, embedding_count=0)
        client = _make_app(svc)

        resp = client.get(f"/api/sources/{SOURCE_ID}")

        assert resp.status_code == 200
        assert resp.json()["processing_stage"] == "ingested"
