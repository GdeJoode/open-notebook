"""Tests for the D.1c ``GET /export-preview`` endpoint.

Covers
======

* Default filter returns ``{entity_count, relation_count}`` honouring
  the D.0 repo projections.
* ``min_connections=10`` post-filter prunes isolated entities AND
  silently drops relations whose endpoints didn't survive (Q-D-4).
* 404 for unknown notebook id (no repo call made).
* 400 for ``min_confidence`` outside [0.0, 1.0] (manual bound check
  -- Query() doesn't propagate the Pydantic Field constraint).

Mocks
=====
``EntityRepository.list_entities_for_notebook`` and
``list_relations_for_notebook`` are AsyncMocked so the test runs
in-process. The notebook lookup is also AsyncMocked.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from shared.models.entity import Entity, Relation

from app_main.api.routers.exports import router
from app_main.dependencies import (
    get_entity_repo,
    get_notebook_service,
)

from tests.conftest import make_notebook


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _entity(eid: str, name: str, confidence: float = 0.95) -> Entity:
    """Minimum-viable Entity for the preview path -- only id/confidence/
    primary_type are read upstream of the count."""
    return Entity(
        id=eid,
        canonical_name=name,
        entity_type="Concept",
        primary_type="Concept",
        type_tags=["Concept"],
        confidence=confidence,
        source_documents=["source:s1"],
    )


def _relation(src: str, dst: str, confidence: float = 0.9) -> Relation:
    return Relation(
        **{
            "in": src,
            "out": dst,
            "relation_type": "RELATED_TO",
            "confidence": confidence,
            "properties": {},
            "source_documents": ["source:s1"],
        }
    )


def _make_app(notebook_svc, entity_repo):
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    app.dependency_overrides[get_entity_repo] = lambda: entity_repo
    return TestClient(app)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExportPreviewCounts:

    def test_default_filter_returns_counts(self):
        """Happy path: default filter → entity_count + relation_count.

        Three entities + two relations, all confidence >= default
        ``min_confidence=0.9``. ``min_connections`` defaults to 5 which
        is too high for the fixture; we lower it via query string so
        the survivors aren't all pruned.
        """
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = make_notebook(id="notebook:abc")

        entity_repo = AsyncMock()
        entity_repo.list_entities_for_notebook.return_value = [
            _entity("entity:a", "Alice"),
            _entity("entity:b", "Bob"),
            _entity("entity:c", "Carol"),
        ]
        entity_repo.list_relations_for_notebook.return_value = [
            _relation("entity:a", "entity:b"),
            _relation("entity:b", "entity:c"),
        ]

        client = _make_app(notebook_svc, entity_repo)
        # min_connections=0 keeps every entity so the count assertion
        # reflects the repo output directly.
        resp = client.get(
            "/api/notebooks/notebook:abc/export-preview?min_connections=0"
        )

        assert resp.status_code == 200
        body = resp.json()
        assert body == {"entity_count": 3, "relation_count": 2}

        # Repo received an ExportFilter built from the query knobs.
        entity_repo.list_entities_for_notebook.assert_awaited_once()
        args = entity_repo.list_entities_for_notebook.await_args
        assert args.args[0] == "notebook:abc"
        assert args.args[1].min_connections == 0
        # Default min_confidence flows through unchanged.
        assert args.args[1].min_confidence == 0.9

    def test_min_connections_post_filter_prunes_isolated_entities(self):
        """``min_connections=10`` drops everyone (no entity has degree 10).

        Verifies BOTH halves of the post-filter:
        1. Entities below the degree threshold are dropped.
        2. Relations whose endpoint we just dropped don't get counted
           (Q-D-4 silent-drop applied at the preview boundary).

        Fixture: 3 entities (Alice, Bob, Carol) connected as A-B-C with
        max degree 2. With min_connections=10 every entity gets pruned
        AND every relation is silently dropped because its endpoints
        no longer survive.
        """
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = make_notebook(id="notebook:abc")

        entity_repo = AsyncMock()
        entity_repo.list_entities_for_notebook.return_value = [
            _entity("entity:a", "Alice"),
            _entity("entity:b", "Bob"),
            _entity("entity:c", "Carol"),
        ]
        entity_repo.list_relations_for_notebook.return_value = [
            _relation("entity:a", "entity:b"),
            _relation("entity:b", "entity:c"),
        ]

        client = _make_app(notebook_svc, entity_repo)
        resp = client.get(
            "/api/notebooks/notebook:abc/export-preview?min_connections=10"
        )

        assert resp.status_code == 200
        # Everyone pruned -> zero entities AND zero surviving relations.
        # If the endpoint counted relations without applying the
        # silent-drop, this would return relation_count=2.
        assert resp.json() == {"entity_count": 0, "relation_count": 0}

    def test_404_for_unknown_notebook(self):
        """Unknown notebook -> 404 BEFORE the repo is touched."""
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = None  # not found

        entity_repo = AsyncMock()

        client = _make_app(notebook_svc, entity_repo)
        resp = client.get(
            "/api/notebooks/notebook:nope/export-preview?min_connections=0"
        )

        assert resp.status_code == 404
        assert resp.json() == {"detail": "Notebook not found"}
        entity_repo.list_entities_for_notebook.assert_not_awaited()
        entity_repo.list_relations_for_notebook.assert_not_awaited()

    def test_400_for_min_confidence_out_of_range(self):
        """``min_confidence=5.0`` -> 400 with a clear message.

        Query() doesn't auto-apply the ExportFilter Pydantic bound, so
        the router validates manually. Without that branch the bad
        value would flow through to SurrealQL (still safe, but silently
        returning empty results would be confusing).
        """
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = make_notebook(id="notebook:abc")
        entity_repo = AsyncMock()

        client = _make_app(notebook_svc, entity_repo)
        resp = client.get(
            "/api/notebooks/notebook:abc/export-preview"
            "?min_connections=0&min_confidence=5.0"
        )

        assert resp.status_code == 400
        assert "min_confidence" in resp.json()["detail"]
        # No repo work happens after the bound rejection.
        entity_repo.list_entities_for_notebook.assert_not_awaited()
