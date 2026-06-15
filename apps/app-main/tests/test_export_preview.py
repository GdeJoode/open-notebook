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


def _entity(
    eid: str,
    name: str,
    confidence: float = 0.95,
    status: str = "active",
) -> Entity:
    """Minimum-viable Entity for the preview path -- only id/confidence/
    primary_type/status are read upstream of the count."""
    return Entity(
        id=eid,
        canonical_name=name,
        entity_type="Concept",
        primary_type="Concept",
        type_tags=["Concept"],
        confidence=confidence,
        source_documents=["source:s1"],
        status=status,
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

    def test_status_archived_and_merged_excluded_from_preview(self):
        """Status filter must run BEFORE degree -- regression guard.

        The bug: without the status filter the preview counted
        archived/merged entities and reported "3 entities" while the
        actual ``ObsidianExportService.export`` would only emit 1.
        That's a user-facing contract break -- the dialog promises N,
        delivers fewer.

        Fixture: three entities (active / archived / merged), all
        wired into a triangle so each has degree 2. With
        ``min_connections=0`` the degree filter is a no-op, so the
        *only* thing that can reduce the count is the status filter.

        Asserts the surviving count is 1, not 3.
        """
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = make_notebook(id="notebook:abc")

        entity_repo = AsyncMock()
        entity_repo.list_entities_for_notebook.return_value = [
            _entity("entity:a", "Active", status="active"),
            _entity("entity:b", "Merged", status="merged"),
            _entity("entity:c", "Archived", status="archived"),
        ]
        # Triangle so every entity has degree 2 -- if the status
        # filter were absent and we relied on min_connections to do
        # the work, we'd count 3 entities + 3 relations here.
        entity_repo.list_relations_for_notebook.return_value = [
            _relation("entity:a", "entity:b"),
            _relation("entity:b", "entity:c"),
            _relation("entity:c", "entity:a"),
        ]

        client = _make_app(notebook_svc, entity_repo)
        resp = client.get(
            "/api/notebooks/notebook:abc/export-preview?min_connections=0"
        )

        assert resp.status_code == 200
        # Only the active entity survives. Both relations touching it
        # have a non-surviving endpoint, so the Q-D-4 silent-drop
        # zeroes the relation count too.
        assert resp.json() == {"entity_count": 1, "relation_count": 0}

    def test_status_filter_does_not_count_relation_endpoints(self):
        """Relations touching archived entities must NOT survive.

        An active entity connected ONLY to archived entities should
        still pass the entity filter (degree on raw relations counts
        the archived edges) but the Q-D-4 silent-drop must zero those
        relations because the archived endpoint didn't survive the
        status filter.

        Fixture: ActiveA with one edge to Archived hub. With
        min_connections=1 the active passes (it has degree 1 on the
        raw relations). The relation gets silently dropped because
        the hub is gone.
        """
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = make_notebook(id="notebook:abc")

        entity_repo = AsyncMock()
        entity_repo.list_entities_for_notebook.return_value = [
            _entity("entity:a", "ActiveA", status="active"),
            _entity("entity:h", "Hub", status="archived"),
        ]
        entity_repo.list_relations_for_notebook.return_value = [
            _relation("entity:a", "entity:h"),
        ]

        client = _make_app(notebook_svc, entity_repo)
        resp = client.get(
            "/api/notebooks/notebook:abc/export-preview?min_connections=1"
        )

        assert resp.status_code == 200
        # Active entity survives (degree 1 on raw relations) but its
        # only relation gets silently dropped because the archived
        # endpoint is gone.
        assert resp.json() == {"entity_count": 1, "relation_count": 0}

    def test_preview_matches_service_collect_on_mixed_fixture(self):
        """Service vs preview parity on a non-trivial fixture (M3).

        The preview endpoint and ``ObsidianExportService._collect``
        must apply the SAME filters in the SAME order. This test
        instantiates both code paths against an identical fixture
        and asserts the survivor counts match. If the preview ever
        re-duplicates the degree filter inline (instead of calling
        the staticmethod) and introduces a subtle bug -- e.g. only
        counting in-degree, mis-applying the threshold, etc. --
        this test catches the drift.

        The fixture is deliberately asymmetric so in-degree alone is
        NOT enough to pass the threshold for some entities; they
        need the combined in+out degree the staticmethod actually
        computes. Specifically: ``entity:hub`` has out-degree 3 but
        in-degree 0. Any "in-degree only" reimplementation in the
        router would drop the hub and yield a different count.
        """
        from app_main.services.obsidian_export_service import (
            ObsidianExportService,
        )

        entities = [
            _entity("entity:hub", "Hub", status="active"),
            _entity("entity:leaf1", "Leaf1", status="active"),
            _entity("entity:leaf2", "Leaf2", status="active"),
            _entity("entity:arch", "Archived", status="archived"),
            _entity("entity:mrg", "Merged", status="merged"),
        ]
        # Asymmetric edges -- hub is purely outbound, leaf1 purely
        # inbound, leaf2 mixed. Degrees (in+out / in-only):
        #   hub        3 / 0   <-- "in-degree only" bug would drop hub
        #   leaf1      2 / 2
        #   leaf2      2 / 1   <-- "in-degree only" bug would drop leaf2
        #   arch       0 / 0
        #   mrg        1 / 1
        # With min_connections=2 and the correct in+out filter, all
        # three actives pass. With an in-degree-only bug, only leaf1
        # would pass -- so service_count vs preview_count diverge.
        relations = [
            _relation("entity:hub", "entity:leaf1"),
            _relation("entity:hub", "entity:leaf2"),
            _relation("entity:hub", "entity:mrg"),
            _relation("entity:leaf2", "entity:leaf1"),
        ]

        # --- Service-side filter pipeline (replicated inline to match
        # ``ObsidianExportService._collect`` step-by-step) ----------
        service_entities = [
            e
            for e in entities
            if (e.status or "active") not in {"archived", "merged"}
        ]
        service_entities = (
            ObsidianExportService._apply_min_connections_filter(
                service_entities, relations, min_connections=2
            )
        )
        service_count = len(service_entities)
        # Sanity-check the fixture: under the correct in+out filter
        # all three actives pass. If this ever changes (e.g. because
        # the staticmethod was rewritten to only count in-degree), the
        # parity assertion below would still fire when preview and
        # service agree -- but this sanity check makes the FIXTURE
        # intent explicit.
        assert service_count == 3, (
            "Fixture sanity: with in+out degree all three actives "
            "should pass min_connections=2"
        )

        # --- Preview-side HTTP call ------------------------------
        notebook_svc = AsyncMock()
        notebook_svc.get.return_value = make_notebook(id="notebook:abc")
        entity_repo = AsyncMock()
        entity_repo.list_entities_for_notebook.return_value = entities
        entity_repo.list_relations_for_notebook.return_value = relations

        client = _make_app(notebook_svc, entity_repo)
        resp = client.get(
            "/api/notebooks/notebook:abc/export-preview?min_connections=2"
        )
        assert resp.status_code == 200
        preview_count = resp.json()["entity_count"]

        # The single source of truth: both code paths agree.
        assert preview_count == service_count, (
            f"Preview/service drift: preview={preview_count}, "
            f"service={service_count}"
        )
