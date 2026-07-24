"""Tests for the OKF bundle import endpoint (Phase OKF.4).

The REST seam OKF.3 deferred: ``POST /notebooks/{id}/import/okf`` accepts a
multipart-uploaded OKF v0.1 bundle zip, runs :class:`OkfImportService`, and
returns the ``OkfImportReport``.

Two layers:

* **Contract (stubbed service)** — TestClient + dependency overrides (mirrors
  ``test_okf_exports_router``): the multipart upload is read and forwarded to
  ``import_bundle``; the report is returned; 404 unknown notebook; 422 on a
  malformed bundle container.
* **Round-trip (``@requires_docker``)** — a bundle produced by the OKF.1
  exporter is POSTed through the ASGI app to a config-bound import service over
  a real SurrealDB container, asserting the report and the persisted rows.
"""

from __future__ import annotations

import io
import uuid
import zipfile
from typing import Dict
from unittest.mock import AsyncMock

import pytest
from app_main.api.routers.exports import router
from app_main.dependencies import get_notebook_service, get_okf_import_service
from app_main.services.okf_export_service import build_okf_bundle
from app_main.services.okf_import_service import (
    OkfImportReport,
    OkfImportService,
    SkippedConcept,
)
from fastapi import FastAPI
from fastapi.testclient import TestClient
from shared.models.entity import Entity, Relation
from shared.models.notebook import Note
from shared.models.source import Asset, Source

from tests.conftest import make_notebook


def _make_app(notebook_svc, import_svc) -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    app.dependency_overrides[get_okf_import_service] = lambda: import_svc
    return TestClient(app)


def _zip(bundle: Dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for path, text in sorted(bundle.items()):
            z.writestr(path, text)
    return buf.getvalue()


def _fixture_bundle(nb: str = "notebook:x", uid: str = "") -> Dict[str, str]:
    entities = [
        Entity(
            id=f"entity:alice{uid}",
            canonical_name=f"Alice{uid}",
            entity_type="person",
            primary_type="Person",
            type_tags=["Person"],
            description="A researcher.",
            confidence=0.95,
        ),
        Entity(
            id=f"entity:bob{uid}",
            canonical_name=f"Bob{uid}",
            entity_type="person",
            primary_type="Person",
            type_tags=["Person"],
            confidence=0.95,
        ),
    ]
    relations = [
        Relation(**{
            "in": f"entity:alice{uid}",
            "out": f"entity:bob{uid}",
            "relation_type": "KNOWS",
            "confidence": 0.9,
        })
    ]
    notes = [
        Note(
            id=f"note:n{uid}",
            title=f"Meeting{uid}",
            content="Body text",
            note_type="human",
        )
    ]
    sources = [
        Source(
            id=f"source:s{uid}",
            title=f"Paper{uid}",
            asset=Asset(url=f"https://example.com/{uid or 'a'}"),
        )
    ]
    return build_okf_bundle(
        entities=entities,
        relations=relations,
        notes=notes,
        sources=sources,
        dropped_relations=0,
        notebook_id=nb,
        timestamp="2026-07-24T00:00:00+00:00",
    ).bundle


# ---------------------------------------------------------------------------
# Contract (stubbed service)
# ---------------------------------------------------------------------------


def test_import_happy_path_returns_report():
    notebook_svc = AsyncMock()
    notebook_svc.get.return_value = make_notebook(id="notebook:abc")

    import_svc = AsyncMock()
    import_svc.import_bundle = AsyncMock(
        return_value=OkfImportReport(
            entities_created=2,
            relations_created=1,
            notes_created=1,
            sources_created=1,
            import_source_id="source:anchor",
            skipped=[SkippedConcept("entities/broken.md", "no frontmatter")],
        )
    )

    client = _make_app(notebook_svc, import_svc)
    payload = _zip(_fixture_bundle())
    resp = client.post(
        "/api/notebooks/notebook:abc/import/okf",
        files={"file": ("bundle.zip", payload, "application/zip")},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["entities_created"] == 2
    assert body["relations_created"] == 1
    assert body["notes_created"] == 1
    assert body["sources_created"] == 1
    assert body["skipped"] == [
        {"path": "entities/broken.md", "reason": "no frontmatter"}
    ]

    # The uploaded bytes were forwarded to the service, notebook-scoped, with
    # dedup on by default.
    import_svc.import_bundle.assert_awaited_once()
    call = import_svc.import_bundle.await_args
    assert call.args[0] == payload
    assert call.kwargs["notebook_id"] == "notebook:abc"
    assert call.kwargs["apply_dedup"] is True


def test_import_apply_dedup_query_flag_forwarded():
    notebook_svc = AsyncMock()
    notebook_svc.get.return_value = make_notebook(id="notebook:abc")
    import_svc = AsyncMock()
    import_svc.import_bundle = AsyncMock(return_value=OkfImportReport())

    client = _make_app(notebook_svc, import_svc)
    resp = client.post(
        "/api/notebooks/notebook:abc/import/okf?apply_dedup=false",
        files={"file": ("b.zip", _zip(_fixture_bundle()), "application/zip")},
    )
    assert resp.status_code == 200
    assert import_svc.import_bundle.await_args.kwargs["apply_dedup"] is False


def test_import_404_unknown_notebook():
    notebook_svc = AsyncMock()
    notebook_svc.get.return_value = None
    import_svc = AsyncMock()

    client = _make_app(notebook_svc, import_svc)
    resp = client.post(
        "/api/notebooks/notebook:nope/import/okf",
        files={"file": ("b.zip", _zip(_fixture_bundle()), "application/zip")},
    )
    assert resp.status_code == 404
    import_svc.import_bundle.assert_not_awaited()


def test_import_malformed_bundle_container_is_422():
    notebook_svc = AsyncMock()
    notebook_svc.get.return_value = make_notebook(id="notebook:abc")
    import_svc = AsyncMock()
    import_svc.import_bundle = AsyncMock(
        side_effect=ValueError("OKF bundle file must be a .zip archive")
    )

    client = _make_app(notebook_svc, import_svc)
    resp = client.post(
        "/api/notebooks/notebook:abc/import/okf",
        files={"file": ("notes.txt", b"not a zip", "text/plain")},
    )
    assert resp.status_code == 422
    assert "zip" in resp.json()["detail"].lower()


# ---------------------------------------------------------------------------
# Round-trip (requires docker) — exporter bundle → HTTP import → real DB
# ---------------------------------------------------------------------------

from surrealdb_service.config import SurrealDBConfig  # noqa: E402
from surrealdb_service.connection import execute_query  # noqa: E402


def _bind_db_import_service(monkeypatch, config: SurrealDBConfig):
    """Wire a DB-backed OkfImportService at the live container.

    Mirrors ``test_okf_import_service._bind_db_service``: the service and the
    persistence service reach SurrealDB through their module-level
    ``execute_query`` (no config arg), so patch those symbols to inject the
    container config; every repository is constructed ``config``-bound.
    """
    from app_main.services import entity_persistence_service as eps_module
    from app_main.services import okf_import_service as okf_module
    from app_main.services.entity_persistence_service import (
        EntityPersistenceService,
    )
    from surrealdb_service.repositories import NoteRepository, SourceRepository
    from surrealdb_service.repositories.entity import EntityRepository

    async def _exec(query, params=None, config_arg=None):
        return await execute_query(query, params, config=config or config_arg)

    monkeypatch.setattr(okf_module, "execute_query", _exec)
    monkeypatch.setattr(eps_module, "execute_query", _exec)

    entity_repo = EntityRepository(config=config)
    return OkfImportService(
        entity_repository=entity_repo,
        persistence_service=EntityPersistenceService(entity_repository=entity_repo),
        note_repository=NoteRepository(config=config),
        source_repository=SourceRepository(config=config),
    )


@pytest.mark.requires_docker
async def test_http_round_trip_import_persists_graph(monkeypatch, live_surrealdb):
    config = live_surrealdb
    import_svc = _bind_db_import_service(monkeypatch, config)

    uid = uuid.uuid4().hex[:8]

    # A real notebook the import links notes/sources into (add_to_notebook
    # RELATEs into it, so it must exist).
    nb_rows = await execute_query(
        "CREATE notebook SET name = $n;", {"n": f"OKF import {uid}"}, config=config
    )
    notebook_id = str(nb_rows[0]["id"])

    bundle = _fixture_bundle(nb=notebook_id, uid=uid)

    notebook_svc = AsyncMock()
    notebook_svc.get.return_value = make_notebook(id=notebook_id)

    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: notebook_svc
    app.dependency_overrides[get_okf_import_service] = lambda: import_svc

    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            f"/api/notebooks/{notebook_id}/import/okf?apply_dedup=false",
            files={"file": ("bundle.zip", _zip(bundle), "application/zip")},
        )

    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["entities_created"] == 2
    assert body["relations_created"] == 1
    assert body["notes_created"] == 1
    assert body["sources_created"] == 1

    # The graph really landed, provenance-tagged.
    ent_rows = await execute_query(
        "SELECT canonical_name, extraction_method, properties FROM entity "
        "WHERE canonical_name IN [$a, $b];",
        {"a": f"Alice{uid}", "b": f"Bob{uid}"},
        config=config,
    )
    assert len(ent_rows) == 2
    for row in ent_rows:
        assert row["extraction_method"] == "okf-import"
        assert row["properties"].get("okf_import") is True

    note_rows = await execute_query(
        "SELECT title FROM note WHERE title = $t;",
        {"t": f"Meeting{uid}"},
        config=config,
    )
    assert len(note_rows) == 1
