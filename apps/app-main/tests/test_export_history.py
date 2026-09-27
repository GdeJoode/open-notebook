"""PC.5c: the ``export.*`` metrics have a reader.

The rows are written the way the exporters write them — through
``record_metric`` — into a real SurrealDB, and read back by
``read_export_history``, the function behind ``GET .../exports/history``.
"""

from __future__ import annotations

import uuid

import pytest
import surrealdb_service.connection as connection_module
from app_main.api.routers import exports
from app_main.dependencies import get_notebook_service
from app_main.services.export_history import read_export_history
from fastapi import FastAPI
from fastapi.testclient import TestClient
from shared.services.metrics import record_metric
from surrealdb_service.config import SurrealDBConfig

pytestmark = [pytest.mark.asyncio]


@pytest.fixture
def bound_metrics(live_surrealdb: SurrealDBConfig, monkeypatch):
    """``record_metric`` writes through the global pool; point it at the container."""
    real = connection_module.execute_query

    async def _bound(q, params=None, config=None):
        return await real(q, params, config or live_surrealdb)

    monkeypatch.setattr(connection_module, "execute_query", _bound)
    monkeypatch.delenv("OPEN_NOTEBOOK_DISABLE_METRICS", raising=False)
    return live_surrealdb


@pytest.mark.requires_docker
async def test_the_newest_run_per_notebook_and_format(bound_metrics):
    nb = f"notebook:{uuid.uuid4().hex[:8]}"
    other = f"notebook:{uuid.uuid4().hex[:8]}"
    await record_metric("export.jsonl", {"entities_written": 3, "relations_written": 2,
                                         "bytes_written": 100, "duration_ms": 12}, notebook=nb)
    await record_metric("export.obsidian", {"entities_written": 5, "files_written": 5,
                                            "duration_ms": 30}, notebook=nb)
    await record_metric("export.jsonl", {"entities_written": 0, "duration_ms": 4,
                                         "error": "RuntimeError: boom"}, notebook=nb)
    await record_metric("export.jsonl", {"entities_written": 99}, notebook=other)
    await record_metric("extraction.complete", {"entities_written": 7}, notebook=nb)

    runs = await read_export_history(nb, config=bound_metrics)
    assert [r["format"] for r in runs] == ["jsonl", "obsidian", "jsonl"]
    assert runs[0]["error"] == "RuntimeError: boom"  # a failed run is a run
    assert runs[2]["entities_written"] == 3 and runs[2]["bytes_written"] == 100

    [last] = await read_export_history(nb, ["obsidian"], limit=1, config=bound_metrics)
    assert last["files_written"] == 5 and last["error"] is None and last["created_at"]


@pytest.mark.requires_docker
async def test_the_bare_id_finds_the_same_rows(bound_metrics):
    """The route gets ``notebook:x`` from the UI; an exporter may have stored ``x``."""
    bare = uuid.uuid4().hex[:8]
    await record_metric("export.okf", {"entities_written": 1}, notebook=bare)
    assert len(await read_export_history(f"notebook:{bare}", config=bound_metrics)) == 1


def _client(monkeypatch, notebook_exists=True, rows=()):
    seen = {}

    async def _read(nb, formats=None, limit=10, config=None):
        seen.update(nb=nb, formats=formats, limit=limit)
        return list(rows)

    class _Nb:
        async def get(self, _id):
            return object() if notebook_exists else None

    monkeypatch.setattr(exports.export_history, "read_export_history", _read)
    app = FastAPI()
    app.include_router(exports.router, prefix="/api")
    app.dependency_overrides[get_notebook_service] = lambda: _Nb()
    return TestClient(app), seen


async def test_the_route_passes_the_format_and_shapes_the_rows(monkeypatch):
    row = {"format": "networkx", "created_at": "2026-09-27T10:00:00Z", "error": None,
           "entities_written": 4, "relations_written": 3, "files_written": 1,
           "bytes_written": 10, "duration_ms": 5, "mode": None}
    client, seen = _client(monkeypatch, rows=[row])
    r = client.get("/api/notebooks/notebook:a/exports/history",
                   params={"format": "networkx", "limit": 1})
    assert r.status_code == 200 and r.json() == [row]
    assert seen == {"nb": "notebook:a", "formats": ["networkx"], "limit": 1}


async def test_the_route_refuses_an_unknown_notebook_and_format(monkeypatch):
    client, _ = _client(monkeypatch, notebook_exists=False)
    assert client.get("/api/notebooks/notebook:a/exports/history").status_code == 404
    client, _ = _client(monkeypatch)
    r = client.get("/api/notebooks/notebook:a/exports/history", params={"format": "pdf"})
    assert r.status_code == 422
