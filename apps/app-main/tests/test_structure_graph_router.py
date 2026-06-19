"""Tests for the structure-graph router + reader (Phase I.F).

The router is thin (validation + delegation); the reader holds the query logic.
Both are tested with the SurrealDB seam mocked — there is no live DB here, so the
latency AC (subgraph for ?page=1 in <200ms on the 50-page fixture) is
live-deferred and documented in the phase self-review.
"""

from __future__ import annotations

from typing import Any, Dict, List

import app_main.api.routers.structure_graph as router_mod
import app_main.services.graph.structure_graph_reader as reader_mod
import pytest
from app_main.api.routers.structure_graph import router
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_app() -> TestClient:
    app = FastAPI()
    app.include_router(router, prefix="/api")
    return TestClient(app)


# ===========================================================================
# Router: page-limit guard (AC7)
# ===========================================================================

class TestPageLimitGuard:

    def test_page_limit_at_cap_is_accepted(self, monkeypatch):
        async def fake_read(source_id, *, page=None, page_limit=100, config=None):
            assert page_limit == 500
            return {"nodes": [], "edges": [], "total_nodes": 0, "truncated": False}

        monkeypatch.setattr(router_mod, "read_structure_graph", fake_read)
        client = _make_app()

        resp = client.get("/api/sources/source:s1/structure-graph?page_limit=500")
        assert resp.status_code == 200

    def test_page_limit_above_cap_is_422(self, monkeypatch):
        called = False

        async def fake_read(*args, **kwargs):
            nonlocal called
            called = True
            return {}

        monkeypatch.setattr(router_mod, "read_structure_graph", fake_read)
        client = _make_app()

        resp = client.get("/api/sources/source:s1/structure-graph?page_limit=501")
        assert resp.status_code == 422
        assert "500" in resp.json()["detail"]
        assert called is False  # rejected before touching the reader

    def test_default_page_limit_is_100(self, monkeypatch):
        seen: Dict[str, Any] = {}

        async def fake_read(source_id, *, page=None, page_limit=100, config=None):
            seen["page_limit"] = page_limit
            seen["page"] = page
            return {"nodes": [], "edges": [], "total_nodes": 0, "truncated": False}

        monkeypatch.setattr(router_mod, "read_structure_graph", fake_read)
        client = _make_app()

        resp = client.get("/api/sources/source:s1/structure-graph")
        assert resp.status_code == 200
        assert seen["page_limit"] == 100
        assert seen["page"] is None

    def test_page_filter_passed_through(self, monkeypatch):
        seen: Dict[str, Any] = {}

        async def fake_read(source_id, *, page=None, page_limit=100, config=None):
            seen["page"] = page
            return {"nodes": [], "edges": [], "total_nodes": 0, "truncated": False}

        monkeypatch.setattr(router_mod, "read_structure_graph", fake_read)
        client = _make_app()

        resp = client.get("/api/sources/source:s1/structure-graph?page=1")
        assert resp.status_code == 200
        assert seen["page"] == 1


# ===========================================================================
# Reader: query shaping with the DB seam mocked
# ===========================================================================

class FakeReaderDB:
    """Returns canned doc_node / edge rows for read_structure_graph."""

    def __init__(self, nodes, parent_of=None, next_node=None, derived_from=None):
        self._nodes = nodes
        self._parent_of = parent_of or []
        self._next_node = next_node or []
        self._derived_from = derived_from or []
        self.calls: List[str] = []
        self.params: List[Dict[str, Any]] = []

    async def execute_query(self, query, params=None, config=None):
        self.calls.append(query)
        self.params.append(params or {})
        if "FROM doc_node" in query:
            return self._nodes
        if "FROM parent_of" in query:
            return self._parent_of
        if "FROM next_node" in query:
            return self._next_node
        if "FROM derived_from" in query:
            return self._derived_from
        return []


@pytest.fixture
def patch_reader(monkeypatch):
    def _apply(db: FakeReaderDB):
        monkeypatch.setattr(reader_mod, "execute_query", db.execute_query)
    return _apply


class TestReader:

    @pytest.mark.asyncio
    async def test_shapes_nodes_and_edges(self, patch_reader):
        db = FakeReaderDB(
            nodes=[
                {"id": "doc_node:1", "self_ref": "#/sections/0",
                 "element_type": "section_header", "text": "Intro", "page": None,
                 "level": 0, "sequence": 0, "bbox": None},
                {"id": "doc_node:2", "self_ref": "#/chunks/0",
                 "element_type": "paragraph", "text": "hi", "page": 0,
                 "level": 1, "sequence": 1, "bbox": [0.1, 0.2, 0.7, 0.6]},
            ],
            parent_of=[{"source": "doc_node:1", "target": "doc_node:2"}],
            next_node=[],
            derived_from=[{"source": "chunk:c0", "target": "doc_node:2"}],
        )
        patch_reader(db)

        result = await reader_mod.read_structure_graph("source:s1")

        assert result["total_nodes"] == 2
        # derived_from attaches the chunk id onto the leaf node (UI highlight)
        leaf = next(n for n in result["nodes"] if n["self_ref"] == "#/chunks/0")
        assert leaf["chunk_id"] == "chunk:c0"
        assert leaf["bbox"] == [0.1, 0.2, 0.7, 0.6]
        edge_types = {e["type"] for e in result["edges"]}
        assert edge_types == {"parent_of", "derived_from"}

    @pytest.mark.asyncio
    async def test_empty_source_returns_empty_payload(self, patch_reader):
        db = FakeReaderDB(nodes=[])
        patch_reader(db)

        result = await reader_mod.read_structure_graph("source:empty")
        assert result == {
            "nodes": [],
            "edges": [],
            "total_nodes": 0,
            "truncated": False,
        }
        # short-circuits before querying edge tables
        assert all("FROM parent_of" not in q for q in db.calls)

    @pytest.mark.asyncio
    async def test_truncated_flag_when_limit_reached(self, patch_reader):
        nodes = [
            {"id": f"doc_node:{i}", "self_ref": f"#/chunks/{i}",
             "element_type": "paragraph", "text": "x", "page": 0, "level": 0,
             "sequence": i, "bbox": None}
            for i in range(reader_mod.MAX_PAGE_LIMIT)
        ]
        db = FakeReaderDB(nodes=nodes)
        patch_reader(db)

        result = await reader_mod.read_structure_graph(
            "source:s1", page_limit=reader_mod.MAX_PAGE_LIMIT
        )
        assert result["truncated"] is True
        assert result["total_nodes"] == reader_mod.MAX_PAGE_LIMIT

    @pytest.mark.asyncio
    async def test_page_limit_clamped_to_max_in_query(self, patch_reader):
        db = FakeReaderDB(nodes=[])
        patch_reader(db)

        await reader_mod.read_structure_graph("source:s1", page_limit=99999)

        # The doc_node SELECT carries LIMIT $limit; the reader clamps to MAX.
        doc_node_params = next(
            p for q, p in zip(db.calls, db.params) if "FROM doc_node" in q
        )
        assert doc_node_params["limit"] == reader_mod.MAX_PAGE_LIMIT
