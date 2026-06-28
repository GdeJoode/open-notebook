"""Unit tests for :meth:`GraphRepository.load_all_edges` (Track W.3).

These pin the uniform-shape contract of the unified edge loader WITHOUT a live
SurrealDB: the repo's ``execute_query`` is patched at the import site
(``surrealdb_service.repositories.graph.execute_query``) so each test scripts
the per-table SELECT results and asserts the normalisation:

- one ``{id, source, target, edge_type, metadata}`` dict per edge row,
- ``edge_type`` is the table name,
- per-table data fields are rolled into ``metadata`` (None-valued ones dropped),
- ``edge_types`` filters which tables are scanned,
- a bad id / empty id short-circuits to ``[]`` with no DB call,
- one table's query failure does not suppress the others.

A docker-backed roundtrip (both-endpoint coverage against real RELATE edges)
lives in ``test_mcp_graph_tools_roundtrip.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import surrealdb_service.repositories.graph as graph_module
from surrealdb_service.repositories.graph import GraphRepository


class _ExecuteQueryStub:
    """Return scripted rows keyed by the edge table named in the query."""

    def __init__(
        self,
        *,
        rows_by_table: Optional[Dict[str, Any]] = None,
        raises_for: Optional[set[str]] = None,
    ) -> None:
        self.rows_by_table = rows_by_table or {}
        self.raises_for = raises_for or set()
        self.tables_queried: List[str] = []

    async def __call__(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        config: Any = None,
    ) -> List[Dict[str, Any]]:
        for table in ("relation", "mentions", "cites"):
            if f"FROM {table} " in query:
                self.tables_queried.append(table)
                if table in self.raises_for:
                    raise RuntimeError(f"boom: {table}")
                return self.rows_by_table.get(table, [])
        raise AssertionError(f"unexpected query: {query}")


async def test_empty_id_short_circuits(monkeypatch) -> None:
    stub = _ExecuteQueryStub()
    monkeypatch.setattr(graph_module, "execute_query", stub)
    edges = await GraphRepository().load_all_edges("")
    assert edges == []
    assert stub.tables_queried == []


async def test_bad_id_returns_empty(monkeypatch) -> None:
    stub = _ExecuteQueryStub()
    monkeypatch.setattr(graph_module, "execute_query", stub)
    # No table/id colon → RecordID.parse raises → handled, no DB call.
    edges = await GraphRepository().load_all_edges("not-a-record-id")
    assert edges == []
    assert stub.tables_queried == []


async def test_uniform_shape_across_all_three_tables(monkeypatch) -> None:
    stub = _ExecuteQueryStub(
        rows_by_table={
            "relation": [
                {
                    "id": "relation:r1",
                    "source": "entity:a",
                    "target": "entity:b",
                    "relation_type": "WORKS_FOR",
                    "confidence": 0.9,
                    "status": "active",
                    "properties": {"k": "v"},
                    "source_documents": ["source:s1"],
                }
            ],
            "mentions": [
                {
                    "id": "mentions:m1",
                    "source": "source:s1",
                    "target": "entity:a",
                    "weight": 0.5,
                    "concept_name": "Alice",
                    "concept_type": "person",
                    "document_frequency": 3,
                }
            ],
            "cites": [
                {
                    "id": "cites:c1",
                    "source": "source:s1",
                    "target": "source:s2",
                    "confidence": 1.0,
                    "reference_text": "Doe 2020",
                    "match_method": "doi",
                }
            ],
        }
    )
    monkeypatch.setattr(graph_module, "execute_query", stub)

    edges = await GraphRepository().load_all_edges("entity:a")

    assert {e["edge_type"] for e in edges} == {"relation", "mentions", "cites"}
    by_type = {e["edge_type"]: e for e in edges}

    rel = by_type["relation"]
    assert rel["id"] == "relation:r1"
    assert rel["source"] == "entity:a"
    assert rel["target"] == "entity:b"
    assert rel["metadata"] == {
        "relation_type": "WORKS_FOR",
        "confidence": 0.9,
        "status": "active",
        "properties": {"k": "v"},
        "source_documents": ["source:s1"],
    }

    men = by_type["mentions"]
    assert men["metadata"]["concept_name"] == "Alice"
    assert men["metadata"]["document_frequency"] == 3

    cit = by_type["cites"]
    assert cit["metadata"]["match_method"] == "doi"
    # Every uniform edge has exactly these keys.
    for e in edges:
        assert set(e) == {"id", "source", "target", "edge_type", "metadata"}


async def test_none_valued_metadata_fields_dropped(monkeypatch) -> None:
    stub = _ExecuteQueryStub(
        rows_by_table={
            "relation": [
                {
                    "id": "relation:r1",
                    "source": "entity:a",
                    "target": "entity:b",
                    "relation_type": "X",
                    "confidence": None,  # dropped
                    "status": None,  # dropped
                    "properties": {},  # kept (not None)
                    "source_documents": [],  # kept
                }
            ],
        },
    )
    monkeypatch.setattr(graph_module, "execute_query", stub)
    edges = await GraphRepository().load_all_edges(
        "entity:a", edge_types=["relation"]
    )
    assert len(edges) == 1
    md = edges[0]["metadata"]
    assert "confidence" not in md and "status" not in md
    assert md == {"relation_type": "X", "properties": {}, "source_documents": []}


async def test_edge_types_filter_restricts_scan(monkeypatch) -> None:
    stub = _ExecuteQueryStub(
        rows_by_table={"mentions": [], "relation": [], "cites": []}
    )
    monkeypatch.setattr(graph_module, "execute_query", stub)
    await GraphRepository().load_all_edges(
        "entity:a", edge_types=["mentions", "bogus"]
    )
    # Only the recognised "mentions" table was scanned; "bogus" ignored.
    assert stub.tables_queried == ["mentions"]


async def test_one_table_failure_does_not_suppress_others(monkeypatch) -> None:
    stub = _ExecuteQueryStub(
        rows_by_table={
            "mentions": [
                {
                    "id": "mentions:m1",
                    "source": "source:s1",
                    "target": "entity:a",
                    "weight": 0.5,
                    "concept_name": "Alice",
                    "concept_type": "person",
                    "document_frequency": 1,
                }
            ],
            "cites": [],
        },
        raises_for={"relation"},
    )
    monkeypatch.setattr(graph_module, "execute_query", stub)
    edges = await GraphRepository().load_all_edges("entity:a")
    # relation raised, but mentions still returned.
    assert [e["edge_type"] for e in edges] == ["mentions"]
    assert "relation" in stub.tables_queried
