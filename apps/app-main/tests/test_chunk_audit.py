"""Unit tests for ChunkAuditService (Track I.H2).

The DB is mocked at the service's ``execute_query`` seam. Chunk snapshots are
stand-in objects exposing the four attributes ``_summarize`` reads (id / order /
chunk_type / text), so no live Chunk model or SurrealDB is needed.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import app_main.services.audit.chunk_audit as audit_mod
import pytest
from app_main.services.audit.chunk_audit import ChunkAuditService

pytestmark = pytest.mark.asyncio


def _chunk(cid: str, order: int, text: str, chunk_type: str = "original"):
    return SimpleNamespace(id=cid, order=order, text=text, chunk_type=chunk_type)


class _RecordingDB:
    """Captures the last execute_query call; returns a canned result."""

    def __init__(self, result: Optional[List[Dict[str, Any]]] = None):
        self.calls: List[Dict[str, Any]] = []
        self.result = result if result is not None else []

    async def execute_query(self, query: str, params=None, config=None):
        self.calls.append({"query": query, "params": params or {}})
        return self.result


@pytest.fixture
def patch_db(monkeypatch):
    def _apply(result=None) -> _RecordingDB:
        db = _RecordingDB(result)
        monkeypatch.setattr(audit_mod, "execute_query", db.execute_query)
        return db

    return _apply


async def test_record_creates_chunk_edit_row(patch_db):
    db = patch_db()
    svc = ChunkAuditService()

    await svc.record(
        source_id="source:doc1",
        op="merge",
        before=[_chunk("chunk:c1", 0, "Alpha"), _chunk("chunk:c2", 1, "Beta")],
        after=[_chunk("chunk:c1", 0, "Alpha\n\nBeta", "merged")],
        chunk_id="chunk:c1",
    )

    assert len(db.calls) == 1
    call = db.calls[0]
    assert call["query"].startswith("CREATE chunk_edit CONTENT")
    data = call["params"]["data"]
    assert data["op"] == "merge"
    assert str(data["source"]) == "source:doc1"
    assert str(data["chunk"]) == "chunk:c1"
    # before/after are JSON strings of the chunk summaries
    before = json.loads(data["before"])
    after = json.loads(data["after"])
    assert [c["id"] for c in before] == ["chunk:c1", "chunk:c2"]
    assert after[0]["text"] == "Alpha\n\nBeta"
    assert after[0]["chunk_type"] == "merged"


async def test_record_without_chunk_id_stores_none(patch_db):
    db = patch_db()
    svc = ChunkAuditService()
    await svc.record(source_id="source:doc1", op="split", before=None, after=None)
    data = db.calls[0]["params"]["data"]
    assert data["chunk"] is None
    assert data["before"] is None
    assert data["after"] is None


async def test_record_is_fail_soft(monkeypatch):
    # A DB error inside record() must be swallowed (an audit write must never
    # break the mutation that already committed).
    async def boom(*a, **k):
        raise RuntimeError("db down")

    monkeypatch.setattr(audit_mod, "execute_query", boom)
    svc = ChunkAuditService()
    # Should NOT raise.
    await svc.record(source_id="source:doc1", op="merge", before=[], after=[])


async def test_history_queries_by_source_desc(patch_db):
    rows = [{"id": "chunk_edit:1", "op": "merge"}]
    db = patch_db(rows)
    svc = ChunkAuditService()

    result = await svc.history("source:doc1")

    assert result == rows
    q = db.calls[0]["query"]
    assert "FROM chunk_edit WHERE source = $source" in q
    assert "ORDER BY ts DESC" in q
    assert str(db.calls[0]["params"]["source"]) == "source:doc1"
