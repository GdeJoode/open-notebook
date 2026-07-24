"""Unit tests for the chunk merge/split mutator (Track I.D-3).

A live SurrealDB isn't available in this sandbox, so the DB is mocked at
the mutator's ``execute_query`` (SELECTs) and ``execute_transaction``
(BEGIN/COMMIT) seams. The fake models the table-level behaviour the mutator
depends on: ordered SELECTs, single-record SELECT, and a transaction applied
*all-or-nothing*.

Atomicity (AC3) is covered at two levels:
* ``test_check_transaction_response_*`` exercise the REAL error-detection
  guard (``surrealdb_service.connection._check_transaction_response``) that
  ``execute_transaction`` uses — proving a statement-level ``ERR`` inside a
  transaction raises rather than being swallowed (the actual bug the seam
  fixes). These don't depend on the fake at all.
* ``test_*_rollback_*`` assert the mutator *propagates* a transaction failure
  (doesn't catch/ignore it) and leaves the store untouched. Live server-side
  rollback itself is a documented SurrealQL guarantee, verified by an
  integration run (deferred — no SurrealDB in sandbox).
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import app_main.services.chunking.chunk_mutator as mutator_mod
import pytest
from app_main.services.chunking.chunk_mutator import (
    MERGE_SEPARATOR,
    ChunkMutationError,
    ChunkMutator,
    _split_positions,
)
from surrealdb_service.connection import _check_transaction_response

SOURCE_ID = "source:doc1"


def _chunk_row(
    n: int,
    order: int,
    text: str,
    positions: Optional[List[List[float]]] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build a chunk row dict shaped like a SurrealDB SELECT result."""
    row: Dict[str, Any] = {
        "id": f"chunk:c{n}",
        "source": SOURCE_ID,
        "text": text,
        "order": order,
        "physical_page": 0,
        "printed_page": 1,
        "chapter": None,
        "paragraph_number": None,
        "element_type": "paragraph",
        "positions": positions if positions is not None else [],
        "metadata": {},
        "section_path": [],
        "section_level": 0,
        "chunk_type": "original",
        "source_element_count": 1,
        "is_content": True,
    }
    row.update(extra)
    return row


class FakeDB:
    """In-memory chunk store that interprets the mutator's queries.

    Dispatches on the query text the mutator actually emits. The
    ``BEGIN TRANSACTION`` block is applied atomically: it stages every
    mutation against a deep copy and only swaps it in if no statement
    raised. ``fail_on`` lets a test inject a mid-transaction failure to
    prove rollback.
    """

    def __init__(self, rows: List[Dict[str, Any]]):
        self.rows: Dict[str, Dict[str, Any]] = {r["id"]: r for r in rows}
        self._counter = 1000
        # If set, the transaction raises when a statement's query contains
        # this substring, simulating a server-side failure mid-block.
        self.fail_on: Optional[str] = None

    # -- helpers -----------------------------------------------------------
    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return copy.deepcopy(self.rows)

    def _next_id(self) -> str:
        self._counter += 1
        return f"chunk:c{self._counter}"

    @staticmethod
    def _rid(value: Any) -> str:
        return str(value)

    # -- query dispatch ----------------------------------------------------
    async def execute_transaction(
        self, query: str, params: Optional[Dict[str, Any]] = None, config=None
    ):
        """Stand in for the real execute_transaction seam (BEGIN/COMMIT)."""
        return await self._run_transaction(query, params or {})

    async def execute_query(
        self, query: str, params: Optional[Dict[str, Any]] = None, config=None
    ):
        params = params or {}
        q = query.strip()

        if q.startswith("SELECT * FROM chunk WHERE source"):
            source = self._rid(params["source"])
            rows = [
                copy.deepcopy(r)
                for r in self.rows.values()
                if self._rid(r["source"]) == source
            ]
            rows.sort(key=lambda r: r["order"])
            return rows

        if q.startswith("SELECT * FROM $id"):
            rid = self._rid(params["id"])
            row = self.rows.get(rid)
            return [copy.deepcopy(row)] if row else []

        raise AssertionError(f"Unexpected query: {q[:80]}")

    async def _run_transaction(self, query: str, params: Dict[str, Any]):
        """Apply the BEGIN/COMMIT block atomically against a staged copy."""
        staged = copy.deepcopy(self.rows)

        # Inject failure before any work if requested — must leave store
        # untouched (rollback).
        if self.fail_on and self.fail_on in query:
            raise RuntimeError("injected transaction failure")

        is_merge = "DELETE $drop_id" in query
        if is_merge:
            keep_id = self._rid(params["keep_id"])
            drop_id = self._rid(params["drop_id"])
            drop_order = params["drop_order"]
            keep = staged[keep_id]
            keep["text"] = params["text"]
            keep["positions"] = params["positions"]
            keep["source_element_count"] += params["drop_elem"]
            keep["chunk_type"] = "merged"
            del staged[drop_id]
            for r in staged.values():
                if (
                    self._rid(r["source"]) == self._rid(params["source"])
                    and r["order"] > drop_order
                ):
                    r["order"] -= 1
        else:  # split
            source = self._rid(params["source"])
            order = params["order"]
            for r in staged.values():
                if self._rid(r["source"]) == source and r["order"] > order:
                    r["order"] += 1
            chunk_id = self._rid(params["chunk_id"])
            original = staged[chunk_id]
            original["text"] = params["first_text"]
            original["positions"] = params["first_positions"]
            original["chunk_type"] = "split"
            new_content = copy.deepcopy(params["new_chunk"])
            new_content["source"] = self._rid(new_content["source"])
            new_id = self._next_id()
            new_content["id"] = new_id
            staged[new_id] = new_content

        # Commit: swap staged copy in.
        self.rows = staged
        return []


@pytest.fixture
def patch_execute(monkeypatch):
    """Wire a FakeDB into the mutator module and return it."""
    def _apply(rows: List[Dict[str, Any]]) -> FakeDB:
        db = FakeDB(rows)
        monkeypatch.setattr(mutator_mod, "execute_query", db.execute_query)
        monkeypatch.setattr(
            mutator_mod, "execute_transaction", db.execute_transaction
        )
        # ensure_record_id is exercised with str ids; keep it identity-ish
        # so FakeDB key lookups match. The real fn parses to RecordID whose
        # str() round-trips to the same "table:id" we keyed on.
        return db
    return _apply


# ---------------------------------------------------------------------------
# Pure helper: proportional position split
# ---------------------------------------------------------------------------


def test_split_positions_proportional_single_box():
    first, second = _split_positions([[1, 0.1, 0.9, 0.2, 0.6]], 0.25)
    # y range 0.2..0.6 (height 0.4); cut at 0.2 + 0.4*0.25 = 0.3
    assert first == [[1, 0.1, 0.9, 0.2, 0.3]]
    assert second == [[1, 0.1, 0.9, 0.3, 0.6]]


def test_split_positions_conserves_union():
    first, second = _split_positions([[2, 0.0, 1.0, 0.0, 1.0]], 0.5)
    assert first[0][3] == 0.0 and second[0][4] == 1.0
    assert first[0][4] == second[0][3]  # contiguous, no gap/overlap


def test_split_positions_empty():
    assert _split_positions([], 0.5) == ([], [])


def test_split_positions_multibox_partitions_whole_boxes():
    """Multi-box chunks must apportion WHOLE line-boxes to each half, not
    cut every box at the same fraction (which would put part of every line
    in both halves). With 4 equal boxes at fraction 0.5, boxes 0-1 go to the
    first half and boxes 2-3 to the second — none are sliced."""
    boxes = [
        [1, 0.0, 1.0, 0.0, 0.1],
        [1, 0.0, 1.0, 0.1, 0.2],
        [1, 0.0, 1.0, 0.2, 0.3],
        [1, 0.0, 1.0, 0.3, 0.4],
    ]
    first, second = _split_positions(boxes, 0.5)
    assert first == boxes[:2]
    assert second == boxes[2:]


def test_split_positions_multibox_splits_only_straddling_box():
    """When the cut lands inside a box, only that box is sliced; the others
    stay whole on their respective side."""
    boxes = [
        [1, 0.0, 1.0, 0.0, 0.2],
        [1, 0.0, 1.0, 0.2, 0.4],
    ]
    # fraction 0.75 → split_point 1.5 → box 0 whole to first, box 1 cut at 50%.
    first, second = _split_positions(boxes, 0.75)
    assert first[0] == [1, 0.0, 1.0, 0.0, 0.2]
    assert first[1][:4] == [1, 0.0, 1.0, 0.2]
    assert first[1][4] == pytest.approx(0.3)
    assert len(second) == 1
    assert second[0][:3] == [1, 0.0, 1.0]
    assert second[0][3] == pytest.approx(0.3)
    assert second[0][4] == 0.4


# ---------------------------------------------------------------------------
# Transaction error guard (AC3) — the real anti-silent-no-op check
# ---------------------------------------------------------------------------


def test_check_transaction_response_ok_returns_results():
    resp = {
        "result": [
            {"status": "OK", "result": "a"},
            {"status": "OK", "result": "b"},
        ]
    }
    assert _check_transaction_response(resp) == ["a", "b"]


def test_check_transaction_response_raises_on_statement_err():
    """A failed statement inside the transaction must raise — the bug the
    guard exists to catch (plain query() returns only statement[0])."""
    resp = {
        "result": [
            {"status": "OK", "result": None},
            {"status": "ERR", "result": "Some delete error"},
            {"status": "OK", "result": None},
        ]
    }
    with pytest.raises(RuntimeError, match="Some delete error"):
        _check_transaction_response(resp)


def test_check_transaction_response_raises_on_top_level_error():
    with pytest.raises(RuntimeError, match="boom"):
        _check_transaction_response({"error": "boom"})


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------


async def test_merge_concats_text_unions_positions_resequences(patch_execute):
    db = patch_execute(
        [
            _chunk_row(1, 0, "Alpha", [[1, 0.0, 0.5, 0.0, 0.2]]),
            _chunk_row(2, 1, "Beta", [[1, 0.0, 0.5, 0.2, 0.4]]),
            _chunk_row(3, 2, "Gamma", [[1, 0.0, 0.5, 0.4, 0.6]]),
        ]
    )
    mut = ChunkMutator()

    merged = await mut.merge(SOURCE_ID, "chunk:c1")

    # AC1: text = concat with separator
    assert merged.text == f"Alpha{MERGE_SEPARATOR}Beta"
    # AC1: positions = union (concatenation) of both boxes
    assert merged.positions == [
        [1, 0.0, 0.5, 0.0, 0.2],
        [1, 0.0, 0.5, 0.2, 0.4],
    ]
    # AC1: the other chunk is removed
    assert "chunk:c2" not in db.rows
    # AC1: order resequenced — survivor keeps 0, Gamma compacted 2 -> 1
    assert db.rows["chunk:c1"]["order"] == 0
    assert db.rows["chunk:c3"]["order"] == 1
    assert db.rows["chunk:c1"]["source_element_count"] == 2
    assert db.rows["chunk:c1"]["chunk_type"] == "merged"


async def test_merge_with_explicit_adjacent_target(patch_execute):
    db = patch_execute(
        [
            _chunk_row(1, 0, "One"),
            _chunk_row(2, 1, "Two"),
        ]
    )
    mut = ChunkMutator()
    merged = await mut.merge(SOURCE_ID, "chunk:c2", target_chunk_id="chunk:c1")
    # Lower-order chunk (c1) survives.
    assert str(merged.id) == "chunk:c1"
    assert merged.text == f"One{MERGE_SEPARATOR}Two"
    assert "chunk:c2" not in db.rows


async def test_merge_last_chunk_has_no_successor(patch_execute):
    patch_execute([_chunk_row(1, 0, "Only")])
    mut = ChunkMutator()
    with pytest.raises(ChunkMutationError, match="No following chunk"):
        await mut.merge(SOURCE_ID, "chunk:c1")


async def test_merge_non_adjacent_target_rejected(patch_execute):
    patch_execute(
        [
            _chunk_row(1, 0, "A"),
            _chunk_row(2, 1, "B"),
            _chunk_row(3, 2, "C"),
        ]
    )
    mut = ChunkMutator()
    with pytest.raises(ChunkMutationError, match="adjacent"):
        await mut.merge(SOURCE_ID, "chunk:c1", target_chunk_id="chunk:c3")


# ---------------------------------------------------------------------------
# Split
# ---------------------------------------------------------------------------


async def test_split_at_offset_creates_two_chunks(patch_execute):
    db = patch_execute(
        [
            _chunk_row(1, 0, "HelloWorld", [[1, 0.0, 1.0, 0.0, 1.0]]),
            _chunk_row(2, 1, "Next", [[1, 0.0, 1.0, 0.0, 0.1]]),
        ]
    )
    mut = ChunkMutator()

    original, new_chunk = await mut.split(SOURCE_ID, "chunk:c1", 5)

    # AC2: text split at the offset
    assert original.text == "Hello"
    assert new_chunk.text == "World"
    # AC2: proportional positions (fraction 5/10 = 0.5 → cut at y=0.5)
    assert original.positions == [[1, 0.0, 1.0, 0.0, 0.5]]
    assert new_chunk.positions == [[1, 0.0, 1.0, 0.5, 1.0]]
    # AC2: new chunk inserted right after (order+1); later chunk shifted up
    assert original.order == 0
    assert new_chunk.order == 1
    assert db.rows["chunk:c2"]["order"] == 2
    assert new_chunk.metadata.get("split_from") == "chunk:c1"


async def test_split_offset_out_of_range_rejected(patch_execute):
    patch_execute([_chunk_row(1, 0, "abc")])
    mut = ChunkMutator()
    with pytest.raises(ChunkMutationError, match="strictly inside"):
        await mut.split(SOURCE_ID, "chunk:c1", 0)
    with pytest.raises(ChunkMutationError, match="strictly inside"):
        await mut.split(SOURCE_ID, "chunk:c1", 3)


async def test_split_unknown_chunk_rejected(patch_execute):
    patch_execute([_chunk_row(1, 0, "abc")])
    mut = ChunkMutator()
    with pytest.raises(ChunkMutationError, match="not found"):
        await mut.split(SOURCE_ID, "chunk:does-not-exist", 1)


# ---------------------------------------------------------------------------
# Atomicity (AC3)
# ---------------------------------------------------------------------------


async def test_merge_rollback_leaves_state_intact(patch_execute):
    db = patch_execute(
        [
            _chunk_row(1, 0, "Alpha", [[1, 0.0, 0.5, 0.0, 0.2]]),
            _chunk_row(2, 1, "Beta", [[1, 0.0, 0.5, 0.2, 0.4]]),
            _chunk_row(3, 2, "Gamma"),
        ]
    )
    before = db.snapshot()
    db.fail_on = "DELETE $drop_id"  # blow up inside the merge transaction

    mut = ChunkMutator()
    with pytest.raises(RuntimeError, match="injected"):
        await mut.merge(SOURCE_ID, "chunk:c1")

    # No partial write: every row identical to the pre-merge snapshot.
    assert db.rows == before


async def test_split_rollback_leaves_state_intact(patch_execute):
    db = patch_execute(
        [
            _chunk_row(1, 0, "HelloWorld", [[1, 0.0, 1.0, 0.0, 1.0]]),
            _chunk_row(2, 1, "Next"),
        ]
    )
    before = db.snapshot()
    db.fail_on = "CREATE chunk CONTENT"  # blow up inside the split transaction

    mut = ChunkMutator()
    with pytest.raises(RuntimeError, match="injected"):
        await mut.split(SOURCE_ID, "chunk:c1", 5)

    assert db.rows == before


# ---------------------------------------------------------------------------
# Audit wiring (Track I.H2) — an injected ChunkAuditService is invoked after
# a successful mutation with the before/after chunk state.
# ---------------------------------------------------------------------------


async def test_merge_records_audit_row(patch_execute):
    patch_execute([_chunk_row(1, 0, "Alpha"), _chunk_row(2, 1, "Beta")])
    audit = AsyncMock()
    mut = ChunkMutator(audit=audit)

    merged = await mut.merge(SOURCE_ID, "chunk:c1")

    audit.record.assert_awaited_once()
    kwargs = audit.record.await_args.kwargs
    assert kwargs["op"] == "merge"
    assert kwargs["source_id"] == SOURCE_ID
    assert kwargs["chunk_id"] == "chunk:c1"
    # before = the two pre-merge chunks; after = the survivor
    assert [str(c.id) for c in kwargs["before"]] == ["chunk:c1", "chunk:c2"]
    assert [str(c.id) for c in kwargs["after"]] == [str(merged.id)]


async def test_split_records_audit_row(patch_execute):
    patch_execute(
        [
            _chunk_row(1, 0, "HelloWorld", [[1, 0.0, 1.0, 0.0, 1.0]]),
            _chunk_row(2, 1, "Next", [[1, 0.0, 1.0, 0.0, 0.1]]),
        ]
    )
    audit = AsyncMock()
    mut = ChunkMutator(audit=audit)

    result = await mut.split(SOURCE_ID, "chunk:c1", 5)

    audit.record.assert_awaited_once()
    kwargs = audit.record.await_args.kwargs
    assert kwargs["op"] == "split"
    assert kwargs["source_id"] == SOURCE_ID
    # before = the single original chunk; after = [original, new]
    assert [str(c.id) for c in kwargs["before"]] == ["chunk:c1"]
    assert [str(c.id) for c in kwargs["after"]] == [str(c.id) for c in result]


async def test_mutator_without_audit_still_mutates(patch_execute):
    # No audit injected → the mutation must still complete (loguru-only path).
    db = patch_execute([_chunk_row(1, 0, "One"), _chunk_row(2, 1, "Two")])
    mut = ChunkMutator()  # audit defaults to None
    merged = await mut.merge(SOURCE_ID, "chunk:c1")
    assert merged.text == f"One{MERGE_SEPARATOR}Two"
    assert "chunk:c2" not in db.rows


async def test_failed_merge_records_no_audit(patch_execute):
    # A rejected mutation (no successor) must not append an audit row.
    patch_execute([_chunk_row(1, 0, "Only")])
    audit = AsyncMock()
    mut = ChunkMutator(audit=audit)
    with pytest.raises(ChunkMutationError):
        await mut.merge(SOURCE_ID, "chunk:c1")
    audit.record.assert_not_awaited()
