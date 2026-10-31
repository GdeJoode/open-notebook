"""Unit tests for the chunk mutator helpers (Track I.D-3).

Since F2-06 merge and split are a manual run with a new chunk set; that is
tested against a real SurrealDB in ``test_handmatige_run.py``. What stays here:

* ``_split_positions``: the proportional split of bounding boxes;
* ``_check_transaction_response``: the guard ``execute_transaction`` uses, so
  that a statement-level ``ERR`` inside a transaction raises rather than being
  swallowed.
"""

from __future__ import annotations

import pytest
from app_main.services.chunking.chunk_mutator import _split_positions
from surrealdb_service.connection import _check_transaction_response


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
