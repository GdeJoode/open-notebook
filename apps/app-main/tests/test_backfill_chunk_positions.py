"""Unit tests for the chunk-position backfill (Track I.C, AC3).

A live 10K-row run isn't possible in this sandbox, so these tests pin the
load-bearing logic instead: legacy-detection, normalization math, the
paginated batch loop, idempotency, and the missing-PDF fallback. The DB is
mocked at the module's ``execute_query`` seam; page dimensions are stubbed
on ``_PageDimensionCache``.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

# The script lives outside the importable package tree; load it by path.
_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "backfill_chunk_positions.py"
)
_spec = importlib.util.spec_from_file_location("backfill_chunk_positions", _SCRIPT)
backfill_mod = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
# Register before exec so dataclass annotation resolution can find the module.
sys.modules["backfill_chunk_positions"] = backfill_mod
_spec.loader.exec_module(backfill_mod)


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_needs_normalization_detects_raw_points():
    assert backfill_mod._needs_normalization([1, 119.0, 297.5, 168.0, 336.0]) is True


def test_needs_normalization_false_for_already_0_1():
    assert backfill_mod._needs_normalization([1, 0.2, 0.5, 0.2, 0.4]) is False


def test_needs_normalization_ignores_page_index():
    # page=5 is large but coords are 0–1 → not legacy.
    assert backfill_mod._needs_normalization([5, 0.1, 0.2, 0.3, 0.4]) is False


def test_normalize_position_divides_and_clamps():
    out = backfill_mod._normalize_position([2, 119.0, 297.5, 168.0, 336.0], 595.0, 842.0)
    assert out[0] == 2
    assert out[1] == pytest.approx(0.2)
    assert out[2] == pytest.approx(0.5)
    assert out[3] == pytest.approx(168.0 / 842.0)
    assert out[4] == pytest.approx(336.0 / 842.0)
    assert all(0.0 <= c <= 1.0 for c in out[1:5])


def test_normalize_position_clamps_overflow():
    out = backfill_mod._normalize_position([1, -10.0, 700.0, 0.0, 900.0], 595.0, 842.0)
    assert out[1] == 0.0   # negative clamps to 0
    assert out[2] == 1.0   # > width clamps to 1


def test_page_dims_resolves_zero_or_one_indexed():
    dims = {1: (595.0, 842.0)}
    assert backfill_mod._page_dims_for(dims, [1, 0, 0, 0, 0]) == (595.0, 842.0)
    # 0-indexed legacy page → page+1 lookup.
    assert backfill_mod._page_dims_for(dims, [0, 0, 0, 0, 0]) == (595.0, 842.0)
    assert backfill_mod._page_dims_for(dims, [9, 0, 0, 0, 0]) is None


# ---------------------------------------------------------------------------
# Batch loop with mocked DB + dimensions
# ---------------------------------------------------------------------------


class _FakeDB:
    """Records UPDATEs and serves chunk rows in pages."""

    def __init__(self, rows: list[dict]):
        self._rows = rows
        self.updates: dict = {}

    async def execute_query(self, query: str, params=None):
        params = params or {}
        if query.strip().startswith("SELECT"):
            start = params["start"]
            limit = params["limit"]
            return self._rows[start : start + limit]
        if query.strip().startswith("UPDATE"):
            self.updates[str(params["id"])] = params["positions"]
            return []
        return []


@pytest.fixture
def patched(monkeypatch):
    """Wire a fake DB + fixed page dimensions (595x842) into the module."""
    def _apply(rows):
        db = _FakeDB(rows)
        monkeypatch.setattr(backfill_mod, "execute_query", db.execute_query)
        monkeypatch.setattr(
            backfill_mod._PageDimensionCache,
            "get",
            lambda self, pdf_path: {1: (595.0, 842.0), 2: (595.0, 842.0)}
            if pdf_path
            else None,
        )
        return db
    return _apply


async def test_backfill_normalizes_only_legacy_rows(patched):
    rows = [
        {"id": "chunk:a", "positions": [[1, 119.0, 297.5, 168.0, 336.0]], "pdf_path": "/x.pdf"},
        {"id": "chunk:b", "positions": [[1, 0.2, 0.5, 0.2, 0.4]], "pdf_path": "/x.pdf"},
        {"id": "chunk:c", "positions": [], "pdf_path": "/x.pdf"},
    ]
    db = patched(rows)
    stats = await backfill_mod.backfill(batch_size=10)

    assert stats.processed == 3
    assert stats.normalized == 1
    assert stats.already_normalized == 1
    assert stats.skipped_no_positions == 1
    # Only the legacy row was written.
    assert list(db.updates.keys()) == ["chunk:a"]
    written = db.updates["chunk:a"][0]
    assert written[1] == pytest.approx(0.2)
    assert written[2] == pytest.approx(0.5)


async def test_backfill_is_idempotent(patched):
    """Running over already-normalized output produces no further writes."""
    rows = [
        {"id": "chunk:a", "positions": [[1, 0.2, 0.5, 0.2, 0.4]], "pdf_path": "/x.pdf"},
    ]
    db = patched(rows)
    stats = await backfill_mod.backfill(batch_size=10)
    assert stats.normalized == 0
    assert stats.already_normalized == 1
    assert db.updates == {}


async def test_backfill_paginates_across_batches(patched):
    rows = [
        {"id": f"chunk:{i}", "positions": [[1, 119.0, 297.5, 168.0, 336.0]], "pdf_path": "/x.pdf"}
        for i in range(5)
    ]
    db = patched(rows)
    stats = await backfill_mod.backfill(batch_size=2)  # forces 3 SELECT pages
    assert stats.processed == 5
    assert stats.normalized == 5
    assert len(db.updates) == 5


async def test_backfill_skips_rows_without_pdf(monkeypatch):
    rows = [
        {"id": "chunk:a", "positions": [[1, 119.0, 297.5, 168.0, 336.0]], "pdf_path": None},
    ]
    db = _FakeDB(rows)
    monkeypatch.setattr(backfill_mod, "execute_query", db.execute_query)
    monkeypatch.setattr(
        backfill_mod._PageDimensionCache, "get", lambda self, pdf_path: None
    )
    stats = await backfill_mod.backfill(batch_size=10)
    assert stats.normalized == 0
    assert stats.skipped_missing_pdf == 1
    assert db.updates == {}


async def test_backfill_dry_run_writes_nothing(patched):
    rows = [
        {"id": "chunk:a", "positions": [[1, 119.0, 297.5, 168.0, 336.0]], "pdf_path": "/x.pdf"},
    ]
    db = patched(rows)
    stats = await backfill_mod.backfill(batch_size=10, dry_run=True)
    assert stats.normalized == 1  # counted as would-normalize
    assert db.updates == {}       # but nothing written
