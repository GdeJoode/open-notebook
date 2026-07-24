"""Tests for F.7 failure-provenance on the resumable source pipeline.

F.7 delivers the failure-provenance half of the resumable-pipeline phase: on a
stage failure the source records WHICH stage failed (``metadata.failed_stage``)
so a resume knows where to retry, instead of a bare ``failed``. Resumability
itself is already provided by the shipped ``advance_source`` driver (a source
resumes from its ``processing_stage``), which this does not change — ``failed``
stays a parked value from which the operator re-runs the recorded stage.

The `chunked` sub-stage split is deliberately NOT done here (see the PR / plan):
parse+chunk is one atomic INGEST handler, so ``ingested`` already means
"parsed + chunked"; splitting it is a core-ingest refactor with regression risk
and marginal benefit, deferred as F.7b.
"""

from __future__ import annotations

import uuid

import pytest
from app_main.services.source_pipeline import (
    _TERMINAL_OR_PARKED,
    read_failed_stage,
    set_failed_stage,
)

pytestmark = pytest.mark.asyncio


def _uid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:8]}"


async def test_failed_is_a_parked_stage_regression_guard():
    # advance_source must NOT auto-progress a failed source (it needs an explicit
    # re-run of the recorded failed_stage). This locks the resume model F.7 relies on.
    assert "failed" in _TERMINAL_OR_PARKED


@pytest.mark.requires_docker
async def test_set_failed_stage_records_which_stage_failed(live_surrealdb):
    import surrealdb_service.connection as conn
    from surrealdb_service.connection import execute_query

    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    try:
        created = await execute_query(
            "CREATE source SET title = $t, full_text = 'body';",
            {"t": _uid("f7")},
            config=live_surrealdb,
        )
        sid = str(created[0]["id"])

        await set_failed_stage(sid, "embed")

        from app_main.dependencies import get_source_repo

        # The source is marked failed AND the failing stage is recorded.
        assert await get_source_repo().get_processing_stage(sid) == "failed"
        assert await read_failed_stage(sid) == "embed"
    finally:
        conn._pool = orig


@pytest.mark.requires_docker
async def test_read_failed_stage_none_when_not_failed(live_surrealdb):
    import surrealdb_service.connection as conn
    from surrealdb_service.connection import execute_query

    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    try:
        created = await execute_query(
            "CREATE source SET title = $t, full_text = 'body';",
            {"t": _uid("f7-ok")},
            config=live_surrealdb,
        )
        sid = str(created[0]["id"])
        # Never failed → no failed_stage recorded.
        assert await read_failed_stage(sid) is None
    finally:
        conn._pool = orig
