"""Tests for the F.5 librarian: enqueue-at-seam + consumer runs the audit.

Follows the job-queue-singleton test pattern: assert the ENQUEUE by mocking
``CommandService.submit_command_job`` (the module-level singleton uses the env DB,
not a testcontainer), and exercise the CONSUMER via the real handler with the
global pool swapped to ``live_surrealdb``.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from app_main.services.audit.librarian_service import LibrarianService

pytestmark = pytest.mark.asyncio


def _service(notebook_ids):
    repo = AsyncMock()
    repo.list_librarian_notebook_ids = AsyncMock(return_value=notebook_ids)
    return LibrarianService(notebook_repo=repo)


async def test_enqueue_due_enqueues_one_job_per_opt_in_notebook(monkeypatch):
    submit = AsyncMock(return_value="job:x")
    monkeypatch.setattr(
        "app_main.services.audit.librarian_service.CommandService.submit_command_job",
        submit,
    )
    svc = _service(["notebook:a", "notebook:b"])

    enqueued = await svc.enqueue_due()

    assert enqueued == ["notebook:a", "notebook:b"]
    assert submit.await_count == 2
    # Each call carries the right command + the notebook id.
    kwargs = submit.await_args_list[0].kwargs
    assert kwargs["command_name"] == "run_librarian_audit"
    assert kwargs["command_args"]["notebook_id"] == "notebook:a"


async def test_enqueue_due_enqueues_nothing_when_no_opt_in(monkeypatch):
    submit = AsyncMock()
    monkeypatch.setattr(
        "app_main.services.audit.librarian_service.CommandService.submit_command_job",
        submit,
    )
    svc = _service([])
    assert await svc.enqueue_due() == []
    submit.assert_not_awaited()


async def test_enqueue_due_is_fail_soft(monkeypatch):
    # First submit raises; the second must still be enqueued.
    submit = AsyncMock(side_effect=[RuntimeError("queue down"), "job:2"])
    monkeypatch.setattr(
        "app_main.services.audit.librarian_service.CommandService.submit_command_job",
        submit,
    )
    svc = _service(["notebook:a", "notebook:b"])

    enqueued = await svc.enqueue_due()

    assert enqueued == ["notebook:b"]  # the failed one is dropped, not fatal
    assert submit.await_count == 2


@pytest.mark.requires_docker
async def test_consumer_runs_audit_and_writes_snapshot(live_surrealdb):
    import uuid

    import surrealdb_service.connection as conn
    from surrealdb_service.connection import ensure_record_id, execute_query

    from app_main.handlers import handle_librarian_audit

    nb = f"notebook:lib-{uuid.uuid4().hex[:8]}"
    orig = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    try:
        result = await handle_librarian_audit({"notebook_id": nb})
        assert result["success"] is True
        assert result["run_id"]

        # A fresh audit_findings run marker was written for this notebook.
        markers = await execute_query(
            "SELECT run_id FROM audit_findings "
            "WHERE notebook = $nb AND check_id = '_run'",
            {"nb": ensure_record_id(nb)},
            config=live_surrealdb,
        )
        assert markers and str(markers[0]["run_id"]) == result["run_id"]
    finally:
        conn._pool = orig
