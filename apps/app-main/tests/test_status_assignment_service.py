"""Unit tests for the Q.3 status assignment service.

Covers the full tier x effective-degree truth table, the manual_override-wins
guarantee, and the apply() side-effect gating (write + log emitted on a real
change, suppressed on an unchanged re-run or an operator pin). No database: the
EntityRepository and the status-change log are mocked, so these tests exercise
ONLY the decision logic and the apply gating (migration-level behaviour lives in
test_migration_59.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from unittest.mock import AsyncMock

import pytest
from app_main.services.triage.status_assignment_service import (
    FLAG_CANDIDATE_ACTIVE,
    FLAG_ISOLATED_CORE,
    FLAG_UNSURE_REVIEW,
    STATUS_ACTIVE,
    STATUS_REFERENCE,
    StatusAssignmentService,
)
from app_main.services.triage.triage_config import load_triage_config


@dataclass
class FakeEntity:
    """Minimal entity stand-in for the pure ``assign`` path."""

    id: str = "entity:x"
    status: str = "active"
    manual_override: bool = False


@pytest.fixture
def config():
    """The committed Q.1 config (core_active_min=1, well_connected=3)."""
    return load_triage_config()


def _service(config, *, set_status_returns: bool = True):
    """Build a service with a mocked repo + log; return (service, repo, log)."""
    repo = AsyncMock()
    repo.config = None
    repo.set_status = AsyncMock(return_value=set_status_returns)
    log = AsyncMock()
    log.append = AsyncMock(return_value="status_change_log:1")
    svc = StatusAssignmentService(config, repo, status_change_log=log)
    return svc, repo, log


# ---------------------------------------------------------------------------
# AC1 — active tier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("degree", [1, 2, 3, 7])
def test_active_tier_connected_is_active_no_flags(config, degree):
    svc, _, _ = _service(config)
    d = svc.assign(FakeEntity(status="reference"), "active", degree)
    assert d.status == STATUS_ACTIVE
    assert d.queue_flags == []


def test_active_tier_isolated_is_active_with_gap_flag(config):
    svc, _, _ = _service(config)
    d = svc.assign(FakeEntity(status="reference"), "active", 0)
    assert d.status == STATUS_ACTIVE
    assert d.queue_flags == [FLAG_ISOLATED_CORE]


# ---------------------------------------------------------------------------
# AC2 — reference tier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("degree", [0, 1, 2])
def test_reference_tier_below_threshold_is_reference_no_flags(config, degree):
    svc, _, _ = _service(config)
    d = svc.assign(FakeEntity(status="active"), "reference", degree)
    assert d.status == STATUS_REFERENCE
    assert d.queue_flags == []


@pytest.mark.parametrize("degree", [3, 4, 9])
def test_reference_tier_well_connected_flags_candidate_active(config, degree):
    svc, _, _ = _service(config)
    d = svc.assign(FakeEntity(status="active"), "reference", degree)
    assert d.status == STATUS_REFERENCE
    assert d.queue_flags == [FLAG_CANDIDATE_ACTIVE]


# ---------------------------------------------------------------------------
# AC3 — unsure_review tier
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("degree", [0, 1, 5])
def test_unsure_review_tier_is_reference_always_queued(config, degree):
    svc, _, _ = _service(config)
    d = svc.assign(FakeEntity(status="active"), "unsure_review", degree)
    assert d.status == STATUS_REFERENCE
    assert d.queue_flags == [FLAG_UNSURE_REVIEW]


# ---------------------------------------------------------------------------
# AC4 — manual_override wins (pure decision)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "tier,degree",
    [("active", 5), ("reference", 0), ("unsure_review", 9), ("active", 0)],
)
def test_manual_override_keeps_operator_status_no_flags(config, tier, degree):
    svc, _, _ = _service(config)
    ent = FakeEntity(status="reference", manual_override=True)
    d = svc.assign(ent, tier, degree)
    # Operator pinned 'reference'; even an active-tier degree-5 entity stays put.
    assert d.status == "reference"
    assert d.manual_override is True
    assert d.queue_flags == []
    assert d.changes_status is False


# ---------------------------------------------------------------------------
# AC5 — apply() emits exactly one log row on a real change, none otherwise
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_apply_writes_and_logs_on_status_change(config):
    svc, repo, log = _service(config)
    # active-tier degree 2 on an entity currently 'reference' -> change to active.
    d = svc.assign(FakeEntity(status="reference"), "active", 2)
    assert d.changes_status is True

    wrote = await svc.apply(d, batch_id="batch:1")

    assert wrote is True
    repo.set_status.assert_awaited_once_with(
        "entity:x", STATUS_ACTIVE, respect_override=True
    )
    log.append.assert_awaited_once()
    kwargs = log.append.await_args.kwargs
    assert kwargs["old"] == "reference"
    assert kwargs["new"] == STATUS_ACTIVE
    assert kwargs["batch_id"] == "batch:1"


@pytest.mark.asyncio
async def test_apply_is_noop_when_status_unchanged(config):
    svc, repo, log = _service(config)
    # active-tier degree 2 on an entity ALREADY 'active' -> no change.
    d = svc.assign(FakeEntity(status="active"), "active", 2)
    assert d.changes_status is False

    wrote = await svc.apply(d, batch_id="batch:2")

    assert wrote is False
    repo.set_status.assert_not_awaited()
    log.append.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_is_noop_under_manual_override(config):
    svc, repo, log = _service(config)
    ent = FakeEntity(status="active", manual_override=True)
    # Automation would pick 'reference', but the pin must win: no write, no log.
    d = svc.assign(ent, "unsure_review", 0)

    wrote = await svc.apply(d, batch_id="batch:3")

    assert wrote is False
    repo.set_status.assert_not_awaited()
    log.append.assert_not_awaited()


@pytest.mark.asyncio
async def test_apply_does_not_log_when_db_override_suppresses_write(config):
    """If set_status returns False (pinned at DB layer), no log row is written.

    Guards the race where an entity is pinned AFTER the decision is computed:
    the override-conditioned UPDATE writes nothing, so apply must not record a
    change that did not happen.
    """
    svc, repo, log = _service(config, set_status_returns=False)
    d = svc.assign(FakeEntity(status="reference"), "active", 2)
    assert d.changes_status is True

    wrote = await svc.apply(d, batch_id="batch:4")

    assert wrote is False
    repo.set_status.assert_awaited_once()
    log.append.assert_not_awaited()


# ---------------------------------------------------------------------------
# Idempotency — re-running the same assignment twice writes the log once
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_rerun_after_change_is_idempotent(config):
    """First run flips the status (writes + logs); the second sees it already
    at the target and writes nothing."""
    svc, repo, log = _service(config)

    first = svc.assign(FakeEntity(status="reference"), "active", 2)
    assert await svc.apply(first, batch_id="b") is True

    # Second run: the entity now reads back as 'active' (the prior write).
    second = svc.assign(FakeEntity(status="active"), "active", 2)
    assert await svc.apply(second, batch_id="b") is False

    assert log.append.await_count == 1


# ---------------------------------------------------------------------------
# dict-row support — the service accepts a raw DB row as well as an Entity
# ---------------------------------------------------------------------------

def test_assign_accepts_dict_row(config):
    svc, _, _ = _service(config)
    row = {"id": "entity:d", "status": "active", "manual_override": True}
    d = svc.assign(row, "active", 0)
    assert d.manual_override is True
    assert d.status == "active"
    assert d.entity_id == "entity:d"
