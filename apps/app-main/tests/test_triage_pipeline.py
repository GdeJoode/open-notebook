"""Q.4 pipeline orchestration: B1->B6 with mocked collaborators.

These tests exercise the orchestration logic of ``TriagePipeline.run`` in
isolation — the FROZEN reuse services (signals, status, queue, backbone,
candidate dedup) are mocked, so the assertions are about the FLOW: affected-set
construction (batch + neighbours), status assignment per affected entity, queue
upserts gated on queue-flags, match-conflict surfacing, backbone wiring, and the
non-blocking failure guard. DB-backed idempotency is proven separately in
``test_triage_pipeline_integration.py`` (run-twice).
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock

import pytest
from app_main.services.triage.signals_service import EntitySignals, SignalsReport
from app_main.services.triage.status_assignment_service import (
    FLAG_ISOLATED_CORE,
    FLAG_UNSURE_REVIEW,
    STATUS_ACTIVE,
    STATUS_REFERENCE,
    StatusDecision,
)
from app_main.services.triage.triage_config import load_triage_config
from app_main.services.triage.triage_pipeline import TriagePipeline

pytestmark = pytest.mark.asyncio


def _sig(eid: str, degree: int, doc_count: int = 1) -> EntitySignals:
    bucket = "0" if degree == 0 else ("1-2" if degree < 3 else "3+")
    return EntitySignals(
        entity_id=eid,
        effective_degree=degree,
        bucket=bucket,  # type: ignore[arg-type]
        doc_count=doc_count,
        is_new=True,
        degree_changed=True,
        doc_count_changed=True,
    )


def _build(
    *,
    rows: List[Dict[str, Any]],
    neighbours_edges: List[Dict[str, Any]] | None = None,
    signals_by_id: Dict[str, EntitySignals],
    decisions: Dict[str, StatusDecision],
    candidate_service: Any = None,
):
    """Assemble a pipeline with fully-mocked collaborators."""
    repo = AsyncMock()
    repo.triage_rows_for_entities = AsyncMock(return_value=rows)
    repo.relations_for_entities = AsyncMock(return_value=neighbours_edges or [])

    signals = AsyncMock()
    signals.compute_report = AsyncMock(
        return_value=SignalsReport(
            by_id=signals_by_id, affected=set(signals_by_id)
        )
    )

    status = AsyncMock()
    status.assign = lambda entity, tier, degree: decisions[str(
        entity.get("id") if isinstance(entity, dict) else entity
    )]
    status.apply = AsyncMock(
        side_effect=lambda d, b: d.changes_status
    )

    queue = AsyncMock()
    queue.upsert = AsyncMock(return_value="triage_queue:1")

    backbone = AsyncMock()
    backbone.warnings_for = AsyncMock(return_value=[])

    pipeline = TriagePipeline(
        entity_repository=repo,
        signals_service=signals,
        status_service=status,
        queue_repository=queue,
        backbone_service=backbone,
        candidate_service=candidate_service,
        recanonicalization_service=None,
        config=load_triage_config(),
    )
    return pipeline, repo, signals, status, queue, backbone


async def test_empty_batch_is_noop():
    pipeline, repo, signals, *_ = _build(
        rows=[], signals_by_id={}, decisions={}
    )
    report = await pipeline.run("source:s", [], "b1")
    assert report.ok
    assert report.affected == set()
    signals.compute_report.assert_not_awaited()


async def test_assigns_status_to_every_affected_entity():
    """AC1: B1->B6 runs and a status is assigned per affected entity."""
    rows = [
        {"id": "entity:a", "canonical_name": "Thema", "primary_type": "BeleidsThema",
         "entity_type": "concept", "status": "reference", "manual_override": False,
         "source_documents": ["s:1"]},
        {"id": "entity:b", "canonical_name": "Persoon X", "primary_type": "Person",
         "entity_type": "person", "status": "active", "manual_override": False,
         "source_documents": ["s:1"]},
    ]
    decisions = {
        "entity:a": StatusDecision(
            entity_id="entity:a", status=STATUS_ACTIVE,
            reason="active-tier", queue_flags=[], current_status="reference",
        ),
        "entity:b": StatusDecision(
            entity_id="entity:b", status=STATUS_REFERENCE,
            reason="unsure", queue_flags=[FLAG_UNSURE_REVIEW],
            current_status="active",
        ),
    }
    signals_by_id = {"entity:a": _sig("entity:a", 2), "entity:b": _sig("entity:b", 0)}
    pipeline, repo, signals, status, queue, backbone = _build(
        rows=rows, signals_by_id=signals_by_id, decisions=decisions
    )

    report = await pipeline.run("source:s", ["entity:a", "entity:b"], "b1")
    assert report.ok
    assert report.affected == {"entity:a", "entity:b"}
    # Both decisions changed status -> two writes.
    assert report.statuses_changed == 2
    # Only entity:b raised a queue flag -> one queue upsert.
    assert report.queued == 1
    queue.upsert.assert_awaited_once()
    backbone.warnings_for.assert_awaited_once()


async def test_queue_upsert_uses_decision_flags_and_signals():
    """B5: the queued row carries the reason + degree/doc-count signals."""
    rows = [
        {"id": "entity:core", "canonical_name": "Ministerie", "primary_type": "Ministerie",
         "entity_type": "organization", "status": "reference",
         "manual_override": False, "source_documents": ["s:1"]},
    ]
    decisions = {
        "entity:core": StatusDecision(
            entity_id="entity:core", status=STATUS_ACTIVE,
            reason="active isolated", queue_flags=[FLAG_ISOLATED_CORE],
            current_status="reference",
        ),
    }
    signals_by_id = {"entity:core": _sig("entity:core", 0, doc_count=2)}
    pipeline, *_rest, queue, _backbone = _build(
        rows=rows, signals_by_id=signals_by_id, decisions=decisions
    )

    await pipeline.run("source:s", ["entity:core"], "b1")
    _, kwargs = queue.upsert.call_args
    assert kwargs["reason"] == FLAG_ISOLATED_CORE
    assert kwargs["type"] == "Ministerie"
    assert kwargs["doc_count"] == 2
    assert kwargs["structural_degree"] == 0


async def test_affected_set_includes_relation_neighbours():
    """A neighbour of a batch entity is re-triaged (its degree may have changed)."""
    batch_rows = [
        {"id": "entity:a", "canonical_name": "A", "primary_type": "BeleidsThema",
         "entity_type": "concept", "status": "active", "manual_override": False,
         "source_documents": ["s:1"]},
        {"id": "entity:nbr", "canonical_name": "N", "primary_type": "Ministerie",
         "entity_type": "organization", "status": "reference",
         "manual_override": False, "source_documents": ["s:1"]},
    ]
    neighbours_edges = [
        {"in": "entity:a", "out": "entity:nbr", "relation_type": "BIJDRAGT_AAN",
         "source_documents": ["s:1"]},
    ]
    decisions = {
        "entity:a": StatusDecision("entity:a", STATUS_ACTIVE, "r", [], current_status="active"),
        "entity:nbr": StatusDecision("entity:nbr", STATUS_ACTIVE, "r", [], current_status="reference"),
    }
    signals_by_id = {"entity:a": _sig("entity:a", 1), "entity:nbr": _sig("entity:nbr", 1)}
    pipeline, repo, signals, *_ = _build(
        rows=batch_rows, neighbours_edges=neighbours_edges,
        signals_by_id=signals_by_id, decisions=decisions,
    )

    await pipeline.run("source:s", ["entity:a"], "b1")
    # The signals call must have been handed BOTH the batch id and the neighbour.
    called_rows = signals.compute_report.call_args.kwargs["batch_entities"]
    assert set(called_rows.keys()) == {"entity:a", "entity:nbr"}
    # Empty pre-batch snapshot (R4 choice b — superset re-triage).
    assert signals.compute_report.call_args.kwargs["pre_batch_snapshot"] == {}


async def test_match_conflicts_surfaced_into_queue():
    """B1: a review-band candidate touching the batch becomes a queue row."""

    class _Cand:
        winner_id = "entity:a"
        loser_id = "entity:other"
        id_a = "entity:a"
        id_b = "entity:other"
        name_a = "Org A"
        name_b = "Org A."
        method = "fuzzy"
        score = 0.91

    class _Report:
        review = [_Cand()]
        auto_merge: list = []

    candidate_service = AsyncMock()
    candidate_service.propose_candidates = AsyncMock(return_value=_Report())

    rows = [
        {"id": "entity:a", "canonical_name": "Org A", "primary_type": "Organisatie",
         "entity_type": "organization", "status": "reference",
         "manual_override": False, "source_documents": ["s:1"]},
    ]
    decisions = {
        "entity:a": StatusDecision(
            "entity:a", STATUS_REFERENCE, "unsure", [FLAG_UNSURE_REVIEW],
            current_status="reference",
        ),
    }
    signals_by_id = {"entity:a": _sig("entity:a", 0)}
    pipeline, *_rest, queue, _backbone = _build(
        rows=rows, signals_by_id=signals_by_id, decisions=decisions,
        candidate_service=candidate_service,
    )

    report = await pipeline.run("source:s", ["entity:a"], "b1")
    assert report.match_conflicts == 1
    # Two upserts: the unsure-review flag + the match conflict.
    assert queue.upsert.await_count == 2


async def test_step_failure_is_non_blocking():
    """AC6: a step raising sets ok=False but run() never propagates."""
    rows = [
        {"id": "entity:a", "canonical_name": "A", "primary_type": "BeleidsThema",
         "entity_type": "concept", "status": "active", "manual_override": False,
         "source_documents": ["s:1"]},
    ]
    pipeline, repo, signals, *_ = _build(
        rows=rows,
        signals_by_id={"entity:a": _sig("entity:a", 1)},
        decisions={},
    )
    signals.compute_report = AsyncMock(side_effect=RuntimeError("boom"))

    report = await pipeline.run("source:s", ["entity:a"], "b1")
    assert report.ok is False
    assert "boom" in (report.error or "")
