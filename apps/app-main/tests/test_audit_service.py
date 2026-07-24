"""Unit tests for the F.1 audit checks — pure functions over in-memory lists.

Each check has a positive fixture (triggers it) and a negative (does not). No DB:
the checks are pure, so they take plain Entity / Relation objects + dicts. The
persistence + fetch (run_all / latest) are exercised by the router integration
test against a testcontainer.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app_main.config import AuditThresholds
from app_main.services.audit.audit_service import (
    check_citation_completeness,
    check_community_cohesion,
    check_low_confidence_survivors,
    check_long_pending_orphans,
    check_schema_drift,
    check_stale_sources,
)
from shared.models.entity import Entity, Relation

_T = AuditThresholds()
_NOW = datetime(2026, 7, 24, tzinfo=timezone.utc)


def _entity(**kw) -> Entity:
    defaults = dict(
        id=kw.pop("id", "entity:e1"),
        canonical_name=kw.pop("canonical_name", "Alice"),
        entity_type=kw.pop("entity_type", "Person"),
        confidence=kw.pop("confidence", 1.0),
        status=kw.pop("status", "active"),
        source_documents=kw.pop("source_documents", ["source:s1"]),
    )
    defaults.update(kw)
    return Entity(**defaults)


def _relation(a: str, b: str) -> Relation:
    return Relation(id=f"relation:{a}-{b}", in_entity=a, out_entity=b, relation_type="rel")


# -- citation_completeness --------------------------------------------------


def test_citation_completeness_flags_entity_without_sources():
    findings = check_citation_completeness(
        [_entity(id="entity:x", source_documents=[])], _T
    )
    assert len(findings) == 1
    assert findings[0].check_id == "citation_completeness"
    assert findings[0].severity == "error"
    assert findings[0].subject == "entity:x"


def test_citation_completeness_ignores_cited_and_non_active():
    findings = check_citation_completeness(
        [
            _entity(source_documents=["source:s1"]),  # cited → ok
            _entity(id="entity:m", status="merged", source_documents=[]),  # merged → skip
        ],
        _T,
    )
    assert findings == []


# -- stale_sources ----------------------------------------------------------


def test_stale_sources_flags_old_source():
    old = (_NOW - timedelta(days=_T.stale_days + 10)).isoformat()
    findings = check_stale_sources(
        [{"id": "source:s1", "updated": old, "title": "Old"}], _T, now=_NOW
    )
    assert len(findings) == 1 and findings[0].check_id == "stale_sources"


def test_stale_sources_ignores_fresh_source():
    fresh = (_NOW - timedelta(days=1)).isoformat()
    findings = check_stale_sources(
        [{"id": "source:s1", "updated": fresh}], _T, now=_NOW
    )
    assert findings == []


# -- low_confidence_survivors -----------------------------------------------


def test_low_confidence_flags_barely_surviving_entity():
    c = _T.filter_min_confidence + _T.low_confidence_band / 2
    findings = check_low_confidence_survivors([_entity(confidence=c)], _T)
    assert len(findings) == 1 and findings[0].severity == "warn"


def test_low_confidence_ignores_high_confidence_entity():
    findings = check_low_confidence_survivors([_entity(confidence=0.99)], _T)
    assert findings == []


# -- long_pending_orphans ---------------------------------------------------


def test_long_pending_orphans_flags_old_orphan():
    old = (_NOW - timedelta(days=_T.orphan_days + 5)).isoformat()
    findings = check_long_pending_orphans(
        [{"id": "entity:o", "first_orphaned_at": old}], _T, now=_NOW
    )
    assert len(findings) == 1 and findings[0].check_id == "long_pending_orphans"


def test_long_pending_orphans_ignores_recent_orphan():
    recent = (_NOW - timedelta(days=1)).isoformat()
    findings = check_long_pending_orphans(
        [{"id": "entity:o", "first_orphaned_at": recent}], _T, now=_NOW
    )
    assert findings == []


# -- community_cohesion -----------------------------------------------------


def test_community_cohesion_flags_loose_community():
    # 4 members, only 1 internal edge → density 1/6 ≈ 0.17; floor default 0.15
    # → use a sparser community to be safely under the floor: 5 members, 0 edges.
    entities = [_entity(id=f"entity:{i}", community_id=1) for i in range(5)]
    findings = check_community_cohesion(entities, [], _T)
    assert len(findings) == 1
    assert findings[0].check_id == "community_cohesion"
    assert findings[0].subject == "community:1"


def test_community_cohesion_ignores_dense_community():
    # 3 fully-connected members (3 edges / 3 possible = 1.0) → above floor.
    entities = [_entity(id=f"entity:{i}", community_id=2) for i in range(3)]
    rels = [
        _relation("entity:0", "entity:1"),
        _relation("entity:1", "entity:2"),
        _relation("entity:0", "entity:2"),
    ]
    assert check_community_cohesion(entities, rels, _T) == []


def test_community_cohesion_ignores_tiny_community():
    # 2 members < cohesion_min_size (3) → never judged.
    entities = [_entity(id=f"entity:{i}", community_id=3) for i in range(2)]
    assert check_community_cohesion(entities, [], _T) == []


# -- schema_drift -----------------------------------------------------------


def test_schema_drift_flags_high_fallback_ratio():
    # 3 of 4 active entities use a fallback type → 75% > 30%.
    entities = [
        _entity(id="entity:1", entity_type="unknown"),
        _entity(id="entity:2", entity_type="Entity"),
        _entity(id="entity:3", entity_type=""),
        _entity(id="entity:4", entity_type="Person"),
    ]
    findings = check_schema_drift(entities, _T)
    assert len(findings) == 1 and findings[0].check_id == "schema_drift"
    assert findings[0].count == 3


def test_schema_drift_ignores_healthy_schema():
    entities = [
        _entity(id="entity:1", entity_type="Person"),
        _entity(id="entity:2", entity_type="Organization"),
        _entity(id="entity:3", entity_type="unknown"),  # 1/3 = 33%... > 30%
    ]
    # Make it clearly healthy: 1 fallback of 5.
    entities += [
        _entity(id="entity:4", entity_type="Place"),
        _entity(id="entity:5", entity_type="Event"),
    ]
    assert check_schema_drift(entities, _T) == []


# ---------------------------------------------------------------------------
# Persistence / latest / prune — against a real testcontainer (migration 75).
# ---------------------------------------------------------------------------

import pytest  # noqa: E402
from app_main.services.audit.audit_service import AuditService, _KEEP_RUNS  # noqa: E402
from shared.models.audit import AuditFinding  # noqa: E402

pytestmark = pytest.mark.asyncio


def _uid(prefix: str) -> str:
    # Unique per run — live_surrealdb is session-scoped, so rows accumulate.
    import uuid

    return f"{prefix}-{uuid.uuid4().hex[:8]}"


@pytest.mark.requires_docker
async def test_latest_empty_for_unaudited_notebook(live_surrealdb):
    svc = AuditService(config=live_surrealdb)
    resp = await svc.latest(f"notebook:{_uid('never')}")
    assert resp.run_id == ""
    assert resp.findings == []


@pytest.mark.requires_docker
async def test_clean_run_supersedes_older_findings(live_surrealdb):
    svc = AuditService(config=live_surrealdb)
    nb = f"notebook:{_uid('audit')}"

    await svc._persist(
        nb, "run-1", [AuditFinding(check_id="schema_drift", severity="info", title="drift")]
    )
    # A later CLEAN run must supersede the earlier findings (marker-anchored).
    await svc._persist(nb, "run-2", [])

    latest = await svc.latest(nb)
    assert latest.run_id == "run-2"
    assert latest.findings == []  # the run-1 drift finding is NOT returned


@pytest.mark.requires_docker
async def test_prune_keeps_only_recent_runs(live_surrealdb):
    svc = AuditService(config=live_surrealdb)
    nb = f"notebook:{_uid('prune')}"
    for i in range(_KEEP_RUNS + 3):
        await svc._persist(
            nb, f"run-{i}", [AuditFinding(check_id="stale_sources", severity="info", title="s")]
        )
    from surrealdb_service.connection import ensure_record_id, execute_query

    markers = await execute_query(
        "SELECT run_id FROM audit_findings WHERE notebook = $nb AND check_id = '_run'",
        {"nb": ensure_record_id(nb)},
        live_surrealdb,
    )
    assert len(markers or []) <= _KEEP_RUNS
