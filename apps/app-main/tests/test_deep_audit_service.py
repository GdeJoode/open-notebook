"""Tests for the F.3 deep audit — conflicting facts + provenance gaps.

The two checks are I/O (a source_verdict query + a relations fetch), so they are
unit-tested by mocking those seams; a light live test exercises run_deep's
append-to-latest path against a testcontainer.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import app_main.services.audit.deep_audit_service as deep_mod
import pytest
from app_main.services.audit.deep_audit_service import DeepAuditService
from shared.models.entity import Relation

pytestmark = pytest.mark.asyncio


def _relation(rid: str, sources: list[str], status: str = "active") -> Relation:
    return Relation(
        id=rid,
        in_entity="entity:a",
        out_entity="entity:b",
        relation_type="WORKS_AT",
        source_documents=sources,
        status=status,
    )


def _service_with_mocks(*, source_ids=None, verdict_rows=None, relations=None):
    audit = AsyncMock()
    audit.source_repo.load_notebook_source_ids = AsyncMock(
        return_value=source_ids or []
    )
    audit.entity_repo.list_relations_for_notebook = AsyncMock(
        return_value=relations or []
    )
    svc = DeepAuditService(audit_service=audit)
    return svc, audit


async def test_conflicting_facts_surfaces_contradiction_verdicts(monkeypatch):
    svc, _ = _service_with_mocks(source_ids=["source:a", "source:b"])
    monkeypatch.setattr(
        deep_mod,
        "execute_query",
        AsyncMock(
            return_value=[
                {"in": "source:a", "out": "source:b", "verdict": "contradicts",
                 "confidence": 0.9}
            ]
        ),
    )
    findings = await svc._conflicting_facts("notebook:n")
    assert len(findings) == 1
    assert findings[0].check_id == "conflicting_facts"
    assert findings[0].severity == "warn"
    assert "source:a" in findings[0].title and "source:b" in findings[0].title


async def test_conflicting_facts_empty_without_sources(monkeypatch):
    svc, _ = _service_with_mocks(source_ids=[])
    # No sources → the query is never issued, no findings.
    monkeypatch.setattr(deep_mod, "execute_query", AsyncMock(return_value=[]))
    assert await svc._conflicting_facts("notebook:n") == []


async def test_provenance_gaps_flags_uncited_relations():
    svc, _ = _service_with_mocks(
        relations=[
            _relation("relation:1", []),  # no evidence → flagged
            _relation("relation:2", ["source:x"]),  # cited → ok
            _relation("relation:3", [], status="merged"),  # not active → skip
        ]
    )
    findings = await svc._provenance_gaps("notebook:n")
    assert len(findings) == 1
    assert findings[0].check_id == "provenance_gaps"
    assert findings[0].subject == "relation:1"


async def test_run_deep_is_fail_soft(monkeypatch):
    # A check that raises must degrade to a check_error finding, not blow up.
    svc, audit = _service_with_mocks(source_ids=["source:a"])
    audit.latest = AsyncMock(
        return_value=type("R", (), {"run_id": "run-1"})()
    )
    audit.append_findings = AsyncMock()
    # Return the enriched snapshot on the final latest() call.
    from shared.models.audit import AuditRunResponse

    audit.latest.side_effect = [
        type("R", (), {"run_id": "run-1"})(),
        AuditRunResponse(notebook_id="notebook:n", run_id="run-1", findings=[]),
    ]
    monkeypatch.setattr(
        deep_mod, "execute_query", AsyncMock(side_effect=RuntimeError("db down"))
    )
    # list_relations also raises to force the second check_error.
    audit.entity_repo.list_relations_for_notebook = AsyncMock(
        side_effect=RuntimeError("db down")
    )

    result = await svc.run_deep("notebook:n")
    # run_deep returns the (mocked) enriched snapshot without raising.
    assert result.run_id == "run-1"
    # Both checks failed → two check_error findings were appended.
    appended = audit.append_findings.await_args.args[2]
    assert len(appended) == 2
    assert all(f.check_id == "check_error" for f in appended)


# -- light live path --------------------------------------------------------


@pytest.mark.requires_docker
async def test_run_deep_on_empty_notebook_is_clean(live_surrealdb):
    import uuid

    from app_main.services.audit.audit_service import AuditService

    nb = f"notebook:deep-{uuid.uuid4().hex[:8]}"
    svc = DeepAuditService(
        audit_service=AuditService(config=live_surrealdb), config=live_surrealdb
    )
    # No verdicts, no relations → a clean deep run that still establishes a run
    # and returns without error.
    result = await svc.run_deep(nb)
    assert result.notebook_id == nb
    assert result.run_id  # a run was established
    assert all(f.check_id != "check_error" for f in result.findings)
