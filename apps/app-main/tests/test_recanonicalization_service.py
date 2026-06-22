"""Unit tests for the K.3 RecanonicalizationService.

Covers the dry-run plan path (cluster grouping, winner-selection rule,
type-aware must-NOT separation, zero-writes) and the apply path's idempotency
using a fake repository that records every write. No DB — the testcontainer
roundtrip lives in ``test_entity_merge_roundtrip.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from app_main.services.entity_resolution import (
    MergeCluster,
    RecanonicalizationService,
)
from shared.models.entity import Entity

# Repo root → tests/fixtures/entity_resolution/must_not_merge.jsonl
_FIXTURES = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "entity_resolution"
)


class FakeEntityRepo:
    """In-memory stand-in for EntityRepository.

    Stores entity rows keyed by id and records every mutating call so tests can
    assert the dry-run path performs zero writes and the apply path is
    idempotent.
    """

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows: Dict[str, Dict[str, Any]] = {
            str(r["id"]): dict(r) for r in rows
        }
        self.writes: List[str] = []  # ordered log of mutating ops

    # --- read paths ---
    async def list_active_entities(
        self, source_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        out = [
            dict(r)
            for r in self._rows.values()
            if r.get("status", "active") == "active"
        ]
        if source_ids is not None:
            out = [
                r
                for r in out
                if set(r.get("source_documents") or []) & set(source_ids)
            ]
        return out

    async def get_entity(self, record_id: str) -> Optional[Entity]:
        row = self._rows.get(str(record_id))
        if row is None:
            return None
        return Entity(**row)

    # --- write paths (all recorded) ---
    async def repoint_relations(self, loser_id: str, winner_id: str) -> int:
        self.writes.append(f"repoint:{loser_id}->{winner_id}")
        return 0

    async def merge_into_winner(
        self, winner_id: str, losers: List[Dict[str, Any]]
    ) -> bool:
        self.writes.append(f"merge_into:{winner_id}")
        winner = self._rows[str(winner_id)]
        srcs = list(winner.get("source_documents") or [])
        aliases = list(winner.get("aliases") or [])
        conf = float(winner.get("confidence") or 0.0)
        for loser in losers:
            for s in loser.get("source_documents") or []:
                if s not in srcs:
                    srcs.append(s)
            name = loser.get("canonical_name")
            if name and name != winner.get("canonical_name") and name not in aliases:
                aliases.append(name)
            conf = max(conf, float(loser.get("confidence") or 0.0))
        winner["source_documents"] = srcs
        winner["aliases"] = aliases
        winner["confidence"] = conf
        return True

    async def register_alias(
        self,
        canonical_entity_id: str,
        alias_text: str,
        match_type: str,
        similarity_score: float,
        method: str = "",
    ) -> bool:
        self.writes.append(f"alias:{canonical_entity_id}:{alias_text}")
        return True

    async def mark_merged(self, loser_id: str, winner_id: str) -> bool:
        self.writes.append(f"mark_merged:{loser_id}")
        self._rows[str(loser_id)]["status"] = "merged"
        self._rows[str(loser_id)]["merged_into"] = str(winner_id)
        return True


def _entity_row(
    rid: str,
    name: str,
    etype: str,
    *,
    confidence: float = 1.0,
    sources: Optional[List[str]] = None,
    created_at: str = "2026-01-01T00:00:00Z",
) -> Dict[str, Any]:
    return {
        "id": rid,
        "canonical_name": name,
        "entity_type": etype,
        "confidence": confidence,
        "source_documents": sources or [],
        "type_tags": [],
        "primary_type": None,
        "provenance_chain": [],
        "properties": {},
        "aliases": [],
        "status": "active",
        "created_at": created_at,
    }


# ---------------------------------------------------------------------------
# plan_merges — grouping
# ---------------------------------------------------------------------------


async def test_plan_groups_bzk_variants_into_one_cluster():
    """BZK / ministerie van BZK / full-form (all org) collapse to one cluster."""
    rows = [
        _entity_row("entity:1", "BZK", "organization", confidence=0.9),
        _entity_row("entity:2", "ministerie van BZK", "organization", confidence=0.8),
        _entity_row(
            "entity:3",
            "Binnenlandse Zaken en Koninkrijksrelaties",
            "organization",
            confidence=0.7,
        ),
    ]
    svc = RecanonicalizationService(entity_repo=FakeEntityRepo(rows))
    plan = await svc.plan_merges()

    assert plan.cluster_count == 1
    cluster = plan.clusters[0]
    assert cluster.size == 3
    assert set(cluster.member_surface_forms) == {
        "BZK",
        "ministerie van BZK",
        "Binnenlandse Zaken en Koninkrijksrelaties",
    }
    # All three normalize to the same key.
    assert cluster.new_canonical == "binnenlandse zaken en koninkrijksrelaties"


async def test_plan_singletons_are_not_clusters():
    """An entity whose normalized form is unique never becomes a cluster."""
    rows = [
        _entity_row("entity:1", "Regio Deal Groningen", "other"),
        _entity_row("entity:2", "Regio Deal Drenthe", "other"),
    ]
    svc = RecanonicalizationService(entity_repo=FakeEntityRepo(rows))
    plan = await svc.plan_merges()
    assert plan.cluster_count == 0


# ---------------------------------------------------------------------------
# plan_merges — winner-selection rule
# ---------------------------------------------------------------------------


async def test_winner_is_highest_confidence():
    rows = [
        _entity_row("entity:low", "BZK", "organization", confidence=0.5),
        _entity_row("entity:high", "ministerie van BZK", "organization", confidence=0.95),
    ]
    svc = RecanonicalizationService(entity_repo=FakeEntityRepo(rows))
    plan = await svc.plan_merges()
    assert plan.clusters[0].winner_id == "entity:high"
    assert plan.clusters[0].loser_ids == ["entity:low"]


async def test_winner_tiebreak_by_source_doc_count():
    """Equal confidence → most source_documents wins."""
    rows = [
        _entity_row("entity:few", "BZK", "organization", confidence=0.8, sources=["s:a"]),
        _entity_row(
            "entity:many",
            "ministerie van BZK",
            "organization",
            confidence=0.8,
            sources=["s:a", "s:b", "s:c"],
        ),
    ]
    svc = RecanonicalizationService(entity_repo=FakeEntityRepo(rows))
    plan = await svc.plan_merges()
    assert plan.clusters[0].winner_id == "entity:many"


async def test_winner_tiebreak_by_oldest_created_at():
    """Equal confidence + equal source count → oldest created_at wins."""
    rows = [
        _entity_row(
            "entity:new",
            "BZK",
            "organization",
            confidence=0.8,
            sources=["s:a"],
            created_at="2026-05-01T00:00:00Z",
        ),
        _entity_row(
            "entity:old",
            "ministerie van BZK",
            "organization",
            confidence=0.8,
            sources=["s:b"],
            created_at="2026-01-01T00:00:00Z",
        ),
    ]
    svc = RecanonicalizationService(entity_repo=FakeEntityRepo(rows))
    plan = await svc.plan_merges()
    assert plan.clusters[0].winner_id == "entity:old"


# ---------------------------------------------------------------------------
# plan_merges — type-aware must-NOT separation (the over-merge guard)
# ---------------------------------------------------------------------------


async def test_person_and_org_same_name_never_group():
    """Grouping is by (new_canonical, ENTITY_TYPE) — cross-type never merges."""
    rows = [
        _entity_row("entity:person", "Minister van BZK", "person"),
        _entity_row("entity:org", "Ministerie van BZK", "organization"),
    ]
    svc = RecanonicalizationService(entity_repo=FakeEntityRepo(rows))
    plan = await svc.plan_merges()
    # Either no cluster (distinct normalized forms) or, if they ever share a
    # form, they must be in DIFFERENT clusters — never the same one.
    for cluster in plan.clusters:
        members = set(cluster.loser_ids) | {cluster.winner_id}
        assert not (
            "entity:person" in members and "entity:org" in members
        ), "person and org with same name must never share a cluster"


async def test_must_not_merge_corpus_no_false_clusters():
    """Every must-NOT pair in the fixture lands in separate clusters."""
    pairs: List[Dict[str, Any]] = []
    with (_FIXTURES / "must_not_merge.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))

    # Build one entity per distinct (name, type) appearing in any pair.
    seen: Dict[tuple, str] = {}
    rows: List[Dict[str, Any]] = []
    for i, pair in enumerate(pairs):
        for side in ("a", "b"):
            ent = pair[side]
            key = (ent["name"], ent["type"])
            if key not in seen:
                rid = f"entity:{len(seen)}"
                seen[key] = rid
                rows.append(_entity_row(rid, ent["name"], ent["type"]))

    svc = RecanonicalizationService(entity_repo=FakeEntityRepo(rows))
    plan = await svc.plan_merges()

    # Map each entity id to its cluster index; pairs must not collide.
    cluster_of: Dict[str, int] = {}
    for idx, cluster in enumerate(plan.clusters):
        for rid in [cluster.winner_id, *cluster.loser_ids]:
            cluster_of[rid] = idx

    for pair in pairs:
        a_id = seen[(pair["a"]["name"], pair["a"]["type"])]
        b_id = seen[(pair["b"]["name"], pair["b"]["type"])]
        if a_id in cluster_of and b_id in cluster_of:
            assert cluster_of[a_id] != cluster_of[b_id], (
                f"FALSE MERGE: {pair['a']} and {pair['b']} share a cluster — "
                f"{pair.get('why', '')}"
            )


# ---------------------------------------------------------------------------
# dry-run guard — zero writes
# ---------------------------------------------------------------------------


async def test_plan_merges_performs_zero_writes():
    rows = [
        _entity_row("entity:1", "BZK", "organization", confidence=0.9),
        _entity_row("entity:2", "ministerie van BZK", "organization", confidence=0.8),
    ]
    repo = FakeEntityRepo(rows)
    svc = RecanonicalizationService(entity_repo=repo)
    await svc.plan_merges()
    assert repo.writes == [], "plan_merges must not write"


async def test_apply_plan_default_is_dry_run_zero_writes():
    """apply_plan() with the default dry_run=True performs ZERO writes."""
    rows = [
        _entity_row("entity:1", "BZK", "organization", confidence=0.9),
        _entity_row("entity:2", "ministerie van BZK", "organization", confidence=0.8),
    ]
    repo = FakeEntityRepo(rows)
    svc = RecanonicalizationService(entity_repo=repo)
    plan = await svc.plan_merges()
    assert plan.cluster_count == 1

    results = await svc.apply_plan(plan)  # default dry_run=True
    assert all(r.skipped for r in results)
    assert repo.writes == [], "apply_plan dry-run must not write"
    # Entities remain active.
    assert repo._rows["entity:2"]["status"] == "active"


# ---------------------------------------------------------------------------
# apply_merge — happy path + idempotency
# ---------------------------------------------------------------------------


async def test_apply_merge_marks_losers_and_creates_aliases():
    rows = [
        _entity_row(
            "entity:win", "BZK", "organization", confidence=0.9, sources=["s:a"]
        ),
        _entity_row(
            "entity:lose",
            "ministerie van BZK",
            "organization",
            confidence=0.8,
            sources=["s:b"],
        ),
    ]
    repo = FakeEntityRepo(rows)
    svc = RecanonicalizationService(entity_repo=repo)
    plan = await svc.plan_merges()
    cluster = plan.clusters[0]

    result = await svc.apply_merge(cluster)
    assert not result.skipped
    assert result.merged_loser_ids == ["entity:lose"]
    assert result.aliases_created == 1
    # Loser is soft-merged.
    assert repo._rows["entity:lose"]["status"] == "merged"
    assert repo._rows["entity:lose"]["merged_into"] == "entity:win"
    # Winner gained the loser's source doc + the loser surface as an alias.
    assert "s:b" in repo._rows["entity:win"]["source_documents"]
    assert "ministerie van BZK" in repo._rows["entity:win"]["aliases"]


async def test_apply_merge_is_idempotent():
    rows = [
        _entity_row("entity:win", "BZK", "organization", confidence=0.9),
        _entity_row("entity:lose", "ministerie van BZK", "organization", confidence=0.8),
    ]
    repo = FakeEntityRepo(rows)
    svc = RecanonicalizationService(entity_repo=repo)
    cluster: MergeCluster = (await svc.plan_merges()).clusters[0]

    first = await svc.apply_merge(cluster)
    assert not first.skipped
    writes_after_first = len(repo.writes)

    # Re-apply the same cluster: losers are already merged → no-op.
    second = await svc.apply_merge(cluster)
    assert second.skipped is True
    assert second.merged_loser_ids == []
    assert len(repo.writes) == writes_after_first, (
        "re-apply must not produce additional writes"
    )
