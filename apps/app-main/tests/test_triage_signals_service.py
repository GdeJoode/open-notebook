"""Q.2: effective-degree (weak-edge promotion) + recurrence + affected-set signals.

The operator's promotion model (load-bearing): a weak ``RELATED`` edge does NOT
count toward connectivity from a single document, but the SAME edge recurring
across ``weak_promotion_min_docs`` documents PROMOTES and counts. These tests
assert that flip directly, plus doc-count dedup, affected-set membership, the
single-batched-query guarantee, and the both-endpoints rule.

Two layers:

* **Unit** — a seeded mini-KG behind a fake ``EntityRepository`` that runs the
  REAL promotion arithmetic (structural always counts; weak counts only at
  ``source_documents`` length >= threshold). Fast, no DB; asserts the flip,
  buckets, doc-count dedup, affected set, and that degree is computed with
  exactly ONE call to the repo helper for the batch.
* **Integration** — gated behind ``@requires_docker``; exercises the REAL
  ``effective_degree_for_entities`` SurrealQL (the ``array::len(...) >= min_docs``
  promotion filter) against a live container, so the SQL promotion semantics are
  verified, not mocked.
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Tuple

import pytest
from app_main.services.triage.signals_service import (
    EntitySignals,
    SignalsReport,
    TriageSignalsService,
)
from app_main.services.triage.triage_config import load_triage_config

# Async tests are marked individually so the two pure (sync) tests below do not
# trip pytest-asyncio's "marked asyncio but not async" warning.
asyncio_test = pytest.mark.asyncio


# ----------------------------------------------------------------------------
# Fake repository: a seeded mini-KG that runs the real promotion arithmetic.
# ----------------------------------------------------------------------------


class FakeEntityRepository:
    """In-memory stand-in mirroring ``effective_degree_for_entities`` semantics.

    Edges are ``(src, tgt, predicate, source_documents)``. The promotion rule is
    applied exactly like the SurrealQL: structural predicates always count; a
    non-structural (weak/unmapped) predicate counts only when its
    ``source_documents`` has reached ``weak_promotion_min_docs``. An edge counts
    once for EACH endpoint that is in the batch (both-endpoints rule).

    ``call_count`` records how many times the batched helper was invoked, so the
    test can assert the affected-set degree is ONE batched call (not N).
    """

    def __init__(self) -> None:
        self.edges: List[Tuple[str, str, str, List[str]]] = []
        self.call_count = 0

    def add_edge(
        self, src: str, tgt: str, predicate: str, source_documents: List[str]
    ) -> None:
        self.edges.append((src, tgt, predicate, list(source_documents)))

    async def effective_degree_for_entities(
        self,
        ids: List[str],
        structural_predicates: List[str],
        weak_predicates: List[str],
        weak_promotion_min_docs: int,
    ) -> Dict[str, int]:
        self.call_count += 1
        structural = set(structural_predicates)
        id_set = {str(i) for i in ids}
        degree: Dict[str, int] = {i: 0 for i in id_set}
        for src, tgt, predicate, docs in self.edges:
            is_structural = predicate in structural
            promoted = (not is_structural) and (
                len(set(docs)) >= weak_promotion_min_docs
            )
            if not (is_structural or promoted):
                continue
            for endpoint in {src, tgt}:
                if endpoint in degree:
                    degree[endpoint] += 1
        return degree


def _u(prefix: str) -> str:
    return f"entity:{prefix}{uuid.uuid4().hex[:8]}"


def _service(repo: FakeEntityRepository) -> TriageSignalsService:
    # Real committed config: BIJDRAGT_AAN is structural, RELATED is weak,
    # weak_promotion_min_docs == 2, well_connected_threshold == 3.
    return TriageSignalsService(repo, config=load_triage_config())


# ----------------------------------------------------------------------------
# AC1 — promotion flip (the load-bearing guarantee)
# ----------------------------------------------------------------------------


@asyncio_test
async def test_single_doc_related_does_not_count_then_promotes() -> None:
    """AC1: 2 structural + 1 single-doc RELATED → degree 2 (bucket 1-2);
    once that RELATED reaches source_documents >= 2 → degree 3 (bucket 3+)."""
    repo = FakeEntityRepository()
    svc = _service(repo)

    actor = _u("actor")
    t1, t2, peer = _u("t1"), _u("t2"), _u("peer")

    # Two structural edges (count from a single document each).
    repo.add_edge(actor, t1, "BIJDRAGT_AAN", ["source:doc1"])
    repo.add_edge(actor, t2, "BIJDRAGT_AAN", ["source:doc1"])
    # One weak RELATED edge seen in ONE document → does NOT count yet.
    repo.add_edge(actor, peer, "RELATED", ["source:doc1"])

    degree_before = await svc.effective_degree(actor)
    assert degree_before == 2, "single-doc RELATED must NOT count"
    assert svc.degree_bucket(degree_before) == "1-2"

    # The SAME weak edge now recurs in a second document → PROMOTES.
    repo.edges[-1] = (actor, peer, "RELATED", ["source:doc1", "source:doc2"])

    degree_after = await svc.effective_degree(actor)
    assert degree_after == 3, "multi-doc RELATED must promote and count"
    assert svc.degree_bucket(degree_after) == "3+"


# ----------------------------------------------------------------------------
# AC5 — a structural / promoted-weak edge counts toward BOTH endpoints
# ----------------------------------------------------------------------------


@asyncio_test
async def test_edge_counts_toward_both_endpoints() -> None:
    """AC5: one structural edge contributes degree to both of its endpoints."""
    repo = FakeEntityRepository()
    svc = _service(repo)
    a, b = _u("a"), _u("b")
    repo.add_edge(a, b, "BIJDRAGT_AAN", ["source:doc1"])

    degrees = await repo.effective_degree_for_entities(
        [a, b], svc._structural_predicates, svc._weak_predicates,
        load_triage_config().weak_promotion_min_docs,
    )
    assert degrees[a] == 1
    assert degrees[b] == 1


@asyncio_test
async def test_promoted_weak_edge_counts_for_both_endpoints() -> None:
    """AC5 (weak): a PROMOTED RELATED edge counts for both endpoints."""
    repo = FakeEntityRepository()
    svc = _service(repo)
    a, b = _u("a"), _u("b")
    repo.add_edge(a, b, "RELATED", ["source:d1", "source:d2"])

    da = await svc.effective_degree(a)
    db = await svc.effective_degree(b)
    assert da == 1 and db == 1


# ----------------------------------------------------------------------------
# AC2 — doc-count dedup / idempotency
# ----------------------------------------------------------------------------


def test_doc_count_distinct_and_idempotent() -> None:
    """AC2: an entity in 3 distinct source_documents → doc_count 3; a re-run
    over the same provenance bag keeps it 3 (distinct, no double-count)."""
    repo = FakeEntityRepository()
    svc = _service(repo)

    entity = {"source_documents": ["source:a", "source:b", "source:c"]}
    assert svc.doc_count(entity) == 3
    # Re-run (same bag) is idempotent.
    assert svc.doc_count(entity) == 3
    # Duplicates in the bag collapse.
    dup = {"source_documents": ["source:a", "source:a", "source:b"]}
    assert svc.doc_count(dup) == 2
    # Empty / missing bag → 0.
    assert svc.doc_count({"source_documents": []}) == 0
    assert svc.doc_count({}) == 0


def test_degree_bucket_boundaries() -> None:
    """0 → '0', 1 and 2 → '1-2', well_connected_threshold (3) and up → '3+'."""
    repo = FakeEntityRepository()
    svc = _service(repo)
    assert svc.degree_bucket(0) == "0"
    assert svc.degree_bucket(1) == "1-2"
    assert svc.degree_bucket(2) == "1-2"
    assert svc.degree_bucket(3) == "3+"
    assert svc.degree_bucket(9) == "3+"


# ----------------------------------------------------------------------------
# AC3 — affected set = new + changed-degree + changed-doc-count; AC4 — one call
# ----------------------------------------------------------------------------


@asyncio_test
async def test_affected_set_and_single_batched_query() -> None:
    """AC3: affected = new + changed-degree + changed-doc-count; unchanged
    excluded. AC4: degree is computed in exactly ONE batched repo call."""
    repo = FakeEntityRepository()
    svc = _service(repo)

    new_e = _u("new")
    promoted_a, promoted_b = _u("pa"), _u("pb")
    grew_docs = _u("grew")
    unchanged = _u("same")
    t = _u("theme")

    # `unchanged` has a stable structural edge (degree 1 before AND after).
    repo.add_edge(unchanged, t, "BIJDRAGT_AAN", ["source:d0"])
    # A weak edge between promoted_a/promoted_b crosses the threshold THIS batch.
    repo.add_edge(promoted_a, promoted_b, "RELATED", ["source:d1", "source:d2"])

    batch = {
        new_e: {"source_documents": ["source:dN"]},
        promoted_a: {"source_documents": ["source:d1"]},
        promoted_b: {"source_documents": ["source:d2"]},
        grew_docs: {"source_documents": ["source:dG1", "source:dG2"]},
        unchanged: {"source_documents": ["source:d0"]},
    }

    # Pre-batch snapshot (captured BEFORE the merge, per the contract):
    #  - new_e absent → new
    #  - promoted_a/b: degree 0 pre-merge (weak edge not yet promoted)
    #  - grew_docs: same degree (0) but doc_count grew 1 → 2
    #  - unchanged: degree 1, doc_count 1 (identical post-batch)
    snapshot = {
        promoted_a: {"degree": 0, "doc_count": 1},
        promoted_b: {"degree": 0, "doc_count": 1},
        grew_docs: {"degree": 0, "doc_count": 1},
        unchanged: {"degree": 1, "doc_count": 1},
    }

    repo.call_count = 0
    report = await svc.compute_report(batch, snapshot)

    # AC4: exactly one batched degree call for the whole batch.
    assert repo.call_count == 1, "degree must be ONE batched call, not per-node"

    assert isinstance(report, SignalsReport)
    affected = report.affected
    assert new_e in affected, "new node is affected"
    assert promoted_a in affected and promoted_b in affected, (
        "weak edge crossing the promotion threshold makes both endpoints changed"
    )
    assert grew_docs in affected, "doc-count growth makes a node affected"
    assert unchanged not in affected, "unchanged node excluded"

    # Spot-check the per-id signals shape.
    sig: EntitySignals = report.by_id[promoted_a]
    assert sig.effective_degree == 1  # promoted weak edge now counts
    assert sig.degree_changed is True
    assert report.by_id[unchanged].changed is False


# ----------------------------------------------------------------------------
# Integration — REAL promotion SQL against a live SurrealDB container
# ----------------------------------------------------------------------------


async def _seed_entity(config, name: str, etype: str) -> str:
    from surrealdb_service.connection import execute_query

    rows = await execute_query(
        "CREATE entity SET canonical_name=$n, name=$n, entity_type=$t, "
        "embedding=[], confidence=0.9, source_documents=[], status='active' "
        "RETURN VALUE id;",
        {"n": name, "t": etype},
        config=config,
    )
    return str(rows[0])


async def _relate(config, src: str, tgt: str, rel_type: str, docs: List[str]) -> None:
    from surrealdb_service.connection import execute_query

    await execute_query(
        """
        LET $s = type::thing($sid);
        LET $t = type::thing($tid);
        RELATE $s->relation->$t SET
            relation_type = $rt,
            confidence = 0.5,
            source_documents = $docs,
            properties = {},
            status = 'active';
        """,
        {"sid": src, "tid": tgt, "rt": rel_type, "docs": docs},
        config=config,
    )


@pytest.mark.requires_docker
@asyncio_test
async def test_integration_promotion_flip_and_batched_query(live_surrealdb) -> None:
    """Integration: the REAL ``effective_degree_for_entities`` SQL applies the
    ``array::len(source_documents) >= min_docs`` promotion filter, counts toward
    both endpoints, and resolves the whole batch in ONE round-trip."""
    from surrealdb_service.repositories.entity import EntityRepository

    repo = EntityRepository(config=live_surrealdb)
    cfg = load_triage_config()
    structural = sorted(
        p for p, s in cfg._strength_by_predicate.items() if s == "structural"
    )
    weak = sorted(p for p, s in cfg._strength_by_predicate.items() if s == "weak")

    actor = await _seed_entity(live_surrealdb, _u("Actor"), "Programma")
    t1 = await _seed_entity(live_surrealdb, _u("Theme1"), "BeleidsThema")
    t2 = await _seed_entity(live_surrealdb, _u("Theme2"), "BeleidsThema")
    peer = await _seed_entity(live_surrealdb, _u("Peer"), "Programma")

    await _relate(live_surrealdb, actor, t1, "BIJDRAGT_AAN", ["source:d1"])
    await _relate(live_surrealdb, actor, t2, "BIJDRAGT_AAN", ["source:d1"])
    # Single-doc RELATED → must NOT count.
    await _relate(live_surrealdb, actor, peer, "RELATED", ["source:d1"])

    before = await repo.effective_degree_for_entities(
        [actor], structural, weak, cfg.weak_promotion_min_docs
    )
    assert before[actor] == 2, "single-doc RELATED must not count in SQL"

    # Promote the weak edge to two docs (the Q.2a union outcome).
    from surrealdb_service.connection import execute_query

    await execute_query(
        "UPDATE relation SET source_documents = ['source:d1', 'source:d2'] "
        "WHERE in = type::thing($s) AND out = type::thing($t) "
        "AND relation_type = 'RELATED';",
        {"s": actor, "t": peer},
        config=live_surrealdb,
    )

    after = await repo.effective_degree_for_entities(
        [actor, peer], structural, weak, cfg.weak_promotion_min_docs
    )
    assert after[actor] == 3, "multi-doc RELATED must promote in SQL"
    # AC5: the promoted edge counts toward the peer endpoint too.
    assert after[peer] == 1
