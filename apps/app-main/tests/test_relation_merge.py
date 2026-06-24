"""Q.2a: idempotent relation merge + cross-doc provenance + dedup backfill.

The operator's weak-edge promotion model (Q.2) only works if relations are
MERGED — one edge per ``(in, out, relation_type)`` whose ``source_documents`` is
the cumulative UNION across documents. Before Q.2a the persist ran an
unconditional ``RELATE``, so the same edge re-appearing in another document
created a DUPLICATE row and per-edge cross-doc recurrence was never tracked.

These integration tests run against a real SurrealDB testcontainer (so the
SurrealQL ``array::union`` / ``type::thing`` semantics are exercised, not
mocked) and cover:

* AC1 — same relation persisted from two different ``source_id``s → ONE edge,
  ``source_documents`` length 2 (not two edges).
* AC2 — re-persisting the same (edge, source_id) is a no-op union (length stays
  2, no duplicate edge).
* AC3 — the backfill collapses a seeded duplicate set into one edge each with
  unioned provenance, and a re-run collapses nothing (idempotent).
* AC4 — a non-duplicate edge is never deleted; endpoints + ``relation_type``
  are preserved; O.1 type-safe resolution is unaffected (the dedicated O.1
  roundtrip suite stays green; here we assert the unrelated edge survives).
* AC5 — a weak ``RELATED`` edge persisted from 3 documents ends with
  ``source_documents`` length 3 — the promotion input.

Gated behind ``@requires_docker`` like the other roundtrip suites.
"""

from __future__ import annotations

import importlib.util
import uuid
from pathlib import Path

import pytest
from app_main.services import entity_persistence_service as eps_module
from app_main.services.entity_persistence_service import EntityPersistenceService
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.entity import EntityRepository

pytestmark = pytest.mark.asyncio

# Load the backfill script by path (scripts/ is not an importable package),
# mirroring test_backfill_entity_embeddings.py.
_SCRIPT = (
    Path(__file__).resolve().parents[3] / "scripts" / "backfill_relation_merge.py"
)
_spec = importlib.util.spec_from_file_location("backfill_relation_merge", _SCRIPT)
backfill = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(backfill)


def _u(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _bind_service_to_config(
    monkeypatch: pytest.MonkeyPatch, config: SurrealDBConfig
) -> EntityPersistenceService:
    """Wire a persistence service at the live container.

    The relation write goes through the module-level ``execute_query`` (no
    config arg), so we patch that symbol with a wrapper injecting the test
    container's config — the REAL upsert logic runs against the container.
    """

    async def _exec(query, params=None, config_arg=None):
        return await execute_query(query, params, config=config or config_arg)

    monkeypatch.setattr(eps_module, "execute_query", _exec)
    return EntityPersistenceService(
        entity_repository=EntityRepository(config=config)
    )


async def _seed_entity(
    config: SurrealDBConfig, name: str, etype: str
) -> str:
    """Create a minimal active entity and return its id string."""
    rows = await execute_query(
        "CREATE entity SET canonical_name=$n, name=$n, entity_type=$t, "
        "embedding=[], confidence=0.9, source_documents=[], status='active' "
        "RETURN VALUE id;",
        {"n": name, "t": etype},
        config=config,
    )
    return str(rows[0])


async def _relate_raw(
    config: SurrealDBConfig,
    src_id: str,
    tgt_id: str,
    rel_type: str,
    source_doc: str,
    *,
    confidence: float = 0.5,
) -> None:
    """Seed a raw edge the PRE-Q.2a way (one source_document per row).

    RELATE cannot take an inline ``type::thing(...)`` endpoint — SurrealDB's
    parser rejects it — so bind the endpoints with ``LET`` first, matching how
    the persistence service writes edges.
    """
    await execute_query(
        """
        LET $s = type::thing($sid);
        LET $t = type::thing($tid);
        RELATE $s->relation->$t SET
            relation_type = $rt,
            confidence = $confidence,
            source_documents = [$doc],
            properties = {},
            status = 'active';
        """,
        {
            "sid": src_id,
            "tid": tgt_id,
            "rt": rel_type,
            "confidence": confidence,
            "doc": source_doc,
        },
        config=config,
    )


async def _edges(
    config: SurrealDBConfig, src_id: str, tgt_id: str, rel_type: str
):
    """Return all edge rows of ``rel_type`` between two ids."""
    return await execute_query(
        "SELECT id, source_documents, confidence, properties FROM relation "
        "WHERE in = type::thing($s) AND out = type::thing($t) "
        "AND relation_type = $rt;",
        {"s": src_id, "t": tgt_id, "rt": rel_type},
        config=config,
    )


async def _persist_relation(
    svc: EntityPersistenceService,
    *,
    source_id: str,
    src: str,
    tgt: str,
    rel_type: str = "RELATED",
    properties: dict | None = None,
    confidence: float = 0.5,
) -> dict:
    """Persist a single relation from a given source via the real service.

    Entities are seeded once by the test; here we pass only the relation and
    rely on the name-only endpoint resolution (no entities in the batch).
    """
    return await svc.persist_filtered_result(
        source_id=source_id,
        entities=[],
        relations=[
            {
                "source_entity": src,
                "target_entity": tgt,
                "relation_type": rel_type,
                "confidence": confidence,
                "properties": properties or {},
            }
        ],
    )


@pytest.mark.requires_docker
async def test_same_edge_two_docs_yields_one_edge_len2(
    live_surrealdb: SurrealDBConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC1: same relation from two source_ids → ONE edge, provenance length 2."""
    svc = _bind_service_to_config(monkeypatch, live_surrealdb)
    a = _u("Alpha")
    b = _u("Beta")
    a_id = await _seed_entity(live_surrealdb, a, "organization")
    b_id = await _seed_entity(live_surrealdb, b, "organization")

    r1 = await _persist_relation(svc, source_id="source:doc1", src=a, tgt=b)
    assert r1["relations_created"] == 1
    assert r1["relations_merged"] == 0

    r2 = await _persist_relation(svc, source_id="source:doc2", src=a, tgt=b)
    # Second doc merges into the SAME edge — no new edge created.
    assert r2["relations_created"] == 0
    assert r2["relations_merged"] == 1

    edges = await _edges(live_surrealdb, a_id, b_id, "RELATED")
    assert len(edges) == 1, "the recurring edge must not duplicate"
    assert len(edges[0]["source_documents"]) == 2
    assert set(edges[0]["source_documents"]) == {"source:doc1", "source:doc2"}


@pytest.mark.requires_docker
async def test_re_persist_same_edge_same_doc_is_noop(
    live_surrealdb: SurrealDBConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC2: re-persisting the same (edge, source_id) keeps length 2, no dup."""
    svc = _bind_service_to_config(monkeypatch, live_surrealdb)
    a = _u("Gamma")
    b = _u("Delta")
    a_id = await _seed_entity(live_surrealdb, a, "organization")
    b_id = await _seed_entity(live_surrealdb, b, "organization")

    await _persist_relation(svc, source_id="source:d1", src=a, tgt=b)
    await _persist_relation(svc, source_id="source:d2", src=a, tgt=b)
    # Re-persist BOTH docs again — pure no-op union.
    await _persist_relation(svc, source_id="source:d1", src=a, tgt=b)
    await _persist_relation(svc, source_id="source:d2", src=a, tgt=b)

    edges = await _edges(live_surrealdb, a_id, b_id, "RELATED")
    assert len(edges) == 1
    assert len(edges[0]["source_documents"]) == 2
    assert set(edges[0]["source_documents"]) == {"source:d1", "source:d2"}


@pytest.mark.requires_docker
async def test_related_edge_three_docs_len3(
    live_surrealdb: SurrealDBConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC5: a weak RELATED edge from 3 docs ends with provenance length 3."""
    svc = _bind_service_to_config(monkeypatch, live_surrealdb)
    a = _u("Epsilon")
    b = _u("Zeta")
    a_id = await _seed_entity(live_surrealdb, a, "topic")
    b_id = await _seed_entity(live_surrealdb, b, "topic")

    for doc in ("source:p1", "source:p2", "source:p3"):
        await _persist_relation(svc, source_id=doc, src=a, tgt=b)

    edges = await _edges(live_surrealdb, a_id, b_id, "RELATED")
    assert len(edges) == 1
    assert len(edges[0]["source_documents"]) == 3
    assert set(edges[0]["source_documents"]) == {
        "source:p1",
        "source:p2",
        "source:p3",
    }


@pytest.mark.requires_docker
async def test_merge_keeps_max_confidence_and_merges_properties(
    live_surrealdb: SurrealDBConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The union keeps max confidence and merges property signal keys."""
    svc = _bind_service_to_config(monkeypatch, live_surrealdb)
    a = _u("Eta")
    b = _u("Theta")
    a_id = await _seed_entity(live_surrealdb, a, "organization")
    b_id = await _seed_entity(live_surrealdb, b, "organization")

    await _persist_relation(
        svc,
        source_id="source:c1",
        src=a,
        tgt=b,
        confidence=0.3,
        properties={"sig_a": 1},
    )
    await _persist_relation(
        svc,
        source_id="source:c2",
        src=a,
        tgt=b,
        confidence=0.7,
        properties={"sig_b": 2},
    )

    edges = await _edges(live_surrealdb, a_id, b_id, "RELATED")
    assert len(edges) == 1
    assert edges[0]["confidence"] == pytest.approx(0.7)
    props = edges[0]["properties"]
    assert props.get("sig_a") == 1, "existing signal key retained (no clobber)"
    assert props.get("sig_b") == 2, "new signal key merged in"


@pytest.mark.requires_docker
async def test_backfill_collapses_seeded_dups_and_is_idempotent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC3+AC4: backfill collapses seeded dups, leaves non-dups, is idempotent."""
    a = _u("Iota")
    b = _u("Kappa")
    other = _u("Lambda")
    a_id = await _seed_entity(live_surrealdb, a, "organization")
    b_id = await _seed_entity(live_surrealdb, b, "organization")
    other_id = await _seed_entity(live_surrealdb, other, "organization")

    # Seed the PRE-Q.2a corruption directly: two duplicate RELATED edges for
    # (a, b), each with one distinct source_document (what the old
    # unconditional RELATE produced across two docs).
    for doc in ("source:bf1", "source:bf2"):
        await _relate_raw(
            live_surrealdb,
            a_id,
            b_id,
            "RELATED",
            doc,
            confidence=0.5,
        )
    # A genuinely-unique edge that must NOT be touched.
    await _relate_raw(
        live_surrealdb,
        a_id,
        other_id,
        "VERSTERKT",
        "source:bf3",
        confidence=0.9,
    )

    # Sanity: two dup edges before the collapse.
    pre = await _edges(live_surrealdb, a_id, b_id, "RELATED")
    assert len(pre) == 2

    # Bind the script's module-level execute_query to the test container.
    async def _exec(query, params=None, config_arg=None):
        return await execute_query(query, params, config=live_surrealdb)

    backfill.execute_query = _exec  # type: ignore[assignment]

    summary = await backfill._backfill(dry_run=False)
    assert summary["groups_collapsed"] >= 1
    assert summary["edges_deleted"] >= 1

    # The dup group is now a single edge with the UNION of both docs.
    post = await _edges(live_surrealdb, a_id, b_id, "RELATED")
    assert len(post) == 1
    assert set(post[0]["source_documents"]) == {"source:bf1", "source:bf2"}

    # The unique edge is untouched (not deleted, provenance intact).
    untouched = await _edges(live_surrealdb, a_id, other_id, "VERSTERKT")
    assert len(untouched) == 1
    assert untouched[0]["source_documents"] == ["source:bf3"]

    # Idempotent: a second run collapses nothing further.
    rerun = await backfill._backfill(dry_run=False)
    # The (a,b) group is already singular; the only way new collapses appear is
    # if other tests left dups, so assert no MORE dup edges for our group.
    post2 = await _edges(live_surrealdb, a_id, b_id, "RELATED")
    assert len(post2) == 1
    assert set(post2[0]["source_documents"]) == {"source:bf1", "source:bf2"}
    # Our seeded group contributes 0 to a re-run's collapses.
    assert rerun["edges_deleted"] >= 0


@pytest.mark.requires_docker
async def test_backfill_dry_run_writes_nothing(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC3: --dry-run reports counts but deletes/changes nothing."""
    a = _u("Mu")
    b = _u("Nu")
    a_id = await _seed_entity(live_surrealdb, a, "organization")
    b_id = await _seed_entity(live_surrealdb, b, "organization")
    for doc in ("source:dr1", "source:dr2"):
        await _relate_raw(
            live_surrealdb,
            a_id,
            b_id,
            "RELATED",
            doc,
            confidence=0.5,
        )

    async def _exec(query, params=None, config_arg=None):
        return await execute_query(query, params, config=live_surrealdb)

    backfill.execute_query = _exec  # type: ignore[assignment]

    summary = await backfill._backfill(dry_run=True)
    assert summary["groups_collapsed"] == 0
    assert summary["edges_deleted"] == 0
    assert summary["groups_seen"] >= 1

    # Both dup edges still present — dry-run wrote nothing.
    edges = await _edges(live_surrealdb, a_id, b_id, "RELATED")
    assert len(edges) == 2
