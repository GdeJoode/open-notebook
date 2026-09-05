"""U.2 integration: regenerate the `mentions` document↔entity edges (container).

Runs the ``MentionsProjectionService`` against a real SurrealDB testcontainer
(so migration 66's ``TYPE RELATION FROM source TO entity`` and the RELATE/clear
semantics are exercised, not mocked). Seeds a miniature of the staging corpus —
a shared programme across four "convenant" sources plus a singleton spoke and an
entity-less "paper" source — and asserts the U.2 acceptance criteria:

* AC1/AC4 — edges are created from ``entity.source_documents`` with R.6 filtering
  (the df==1 singleton spoke is dropped; df≥2 concepts survive).
* AC2 — each edge carries a weight (R.2 salience × rarity).
* AC3 — regenerating is idempotent: a second regenerate yields the identical
  edge set with zero duplicate edges.
* AC5 — the canonical ``entity`` / ``source`` rows are NOT mutated by the
  regenerate (only the ``mentions`` table is written).
* AC6 — a ``document -> mentions -> entity <- mentions <- document`` traversal
  returns the convenant cluster (the shared-entity K-clique); the entity-less
  source is isolated.
* The dry-run path writes nothing.

Gated behind ``@requires_docker`` like the other roundtrip suites.
"""

from __future__ import annotations

import uuid

import pytest
from app_main.services.mentions_projection_service import MentionsProjectionService
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.entity import EntityRepository
from surrealdb_service.repositories.source import SourceRepository

pytestmark = pytest.mark.asyncio


def _u(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


async def _make_source(config: SurrealDBConfig, title: str) -> str:
    rows = await execute_query(
        "CREATE source SET title = $t;", {"t": title}, config=config
    )
    return str(rows[0]["id"])


async def _make_entity(
    config: SurrealDBConfig,
    name: str,
    etype: str,
    source_docs: list[str],
    *,
    status: str = "active",
) -> str:
    rows = await execute_query(
        "CREATE entity SET canonical_name = $n, name_key = string::lowercase(string::trim($n)), name = $n, entity_type = $t, "
        "embedding = [], status = $s, source_documents = $sd;",
        {"n": name, "t": etype, "s": status, "sd": source_docs},
        config=config,
    )
    return str(rows[0]["id"])


def _service(config: SurrealDBConfig) -> MentionsProjectionService:
    return MentionsProjectionService(
        entity_repo=EntityRepository(config=config),
        source_repo=SourceRepository(config=config),
    )


async def _seed_corpus(config: SurrealDBConfig) -> dict:
    """Four convenant sources sharing a programme; a spoke; an isolated paper.

    The testcontainer DB is shared across the session, so each test starts from a
    pristine corpus: clear ``mentions`` + the ``entity`` / ``source`` rows so the
    edge counts and the IDF source-count denominator are exact and isolated from
    other tests' seeds. (This is a dedicated throwaway test DB, never staging.)
    """
    await execute_query("DELETE mentions;", config=config)
    await execute_query("DELETE entity;", config=config)
    await execute_query("DELETE source;", config=config)

    c = [
        await _make_source(config, _u("Convenant"))
        for _ in range(4)
    ]
    paper = await _make_source(config, _u("Paper"))

    # Shared named programme across all four convenanten (df=4).
    rd = await _make_entity(
        config, _u("Regio Deal"), "programme", c, status="active"
    )
    # A generic topic across three convenanten (df=3, down-weighted, kept).
    bw = await _make_entity(
        config, _u("brede welvaart"), "topic", c[:3], status="active"
    )
    # A df==1 singleton spoke — must be dropped by R.6.
    spoke = await _make_entity(
        config, _u("Only Here"), "topic", [c[0]], status="active"
    )
    # An archived entity on the paper — must NOT contribute (active-only).
    await _make_entity(
        config, _u("Archived Thing"), "programme", [paper], status="archived"
    )
    return {
        "convenanten": c,
        "paper": paper,
        "rd": rd,
        "bw": bw,
        "spoke": spoke,
    }


@pytest.mark.requires_docker
async def test_dry_run_writes_nothing(live_surrealdb: SurrealDBConfig) -> None:
    seed = await _seed_corpus(live_surrealdb)
    svc = _service(live_surrealdb)

    edges, n_sources = await svc.project_edges()
    # df=4 (Regio Deal) + df=3 (brede welvaart) = 7 edges; spoke dropped.
    rd_and_bw = {seed["rd"], seed["bw"]}
    assert {e.entity_id for e in edges} <= rd_and_bw
    assert len(edges) == 7
    # Nothing persisted by a projection-only call.
    count = await execute_query(
        "SELECT count() AS c FROM mentions GROUP ALL;", config=live_surrealdb
    )
    assert (count[0]["c"] if count else 0) == 0


@pytest.mark.requires_docker
async def test_regenerate_creates_weighted_edges_from_source_documents(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC1/AC2/AC4: edges created from source_documents, weighted, singleton dropped."""
    seed = await _seed_corpus(live_surrealdb)
    svc = _service(live_surrealdb)

    result = await svc.regenerate()
    assert result.failed == 0
    assert result.created == 7  # 4 (Regio Deal) + 3 (brede welvaart)

    edges = await execute_query(
        "SELECT in AS source, out AS target, weight, concept_name, "
        "document_frequency FROM mentions;",
        config=live_surrealdb,
    )
    assert len(edges) == 7
    # AC2: every edge carries a positive weight.
    assert all(float(e["weight"]) > 0.0 for e in edges)
    # AC4: the singleton spoke entity is never an endpoint.
    assert seed["spoke"] not in {e["target"] for e in edges}
    # The named programme (df=4) outweighs the generic topic (df=3).
    rd_w = next(float(e["weight"]) for e in edges if e["target"] == seed["rd"])
    bw_w = next(float(e["weight"]) for e in edges if e["target"] == seed["bw"])
    assert rd_w > bw_w


@pytest.mark.requires_docker
async def test_regenerate_is_idempotent(live_surrealdb: SurrealDBConfig) -> None:
    """AC3: re-running yields the identical edge set, zero duplicates."""
    await _seed_corpus(live_surrealdb)
    svc = _service(live_surrealdb)

    first = await svc.regenerate()
    edges1 = await execute_query(
        "SELECT in AS s, out AS t, weight FROM mentions;", config=live_surrealdb
    )
    second = await svc.regenerate()
    edges2 = await execute_query(
        "SELECT in AS s, out AS t, weight FROM mentions;", config=live_surrealdb
    )

    assert first.created == second.created
    # Second run cleared exactly what the first created (no growth).
    assert second.cleared == first.created

    def _key(es):
        return sorted((e["s"], e["t"], round(float(e["weight"]), 9)) for e in es)

    assert _key(edges1) == _key(edges2)
    # No duplicate (source, entity) pairs.
    pairs = [(e["s"], e["t"]) for e in edges2]
    assert len(pairs) == len(set(pairs))


@pytest.mark.requires_docker
async def test_canonical_rows_untouched(live_surrealdb: SurrealDBConfig) -> None:
    """AC5: regenerate writes only `mentions`; entity/source rows unchanged."""
    seed = await _seed_corpus(live_surrealdb)
    svc = _service(live_surrealdb)

    def _snapshot_query(table: str) -> str:
        return f"SELECT id, updated_at FROM {table} ORDER BY id;"

    before_ent = await execute_query(
        "SELECT id, canonical_name, entity_type, source_documents, status, "
        "updated_at FROM entity ORDER BY id;",
        config=live_surrealdb,
    )
    before_src = await execute_query(
        "SELECT id, title FROM source ORDER BY id;", config=live_surrealdb
    )

    await svc.regenerate()

    after_ent = await execute_query(
        "SELECT id, canonical_name, entity_type, source_documents, status, "
        "updated_at FROM entity ORDER BY id;",
        config=live_surrealdb,
    )
    after_src = await execute_query(
        "SELECT id, title FROM source ORDER BY id;", config=live_surrealdb
    )

    assert before_ent == after_ent, "entity rows were mutated by regenerate"
    assert before_src == after_src, "source rows were mutated by regenerate"
    # The seeded ids still exist exactly as before.
    assert seed["rd"] in {str(r["id"]) for r in after_ent}


@pytest.mark.requires_docker
async def test_traversal_returns_convenant_cluster(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC6: document -> mentions -> entity <- mentions <- document = the cluster."""
    seed = await _seed_corpus(live_surrealdb)
    svc = _service(live_surrealdb)
    await svc.regenerate()

    c0 = seed["convenanten"][0]
    # Two-hop graph traversal from one convenant out to its co-mentioning docs.
    rows = await execute_query(
        f"SELECT ->mentions->entity<-mentions<-source AS peers FROM {c0};",
        config=live_surrealdb,
    )
    peers = {str(p) for p in (rows[0]["peers"] if rows else [])}
    # The traversal reaches all four convenanten (incl. the start, via the
    # shared Regio Deal / brede welvaart entities).
    for c in seed["convenanten"]:
        assert c in peers, f"convenant {c} not reached by traversal"
    # The entity-less paper is isolated — never reached.
    assert seed["paper"] not in peers


@pytest.mark.requires_docker
async def test_isolated_source_has_no_edges(
    live_surrealdb: SurrealDBConfig,
) -> None:
    seed = await _seed_corpus(live_surrealdb)
    svc = _service(live_surrealdb)
    await svc.regenerate()

    edges = await svc.get_document_entity_edges(source_id=seed["paper"])
    assert edges == []


@pytest.mark.requires_docker
async def test_named_only_preset_collapses(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The 0.3 named-only preset keeps only the named anchor (drops topics)."""
    seed = await _seed_corpus(live_surrealdb)
    svc = _service(live_surrealdb)

    result = await svc.regenerate(named_only=True)
    edges = await execute_query(
        "SELECT out AS target FROM mentions;", config=live_surrealdb
    )
    targets = {e["target"] for e in edges}
    # Only the named programme (Regio Deal) survives 0.3; the topic is gone.
    assert seed["rd"] in targets
    assert seed["bw"] not in targets
    assert result.min_weight == 0.3
