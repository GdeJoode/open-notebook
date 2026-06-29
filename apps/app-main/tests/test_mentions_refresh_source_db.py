"""PL.3 integration: source-scoped ``mentions`` refresh (container).

Exercises ``MentionsProjectionService.refresh_source`` (the incremental,
source-scoped graph refresh the PL.3 auto-chain runs after a source's entity
extraction) against a real SurrealDB testcontainer. Seeds the same miniature
convenant corpus as the global-regenerate suite and asserts the PL.3 acceptance
criteria:

* AC1 — after a refresh for ONE source, that source's ``mentions`` edges exist
  (with NO global regenerate) and the document-graph read returns them.
* AC2 — the refresh is source-SCOPED: only the refreshed source's edges change;
  another source's pre-existing edges are untouched. It is idempotent: re-running
  for the same source yields the identical edge set with no duplicates. R.6 noise
  normalization is preserved (the df==1 singleton spoke never becomes an edge).

The weights are still computed corpus-wide (global IDF / df) — refresh_source
filters the global projection to this source — so a refreshed edge carries the
SAME weight the global regenerate would give it.
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
        "CREATE entity SET canonical_name = $n, name = $n, entity_type = $t, "
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
    await execute_query("DELETE mentions;", config=config)
    await execute_query("DELETE entity;", config=config)
    await execute_query("DELETE source;", config=config)

    c = [await _make_source(config, _u("Convenant")) for _ in range(4)]
    paper = await _make_source(config, _u("Paper"))

    rd = await _make_entity(config, _u("Regio Deal"), "programme", c, status="active")
    bw = await _make_entity(
        config, _u("brede welvaart"), "topic", c[:3], status="active"
    )
    spoke = await _make_entity(
        config, _u("Only Here"), "topic", [c[0]], status="active"
    )
    return {"convenanten": c, "paper": paper, "rd": rd, "bw": bw, "spoke": spoke}


@pytest.mark.requires_docker
async def test_refresh_source_creates_only_this_sources_edges(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC1: refresh ONE source -> its edges exist, with no global regenerate."""
    seed = await _seed_corpus(live_surrealdb)
    svc = _service(live_surrealdb)
    c0 = seed["convenanten"][0]

    result = await svc.refresh_source(c0)
    assert result.failed == 0
    # c0 shares Regio Deal (df=4) + brede welvaart (df=3) -> 2 edges for c0.
    assert result.created == 2

    # AC1: the document-graph read returns this source's edges.
    edges = await svc.get_document_entity_edges(source_id=c0)
    assert {e["target"] for e in edges} == {seed["rd"], seed["bw"]}
    assert all(float(e["weight"]) > 0.0 for e in edges)

    # The refresh is source-scoped: NO other source got edges (only c0 refreshed).
    total = await execute_query(
        "SELECT count() AS c FROM mentions GROUP ALL;", config=live_surrealdb
    )
    assert (total[0]["c"] if total else 0) == 2

    # R.6 preserved: the df==1 spoke is never an edge endpoint.
    assert seed["spoke"] not in {e["target"] for e in edges}


@pytest.mark.requires_docker
async def test_refresh_source_is_scoped_and_idempotent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC2: refreshing one source leaves another's edges intact; re-run is a no-op."""
    seed = await _seed_corpus(live_surrealdb)
    svc = _service(live_surrealdb)
    c0, c1 = seed["convenanten"][0], seed["convenanten"][1]

    # Build edges for two distinct sources.
    await svc.refresh_source(c0)
    await svc.refresh_source(c1)

    c1_before = await svc.get_document_entity_edges(source_id=c1)
    assert len(c1_before) == 2  # Regio Deal + brede welvaart

    # Re-refresh c0 only: c1's edges must be byte-identical (scoped write).
    second = await svc.refresh_source(c0)
    assert second.cleared == 2  # cleared exactly c0's prior 2 edges
    assert second.created == 2  # rebuilt the same 2

    c1_after = await svc.get_document_entity_edges(source_id=c1)

    def _key(es):
        return sorted(
            (str(e["source"]), str(e["target"]), round(float(e["weight"]), 9))
            for e in es
        )

    assert _key(c1_before) == _key(c1_after), "another source's edges were disturbed"

    # Idempotent for c0: re-run -> same edge set, no duplicate pairs.
    c0_edges = await svc.get_document_entity_edges(source_id=c0)
    pairs = [(str(e["source"]), str(e["target"])) for e in c0_edges]
    assert len(pairs) == len(set(pairs))
    assert {e["target"] for e in c0_edges} == {seed["rd"], seed["bw"]}


@pytest.mark.requires_docker
async def test_refresh_weights_match_global_regenerate(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The scoped refresh carries the SAME (corpus-wide IDF) weights as the
    global regenerate — refresh_source filters the global projection, it does not
    recompute a single-source figure."""
    seed = await _seed_corpus(live_surrealdb)
    svc = _service(live_surrealdb)
    c0 = seed["convenanten"][0]

    # Global regenerate, snapshot c0's weights.
    await svc.regenerate()
    global_edges = await svc.get_document_entity_edges(source_id=c0)
    global_w = {
        str(e["target"]): round(float(e["weight"]), 9) for e in global_edges
    }

    # Re-seed clean, then source-scoped refresh c0.
    await execute_query("DELETE mentions;", config=live_surrealdb)
    await svc.refresh_source(c0)
    scoped_edges = await svc.get_document_entity_edges(source_id=c0)
    scoped_w = {
        str(e["target"]): round(float(e["weight"]), 9) for e in scoped_edges
    }

    assert scoped_w == global_w
