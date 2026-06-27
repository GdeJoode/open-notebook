"""U.3 integration: materialize `cites` document↔document edges (container).

Runs the ``CitesMaterializationService`` against a real SurrealDB testcontainer
so migration 67's ``TYPE RELATION FROM source TO source`` + U.3 fields and the
RELATE/clear semantics are exercised, not mocked. Feeds SYNTHETIC
``ParsedReference``s (the Track V boundary) — some confidently matching seeded
in-corpus sources, some external — and asserts only the confident intra-corpus
ones become `cites` edges (the precision contract), with no self-citation.

Acceptance criteria mapped:

* AC1 — confident references (DOI exact / strong title+author) matching seeded
  sources → `cites` source→source edges, each carrying confidence + the matched
  reference text + the match method.
* AC2 — external refs → NO edge; a source never cites itself.
* AC3 — a near-miss (similar but not confident) → NO edge (precision guard).
* AC4 — idempotent: re-running yields the identical edge set, zero duplicates.
* AC6 — canonical `source` rows are NOT mutated by materialization.
* AC7 — empty reference input → clean 0-edge no-op (the current-corpus reality).

Gated behind ``@requires_docker`` like the other roundtrip suites.
"""

from __future__ import annotations

import uuid

import pytest
from app_main.services.cites_materialization_service import (
    CitesMaterializationService,
)
from shared.retrieval.cites_matching import ParsedReference
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.source import SourceRepository

pytestmark = pytest.mark.asyncio


def _u(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


async def _make_source(
    config: SurrealDBConfig,
    title: str,
    *,
    authors: list[str] | None = None,
    doi: str | None = None,
) -> str:
    """Seed a source with bibliographic detail in the FLEXIBLE `metadata` object.

    The `source` table is SCHEMAFULL with no top-level `authors`/`doi` column (a
    `SET authors=...` would be dropped). The enrichment home is the flexible
    `metadata` object — exactly where the loader reads `authors`/`doi` from — so
    the test seeds them there, mirroring how a Zotero/Docling pass would.
    """
    meta: dict = {}
    if authors:
        meta["authors"] = authors
    if doi:
        meta["doi"] = doi
    rows = await execute_query(
        "CREATE source SET title = $t, metadata = $m;",
        {"t": title, "m": meta},
        config=config,
    )
    return str(rows[0]["id"])


def _service(config: SurrealDBConfig) -> CitesMaterializationService:
    return CitesMaterializationService(source_repo=SourceRepository(config=config))


async def _seed_corpus(config: SurrealDBConfig) -> dict:
    """Two papers (citable, with DOIs) + a policy doc — a pristine corpus.

    The testcontainer DB is shared, so clear `cites` + `source` first so the edge
    counts are exact and isolated. (A throwaway test DB, never staging.)
    """
    await execute_query("DELETE cites;", config=config)
    await execute_query("DELETE source;", config=config)

    ali = await _make_source(
        config,
        "Cohesion Policy and Institutional Quality",
        authors=["Ali"],
        doi="10.1111/jcms.13501",
    )
    brakman = await _make_source(
        config,
        "Economics without equilibrium",
        authors=["Brakman", "Garretsen"],
        doi="10.1093/cjres/rsab012",
    )
    policy = await _make_source(
        config, "Convenant Regio Deal Het Hogeland", authors=[], doi=None
    )
    return {"ali": ali, "brakman": brakman, "policy": policy}


@pytest.mark.requires_docker
async def test_confident_intra_corpus_refs_become_cites_edges(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC1: confident matches → cites edges with confidence + reference text."""
    seed = await _seed_corpus(live_surrealdb)
    svc = _service(live_surrealdb)

    refs = {
        # Brakman cites Ali by DOI (intra-corpus) → 1 edge.
        seed["brakman"]: [
            ParsedReference(
                raw_text="Ali, S. (2025). Cohesion Policy. doi:10.1111/jcms.13501",
                doi="10.1111/jcms.13501",
            ),
            # ...and an external work → no edge.
            ParsedReference(
                raw_text="Kaldor, N. (1970). The case for regional policies.",
                title="The case for regional policies",
                authors=("Kaldor",),
            ),
        ],
        # Ali cites Brakman by title+author (no DOI) → 1 edge.
        seed["ali"]: [
            ParsedReference(
                raw_text="Brakman & Garretsen, Economics without equilibrium.",
                title="Economics without equilibrium",
                authors=("Brakman", "Garretsen"),
            ),
        ],
    }
    result = await svc.materialize(refs)

    assert result.failed == 0
    assert result.created == 2
    assert result.external == 1
    assert result.total_references == 3

    edges = await execute_query(
        "SELECT in AS src, out AS tgt, confidence, reference_text, match_method "
        "FROM cites;",
        config=live_surrealdb,
    )
    assert len(edges) == 2
    by_pair = {(e["src"], e["tgt"]): e for e in edges}
    # Brakman -> Ali (DOI, confidence 1.0).
    bru = by_pair[(seed["brakman"], seed["ali"])]
    assert bru["match_method"] == "doi"
    assert abs(float(bru["confidence"]) - 1.0) < 1e-9
    assert "10.1111/jcms.13501" in bru["reference_text"]
    # Ali -> Brakman (title+author).
    alb = by_pair[(seed["ali"], seed["brakman"])]
    assert alb["match_method"] == "title_author"
    assert float(alb["confidence"]) > 0.0
    assert "Economics without equilibrium" in alb["reference_text"]


@pytest.mark.requires_docker
async def test_external_and_self_citation_produce_no_edge(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC2: external refs → no edge; a source never cites itself."""
    seed = await _seed_corpus(live_surrealdb)
    svc = _service(live_surrealdb)

    refs = {
        # Ali's bibliography lists ITSELF (self-citation, must be suppressed)
        # plus a genuine external work.
        seed["ali"]: [
            ParsedReference(
                raw_text="Ali (2025) doi:10.1111/jcms.13501",
                doi="10.1111/jcms.13501",
            ),
            ParsedReference(
                raw_text="Acemoglu & Robinson (2012) Why Nations Fail.",
                title="Why Nations Fail",
                authors=("Acemoglu", "Robinson"),
            ),
        ],
    }
    result = await svc.materialize(refs)

    assert result.created == 0
    assert result.self_citations_skipped == 1
    assert result.external == 1

    edges = await execute_query(
        "SELECT count() AS c FROM cites GROUP ALL;", config=live_surrealdb
    )
    assert (edges[0]["c"] if edges else 0) == 0


@pytest.mark.requires_docker
async def test_near_miss_reference_rejected(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC3: a similar-but-not-confident reference → NO edge (precision guard)."""
    seed = await _seed_corpus(live_surrealdb)
    svc = _service(live_surrealdb)

    refs = {
        # Right author (Ali) but a DIFFERENT title — the dangerous near-miss.
        seed["brakman"]: [
            ParsedReference(
                raw_text="Ali (2024) Cohesion Policy and Regional Development.",
                title="Cohesion Policy and Regional Development Outcomes",
                authors=("Ali",),
            ),
        ],
    }
    result = await svc.materialize(refs)

    assert result.created == 0, "near-miss must NOT fabricate a citation"
    assert result.external == 1
    edges = await execute_query(
        "SELECT count() AS c FROM cites GROUP ALL;", config=live_surrealdb
    )
    assert (edges[0]["c"] if edges else 0) == 0


@pytest.mark.requires_docker
async def test_materialize_is_idempotent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC4: re-running yields the identical edge set, zero duplicates."""
    seed = await _seed_corpus(live_surrealdb)
    svc = _service(live_surrealdb)

    refs = {
        seed["brakman"]: [
            ParsedReference(
                raw_text="Ali doi:10.1111/jcms.13501",
                doi="10.1111/jcms.13501",
            ),
        ],
    }
    first = await svc.materialize(refs)
    edges1 = await execute_query(
        "SELECT in AS s, out AS t, confidence FROM cites;", config=live_surrealdb
    )
    second = await svc.materialize(refs)
    edges2 = await execute_query(
        "SELECT in AS s, out AS t, confidence FROM cites;", config=live_surrealdb
    )

    assert first.created == second.created == 1
    # Second run cleared exactly what the first created (no growth).
    assert second.cleared == first.created

    def _key(es):
        return sorted((e["s"], e["t"], round(float(e["confidence"]), 9)) for e in es)

    assert _key(edges1) == _key(edges2)
    pairs = [(e["s"], e["t"]) for e in edges2]
    assert len(pairs) == len(set(pairs))


@pytest.mark.requires_docker
async def test_canonical_source_rows_untouched(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC6: materialize writes only `cites`; source rows are unchanged."""
    seed = await _seed_corpus(live_surrealdb)
    svc = _service(live_surrealdb)

    before = await execute_query(
        "SELECT id, title, metadata FROM source ORDER BY id;",
        config=live_surrealdb,
    )
    refs = {
        seed["brakman"]: [
            ParsedReference(
                raw_text="Ali doi:10.1111/jcms.13501",
                doi="10.1111/jcms.13501",
            ),
        ],
    }
    await svc.materialize(refs)
    after = await execute_query(
        "SELECT id, title, metadata FROM source ORDER BY id;",
        config=live_surrealdb,
    )
    assert before == after, "source rows were mutated by materialize"


@pytest.mark.requires_docker
async def test_empty_input_is_clean_no_op(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC7: no reference input → 0 edges, no crash (the current-corpus reality).

    This is the load-bearing infrastructure check: the mechanism runs against the
    seeded corpus with NO references (as on staging today, where Track V doesn't
    exist) and produces a clean 0-edge result rather than crashing.
    """
    await _seed_corpus(live_surrealdb)
    svc = _service(live_surrealdb)

    result = await svc.materialize({})
    assert result.created == 0
    assert result.total_references == 0
    assert result.external == 0
    assert result.self_citations_skipped == 0
    # sources were loaded (the matcher had a corpus), but produced no edges.
    assert result.sources_in_corpus == 3

    edges = await execute_query(
        "SELECT count() AS c FROM cites GROUP ALL;", config=live_surrealdb
    )
    assert (edges[0]["c"] if edges else 0) == 0


@pytest.mark.requires_docker
async def test_get_citation_edges_reads_back(
    live_surrealdb: SurrealDBConfig,
) -> None:
    seed = await _seed_corpus(live_surrealdb)
    svc = _service(live_surrealdb)

    refs = {
        seed["brakman"]: [
            ParsedReference(
                raw_text="Ali doi:10.1111/jcms.13501",
                doi="10.1111/jcms.13501",
            ),
        ],
    }
    await svc.materialize(refs)

    all_edges = await svc.get_citation_edges()
    assert len(all_edges) == 1
    assert all_edges[0]["source"] == seed["brakman"]
    assert all_edges[0]["target"] == seed["ali"]

    # Scoped to the cited source — the edge is incident on Ali too.
    scoped = await svc.get_citation_edges(source_id=seed["ali"])
    assert len(scoped) == 1

    # A confidence floor above the edge's confidence filters it out.
    above = await svc.get_citation_edges(min_confidence=1.0001)
    assert above == []
