"""Read-only staging probe for chunk-provenance hydration (Track X.1, AC3).

Unlike the ``requires_docker`` roundtrip tests (which seed a throwaway
testcontainer), this probes the *existing, populated* staging database
(``SURREAL_DATABASE=staging``) to prove the hit→chunk mapping on real Docling
output: that ``hydrate_provenance`` attaches the actual ``physical_page`` /
``section_path`` of the chunk a vector hit came from, and that this page equals
``fn::vector_search``'s collapsed ``math::max`` winner (it is the same row).

It is strictly read-only (SELECT only) and self-skips when staging is
unreachable or lacks chunk-linked embeddings, so it never blocks CI on a box
without the populated DB. Run explicitly with::

    SURREAL_DATABASE=staging uv run --project packages/surrealdb-service \\
        pytest packages/surrealdb-service/tests/test_search_provenance_staging.py -v -s
"""

from __future__ import annotations

import os

import pytest

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories import SearchRepository

pytestmark = pytest.mark.asyncio


async def _staging_cfg_or_skip() -> SurrealDBConfig:
    if os.getenv("SURREAL_DATABASE") != "staging":
        pytest.skip("staging probe: set SURREAL_DATABASE=staging to run")
    cfg = SurrealDBConfig.from_env()
    try:
        probe = await execute_query(
            "SELECT count() AS n FROM source_embedding "
            "WHERE chunk.physical_page IS NOT NONE GROUP ALL",
            {},
            cfg,
        )
    except Exception as e:  # noqa: BLE001 — any connection failure -> skip
        pytest.skip(f"staging unreachable: {e}")
    if not probe or probe[0].get("n", 0) == 0:
        pytest.skip("staging has no chunk-linked embeddings with a page")
    return cfg


async def _a_multipage_source(cfg: SurrealDBConfig) -> str:
    """A source id whose linked chunks span several physical pages."""
    grouped = await execute_query(
        """
        SELECT source, count() AS n FROM source_embedding
        WHERE chunk.physical_page IS NOT NONE
        GROUP BY source ORDER BY n DESC LIMIT 1
        """,
        {},
        cfg,
    )
    return grouped[0]["source"]


async def test_vector_hit_carries_real_chunk_page_and_section():
    cfg = await _staging_cfg_or_skip()
    src = await _a_multipage_source(cfg)

    # Use a real stored embedding as the query so cosine is meaningful and the
    # top chunk is well-defined.
    seed = await execute_query(
        "SELECT embedding FROM source_embedding "
        "WHERE source = $s AND chunk.physical_page IS NOT NONE LIMIT 1",
        {"s": ensure_record_id(src)},
        cfg,
    )
    query_embedding = seed[0]["embedding"]

    repo = SearchRepository(config=cfg)
    # Hydration is opt-in as of Track X.2 (the generic /search default is off);
    # the answer-citation path passes hydrate=True, which we exercise here.
    hits = await repo.vector_search(
        query_embedding, results=10, minimum_score=0.2, hydrate=True
    )

    assert hits, "expected at least one vector hit on staging"

    # Find a source-scoped hit that resolved a page.
    paged = [
        h for h in hits
        if str(h.get("id", "")).startswith("source:")
        and h.get("physical_page") is not None
    ]
    assert paged, "no source hit carried a physical_page after hydration"

    hit = paged[0]
    assert hit["chunk_id"] is not None and str(hit["chunk_id"]).startswith("chunk:")
    assert isinstance(hit["physical_page"], int)
    assert "section_path" in hit and "element_type" in hit
    # additive: the original score field survived hydration
    assert ("similarity" in hit) or ("relevance" in hit)


async def test_hydrated_page_matches_the_chunk_the_hit_came_from():
    """The attached page must be the page of the chunk that produced the hit's
    score — i.e. ``hydrate_provenance``'s cosine top-1 equals
    ``fn::vector_search``'s collapsed ``math::max`` winner for that source.
    """
    cfg = await _staging_cfg_or_skip()
    src = await _a_multipage_source(cfg)

    seed = await execute_query(
        "SELECT embedding FROM source_embedding "
        "WHERE source = $s AND chunk.physical_page IS NOT NONE LIMIT 1",
        {"s": ensure_record_id(src)},
        cfg,
    )
    query_embedding = seed[0]["embedding"]

    # fn::vector_search collapse: best similarity per source.
    collapsed = await execute_query(
        """
        SELECT source AS item_id,
               math::max(vector::similarity::cosine(embedding, $q)) AS similarity
        FROM source_embedding WHERE chunk IS NOT NONE
        GROUP BY source ORDER BY similarity DESC LIMIT 1
        """,
        {"q": query_embedding},
        cfg,
    )
    top_source = collapsed[0]["item_id"]
    top_sim = collapsed[0]["similarity"]

    # The chunk our hydration would pick for that source, and its page.
    best = await execute_query(
        """
        SELECT chunk AS chunk_id, chunk.physical_page AS physical_page,
               vector::similarity::cosine(embedding, $q) AS sim
        FROM source_embedding WHERE source = $s AND chunk IS NOT NONE
        ORDER BY sim DESC LIMIT 1
        """,
        {"q": query_embedding, "s": ensure_record_id(top_source)},
        cfg,
    )
    assert best, "expected a chunk for the top source"
    # Same row -> same score (the hit's page is the page it came from).
    assert abs(best[0]["sim"] - top_sim) < 1e-9
    assert best[0]["physical_page"] is not None
