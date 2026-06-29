"""
Search repositories for text and vector search operations.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query

# Provenance keys attached to each hit by ``hydrate_provenance``. Every hit
# carries all of these (Track X.1): present-with-value when the underlying
# chunk has them, ``None`` otherwise. Keeping the key set fixed lets callers
# rely on a stable shape rather than ``key in hit`` probing.
_PROVENANCE_KEYS = (
    "chunk_id",
    "physical_page",
    "printed_page",
    "section_path",
    "element_type",
)


def _hit_source_id(hit: Dict[str, Any]) -> Optional[str]:
    """The ``source:...`` id a hit belongs to, or ``None`` for non-source hits.

    The ``fn::`` functions return ``parent_id`` == the source id for
    source-derived hits (source/source_embedding/source_insight) and the note's
    own id for note hits. We treat a hit as source-scoped only when its
    ``parent_id`` (falling back to ``id``) is a ``source:`` record — notes and
    anything else yield ``None`` and get null provenance.
    """
    candidate = hit.get("parent_id") or hit.get("id")
    if isinstance(candidate, str) and candidate.startswith("source:"):
        return candidate
    return None


class SearchRepository:
    """Repository for search operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        self.config = config

    async def hydrate_provenance(
        self,
        hits: List[Dict[str, Any]],
        embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Attach per-hit chunk provenance to search results in place.

        The ``fn::text_search``/``fn::vector_search`` functions collapse a
        source's many matching ``source_embedding`` rows to a single
        source-level hit (``id`` == the ``source`` id, scored by
        ``math::max(...)``). The page/section provenance, however, lives on the
        ``chunk`` table, reachable via ``source_embedding.chunk`` (migration 27,
        a ``record<chunk>`` link populated by the Docling ingest). This method
        recovers it *without* touching the load-bearing ``fn::`` functions: a
        single batched follow-up ``SELECT`` over ``source_embedding`` joined to
        ``chunk``.

        Precision depends on whether the query embedding is available:

        * **embedding given** (vector / hybrid): re-score each hit-source's
          ``source_embedding`` rows by cosine against ``embedding`` and take the
          top one per source. Because ``fn::vector_search`` collapses with
          ``math::max(cosine)``, this top chunk is *exactly* the row that
          produced the source's winning score — so the attached
          ``physical_page`` is the page of the actual chunk the hit came from
          (verified against staging to 1e-9).
        * **embedding ``None``** (text-only): BM25 ``search::score`` is not
          reproducible outside its originating query context, so we do **not**
          fabricate a specific chunk/page. We attach the source's *first*
          chunk's structural provenance (``section_path``/``element_type``) as a
          best-effort source-level hint and leave ``physical_page`` ``None``
          rather than assert a page we cannot verify.

        Hits whose ``id`` is not a ``source`` record (notes, and any hit lacking
        a chunk-bearing embedding) degrade gracefully: every provenance key is
        set to ``None``. This is additive — existing keys are untouched and
        callers that ignore the new keys are unaffected.

        Args:
            hits: Search-result dicts (each with an ``id``). Mutated in place.
            embedding: The query embedding used for the search, if any. Enables
                exact chunk resolution for vector/hybrid hits.

        Returns:
            The same ``hits`` list, each dict now carrying the provenance keys.
        """
        # Seed every hit with the full key set so the shape is stable even when
        # a hit has no chunk (notes, source-only, lookup failure).
        for hit in hits:
            for key in _PROVENANCE_KEYS:
                hit.setdefault(key, None)
            hit.setdefault("source", _hit_source_id(hit))

        # Only ``source:...`` hits can carry chunk provenance.
        source_ids = sorted(
            {sid for hit in hits if (sid := _hit_source_id(hit)) is not None}
        )
        if not source_ids:
            return hits

        try:
            best_by_source = await self._best_chunk_per_source(
                source_ids, embedding
            )
        except Exception as e:  # provenance is additive — never break a search
            logger.warning(f"Provenance hydration failed (continuing): {e}")
            return hits

        for hit in hits:
            sid = _hit_source_id(hit)
            prov = best_by_source.get(sid) if sid is not None else None
            if prov is None:
                continue
            for key in _PROVENANCE_KEYS:
                if prov.get(key) is not None:
                    hit[key] = prov[key]
        return hits

    async def _best_chunk_per_source(
        self,
        source_ids: List[str],
        embedding: Optional[List[float]],
    ) -> Dict[str, Dict[str, Any]]:
        """Return the provenance of the best chunk per source (batched).

        With ``embedding`` the "best" chunk is the highest-cosine one
        (reproducing ``fn::vector_search``'s ``math::max``); without it we take
        the source's first ``source_embedding`` row as a source-level fallback
        and suppress ``physical_page``/``printed_page`` (unverifiable for BM25).
        """
        record_ids = [ensure_record_id(sid) for sid in source_ids]

        if embedding is not None:
            rows = await execute_query(
                """
                SELECT source,
                       chunk AS chunk_id,
                       chunk.physical_page AS physical_page,
                       chunk.printed_page AS printed_page,
                       chunk.section_path AS section_path,
                       chunk.element_type AS element_type,
                       vector::similarity::cosine(embedding, $embed) AS _sim
                FROM source_embedding
                WHERE source IN $sources AND chunk IS NOT NONE
                ORDER BY _sim DESC
                """,
                {"embed": embedding, "sources": record_ids},
                self.config,
            )
        else:
            rows = await execute_query(
                """
                SELECT source,
                       chunk AS chunk_id,
                       chunk.section_path AS section_path,
                       chunk.element_type AS element_type
                FROM source_embedding
                WHERE source IN $sources AND chunk IS NOT NONE
                ORDER BY order ASC
                """,
                {"sources": record_ids},
                self.config,
            )

        # Rows are pre-sorted (cosine desc, or order asc) — first per source wins.
        best: Dict[str, Dict[str, Any]] = {}
        for row in rows:
            sid = row.get("source")
            if sid is None or sid in best:
                continue
            best[sid] = row
        return best

    async def text_search(
        self,
        keyword: str,
        results: int = 10,
        include_sources: bool = True,
        include_notes: bool = True,
        hydrate: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Perform a text search across sources and notes.

        Args:
            keyword: Search keyword.
            results: Maximum number of results.
            include_sources: Whether to search sources.
            include_notes: Whether to search notes.
            hydrate: Attach per-hit chunk provenance (Track X.1). No embedding
                is available for a text search, so provenance is source-level
                (``section_path``/``element_type``); ``physical_page`` stays
                ``None``. Disabled internally by ``hybrid_search`` so the fused
                result set is hydrated once, with the embedding.

        Returns:
            List of search results.
        """
        if not keyword:
            return []

        try:
            hits = await execute_query(
                """
                SELECT *
                FROM fn::text_search($keyword, $results, $source, $note)
                """,
                {
                    "keyword": keyword,
                    "results": results,
                    "source": include_sources,
                    "note": include_notes,
                },
                self.config,
            )
        except Exception as e:
            logger.error(f"Text search failed: {e}")
            return []

        if hydrate:
            return await self.hydrate_provenance(hits, embedding=None)
        return hits

    async def vector_search(
        self,
        embedding: List[float],
        results: int = 10,
        include_sources: bool = True,
        include_notes: bool = True,
        minimum_score: float = 0.2,
        hydrate: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Perform a vector similarity search.

        Args:
            embedding: Query embedding vector.
            results: Maximum number of results.
            include_sources: Whether to search sources.
            include_notes: Whether to search notes.
            minimum_score: Minimum similarity score.
            hydrate: Attach per-hit chunk provenance (Track X.1) using
                ``embedding`` to resolve the exact matching chunk per source.
                Disabled internally by ``hybrid_search`` so the fused result set
                is hydrated once.

        Returns:
            List of search results with similarity scores.
        """
        try:
            hits = await execute_query(
                """
                SELECT * FROM fn::vector_search($embed, $results, $source, $note, $minimum_score)
                """,
                {
                    "embed": embedding,
                    "results": results,
                    "source": include_sources,
                    "note": include_notes,
                    "minimum_score": minimum_score,
                },
                self.config,
            )
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []

        if hydrate:
            return await self.hydrate_provenance(hits, embedding=embedding)
        return hits

    # Reciprocal Rank Fusion damping constant ``k`` in ``w / (k + rank)``.
    # 60 is the value from the original RRF paper (Cormack, Clarke & Buettcher,
    # SIGIR 2009) and the de-facto default in hybrid-search systems; it matches
    # ``shared.retrieval.hybrid_fusion.DEFAULT_RRF_K`` used by Track R's
    # source-level fusion.
    _RRF_K: int = 60

    async def hybrid_search(
        self,
        keyword: str,
        embedding: List[float],
        results: int = 10,
        include_sources: bool = True,
        include_notes: bool = True,
        minimum_score: float = 0.2,
        text_weight: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """
        Perform a hybrid text + vector search via Reciprocal Rank Fusion.

        The two signals are on different, incomparable scales — BM25
        ``relevance`` (from ``fn::text_search``) is unbounded, cosine
        ``similarity`` (from ``fn::vector_search``) is ``[0, 1]`` — so a linear
        combination of raw scores would let BM25 silently dominate. We fuse by
        **RRF** instead, which consumes each signal's *rank* (not its raw
        score) and is therefore scale-independent::

            fused(item) = text_weight     · 1/(k + rank_text)
                        + (1-text_weight) · 1/(k + rank_vector)

        Each ``fn::`` function already returns its list best-first, so the
        1-based list position is the rank. An item present in only one list
        still ranks (its missing signal contributes ``0``); an item found by
        both is tagged ``hybrid`` and accrues both terms — biasing corroborated
        results upward.

        Args:
            keyword: Search keyword.
            embedding: Query embedding vector.
            results: Maximum number of results.
            include_sources: Whether to search sources.
            include_notes: Whether to search notes.
            minimum_score: Minimum similarity score for vector search.
            text_weight: RRF weight for the text (BM25) signal (0-1); the
                vector signal gets ``1 - text_weight``.

        Returns:
            Combined, deduplicated, RRF-ranked search results. Each carries
            ``_search_type`` (``text``/``vector``/``hybrid``), ``_combined_score``
            (the fused RRF score), ``_text_rank``/``_vector_rank`` provenance
            (the 1-based rank in each list, or ``None`` if absent), and the
            chunk provenance keys (``chunk_id``/``physical_page``/... — Track
            X.1), resolved with ``embedding`` so the page matches the chunk the
            hit came from.
        """
        vector_weight = 1 - text_weight

        # Each fn:: function returns best-first, so list position == rank.
        # Hydrate once on the fused set below (with the embedding) rather than
        # per-leg — so the page reflects the vector match, not the text leg.
        text_results = await self.text_search(
            keyword, results * 2, include_sources, include_notes, hydrate=False
        )
        vector_results = await self.vector_search(
            embedding,
            results * 2,
            include_sources,
            include_notes,
            minimum_score,
            hydrate=False,
        )

        combined: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(text_results, start=1):
            item_id = item.get("id")
            if not item_id or item_id in combined:
                continue
            item["_search_type"] = "text"
            item["_text_rank"] = rank
            item["_vector_rank"] = None
            item["_combined_score"] = text_weight / (self._RRF_K + rank)
            combined[item_id] = item

        for rank, item in enumerate(vector_results, start=1):
            item_id = item.get("id")
            if not item_id:
                continue
            contribution = vector_weight / (self._RRF_K + rank)
            existing = combined.get(item_id)
            if existing is not None:
                existing["_search_type"] = "hybrid"
                existing["_vector_rank"] = rank
                existing["_combined_score"] += contribution
            else:
                item["_search_type"] = "vector"
                item["_text_rank"] = None
                item["_vector_rank"] = rank
                item["_combined_score"] = contribution
                combined[item_id] = item

        # Sort by fused score desc, with a deterministic id tie-break.
        ranked = sorted(
            combined.values(),
            key=lambda x: (-x.get("_combined_score", 0.0), str(x.get("id"))),
        )
        top = ranked[:results]
        # Hydrate the final set once, using the query embedding for exact
        # chunk resolution. Additive — leaves the fusion fields intact.
        return await self.hydrate_provenance(top, embedding=embedding)
