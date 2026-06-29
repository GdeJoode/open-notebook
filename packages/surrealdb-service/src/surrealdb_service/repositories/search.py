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


def _hit_parent_source(hit: Dict[str, Any]) -> Optional[str]:
    """The ``source:...`` a hit belongs to (its ``parent_id``), or ``None``.

    The ``fn::`` functions set ``parent_id`` == the owning source id for every
    source-derived hit class (``source_embedding``/``source_insight``/source
    title/full_text) and the note's own id for note hits. This is the
    *source-level* anchor — used to populate the ``source`` field on any
    source-derived hit. It is **not** sufficient to decide chunk-level
    provenance (see ``_hit_is_chunk_backed``): a ``source_insight`` hit also
    carries a ``source:`` parent but has no single originating chunk.
    """
    parent = hit.get("parent_id")
    if isinstance(parent, str) and parent.startswith("source:"):
        return parent
    return None


def _hit_is_chunk_backed(hit: Dict[str, Any]) -> bool:
    """Whether a hit maps to a single, identifiable originating chunk.

    The distinguisher is the hit's **own** ``id`` prefix, not ``parent_id``:

    * ``source:...``  — emitted by the ``source_embedding`` leg of
      ``fn::vector_search`` (``SELECT source.id AS id``); collapses many chunk
      embeddings of one source via ``math::max(cosine)``. The winning chunk is
      recoverable exactly by cosine-argmax, so this class is chunk-backed.
    * ``source_insight:...`` — a synthesized source-level summary with no single
      originating chunk; chunk-level provenance must stay ``None``.
    * ``note:...`` / anything else — not chunk-backed.

    (For a *text* search even a ``source:`` hit may come from the title or
    full_text leg, which has no chunk; chunk-level provenance is suppressed for
    the whole text path separately, by passing ``embedding=None``.)
    """
    own_id = hit.get("id")
    return isinstance(own_id, str) and own_id.startswith("source:")


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
        source-level hit (own ``id`` == the ``source`` id, scored by
        ``math::max(...)``). The page/section provenance, however, lives on the
        ``chunk`` table, reachable via ``source_embedding.chunk`` (migration 27,
        a ``record<chunk>`` link populated by the Docling ingest). This method
        recovers it *without* touching the load-bearing ``fn::`` functions: a
        single batched follow-up ``SELECT`` over ``source_embedding`` joined to
        ``chunk``.

        **One rule — attach chunk-level provenance
        (``chunk_id``/``physical_page``/``printed_page``/``section_path``/
        ``element_type``) only when the EXACT originating chunk is identifiable;
        otherwise leave those keys ``None`` and attach the ``source`` only.**
        Applied by hit class (distinguished by the hit's *own* ``id`` prefix,
        not ``parent_id``):

        * **``source:`` hit WITH an embedding** (vector / hybrid) — the
          ``source_embedding`` leg collapsed many chunk embeddings of one source
          via ``math::max(cosine)``. The winning chunk is recovered exactly by
          cosine-argmax, so its ``physical_page`` is the page of the actual
          chunk the hit came from (verified against staging to 1e-9).
        * **``source_insight:`` hit** — a synthesized source-level summary with
          no single originating chunk. Chunk keys stay ``None``; ``source`` is
          set (from ``parent_id``). Never routed through the chunk lookup.
        * **text-only path** (``embedding is None``) — BM25 ``search::score`` is
          not reproducible outside its originating query, and a ``source:`` text
          hit may even come from the title/full_text leg (no chunk at all). We
          therefore assert no chunk for *any* text hit: chunk keys ``None``,
          ``source`` set. (We deliberately do NOT attach an arbitrary first
          chunk's ``section_path``/``element_type`` — it is not the chunk that
          produced the hit.)
        * **notes / anything else** — chunk keys ``None``, no ``source``.

        This is additive — existing keys are untouched and callers that ignore
        the new keys are unaffected; a lookup failure degrades to all-``None``
        chunk keys without raising.

        Args:
            hits: Search-result dicts (each with an ``id``). Mutated in place.
            embedding: The query embedding used for the search, if any. Its
                presence is also the text-vs-vector signal: ``None`` means
                text-only, so no chunk-level provenance is attached.

        Returns:
            The same ``hits`` list, each dict now carrying the provenance keys.
        """
        # Seed every hit with the full key set (stable shape) and the
        # source-level anchor for any source-derived hit (incl. insights).
        for hit in hits:
            for key in _PROVENANCE_KEYS:
                hit.setdefault(key, None)
            hit.setdefault("source", _hit_parent_source(hit))

        # Chunk-level provenance is only attachable for chunk-backed hits when
        # we have the embedding to pin the exact chunk. source_insight hits,
        # note hits, and the entire text-only path are excluded by design.
        if embedding is None:
            return hits

        chunk_backed = {
            hit["id"]
            for hit in hits
            if _hit_is_chunk_backed(hit)
        }
        if not chunk_backed:
            return hits

        try:
            best_by_source = await self._best_chunk_per_source(
                sorted(chunk_backed), embedding
            )
        except Exception as e:  # provenance is additive — never break a search
            logger.warning(f"Provenance hydration failed (continuing): {e}")
            return hits

        for hit in hits:
            if not _hit_is_chunk_backed(hit):
                continue
            prov = best_by_source.get(hit["id"])
            if prov is None:
                continue
            for key in _PROVENANCE_KEYS:
                if prov.get(key) is not None:
                    hit[key] = prov[key]
        return hits

    async def _best_chunk_per_source(
        self,
        source_ids: List[str],
        embedding: List[float],
    ) -> Dict[str, Dict[str, Any]]:
        """Best (highest-cosine) chunk per source, batched.

        Reproduces ``fn::vector_search``'s ``math::max(cosine)`` collapse: the
        top-1 chunk-bearing ``source_embedding`` row per source is *exactly* the
        row that produced that source's winning hit score. Only ever called for
        chunk-backed (``source:``) hits with an embedding in hand.
        """
        record_ids = [ensure_record_id(sid) for sid in source_ids]
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

        # Rows are pre-sorted by cosine desc — first per source wins.
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
            hydrate: Attach per-hit provenance (Track X.1). No embedding is
                available for a text search and BM25 scores are not reproducible
                out of context, so no chunk-level keys are attached for any text
                hit (``chunk_id``/``physical_page``/``section_path``/... stay
                ``None``); only the source-level ``source`` is set. Disabled
                internally by ``hybrid_search`` so the fused result set is
                hydrated once, with the embedding.

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
