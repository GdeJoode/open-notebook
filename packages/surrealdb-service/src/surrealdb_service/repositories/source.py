"""
Source repository with specialized operations.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models import Asset, Chunk, Source, SourceEmbedding, SourceInsight
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories.base import BaseRepository, _validate_record_id


class SourceRepository(BaseRepository[Source]):
    """Repository for Source operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(Source, config)

    async def add_to_notebook(self, source_id: str, notebook_id: str) -> bool:
        """
        Add a source to a notebook.

        Args:
            source_id: Source ID.
            notebook_id: Notebook ID.

        Returns:
            True if successful.
        """
        try:
            await self.relate(source_id, "reference", notebook_id)
            return True
        except Exception as e:
            logger.error(f"Failed to add source to notebook: {e}")
            return False

    async def get_notebook_id(self, source_id: str) -> Optional[str]:
        """Return the first notebook that links to ``source_id`` (or ``None``).

        Phase B.1f wires the multi-schema orchestrator into
        ``EntityExtractionService``. The orchestrator wants the owning
        notebook's id so it can load a :class:`NotebookSchema` row. The
        source ↔ notebook link lives on the ``reference`` graph edge
        (``add_to_notebook`` above creates it), so we walk that edge.

        A source may belong to multiple notebooks; we return the first
        one because notebook-schemas are 1:1 with notebooks and the
        downstream call site simply needs *a* notebook to anchor schema
        lookup. The chosen notebook is deterministic because SurrealDB
        returns ``reference`` edges in insertion order (no explicit
        ORDER BY is required for this single-record fetch).

        Returns ``None`` when the source is unlinked (CLI extractions
        or orphaned sources). The caller treats ``None`` as "no
        notebook-schema available → run single-schema legacy path".
        """
        try:
            rows = await execute_query(
                "SELECT VALUE out FROM reference "
                "WHERE in = $source LIMIT 1;",
                {"source": ensure_record_id(source_id)},
                self.config,
            )
            if not rows:
                return None
            # ``execute_query`` parses RecordIDs to strings already.
            return str(rows[0]) if rows[0] is not None else None
        except Exception as e:
            logger.error(
                f"Failed to fetch notebook for source {source_id}: {e}"
            )
            return None

    async def get_notebook_ids(self, source_id: str) -> List[str]:
        """Return every notebook linked to ``source_id`` (possibly empty).

        Unlike :meth:`get_notebook_id` (which returns only the first link for
        schema anchoring), this walks the full ``reference`` edge set. The J.4
        summarization privacy path needs ALL owning notebooks so it can pick the
        most-private one (any ``private`` notebook -> PRIVATE, fail safe): a
        source shared into a private notebook must not be summarized via cloud
        just because some other (cloud) notebook also references it.

        Returns an empty list when the source is unlinked or on any read error
        (the caller then falls through to the document/global privacy layers).
        """
        try:
            rows = await execute_query(
                "SELECT VALUE out FROM reference WHERE in = $source;",
                {"source": ensure_record_id(source_id)},
                self.config,
            )
            return [str(r) for r in (rows or []) if r is not None]
        except Exception as e:
            logger.error(
                f"Failed to fetch notebooks for source {source_id}: {e}"
            )
            return []

    async def get_insights(self, source_id: str) -> List[SourceInsight]:
        """
        Get all insights for a source.

        Args:
            source_id: Source ID.

        Returns:
            List of source insights.
        """
        try:
            result = await execute_query(
                "SELECT * FROM source_insight WHERE source=$id",
                {"id": ensure_record_id(source_id)},
                self.config,
            )
            return [SourceInsight(**item) for item in result]
        except Exception as e:
            logger.error(f"Failed to get insights for source {source_id}: {e}")
            return []

    async def get_chunks(self, source_id: str) -> List[Chunk]:
        """
        Get all chunks for a source, ordered by position.

        Args:
            source_id: Source ID.

        Returns:
            List of chunks ordered by order field.
        """
        try:
            result = await execute_query(
                "SELECT * FROM chunk WHERE source=$id ORDER BY order ASC",
                {"id": ensure_record_id(source_id)},
                self.config,
            )
            return [Chunk(**item) for item in result]
        except Exception as e:
            logger.error(f"Failed to get chunks for source {source_id}: {e}")
            return []

    async def get_embeddings(self, source_id: str) -> List[SourceEmbedding]:
        """
        Get all embeddings for a source.

        Args:
            source_id: Source ID.

        Returns:
            List of source embeddings ordered by order field.
        """
        try:
            result = await execute_query(
                "SELECT * FROM source_embedding WHERE source=$id ORDER BY order ASC",
                {"id": ensure_record_id(source_id)},
                self.config,
            )
            return [SourceEmbedding(**item) for item in result]
        except Exception as e:
            logger.error(f"Failed to get embeddings for source {source_id}: {e}")
            return []

    async def get_embedding_count(self, source_id: str) -> int:
        """
        Get the count of embeddings for a source.

        Args:
            source_id: Source ID.

        Returns:
            Number of embeddings.
        """
        try:
            result = await execute_query(
                "SELECT count() AS chunks FROM source_embedding WHERE source=$id GROUP ALL",
                {"id": ensure_record_id(source_id)},
                self.config,
            )
            if result:
                return result[0].get("chunks", 0)
            return 0
        except Exception as e:
            logger.error(f"Failed to count embeddings for source {source_id}: {e}")
            return 0

    async def delete_embeddings(self, source_id: str) -> int:
        """
        Delete all embeddings for a source.

        Args:
            source_id: Source ID.

        Returns:
            Number of deleted embeddings.
        """
        try:
            result = await execute_query(
                "DELETE source_embedding WHERE source=$source_id",
                {"source_id": ensure_record_id(source_id)},
                self.config,
            )
            return len(result) if result else 0
        except Exception as e:
            logger.error(f"Failed to delete embeddings for source {source_id}: {e}")
            return 0

    async def add_insight(
        self,
        source_id: str,
        insight_type: str,
        content: str,
        embedding: Optional[List[float]] = None,
    ) -> SourceInsight:
        """
        Add an insight to a source.

        Args:
            source_id: Source ID.
            insight_type: Type of insight.
            content: Insight content.
            embedding: Optional embedding vector.

        Returns:
            Created insight.
        """
        try:
            result = await execute_query(
                """
                CREATE source_insight CONTENT {
                    "source": $source_id,
                    "insight_type": $insight_type,
                    "content": $content,
                    "embedding": $embedding
                }
                """,
                {
                    "source_id": ensure_record_id(source_id),
                    "insight_type": insight_type,
                    "content": content,
                    "embedding": embedding or [],
                },
                self.config,
            )
            if result:
                return SourceInsight(**result[0])
            raise RuntimeError("Failed to create insight")
        except Exception as e:
            logger.exception(f"Failed to add insight to source {source_id}: {e}")
            raise

    async def add_embedding(
        self,
        source_id: str,
        content: str,
        order: int,
        embedding: List[float],
        chunk_id: Optional[str] = None,
    ) -> SourceEmbedding:
        """
        Add an embedding to a source.

        Args:
            source_id: Source ID.
            content: Text content.
            order: Order in document.
            embedding: Embedding vector.
            chunk_id: Optional reference to the chunk record.

        Returns:
            Created embedding.
        """
        try:
            params: Dict[str, Any] = {
                "source_id": ensure_record_id(source_id),
                "order": order,
                "content": content,
                "embedding": embedding,
            }
            if chunk_id:
                params["chunk_id"] = ensure_record_id(chunk_id)
                query = """
                CREATE source_embedding CONTENT {
                    "source": $source_id,
                    "order": $order,
                    "content": $content,
                    "embedding": $embedding,
                    "chunk": $chunk_id
                }
                """
            else:
                query = """
                CREATE source_embedding CONTENT {
                    "source": $source_id,
                    "order": $order,
                    "content": $content,
                    "embedding": $embedding
                }
                """
            result = await execute_query(query, params, self.config)
            if result:
                return SourceEmbedding(**result[0])
            raise RuntimeError("Failed to create embedding")
        except Exception as e:
            logger.exception(f"Failed to add embedding to source {source_id}: {e}")
            raise


    async def get_embedding_vectors(self, source_id: str) -> List[List[float]]:
        """Return the source's per-chunk embedding vectors (non-empty only).

        Reads the ``embedding`` column of every ``source_embedding`` row for
        this source, ordered by ``order``. Rows with an empty/NONE vector are
        skipped so the caller (the R.0 mean-pool) never averages in a zero
        vector. Returns ``[]`` when the source has no embedded chunks yet.
        """
        try:
            # No ORDER BY: ``SELECT VALUE embedding`` projects a bare vector, so
            # ``order`` is not in scope to sort on — and mean-pooling is
            # order-invariant anyway.
            result = await execute_query(
                "SELECT VALUE embedding FROM source_embedding WHERE source=$id",
                {"id": ensure_record_id(source_id)},
                self.config,
            )
            return [
                [float(x) for x in vec]
                for vec in (result or [])
                if vec
            ]
        except Exception as e:
            logger.error(
                f"Failed to read embedding vectors for source {source_id}: {e}"
            )
            return []

    async def set_aggregate_embedding(
        self, source_id: str, embedding: Optional[List[float]]
    ) -> bool:
        """Persist the source-level aggregate ``embedding`` (R.0).

        ``embedding`` is the mean-pool of the source's chunk vectors, or
        ``None`` to clear it (a source with no chunk vectors). The field is
        ``option<array<float>>`` (migration 63), so ``None`` writes NONE.
        Returns True on success.
        """
        try:
            await execute_query(
                "UPDATE $id SET embedding = $embedding",
                {
                    "id": ensure_record_id(source_id),
                    "embedding": embedding,
                },
                self.config,
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to set aggregate embedding for source {source_id}: {e}"
            )
            return False

    async def find_related_by_embedding(
        self, source_id: str, k: int
    ) -> List[Dict[str, Any]]:
        """Return the top-``k`` other sources by cosine similarity (R.1).

        Ranks every *other* source that has a populated aggregate
        ``source.embedding`` against this source's aggregate vector using
        SurrealDB's native ``vector::similarity::cosine`` — the same operator
        ``fn::vector_search`` uses for chunk search. Ranking server-side keeps
        all 1024-dim vectors in the DB (no bulk pull into Python) and reuses
        the proven cosine path.

        Behaviour:
          * The query source's own row is excluded (``id != $id``).
          * Sources whose ``embedding`` is NONE (not yet computed / no chunk
            vectors) are excluded — they can never be a result and never crash
            the cosine call.
          * If the *query* source itself has no aggregate embedding, returns
            ``[]`` (nothing to compare). The caller distinguishes this from
            "source not found" via a prior existence check.
          * Ordering is ``score DESC`` with a stable ``id ASC`` tie-break, so
            equal-similarity sources come back in a deterministic order.
          * ``k`` bounds the LIMIT; requesting more than exist returns all.

        The embedding dimension is never hardcoded — cosine reads whatever
        length the stored vectors are (the configured model's dim, 1024 today).

        Returns a list of ``{"id", "title", "score"}`` dicts (ids stringified
        by ``execute_query``), or ``[]`` on error / no aggregate.
        """
        try:
            rid = ensure_record_id(source_id)
            query_vec = await execute_query(
                "SELECT VALUE embedding FROM $id",
                {"id": rid},
                self.config,
            )
            # ``SELECT VALUE embedding`` yields [vector] for a present field,
            # [None] when the field is NONE, and [] when the source is absent.
            if not query_vec or not query_vec[0]:
                return []

            # ``array::len(embedding) = array::len($q)`` guards the cosine
            # call: SurrealDB errors if the two vectors differ in length. In
            # production every source shares one embedding model (one dim), so
            # this is a no-op there — but it keeps the query robust to legacy /
            # mixed-dim rows instead of letting one bad vector fail the whole
            # ranking. The dim is still never hardcoded: it is derived from the
            # query vector itself.
            rows = await execute_query(
                "SELECT id, title, "
                "vector::similarity::cosine(embedding, $q) AS score "
                "FROM source "
                "WHERE embedding != NONE AND id != $id "
                "AND array::len(embedding) = array::len($q) "
                "ORDER BY score DESC, id ASC "
                "LIMIT $k",
                {"q": query_vec[0], "id": rid, "k": int(k)},
                self.config,
            )
            return [
                {
                    "id": str(r["id"]),
                    "title": r.get("title"),
                    "score": float(r["score"]),
                }
                for r in (rows or [])
                if r.get("score") is not None
            ]
        except Exception as e:
            logger.error(
                f"Failed to find related sources for {source_id}: {e}"
            )
            return []

    async def get_titles_by_ids(
        self, source_ids: List[str]
    ) -> Dict[str, Optional[str]]:
        """Batch-fetch source titles for a list of ids (Track R.2 hydration).

        The KG scorer returns ranked source *ids* only; the service hydrates
        them to ``{id, title, ...}`` for the API. Doing it in one ``id IN``
        query (not a per-id loop) keeps the related-KG endpoint a fixed two
        round-trips regardless of ``k``. Missing ids simply don't appear in the
        returned map (the caller falls back to ``None``).

        Args:
            source_ids: Source record-id strings to resolve.

        Returns:
            A dict mapping stringified source id -> title (may be ``None`` for a
            source with no title). Empty on empty input / failure.
        """
        if not source_ids:
            return {}
        try:
            record_ids = [ensure_record_id(s) for s in source_ids]
            rows = await execute_query(
                "SELECT id, title FROM source WHERE id IN $ids",
                {"ids": record_ids},
                self.config,
            )
            return {str(r["id"]): r.get("title") for r in (rows or [])}
        except Exception as e:
            logger.error(f"Failed to batch fetch source titles: {e}")
            return {}

    async def list_with_metadata(
        self,
        notebook_id: Optional[str] = None,
        order_by: str = "updated",
        order_dir: str = "DESC",
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        List sources with insight counts and embedding status.

        Args:
            notebook_id: Optional notebook filter.
            order_by: Field to sort by (created or updated).
            order_dir: Sort direction (ASC or DESC).
            limit: Max results.
            offset: Skip count.

        Returns:
            List of source dicts with metadata fields.
        """
        order_clause = f"ORDER BY {order_by} {order_dir}"

        if notebook_id:
            query = f"""
                SELECT id, asset, created, title, updated, topics, command,
                (SELECT VALUE count() FROM source_insight
                 WHERE source = $parent.id GROUP ALL)[0].count OR 0
                 AS insights_count,
                ((SELECT VALUE id FROM source_embedding
                  WHERE source = $parent.id LIMIT 1)) != NONE AS embedded
                FROM (select value in from reference where out=$notebook_id)
                {order_clause}
                LIMIT $limit START $offset
            """
            return await execute_query(
                query,
                {
                    "notebook_id": ensure_record_id(notebook_id),
                    "limit": limit,
                    "offset": offset,
                },
                self.config,
            )
        else:
            query = f"""
                SELECT id, asset, created, title, updated, topics, command,
                (SELECT VALUE count() FROM source_insight
                 WHERE source = $parent.id GROUP ALL)[0].count OR 0
                 AS insights_count,
                ((SELECT VALUE id FROM source_embedding
                  WHERE source = $parent.id LIMIT 1)) != NONE AS embedded
                FROM source
                {order_clause}
                LIMIT $limit START $offset
            """
            return await execute_query(
                query, {"limit": limit, "offset": offset}, self.config
            )

    async def batch_get_command_status(
        self, command_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]:
        """
        Fetch job statuses for multiple command IDs in a single query.

        Args:
            command_ids: List of job record IDs.

        Returns:
            Dict mapping command_id -> job record dict.
        """
        if not command_ids:
            return {}

        try:
            record_ids = [ensure_record_id(cid) for cid in command_ids]
            result = await execute_query(
                "SELECT * FROM job WHERE id IN $ids",
                {"ids": record_ids},
                self.config,
            )
            return {str(row["id"]): row for row in result} if result else {}
        except Exception as e:
            logger.error(f"Failed to batch fetch command statuses: {e}")
            return {}

    # ------------------------------------------------------------------
    # Track U Phase U.3 — `cites` (source → source) citation edges
    # ------------------------------------------------------------------
    #
    # `cites` is a regenerated projection of the parsed references Track V will
    # feed (matched to in-corpus sources by the pure precision matcher). Migration
    # 67 defines it ``TYPE RELATION FROM source TO source`` with the U.3 fields
    # (`confidence` / `reference_text` / `match_method`). These methods are the DB
    # seam the ``CitesMaterializationService`` composes into one idempotent
    # regenerate; the canonical `source` rows are NEVER mutated — only the `cites`
    # edge table is written. Re-running clears then recreates so the edge set is
    # byte-identical with no duplicates (the U.3 idempotency contract). On the
    # current corpus there are 0 intra-corpus citations (U.1), so a regenerate is
    # a clean 0-edge no-op until Track V supplies references.

    async def load_source_records(self) -> List[Dict[str, Any]]:
        """Load the bibliographic projection of every source (matcher input).

        Returns ``id`` + ``title`` plus the FLEXIBLE metadata objects where any
        bibliographic detail lives: ``metadata`` / ``type_metadata`` (a Zotero/
        Docling enrichment may carry ``authors`` / ``creators`` / ``DOI`` here)
        and ``external_ids`` (the natural home for a ``doi``). The ``source``
        table is SCHEMAFULL (migration 1) with NO top-level ``authors`` / ``doi``
        column — they would be silently dropped — so the caller extracts those
        from these objects. On the current corpus the objects are empty (sources
        carry only titles), so matching falls back to the title alone (and the
        precision guard then correctly declines a title-only match). Read-only.
        """
        try:
            return await execute_query(
                "SELECT id, title, metadata, type_metadata, external_ids "
                "FROM source",
                {},
                self.config,
            )
        except Exception as e:
            logger.error(f"load_source_records failed: {e}")
            return []

    async def load_notebook_source_ids(self, notebook_id: str) -> List[str]:
        """Resolve a notebook's source ids via the ``reference`` edge (U.5).

        Mirrors the notebook→source traversal
        :meth:`EntityRepository.list_entities_for_notebook` opens with: the
        source↔notebook link lives on the ``reference`` RELATE edge (migration 1),
        so the notebook's sources are ``SELECT VALUE in FROM reference WHERE
        out = notebook``.

        The document-layer export (U.5) needs this to scope the global
        ``mentions`` / ``cites`` edge tables — which are notebook-agnostic
        materialized projections — to the sources that belong to the notebook
        being exported, so the entity-graph and document-graph layers describe
        the SAME corpus slice. Read-only; returns stringified ``source:...`` ids.

        Args:
            notebook_id: Record id of the notebook (e.g. ``"notebook:abc"``).

        Returns:
            Stringified source ids referenced by the notebook. Empty on empty
            input, an empty notebook, or query failure.
        """
        if not notebook_id:
            return []
        try:
            rows = await execute_query(
                "SELECT VALUE in FROM reference "
                "WHERE out = type::thing($notebook_id)",
                {"notebook_id": notebook_id},
                self.config,
            )
        except Exception as e:
            logger.error(
                f"load_notebook_source_ids failed for '{notebook_id}': {e}"
            )
            return []
        return [str(r) for r in (rows or []) if r]

    async def count_cites(self) -> int:
        """Count rows in the ``cites`` edge table."""
        try:
            result = await execute_query(
                "SELECT count() AS total FROM cites GROUP ALL", {}, self.config
            )
            if result:
                return int(result[0].get("total", 0))
            return 0
        except Exception as e:
            logger.error(f"count_cites failed: {e}")
            return 0

    async def clear_cites(self) -> int:
        """Delete every ``cites`` edge (the clear half of regeneration).

        Returns the number of edges removed so the regenerator can report the
        before/after delta. A bare ``DELETE cites`` is correct here precisely
        because the table is a regenerated projection the U.3 service OWNS —
        unlike migration 67 (a schema repair that must preserve any healthy
        edges), the regenerator rebuilds the edge set from the matched references
        on the next step, so clearing all of it is the intended, reversible
        operation. Canonical `source` rows are untouched.

        Returns:
            The count of edges deleted (0 on an already-empty table / failure).
        """
        try:
            before = await self.count_cites()
            await execute_query("DELETE cites;", {}, self.config)
            return before
        except Exception as e:
            logger.error(f"clear_cites failed: {e}")
            return 0

    async def relate_cites(
        self,
        citing_source_id: str,
        cited_source_id: str,
        *,
        confidence: float,
        reference_text: str = "",
        match_method: str = "",
    ) -> bool:
        """RELATE one ``source -> cites -> source`` edge with its U.3 provenance.

        Mirrors the RELATE discipline of :meth:`EntityRepository.relate_mention`:
        the arrow positions need bare record-id literals (SurrealDB issue #4232 —
        params in the ``in``/``out`` slots are rejected), and the ids are
        ``source:<alnum>`` record ids straight from the matcher (no user input),
        so interpolating them is safe; all metadata is bound as parameters.

        A wrong edge fabricates a citation, so the caller (the matcher) only ever
        passes a confident, non-self pair; this method does NOT re-check precision
        — it is the persistence seam, not the decision.

        Args:
            citing_source_id: The citing ``source:...`` (edge ``in``).
            cited_source_id: The cited ``source:...`` (edge ``out``).
            confidence: The matcher confidence (1.0 for a DOI-exact hit).
            reference_text: The verbatim cited reference (the "why").
            match_method: The precision path (``doi`` | ``title_author``).

        Returns:
            True on success; False when an endpoint id is missing/malformed or a
            source would cite itself (a final defensive self-citation guard —
            the matcher already excludes it, but never RELATE a self-loop).

        Raises:
            RuntimeError: On a transport-level / SurrealQL failure of the RELATE
                (mirrors the other write helpers). The regenerator wraps this and
                counts a raised edge as ``failed`` rather than aborting.
        """
        if not citing_source_id or not cited_source_id:
            return False
        try:
            src = str(ensure_record_id(citing_source_id))
            tgt = str(ensure_record_id(cited_source_id))
        except Exception as e:
            logger.error(
                f"relate_cites: bad id citing={citing_source_id!r} "
                f"cited={cited_source_id!r}: {e}"
            )
            return False
        if src == tgt:
            # Defensive backstop: never RELATE a source to itself, even if a
            # caller bug got past the matcher's self-citation exclusion.
            logger.warning(f"relate_cites: refused self-citation on {src}")
            return False
        try:
            await execute_query(
                f"RELATE {src}->cites->{tgt} SET "
                "confidence = $confidence, reference_text = $reference_text, "
                "match_method = $match_method;",
                {
                    "confidence": float(confidence),
                    "reference_text": reference_text or "",
                    "match_method": match_method or "",
                },
                self.config,
            )
            return True
        except Exception as e:
            logger.exception(
                f"relate_cites failed for {src}->{tgt}: {e}"
            )
            raise

    async def load_cites_edges(
        self,
        *,
        min_confidence: float = 0.0,
        source_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Load materialized ``cites`` edges for viz / traversal / export (U.4).

        Returns the document↔document citation edges with their endpoints and
        confidence/why metadata. ``in`` / ``out`` are stringified ``source:...``
        record ids.

        Args:
            min_confidence: Optional per-edge confidence floor (default 0.0 —
                return all).
            source_id: Optional scope to edges incident on a single source (the
                citing OR cited side, for a document's citation neighbourhood).

        Returns:
            A list of ``{"id", "source", "target", "confidence", "reference_text",
            "match_method"}`` dicts. Empty on failure / the 0-edge no-op corpus.
        """
        clauses: List[str] = []
        params: Dict[str, Any] = {}
        if min_confidence is not None and min_confidence > 0.0:
            clauses.append("confidence >= $min_confidence")
            params["min_confidence"] = float(min_confidence)
        if source_id:
            try:
                params["sid"] = ensure_record_id(source_id)
                clauses.append("(in = $sid OR out = $sid)")
            except Exception as e:
                logger.error(f"load_cites_edges: bad source_id {source_id!r}: {e}")
                return []
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        try:
            return await execute_query(
                "SELECT id, in AS source, out AS target, confidence, "
                "reference_text, match_method "
                f"FROM cites{where} ORDER BY confidence DESC",
                params,
                self.config,
            )
        except Exception as e:
            logger.error(f"load_cites_edges failed: {e}")
            return []

    async def relate_verdict(
        self,
        from_source: str,
        to_source: str,
        *,
        verdict: str,
        confidence: float,
        reasoning: str = "",
        judge_model: str = "",
    ) -> bool:
        """Idempotently RELATE a ``source -> source_verdict -> source`` edge (Z.1).

        Stores an LLM judge's verdict on whether two RELATED sources reinforce,
        contradict, or are neutral. This is the persistence seam for Track Z
        (contradiction detection); the precision decision (which verdicts are
        confident enough to persist) lives in the judge, not here.

        RELATE is *not* idempotent: a repeated ``RELATE a -> source_verdict -> b``
        writes a SECOND row (the Track W/Y lesson), so a re-judged pair would
        accrue duplicate edges. This helper clears any existing
        ``(from_source, to_source)`` edge first, then RELATEs exactly once — so
        the edge set for a directed pair is always a single, current row carrying
        the latest verdict/confidence.

        Guards (all BEFORE any interpolation — the Y.1 injection lesson):
          * both ids are validated against the strict ``_RECORD_ID_RE`` pattern
            via :func:`_validate_record_id`. A ``;``-bearing / ``REMOVE TABLE``-
            bearing / otherwise malformed id is REFUSED, not interpolated.
            ``RecordID.parse`` splits only on the first colon and round-trips an
            injection payload verbatim, so the SDK's own parsing is NOT a
            validator; the regex is. The endpoints are then interpolated as
            literal record ids because SurrealDB's RELATE graph syntax does not
            bind a ``$param`` in the in/out position (a parameterized
            ``RELATE $from->...->$to`` silently writes 0 rows — issue #4232),
            which is only safe because every accepted id matches ``table:id`` with
            no SurrealQL metacharacters. Only the SET *values* stay parameterized;
          * both ids must be in the ``source`` table (a wrong-table id is rejected
            rather than silently mis-linked);
          * a self-edge (``from_source == to_source``) is refused — a source is
            not "in contradiction" with itself, and the candidate generation
            already excludes self.

        Args:
            from_source: source-A record id (edge ``in``).
            to_source: source-B record id (edge ``out``).
            verdict: the judgment (``reinforces`` | ``contradicts`` | ``neutral``).
            confidence: the judge's [0, 1] confidence in the verdict.
            reasoning: the judge's short rationale (retained for review).
            judge_model: which model produced the verdict (provenance).

        Returns:
            True on success; False if refused (invalid/unsafe id, wrong table,
            self-edge) or on a DB error.
        """
        # Strict-validate the RAW input strings BEFORE the SDK touches them.
        # ``RecordID.parse`` would happily accept ``source:x; REMOVE TABLE source --``
        # (it splits on the first colon and keeps the rest as the id, round-tripped
        # verbatim by ``str()``), so validation must run on the raw string against
        # ``_RECORD_ID_RE`` — which rejects every SurrealQL metacharacter.
        try:
            from_str = _validate_record_id(str(from_source))
            to_str = _validate_record_id(str(to_source))
        except ValueError as e:
            logger.error(
                f"relate_verdict: refusing unsafe/invalid source id "
                f"({from_source!r}/{to_source!r}): {e}"
            )
            return False

        if not (from_str.startswith("source:") and to_str.startswith("source:")):
            logger.error(
                f"relate_verdict: both ids must be sources "
                f"(got {from_str} -> {to_str})"
            )
            return False

        if from_str == to_str:
            logger.warning(f"relate_verdict: refusing self-edge for {from_str}")
            return False

        from_rid = ensure_record_id(from_str)
        to_rid = ensure_record_id(to_str)

        try:
            # Clear-before-relate: drop any prior edge for this exact (in, out)
            # pair so the re-judge replaces rather than duplicates (RELATE is not
            # idempotent). Scoped to the directed pair — other edges untouched.
            await execute_query(
                "DELETE source_verdict WHERE in = $from AND out = $to",
                {"from": from_rid, "to": to_rid},
                self.config,
            )
            # Endpoints interpolated as literal record ids (RELATE graph syntax
            # won't bind a ``$param`` in the in/out position — see the docstring).
            # Safe because ``from_str``/``to_str`` passed ``_RECORD_ID_RE`` above:
            # ``table:id`` only, no SurrealQL metacharacters. Only the SET *values*
            # stay parameterized.
            await execute_query(
                f"RELATE {from_str}->source_verdict->{to_str} "
                "SET verdict = $verdict, confidence = $confidence, "
                "reasoning = $reasoning, judge_model = $judge_model",
                {
                    "verdict": verdict,
                    "confidence": float(confidence),
                    "reasoning": reasoning or "",
                    "judge_model": judge_model or "",
                },
                self.config,
            )
            return True
        except Exception as e:
            logger.error(
                f"relate_verdict: failed to relate {from_str} -> {to_str}: {e}"
            )
            return False


class ChunkRepository(BaseRepository[Chunk]):
    """Repository for Chunk operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(Chunk, config)

    async def get_by_source(
        self,
        source_id: str,
        page: Optional[int] = None,
    ) -> List[Chunk]:
        """
        Get chunks for a source, optionally filtered by page.

        Args:
            source_id: Source ID.
            page: Optional physical page number.

        Returns:
            List of chunks.
        """
        where = "source=$source_id"
        params: Dict[str, Any] = {"source_id": ensure_record_id(source_id)}

        if page is not None:
            where += " AND physical_page=$page"
            params["page"] = page

        return await self.query(where, params, order_by="order ASC")

    async def delete_by_source(self, source_id: str) -> int:
        """
        Delete all chunks for a source.

        Args:
            source_id: Source ID.

        Returns:
            Number of deleted chunks.
        """
        try:
            result = await execute_query(
                "DELETE chunk WHERE source=$source_id",
                {"source_id": ensure_record_id(source_id)},
                self.config,
            )
            return len(result) if result else 0
        except Exception as e:
            logger.error(f"Failed to delete chunks for source {source_id}: {e}")
            return 0

    async def bulk_create(self, chunks: List[Dict[str, Any]]) -> List[Chunk]:
        """
        Create multiple chunks at once.

        Args:
            chunks: List of chunk data.

        Returns:
            List of created chunks.
        """
        try:
            prepared = []
            for chunk in chunks:
                c = dict(chunk)
                if "source" in c and isinstance(c["source"], str):
                    c["source"] = ensure_record_id(c["source"])
                prepared.append(c)
            result = await execute_query(
                "INSERT INTO chunk $chunks",
                {"chunks": prepared},
                self.config,
            )
            return [Chunk(**item) for item in result] if result else []
        except Exception as e:
            logger.exception(f"Failed to bulk create chunks: {e}")
            raise


class SourceInsightRepository(BaseRepository[SourceInsight]):
    """Repository for SourceInsight operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(SourceInsight, config)

    async def get_by_source(self, source_id: str) -> List[SourceInsight]:
        """Get all insights for a source."""
        return await self.query(
            "source=$source_id",
            {"source_id": ensure_record_id(source_id)},
        )

    async def get_source(self, insight_id: str) -> Optional[Source]:
        """
        Get the source for an insight.

        Args:
            insight_id: Insight ID.

        Returns:
            Source or None.
        """
        try:
            result = await execute_query(
                "SELECT source.* FROM $id FETCH source",
                {"id": ensure_record_id(insight_id)},
                self.config,
            )
            if result and result[0].get("source"):
                return Source(**result[0]["source"])
            return None
        except Exception as e:
            logger.error(f"Failed to get source for insight {insight_id}: {e}")
            return None


class SourceEmbeddingRepository(BaseRepository[SourceEmbedding]):
    """Repository for SourceEmbedding operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(SourceEmbedding, config)

    async def get_by_source(self, source_id: str) -> List[SourceEmbedding]:
        """Get all embeddings for a source."""
        return await self.query(
            "source=$source_id",
            {"source_id": ensure_record_id(source_id)},
            order_by="order ASC",
        )

    async def get_source(self, embedding_id: str) -> Optional[Source]:
        """
        Get the source for an embedding.

        Args:
            embedding_id: Embedding ID.

        Returns:
            Source or None.
        """
        try:
            result = await execute_query(
                "SELECT source.* FROM $id FETCH source",
                {"id": ensure_record_id(embedding_id)},
                self.config,
            )
            if result and result[0].get("source"):
                return Source(**result[0]["source"])
            return None
        except Exception as e:
            logger.error(f"Failed to get source for embedding {embedding_id}: {e}")
            return None
