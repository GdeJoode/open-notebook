"""
Source repository with specialized operations.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models import Asset, Chunk, Source, SourceEmbedding, SourceInsight
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories.base import BaseRepository


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

            rows = await execute_query(
                "SELECT id, title, "
                "vector::similarity::cosine(embedding, $q) AS score "
                "FROM source "
                "WHERE embedding != NONE AND id != $id "
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
