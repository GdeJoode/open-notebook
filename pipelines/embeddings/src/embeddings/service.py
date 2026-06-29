"""
Embedding service for generating and storing vector embeddings.

Uses structural chunks from the ingestion pipeline when available,
falling back to simple text splitting for legacy sources or notes.
"""

import asyncio
from dataclasses import dataclass, field
from typing import List, Optional

from esperanto import EmbeddingModel
from loguru import logger

from embeddings.config import EmbeddingConfig
from shared.models import Chunk, Source, SourceEmbedding
from shared.utils.text import split_text
from surrealdb_service.repositories.source import SourceRepository


@dataclass
class EmbeddingResult:
    """Result of embedding a single source or note."""
    id: str
    embeddings_created: int
    chunks_used: int
    used_structural_chunks: bool
    errors: list[str] = field(default_factory=list)


@dataclass
class RebuildResult:
    """Result of a bulk rebuild operation."""
    sources_processed: int = 0
    notes_processed: int = 0
    insights_processed: int = 0
    total_embeddings: int = 0
    errors: list[str] = field(default_factory=list)


class EmbeddingService:
    """
    Core service for generating vector embeddings.

    Uses structural chunks from the ingestion pipeline when available,
    falling back to text splitting for legacy sources or notes.
    """

    def __init__(
        self,
        source_repo: SourceRepository,
        embedding_model: EmbeddingModel,
        config: Optional[EmbeddingConfig] = None,
    ):
        self.source_repo = source_repo
        self.embedding_model = embedding_model
        self.config = config or EmbeddingConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent)

    async def embed_source(self, source_id: str) -> EmbeddingResult:
        """
        Embed a source using its stored chunks.

        Flow:
        1. Delete existing embeddings (idempotent re-embed)
        2. Load chunks from DB (from ingestion pipeline)
        3. If chunks exist, batch-embed them
        4. If no chunks, fall back to text splitting on full_text
        5. Store SourceEmbedding records
        """
        result = EmbeddingResult(
            id=source_id,
            embeddings_created=0,
            chunks_used=0,
            used_structural_chunks=False,
        )

        # 1. Delete existing embeddings for idempotency
        await self.source_repo.delete_embeddings(source_id)

        # 2. Try structural chunks first
        chunks = await self.source_repo.get_chunks(source_id)

        if chunks:
            # 3a. Use structural chunks from ingestion pipeline
            result.used_structural_chunks = True
            result.chunks_used = len(chunks)
            embeddings = await self._embed_chunks(source_id, chunks)
            result.embeddings_created = len(embeddings)
        else:
            # 3b. Fall back to text splitting
            source = await self.source_repo.get(source_id)
            if not source or not source.full_text:
                logger.warning(f"No text or chunks for source {source_id}")
                # The per-chunk rows were just deleted (idempotent re-embed) and
                # there is nothing to re-embed, so clear any stale aggregate too
                # rather than leaving it pointing at vectors that no longer exist.
                from embeddings.aggregate import populate_source_embedding

                await populate_source_embedding(
                    source_id, source_repo=self.source_repo
                )
                return result

            text_chunks = split_text(
                source.full_text,
                chunk_size=self.config.fallback_chunk_size,
                chunk_overlap=self.config.fallback_chunk_overlap,
            )
            result.chunks_used = len(text_chunks)
            embeddings = await self._embed_text_chunks(source_id, text_chunks)
            result.embeddings_created = len(embeddings)

        # Write the source-level aggregate (mean-pool of the per-chunk vectors)
        # right after the per-chunk rows exist. Without this, a freshly-embedded
        # source has a NULL ``source.embedding`` and is invisible to "Verwante",
        # the document-graph relatedness, and contradiction candidates — all of
        # which read the aggregate, not the per-chunk vectors. Idempotent +
        # graceful: it recomputes from the current chunk vectors and writes NONE
        # when there are none. Imported lazily to avoid a module-load cycle.
        from embeddings.aggregate import populate_source_embedding

        await populate_source_embedding(source_id, source_repo=self.source_repo)

        logger.info(
            f"Embedded source {source_id}: {result.embeddings_created} embeddings "
            f"from {result.chunks_used} chunks "
            f"(structural={result.used_structural_chunks})"
        )
        return result

    @staticmethod
    def _is_empty_text(text: Optional[str]) -> bool:
        """True when ``text`` carries no embeddable content.

        Covers the three forms a no-text chunk takes in the corpus: ``None``,
        the empty string, and whitespace-only. Table/image chunks routinely
        have no extracted text (e.g. ``element_type='table'`` with empty
        ``text``); the embedding model rejects empty input with ``Text cannot
        be empty``, so such chunks must be skipped rather than embedded.
        """
        return text is None or not text.strip()

    async def _embed_chunks(
        self,
        source_id: str,
        chunks: List[Chunk],
    ) -> List[SourceEmbedding]:
        """Generate embeddings for structural chunks with chunk linkage.

        Empty/whitespace-only chunks (typically table/image chunks with no
        extracted text) are skipped: no model call, no ``source_embedding``
        row, no crash. A skipped chunk does not fail its batch-mates — only the
        non-empty chunks of a batch are sent to the model, and output alignment
        is preserved by zipping the vectors back onto exactly those chunks.
        """
        embeddings: List[SourceEmbedding] = []
        batch_size = self.config.batch_size

        for batch_start in range(0, len(chunks), batch_size):
            batch = chunks[batch_start:batch_start + batch_size]

            # Keep (original-order, chunk) only for non-empty chunks so the
            # stored ``order`` still reflects the chunk's position, and one
            # empty chunk never blanks out or misaligns its batch-mates.
            embeddable = [
                (batch_start + i, chunk)
                for i, chunk in enumerate(batch)
                if not self._is_empty_text(chunk.text)
            ]
            skipped = len(batch) - len(embeddable)
            if skipped:
                logger.info(
                    f"Skipping {skipped} empty/whitespace chunk(s) for source "
                    f"{source_id} (no text to embed)"
                )
            if not embeddable:
                continue

            texts = [chunk.text for _, chunk in embeddable]
            vectors = await self._embed_with_retry(texts)

            for (order, chunk), vector in zip(embeddable, vectors):
                chunk_id = chunk.id if hasattr(chunk, "id") and chunk.id else None
                embedding = await self.source_repo.add_embedding(
                    source_id=source_id,
                    content=chunk.text,
                    order=order,
                    embedding=vector,
                    chunk_id=chunk_id,
                )
                embeddings.append(embedding)

        return embeddings

    async def _embed_text_chunks(
        self,
        source_id: str,
        text_chunks: List[str],
    ) -> List[SourceEmbedding]:
        """Generate embeddings for plain text chunks (fallback path).

        Mirrors ``_embed_chunks``: empty/whitespace-only chunks are skipped
        (no model call, no row, no crash) while their batch-mates still embed,
        with output alignment preserved.
        """
        embeddings: List[SourceEmbedding] = []
        batch_size = self.config.batch_size

        for batch_start in range(0, len(text_chunks), batch_size):
            batch = text_chunks[batch_start:batch_start + batch_size]

            embeddable = [
                (batch_start + i, text)
                for i, text in enumerate(batch)
                if not self._is_empty_text(text)
            ]
            skipped = len(batch) - len(embeddable)
            if skipped:
                logger.info(
                    f"Skipping {skipped} empty/whitespace text chunk(s) for "
                    f"source {source_id} (no text to embed)"
                )
            if not embeddable:
                continue

            texts = [text for _, text in embeddable]
            vectors = await self._embed_with_retry(texts)

            for (order, text), vector in zip(embeddable, vectors):
                embedding = await self.source_repo.add_embedding(
                    source_id=source_id,
                    content=text,
                    order=order,
                    embedding=vector,
                )
                embeddings.append(embedding)

        return embeddings

    @staticmethod
    def _is_context_length_error(error: Exception) -> bool:
        """True when ``error`` is Ollama's over-long-input rejection.

        The legacy ``/api/embeddings`` endpoint (used by the esperanto Ollama
        provider) returns HTTP 500 ``"input length exceeds the context
        length"`` and ignores the ``truncate`` option, so we detect it from
        the message and recover by shrinking the text client-side.
        """
        return "input length exceeds the context length" in str(error).lower()

    def _cap_texts(self, texts: List[str]) -> List[str]:
        """Tail-truncate each text to the configured char cap, logging cuts.

        Truncating the tail (keeping the head) is acceptable for retrieval
        embeddings — the leading content dominates the chunk's topical signal.
        This is the cheap first line of defence; it catches the common
        Latin-script case in a single embed call with no extra round-trips.
        """
        cap = self.config.max_input_chars
        if cap is None or cap <= 0:
            return texts
        capped: List[str] = []
        for text in texts:
            if len(text) > cap:
                logger.warning(
                    f"Truncating over-long embedding input: {len(text)} -> "
                    f"{cap} chars (model context guard, tail dropped)"
                )
                capped.append(text[:cap])
            else:
                capped.append(text)
        return capped

    async def _embed_one_with_shrink(self, text: str) -> List[float]:
        """Embed a single text, halving its length until it fits the context.

        Char counts are not a reliable proxy for tokens (dense scripts such as
        CJK pack far more tokens per char), so the static cap can still miss.
        When the model rejects a text for length, we halve and retry until it
        is accepted. This guarantees a vector for every chunk regardless of
        script/density without dropping it. Halving converges quickly (a 9k
        char text is accepted within a few iterations).
        """
        candidate = text
        while True:
            try:
                vectors = await self.embedding_model.aembed([candidate])
                if len(candidate) < len(text):
                    logger.warning(
                        f"Embedded after adaptive shrink: {len(text)} -> "
                        f"{len(candidate)} chars (context-length recovery)"
                    )
                return vectors[0]
            except Exception as e:  # noqa: BLE001 — only shrink for length errs
                if not (
                    self.config.truncate_on_context_error
                    and self._is_context_length_error(e)
                    and len(candidate) > 1
                ):
                    raise
                candidate = candidate[: max(1, len(candidate) // 2)]

    async def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """Embed texts with length-guard, retry logic, and concurrency control.

        1. Cap each text to ``max_input_chars`` up front (cheap, one call).
        2. On transient errors, retry the whole batch.
        3. On a context-length rejection, fall back to per-text adaptive
           shrinking so a single over-long text never fails its batch-mates
           and every chunk still gets a vector.
        """
        texts = self._cap_texts(texts)
        async with self._semaphore:
            last_error = None
            for attempt in range(self.config.retry_attempts + 1):
                try:
                    return (await self.embedding_model.aembed(texts))
                except Exception as e:
                    last_error = e
                    if (
                        self.config.truncate_on_context_error
                        and self._is_context_length_error(e)
                    ):
                        # The char cap did not catch this batch (dense tokens).
                        # Recover per-text so we never drop a chunk; preserves
                        # input order and batch output shape.
                        logger.warning(
                            "Embedding batch hit context-length limit after "
                            "char cap; recovering per-text via adaptive shrink"
                        )
                        return [
                            await self._embed_one_with_shrink(t) for t in texts
                        ]
                    if attempt < self.config.retry_attempts:
                        logger.warning(
                            f"Embedding attempt {attempt + 1} failed: {e}, retrying..."
                        )
                        await asyncio.sleep(
                            self.config.retry_delay * (attempt + 1)
                        )
            raise RuntimeError(
                f"Embedding failed after {self.config.retry_attempts + 1} attempts: {last_error}"
            )

    async def embed_note(self, note_id: str) -> EmbeddingResult:
        """
        Embed a note using simple text splitting.

        Notes don't have structural chunks, so we always use text splitting
        and store the embedding directly on the note record.
        """
        from surrealdb_service.repositories.notebook import NoteRepository

        result = EmbeddingResult(
            id=note_id,
            embeddings_created=0,
            chunks_used=0,
            used_structural_chunks=False,
        )

        note_repo = NoteRepository(self.source_repo.config)
        note = await note_repo.get(note_id)

        if not note or not note.content:
            logger.warning(f"No content for note {note_id}")
            return result

        # Embed the full note content as a single vector
        vectors = await self._embed_with_retry([note.content])
        if vectors:
            note.embedding = vectors[0]
            await note_repo.update(note_id, {"embedding": vectors[0]})
            result.embeddings_created = 1
            result.chunks_used = 1

        return result

    async def embed_insight(self, insight_id: str) -> EmbeddingResult:
        """Embed a source insight."""
        from surrealdb_service.repositories.source import SourceInsightRepository

        result = EmbeddingResult(
            id=insight_id,
            embeddings_created=0,
            chunks_used=0,
            used_structural_chunks=False,
        )

        insight_repo = SourceInsightRepository(self.source_repo.config)
        insight = await insight_repo.get(insight_id)

        if not insight or not insight.content:
            logger.warning(f"No content for insight {insight_id}")
            return result

        vectors = await self._embed_with_retry([insight.content])
        if vectors:
            await insight_repo.update(insight_id, {"embedding": vectors[0]})
            result.embeddings_created = 1
            result.chunks_used = 1

        return result

    async def rebuild_embeddings(
        self,
        include_sources: bool = True,
        include_notes: bool = True,
        include_insights: bool = True,
        mode: str = "all",
    ) -> RebuildResult:
        """Rebuild embeddings across the knowledge base.

        Args:
            include_sources: Whether to re-embed sources.
            include_notes: Whether to re-embed notes.
            include_insights: Whether to re-embed insights.
            mode: "all" to embed every item, "existing" to only re-embed
                  items that already have embeddings.
        """
        from surrealdb_service.connection import execute_query

        rebuild_result = RebuildResult()

        if include_sources:
            if mode == "existing":
                rows = await execute_query(
                    "RETURN array::distinct("
                    "SELECT VALUE source.id FROM source_embedding "
                    "WHERE embedding != NONE AND array::len(embedding) > 0)"
                )
                source_ids = [str(r) for r in rows] if rows else []
            else:
                sources = await self.source_repo.get_all()
                source_ids = [s.id for s in sources if s.id]

            for sid in source_ids:
                try:
                    result = await self.embed_source(sid)
                    rebuild_result.sources_processed += 1
                    rebuild_result.total_embeddings += result.embeddings_created
                except Exception as e:
                    error_msg = f"Failed to embed source {sid}: {e}"
                    logger.error(error_msg)
                    rebuild_result.errors.append(error_msg)

        if include_notes:
            if mode == "existing":
                rows = await execute_query(
                    "SELECT id FROM note "
                    "WHERE embedding != NONE AND array::len(embedding) > 0"
                )
                note_ids = [str(r["id"]) for r in rows] if rows else []
            else:
                from surrealdb_service.repositories.notebook import NoteRepository
                note_repo = NoteRepository(self.source_repo.config)
                notes = await note_repo.get_all()
                note_ids = [n.id for n in notes if n.id and n.content]

            for nid in note_ids:
                try:
                    result = await self.embed_note(nid)
                    rebuild_result.notes_processed += 1
                    rebuild_result.total_embeddings += result.embeddings_created
                except Exception as e:
                    error_msg = f"Failed to embed note {nid}: {e}"
                    logger.error(error_msg)
                    rebuild_result.errors.append(error_msg)

        if include_insights:
            if mode == "existing":
                rows = await execute_query(
                    "SELECT id FROM source_insight "
                    "WHERE embedding != NONE AND array::len(embedding) > 0"
                )
                insight_ids = [str(r["id"]) for r in rows] if rows else []
            else:
                from surrealdb_service.repositories.source import SourceInsightRepository
                insight_repo = SourceInsightRepository(self.source_repo.config)
                insights = await insight_repo.get_all()
                insight_ids = [i.id for i in insights if i.id and i.content]

            for iid in insight_ids:
                try:
                    result = await self.embed_insight(iid)
                    rebuild_result.insights_processed += 1
                    rebuild_result.total_embeddings += result.embeddings_created
                except Exception as e:
                    error_msg = f"Failed to embed insight {iid}: {e}"
                    logger.error(error_msg)
                    rebuild_result.errors.append(error_msg)

        logger.info(
            f"Rebuild complete: {rebuild_result.sources_processed} sources, "
            f"{rebuild_result.notes_processed} notes, "
            f"{rebuild_result.insights_processed} insights, "
            f"{rebuild_result.total_embeddings} total embeddings"
        )
        return rebuild_result
