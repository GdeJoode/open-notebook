"""Tests for the embedding service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from embeddings.config import EmbeddingConfig
from embeddings.service import EmbeddingService
from shared.models import Chunk, Source, SourceEmbedding


def _make_chunk(text: str, order: int, chunk_id: str = None) -> Chunk:
    """Create a test chunk."""
    return Chunk(
        id=chunk_id,
        text=text,
        order=order,
        element_type="paragraph",
        physical_page=1,
    )


def _make_embedding_model():
    """Create a mock embedding model."""
    model = AsyncMock()
    # aembed returns a list of vectors, one per input text
    model.aembed = AsyncMock(
        side_effect=lambda texts: [[0.1, 0.2, 0.3] for _ in texts]
    )
    return model


def _make_source_repo():
    """Create a mock source repository."""
    repo = AsyncMock()
    repo.config = None
    repo.delete_embeddings = AsyncMock(return_value=0)
    repo.get_chunks = AsyncMock(return_value=[])
    repo.get = AsyncMock(return_value=None)
    repo.add_embedding = AsyncMock(
        side_effect=lambda **kwargs: SourceEmbedding(
            source=kwargs.get("source_id"),
            content=kwargs.get("content"),
            order=kwargs.get("order", 0),
            embedding=kwargs.get("embedding"),
            chunk=kwargs.get("chunk_id"),
        )
    )
    return repo


class TestEmbedSourceWithChunks:
    """Test embed_source when structural chunks exist."""

    @pytest.mark.asyncio
    async def test_happy_path_with_chunks(self):
        repo = _make_source_repo()
        repo.get_chunks.return_value = [
            _make_chunk("First paragraph content.", 0, "chunk:1"),
            _make_chunk("Second paragraph content.", 1, "chunk:2"),
            _make_chunk("Third paragraph content.", 2, "chunk:3"),
        ]
        model = _make_embedding_model()
        service = EmbeddingService(repo, model, EmbeddingConfig(batch_size=10))

        result = await service.embed_source("source:test1")

        assert result.embeddings_created == 3
        assert result.chunks_used == 3
        assert result.used_structural_chunks is True
        assert result.id == "source:test1"
        # Should have deleted old embeddings first
        repo.delete_embeddings.assert_called_once_with("source:test1")
        # Should have called aembed once (batch of 3 fits in batch_size=10)
        model.aembed.assert_called_once()
        # Should have stored 3 embeddings
        assert repo.add_embedding.call_count == 3

    @pytest.mark.asyncio
    async def test_chunks_with_batching(self):
        """When chunks exceed batch_size, multiple aembed calls are made."""
        repo = _make_source_repo()
        repo.get_chunks.return_value = [
            _make_chunk(f"Paragraph {i} content.", i, f"chunk:{i}")
            for i in range(5)
        ]
        model = _make_embedding_model()
        service = EmbeddingService(repo, model, EmbeddingConfig(batch_size=2))

        result = await service.embed_source("source:test2")

        assert result.embeddings_created == 5
        # 5 chunks / batch_size 2 = 3 batches
        assert model.aembed.call_count == 3

    @pytest.mark.asyncio
    async def test_chunk_id_passed_to_add_embedding(self):
        """Chunk ID is linked in the embedding record."""
        repo = _make_source_repo()
        repo.get_chunks.return_value = [
            _make_chunk("Content here.", 0, "chunk:abc123"),
        ]
        model = _make_embedding_model()
        service = EmbeddingService(repo, model)

        await service.embed_source("source:test3")

        repo.add_embedding.assert_called_once()
        call_kwargs = repo.add_embedding.call_args.kwargs
        assert call_kwargs["chunk_id"] == "chunk:abc123"


class TestEmbedSourceFallback:
    """Test embed_source fallback when no structural chunks exist."""

    @pytest.mark.asyncio
    async def test_fallback_to_text_splitting(self):
        repo = _make_source_repo()
        repo.get_chunks.return_value = []  # No structural chunks
        repo.get.return_value = Source(
            id="source:legacy",
            full_text="A" * 2500,  # Will be split into multiple chunks
        )
        model = _make_embedding_model()
        service = EmbeddingService(
            repo, model,
            EmbeddingConfig(fallback_chunk_size=1000, fallback_chunk_overlap=200),
        )

        result = await service.embed_source("source:legacy")

        assert result.used_structural_chunks is False
        assert result.embeddings_created > 0
        assert result.chunks_used > 0

    @pytest.mark.asyncio
    async def test_no_text_no_chunks(self):
        """Source with no text and no chunks produces no embeddings."""
        repo = _make_source_repo()
        repo.get_chunks.return_value = []
        repo.get.return_value = Source(id="source:empty", full_text=None)
        model = _make_embedding_model()
        service = EmbeddingService(repo, model)

        result = await service.embed_source("source:empty")

        assert result.embeddings_created == 0
        assert result.chunks_used == 0

    @pytest.mark.asyncio
    async def test_source_not_found(self):
        """Non-existent source produces no embeddings."""
        repo = _make_source_repo()
        repo.get_chunks.return_value = []
        repo.get.return_value = None
        model = _make_embedding_model()
        service = EmbeddingService(repo, model)

        result = await service.embed_source("source:missing")

        assert result.embeddings_created == 0


class TestEmbedSourceIdempotent:
    """Test that re-embedding deletes old embeddings first."""

    @pytest.mark.asyncio
    async def test_deletes_existing_before_reembed(self):
        repo = _make_source_repo()
        repo.get_chunks.return_value = [
            _make_chunk("Content.", 0, "chunk:1"),
        ]
        repo.delete_embeddings.return_value = 5  # Had 5 old embeddings
        model = _make_embedding_model()
        service = EmbeddingService(repo, model)

        result = await service.embed_source("source:reembed")

        repo.delete_embeddings.assert_called_once_with("source:reembed")
        assert result.embeddings_created == 1


class TestEmbedNote:
    """Test note embedding."""

    @pytest.mark.asyncio
    async def test_embed_note_happy_path(self):
        repo = _make_source_repo()
        model = _make_embedding_model()
        service = EmbeddingService(repo, model)

        with patch(
            "surrealdb_service.repositories.notebook.NoteRepository"
        ) as MockNoteRepo:
            mock_note_repo = AsyncMock()
            MockNoteRepo.return_value = mock_note_repo
            mock_note = MagicMock()
            mock_note.content = "This is a note about machine learning."
            mock_note.embedding = None
            mock_note_repo.get.return_value = mock_note

            result = await service.embed_note("note:test1")

            assert result.embeddings_created == 1
            assert result.chunks_used == 1
            mock_note_repo.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_embed_note_no_content(self):
        repo = _make_source_repo()
        model = _make_embedding_model()
        service = EmbeddingService(repo, model)

        with patch(
            "surrealdb_service.repositories.notebook.NoteRepository"
        ) as MockNoteRepo:
            mock_note_repo = AsyncMock()
            MockNoteRepo.return_value = mock_note_repo
            mock_note = MagicMock()
            mock_note.content = None
            mock_note_repo.get.return_value = mock_note

            result = await service.embed_note("note:empty")

            assert result.embeddings_created == 0


class TestRetryBehavior:
    """Test retry logic on embedding failures."""

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        repo = _make_source_repo()
        repo.get_chunks.return_value = [
            _make_chunk("Content.", 0, "chunk:1"),
        ]
        model = AsyncMock()
        # Fail first, succeed second
        model.aembed = AsyncMock(
            side_effect=[
                RuntimeError("API error"),
                [[0.1, 0.2, 0.3]],
            ]
        )
        service = EmbeddingService(
            repo, model,
            EmbeddingConfig(retry_attempts=2, retry_delay=0.01),
        )

        result = await service.embed_source("source:retry")

        assert result.embeddings_created == 1
        assert model.aembed.call_count == 2

    @pytest.mark.asyncio
    async def test_exhausted_retries_raises(self):
        repo = _make_source_repo()
        repo.get_chunks.return_value = [
            _make_chunk("Content.", 0, "chunk:1"),
        ]
        model = AsyncMock()
        model.aembed = AsyncMock(side_effect=RuntimeError("Persistent failure"))
        service = EmbeddingService(
            repo, model,
            EmbeddingConfig(retry_attempts=1, retry_delay=0.01),
        )

        with pytest.raises(RuntimeError, match="Embedding failed after"):
            await service.embed_source("source:fail")


class TestContextLengthGuard:
    """R.0b: chunk texts longer than the model context must still embed.

    The local mxbai-embed-large model (~512-token context) rejects over-long
    input via Ollama's legacy endpoint, which ignores ``truncate``. These tests
    cover the client-side char cap + adaptive shrink with a fake model (the
    real-Ollama proof lives in ``test_service_ollama_context.py``).
    """

    @pytest.mark.asyncio
    async def test_char_cap_truncates_before_embed(self):
        """Texts over ``max_input_chars`` reach the model already truncated."""
        repo = _make_source_repo()
        repo.get_chunks.return_value = [
            _make_chunk("X" * 5000, 0, "chunk:long"),
        ]
        model = _make_embedding_model()
        service = EmbeddingService(
            repo, model, EmbeddingConfig(max_input_chars=2048)
        )

        result = await service.embed_source("source:long")

        assert result.embeddings_created == 1
        sent = model.aembed.call_args.args[0]
        assert len(sent[0]) == 2048  # head kept, tail dropped

    @pytest.mark.asyncio
    async def test_no_chunk_dropped_under_cap(self):
        """Short chunks pass through untouched alongside a capped one."""
        repo = _make_source_repo()
        repo.get_chunks.return_value = [
            _make_chunk("short", 0, "chunk:a"),
            _make_chunk("Y" * 4000, 1, "chunk:b"),
        ]
        model = _make_embedding_model()
        service = EmbeddingService(
            repo, model, EmbeddingConfig(max_input_chars=2048, batch_size=10)
        )

        result = await service.embed_source("source:mixed")

        assert result.embeddings_created == 2
        sent = model.aembed.call_args.args[0]
        assert sent[0] == "short"
        assert len(sent[1]) == 2048

    @pytest.mark.asyncio
    async def test_adaptive_shrink_on_context_error(self):
        """A batch context-length error recovers per-text by halving length.

        Simulates a dense text the char cap did not catch: the model rejects
        the batch with the Ollama message, then accepts each text once it is
        short enough. Every chunk still gets a vector.
        """
        repo = _make_source_repo()
        repo.get_chunks.return_value = [
            _make_chunk("dense" * 600, 0, "chunk:dense"),  # 3000 chars
        ]
        model = AsyncMock()

        async def fake_aembed(texts):
            # Reject anything longer than 1000 chars with Ollama's message.
            if any(len(t) > 1000 for t in texts):
                raise RuntimeError(
                    "Ollama API error: the input length exceeds the context length"
                )
            return [[0.1, 0.2, 0.3] for _ in texts]

        model.aembed = AsyncMock(side_effect=fake_aembed)
        service = EmbeddingService(
            repo, model,
            # cap above the text length so the cap does NOT catch it; the
            # shrink path must do the work.
            EmbeddingConfig(max_input_chars=10000, retry_attempts=1, retry_delay=0.01),
        )

        result = await service.embed_source("source:dense")

        assert result.embeddings_created == 1  # not dropped
        stored = repo.add_embedding.call_args.kwargs["embedding"]
        assert len(stored) == 3

    @pytest.mark.asyncio
    async def test_non_length_error_still_retries_and_raises(self):
        """A non-length error is not treated as a truncation case."""
        repo = _make_source_repo()
        repo.get_chunks.return_value = [_make_chunk("Content.", 0, "chunk:1")]
        model = AsyncMock()
        model.aembed = AsyncMock(side_effect=RuntimeError("connection refused"))
        service = EmbeddingService(
            repo, model,
            EmbeddingConfig(retry_attempts=1, retry_delay=0.01),
        )

        with pytest.raises(RuntimeError, match="Embedding failed after"):
            await service.embed_source("source:conn")
        # Retried (not short-circuited as a length error): 1 + retry_attempts
        assert model.aembed.call_count == 2

    def test_cap_disabled_when_zero(self):
        """``max_input_chars=0`` disables the char cap (passthrough)."""
        repo = _make_source_repo()
        model = _make_embedding_model()
        service = EmbeddingService(repo, model, EmbeddingConfig(max_input_chars=0))
        long = ["Z" * 9000]
        assert service._cap_texts(long) == long
