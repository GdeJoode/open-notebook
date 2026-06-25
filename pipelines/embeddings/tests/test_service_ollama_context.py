"""R.0b real-Ollama proof: over-long chunk text embeds via the LOCAL model.

This is the falsifiable proof for R.0b acceptance criterion 1. Unlike the
fake-model unit tests, it drives a *deliberately* over-long (~9000 char) text
through the genuine esperanto Ollama embedding provider against the local
``mxbai-embed-large`` model — the exact path the live backfill uses — and
asserts a 1024-dim vector returns instead of the "input length exceeds the
context length" failure.

It is skipped automatically when Ollama / the model is not reachable, so it
never breaks CI; it runs wherever a local Ollama with mxbai is up (the R.0b
environment). The embedding dimension is asserted to be > 0 and consistent,
NOT hardcoded to a literal in the production path — the literal 1024 here is an
assertion about *this* model, mirroring the live P.2 backfill (dim 1024).
"""

from __future__ import annotations

import os

import httpx
import pytest

from embeddings.config import EmbeddingConfig
from embeddings.service import EmbeddingService

OLLAMA_BASE_URL = (
    os.getenv("OLLAMA_BASE_URL")
    or os.getenv("OLLAMA_API_BASE")
    or "http://localhost:11434"
)
MODEL_NAME = "mxbai-embed-large:latest"
EXPECTED_DIM = 1024  # mxbai-embed-large native dimension (matches live P.2)


def _ollama_has_model() -> bool:
    try:
        resp = httpx.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        names = {m.get("name", "") for m in resp.json().get("models", [])}
        return MODEL_NAME in names or any(
            n.startswith("mxbai-embed-large") for n in names
        )
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _ollama_has_model(),
    reason=f"local Ollama with {MODEL_NAME} not reachable at {OLLAMA_BASE_URL}",
)


def _real_ollama_embedding_model():
    """Build the genuine esperanto Ollama embedding model (no mocks)."""
    from esperanto import AIFactory

    return AIFactory.create_embedding(
        model_name=MODEL_NAME,
        provider="ollama",
        config={"base_url": OLLAMA_BASE_URL},
    )


def test_baseline_long_text_fails_without_guard():
    """Sanity: the raw provider DOES reject the over-long text.

    Confirms the bug is real on this model/endpoint (otherwise the guard test
    below would prove nothing). Uses the same provider with no length guard.
    """
    model = _real_ollama_embedding_model()
    long_text = "The quick brown fox jumps over the lazy dog. " * 200  # ~9000
    with pytest.raises(Exception) as exc_info:
        import asyncio

        asyncio.run(model.aembed([long_text]))
    assert "context length" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_overlong_chunk_embeds_through_service():
    """A ~9000-char text returns a 1024-dim vector through the real path."""
    from unittest.mock import AsyncMock

    from shared.models import SourceEmbedding

    repo = AsyncMock()
    repo.config = None
    repo.delete_embeddings = AsyncMock(return_value=0)
    captured: dict = {}

    async def _add_embedding(**kwargs):
        captured["embedding"] = kwargs.get("embedding")
        return SourceEmbedding(
            source=kwargs.get("source_id"),
            content=kwargs.get("content"),
            order=kwargs.get("order", 0),
            embedding=kwargs.get("embedding"),
        )

    repo.add_embedding = AsyncMock(side_effect=_add_embedding)

    model = _real_ollama_embedding_model()
    service = EmbeddingService(repo, model, EmbeddingConfig())

    long_text = "The quick brown fox jumps over the lazy dog. " * 200  # ~9000
    assert len(long_text) > 4000

    vectors = await service._embed_with_retry([long_text])

    assert len(vectors) == 1
    assert len(vectors[0]) == EXPECTED_DIM
    assert all(isinstance(v, float) for v in vectors[0][:5])


@pytest.mark.asyncio
async def test_dense_cjk_overlong_embeds_through_service():
    """CJK text (high token density) the char cap misses still embeds.

    CJK packs far more tokens per char, so a 1000-char CJK string can exceed
    512 tokens even though it is well under ``max_input_chars``. This exercises
    the adaptive-shrink recovery against the REAL model, proving no chunk is
    dropped regardless of script.
    """
    model = _real_ollama_embedding_model()
    service = EmbeddingService(
        __import__("unittest.mock", fromlist=["AsyncMock"]).AsyncMock(),
        model,
        EmbeddingConfig(),
    )
    cjk = "知識グラフ埋め込み検索における文脈長制限の問題を解決する必要がある。" * 60
    vector = await service._embed_one_with_shrink(cjk)
    assert len(vector) == EXPECTED_DIM
