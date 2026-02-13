"""Embedding provider abstractions."""

from embeddings.providers.base import EmbeddingProvider
from embeddings.providers.esperanto_provider import EsperantoProvider

__all__ = ["EmbeddingProvider", "EsperantoProvider"]
