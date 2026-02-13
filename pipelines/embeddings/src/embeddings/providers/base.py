"""
Abstract base class for embedding providers.
"""

from abc import ABC, abstractmethod
from typing import Optional


class EmbeddingProvider(ABC):
    """Interface for embedding model providers.

    All embedding providers must implement embed() and embed_single().
    Optionally expose dimension and max_tokens metadata.
    """

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into vectors.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (one per input text).
        """

    async def embed_single(self, text: str) -> list[float]:
        """Embed a single text string.

        Default implementation delegates to embed() with a single-item list.
        """
        vectors = await self.embed([text])
        return vectors[0]

    @property
    def dimension(self) -> Optional[int]:
        """Embedding vector dimension, if known."""
        return None

    @property
    def max_tokens(self) -> Optional[int]:
        """Maximum input tokens per text, if known."""
        return None
