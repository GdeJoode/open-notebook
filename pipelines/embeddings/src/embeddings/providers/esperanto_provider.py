"""
Esperanto embedding provider — wraps esperanto.EmbeddingModel.

Makes the dependency on esperanto explicit and swappable via the
EmbeddingProvider interface.
"""

from typing import Optional

from esperanto import EmbeddingModel

from embeddings.providers.base import EmbeddingProvider


class EsperantoProvider(EmbeddingProvider):
    """Embedding provider backed by an esperanto EmbeddingModel."""

    def __init__(self, model: EmbeddingModel) -> None:
        self._model = model

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts using the esperanto model's aembed method."""
        return await self._model.aembed(texts)

    @property
    def model(self) -> EmbeddingModel:
        """Access the underlying esperanto model."""
        return self._model
