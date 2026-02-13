"""Tests for embedding providers."""

from unittest.mock import AsyncMock

import pytest

from embeddings.providers.base import EmbeddingProvider
from embeddings.providers.esperanto_provider import EsperantoProvider


class TestEmbeddingProviderInterface:
    """Verify the abstract interface contract."""

    def test_cannot_instantiate_base(self):
        with pytest.raises(TypeError):
            EmbeddingProvider()

    def test_default_dimension_is_none(self):
        class Stub(EmbeddingProvider):
            async def embed(self, texts):
                return []

        assert Stub().dimension is None

    def test_default_max_tokens_is_none(self):
        class Stub(EmbeddingProvider):
            async def embed(self, texts):
                return []

        assert Stub().max_tokens is None


class TestEsperantoProvider:
    """Tests for the EsperantoProvider wrapper."""

    def _make_model(self, vectors=None):
        model = AsyncMock()
        model.aembed = AsyncMock(
            side_effect=lambda texts: vectors or [[0.1, 0.2] for _ in texts]
        )
        return model

    @pytest.mark.asyncio
    async def test_embed_delegates_to_aembed(self):
        model = self._make_model()
        provider = EsperantoProvider(model)

        result = await provider.embed(["hello", "world"])

        model.aembed.assert_called_once_with(["hello", "world"])
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_embed_single_returns_single_vector(self):
        model = self._make_model()
        provider = EsperantoProvider(model)

        result = await provider.embed_single("hello")

        assert isinstance(result, list)
        assert result == [0.1, 0.2]

    @pytest.mark.asyncio
    async def test_model_property_returns_underlying_model(self):
        model = self._make_model()
        provider = EsperantoProvider(model)

        assert provider.model is model

    @pytest.mark.asyncio
    async def test_embed_propagates_errors(self):
        model = AsyncMock()
        model.aembed = AsyncMock(side_effect=RuntimeError("API down"))
        provider = EsperantoProvider(model)

        with pytest.raises(RuntimeError, match="API down"):
            await provider.embed(["test"])
