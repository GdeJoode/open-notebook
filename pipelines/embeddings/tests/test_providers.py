"""Tests for embedding providers."""

import httpx
import pytest
from llm_manager.aanroep import PoortAanroeper
from llm_manager.gate import ModelCallRegel, ModelGate
from llm_manager.providers.factory import LlamaSwapClient
from llm_manager.register import ModelRegister, RegisterModel

from embeddings.providers.base import EmbeddingProvider
from embeddings.providers.poort_provider import PoortEmbedder


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


class _Audit:
    def __init__(self):
        self.regels: list[ModelCallRegel] = []

    async def schrijf(self, regel):
        self.regels.append(regel)


class TestPoortEmbedder:
    """F2-08: embeddings gaan via ModelGate naar /v1/embeddings van llama-swap."""

    @pytest.mark.asyncio
    async def test_aembed_gaat_via_de_poort_naar_llama_swap(self):
        verzoeken = []

        def llamaswap(verzoek: httpx.Request) -> httpx.Response:
            verzoeken.append(verzoek)
            return httpx.Response(200, json={"data": [
                {"index": 0, "embedding": [0.1] * 1024},
                {"index": 1, "embedding": [0.2] * 1024},
            ]})

        register = ModelRegister(versie=1, modellen=[RegisterModel(
            naam="embed", versie="v1", aanroepwijze="llama-swap", locatie="lokaal",
            toegestaan_intern="ja", taken=["embedding"])])
        audit = _Audit()
        embedder = PoortEmbedder(PoortAanroeper(
            ModelGate(register, audit),
            LlamaSwapClient(transport=httpx.MockTransport(llamaswap))))

        vectoren = await embedder.aembed(["a", "b"], classificatie="intern", run_id="processing_run:r")

        assert [len(v) for v in vectoren] == [1024, 1024]
        assert str(verzoeken[0].url) == "http://llama-swap:8080/v1/embeddings"
        [regel] = audit.regels
        assert (regel.model, regel.task, regel.classification, regel.run) == (
            "embed", "embedding", "intern", "processing_run:r")

    @pytest.mark.asyncio
    async def test_de_classificatie_is_verplicht(self):
        embedder = PoortEmbedder(object())  # type: ignore[arg-type]
        with pytest.raises(TypeError):
            await embedder.aembed(["a"])  # type: ignore[call-arg]
