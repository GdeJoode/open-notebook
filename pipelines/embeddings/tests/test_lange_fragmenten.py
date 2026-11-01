"""Fragmenten boven 512 tokens: afkappen (F2-08, ADR 0008).

mxbai-embed-large draait in llama-server met ``-c 512``. De keuze is afkappen,
niet opdelen: één fragment krijgt één vector, de kop blijft.

- Vóór elke aanroep gaat een fragment met ten hoogste ``max_input_chars``
  (1 536) tekens naar het model.
- Weigert llama-server de invoer toch wegens lengte
  (``exceed_context_size_error``, en alleen dan), dan halveert de tekst tot hij past, maar
  nooit onder ``ZEKER_TEKENS`` (510): zo'n tekst past altijd in 512 tokens.
  Weigert het model die ook, dan is het geen lengteprobleem en volgt een fout.
- Elke poging is één aanroep via ``ModelGate`` en dus één regel in ``model_call``.

Er draait geen llama-swap: een ``httpx.MockTransport`` speelt llama-server na.
De teksten zijn synthetisch.
"""

from __future__ import annotations

import json
from typing import List, Optional
from unittest.mock import AsyncMock

import httpx
import pytest
from llm_manager.aanroep import PoortAanroeper
from llm_manager.gate import AuditNietGeschreven, ModelCallRegel, ModelGate, ModelGeweigerd
from llm_manager.providers.factory import LlamaSwapClient
from llm_manager.register import ModelRegister, RegisterModel

from embeddings.config import EmbeddingConfig
from embeddings.providers import PoortEmbedder
from embeddings.service import ZEKER_TEKENS, EmbeddingService
from shared.models import Chunk, SourceEmbedding


class Audit:
    def __init__(self, faalt: bool = False) -> None:
        self.regels: List[ModelCallRegel] = []
        self.faalt = faalt

    async def schrijf(self, regel: ModelCallRegel) -> None:
        if self.faalt:
            raise RuntimeError("database weg")
        self.regels.append(regel)


class LlamaServer:
    """Weigert invoer langer dan ``grens`` tekens zoals llama-server dat doet."""

    def __init__(self, grens: Optional[int] = None, status: int = 400,
                 soort: Optional[str] = "exceed_context_size_error") -> None:
        self.grens = grens
        self.status = status
        self.soort = soort
        self.invoer: List[List[str]] = []

    def __call__(self, verzoek: httpx.Request) -> httpx.Response:
        teksten = json.loads(verzoek.content)["input"]
        self.invoer.append(teksten)
        if self.grens is not None and any(len(t) > self.grens for t in teksten):
            fout = {"code": self.status, "message": "te lang"}
            if self.soort:
                fout["type"] = self.soort
            return httpx.Response(self.status, json={"error": fout})
        return httpx.Response(200, json={"data": [
            {"index": i, "embedding": [0.5] * 1024} for i in range(len(teksten))]})


def _service(server: LlamaServer, audit: Audit, intern: str = "ja",
             **config) -> tuple[EmbeddingService, AsyncMock]:
    register = ModelRegister(versie=1, modellen=[RegisterModel(
        naam="embed", versie="v1", aanroepwijze="llama-swap", locatie="lokaal",
        toegestaan_intern=intern, taken=["embedding"])])
    gate = ModelGate(register, audit, max_pogingen=1)
    embedder = PoortEmbedder(PoortAanroeper(
        gate, LlamaSwapClient(transport=httpx.MockTransport(server))))
    repo = AsyncMock()
    repo.delete_embeddings = AsyncMock(return_value=0)
    repo.add_embedding = AsyncMock(side_effect=lambda **kw: SourceEmbedding(
        source=kw["source_id"], content=kw["content"], order=kw["order"],
        embedding=kw["embedding"], chunk=kw.get("chunk_id"), afgekapt=kw.get("afgekapt")))
    return EmbeddingService(repo, embedder, EmbeddingConfig(retry_delay=0, **config)), repo


def _fragment(tekst: str, nr: int, classificatie: str = "intern") -> Chunk:
    return Chunk(id=f"chunk:{nr}", text=tekst, order=nr, element_type="paragraph",
                 physical_page=1, classification=classificatie)


async def _embed(service: EmbeddingService, repo: AsyncMock, *teksten: str) -> None:
    repo.get_chunks = AsyncMock(return_value=[_fragment(t, i) for i, t in enumerate(teksten)])
    await service.embed_source("source:s", run_id="processing_run:r")


@pytest.fixture(autouse=True)
def _geen_aggregaat(monkeypatch):
    monkeypatch.setattr("embeddings.aggregate.populate_source_embedding", AsyncMock())


def test_de_regel_staat_in_de_configuratie():
    assert EmbeddingConfig().max_input_chars == 1536
    assert ZEKER_TEKENS == 510


async def test_een_lang_fragment_wordt_afgekapt_tot_een_vector():
    server, audit = LlamaServer(), Audit()
    service, repo = _service(server, audit)

    await _embed(service, repo, "A" * 5000)

    assert server.invoer == [["A" * 1536]]
    [aanroep] = repo.add_embedding.call_args_list
    assert aanroep.kwargs["content"] == "A" * 5000  # het fragment blijft heel
    assert len(aanroep.kwargs["embedding"]) == 1024
    [regel] = audit.regels
    assert (regel.status, regel.classification, regel.run) == (
        "geslaagd", "intern", "processing_run:r")


async def test_korte_fragmenten_blijven_ongemoeid():
    server, audit = LlamaServer(), Audit()
    service, repo = _service(server, audit)

    await _embed(service, repo, "kort", "B" * 1536)

    assert server.invoer == [["kort", "B" * 1536]]
    assert repo.add_embedding.call_count == 2


async def test_bij_een_lengtefout_halveert_de_tekst_tot_hij_past():
    server, audit = LlamaServer(grens=1000), Audit()
    service, repo = _service(server, audit, retry_attempts=0)

    await _embed(service, repo, "C" * 5000)

    assert [len(t[0]) for t in server.invoer] == [1536, 1536, 768]
    assert repo.add_embedding.call_count == 1
    assert [r.status for r in audit.regels] == ["mislukt", "mislukt", "geslaagd"]
    assert {r.error_code for r in audit.regels[:2]} == {"http_400"}


async def test_onder_de_zekere_lengte_wordt_niet_verder_gehalveerd():
    server, audit = LlamaServer(grens=100), Audit()
    service, repo = _service(server, audit, retry_attempts=0)

    with pytest.raises(RuntimeError):
        await _embed(service, repo, "D" * 5000)

    assert [len(t[0]) for t in server.invoer] == [1536, 1536, 768, ZEKER_TEKENS]
    repo.add_embedding.assert_not_called()


async def test_een_andere_fout_wordt_niet_als_lengtefout_behandeld():
    server, audit = LlamaServer(grens=0, status=500, soort=None), Audit()
    service, repo = _service(server, audit, retry_attempts=1)

    with pytest.raises(RuntimeError, match="Embedding failed after"):
        await _embed(service, repo, "E" * 2000)

    assert [len(t[0]) for t in server.invoer] == [1536, 1536]


async def test_een_server_error_wordt_niet_gehalveerd():
    server, audit = LlamaServer(grens=1000, status=500, soort="server_error"), Audit()
    service, repo = _service(server, audit, retry_attempts=0)

    with pytest.raises(RuntimeError, match="Embedding failed after"):
        await _embed(service, repo, "F" * 5000)

    assert [len(t[0]) for t in server.invoer] == [1536]
    assert [r.status for r in audit.regels] == ["mislukt"]


async def test_een_weigering_van_de_poort_wordt_niet_herhaald():
    server, audit = LlamaServer(), Audit()
    service, repo = _service(server, audit, intern="nee", retry_attempts=2)

    with pytest.raises(ModelGeweigerd):
        await _embed(service, repo, "G" * 100)

    assert server.invoer == []
    assert [r.status for r in audit.regels] == ["geweigerd"]


async def test_een_ontbrekende_auditregel_wordt_niet_herhaald():
    server = LlamaServer()
    service, repo = _service(server, Audit(faalt=True), retry_attempts=2)

    with pytest.raises(AuditNietGeschreven):
        await _embed(service, repo, "H" * 100)

    assert len(server.invoer) == 1


async def test_een_gemengde_batch_telt_als_intern():
    """Een batch met een intern en een extern fragment gaat als ``intern``."""
    server, audit = LlamaServer(), Audit()
    service, repo = _service(server, audit)
    repo.get_chunks = AsyncMock(return_value=[
        _fragment("extern stuk", 0, "extern"), _fragment("intern stuk", 1, "intern")])

    await service.embed_source("source:s")

    assert server.invoer == [["extern stuk", "intern stuk"]]
    [regel] = audit.regels
    assert regel.classification == "intern"


async def test_stap_embed_legt_run_model_en_afgekapt_vast():
    """F2-15: de stap ``embed`` geeft per embedding run, model en ``afgekapt``.

    Afgekapt is zowel de vaste grens (1 536) als het halveren na een lengtefout.
    Een fragment in ``overslaan`` heeft al een embedding in deze run.
    """
    server, audit = LlamaServer(grens=1000), Audit()
    service, repo = _service(server, audit, retry_attempts=0)
    fragmenten = [_fragment("kort", 0), _fragment("I" * 1200, 1), _fragment("J" * 5000, 2),
                  _fragment("al gedaan", 3)]

    uitkomst = await service.embed_fragmenten(
        "source:s", fragmenten, run_id="processing_run:r", overslaan={"chunk:3"})

    gegeven = [c.kwargs for c in repo.add_embedding.call_args_list]
    assert [(k["chunk_id"], k["afgekapt"]) for k in gegeven] == [
        ("chunk:0", False), ("chunk:1", True), ("chunk:2", True)]
    assert {(k["run_id"], k["embedding_model"]) for k in gegeven} == {
        ("processing_run:r", "embed:v1")}
    assert (uitkomst.embeddings_created, uitkomst.afgekapt) == (3, 2)
    assert {r.run for r in audit.regels} == {"processing_run:r"}
    repo.delete_embeddings.assert_not_called()
