"""F2-08: elke modelaanroep uit app-main gaat via ``ModelGate`` naar llama-swap.

Getoetst per soort aanroeper (chatgrafen, extractie, samenvatten,
voorbewerking, de filteringsmatcher):

- de aanvraag gaat naar ``http://llama-swap:8080/v1`` met het model uit het
  register en de juiste payload;
- de classificatie van de invoer gaat mee en staat in de auditregel; precies één
  regel per aanroep;
- een model dat intern niet mag, krijgt geen interne of onbekende invoer: geen
  aanvraag, wel een regel ``geweigerd``; er is geen terugval.

Er draait geen llama-swap: een ``httpx.MockTransport`` legt de aanvragen vast.
Het register is synthetisch.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import List

import httpx
import pytest
from langchain_core.messages import HumanMessage, SystemMessage
from llm_manager import aanroep
from llm_manager.gate import ModelCallRegel, ModelGeweigerd
from llm_manager.providers.factory import LlamaSwapClient
from llm_manager.register import ModelRegister, RegisterModel


def _register(*, intern: str = "ja") -> ModelRegister:
    return ModelRegister(versie=1, modellen=[
        RegisterModel(naam="werkmodel", versie="v1", aanroepwijze="llama-swap",
                      locatie="lokaal", toegestaan_intern=intern,
                      taken=["chat", "entity_extraction", "entity_dedup", "summarization"]),
        RegisterModel(naam="embed", versie="v1", aanroepwijze="llama-swap",
                      locatie="lokaal", toegestaan_intern=intern, taken=["embedding"]),
    ])


def _graphs_utils():
    """Laad ``app_main.graphs.utils`` uit zijn bestand: het pakket ``graphs``
    importeert bij het laden ``ask.py``, dat ``ai_prompter`` nodig heeft (niet in
    de testomgeving), zoals in ``test_ask_graph_citations.py``."""
    naam = "app_main.graphs.utils"
    if naam in sys.modules:
        return sys.modules[naam]
    pad = Path(__file__).resolve().parents[1] / "src" / "app_main" / "graphs" / "utils.py"
    spec = importlib.util.spec_from_file_location(naam, pad)
    module = importlib.util.module_from_spec(spec)
    sys.modules[naam] = module
    spec.loader.exec_module(module)
    return module


class Poort:
    def __init__(self) -> None:
        self.verzoeken: List[httpx.Request] = []
        self.regels: List[ModelCallRegel] = []

    def configureer(self, register: ModelRegister) -> None:
        poort = self

        class _Audit:
            async def schrijf(self, regel: ModelCallRegel) -> None:
                poort.regels.append(regel)

        aanroep.configureer(_Audit(), register=register,
                            client=LlamaSwapClient(transport=httpx.MockTransport(self)))

    def __call__(self, verzoek: httpx.Request) -> httpx.Response:
        self.verzoeken.append(verzoek)
        if verzoek.url.path.endswith("/embeddings"):
            n = len(json.loads(verzoek.content)["input"])
            return httpx.Response(200, json={"data": [
                {"index": i, "embedding": [0.1] * 1024} for i in range(n)]})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": '{"summary": "s", "document_type": "other"}'}}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2}})

    def payload(self, i: int = -1) -> dict:
        return json.loads(self.verzoeken[i].content)


@pytest.fixture
def poort():
    p = Poort()
    p.configureer(_register())
    yield p
    aanroep.reset()


def _alles_naar_llama_swap(poort: Poort, pad: str = "/chat/completions") -> None:
    assert [str(v.url) for v in poort.verzoeken] == [f"http://llama-swap:8080/v1{pad}"]


async def test_chatgraaf_gaat_via_de_poort(poort):
    provision_langchain_model = _graphs_utils().provision_langchain_model

    model = await provision_langchain_model(
        "inhoud", "model:oude-keuze", "tools", classificatie="onbekend",
        max_tokens=100, structured={"type": "json"})
    antwoord = await model.ainvoke([SystemMessage(content="s"), HumanMessage(content="u")])

    assert antwoord.content.startswith('{"summary"')
    _alles_naar_llama_swap(poort)
    assert poort.payload() == {
        "model": "werkmodel", "stream": False, "max_tokens": 100,
        "response_format": {"type": "json_object"},
        "messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]}
    [regel] = poort.regels
    assert (regel.task, regel.classification, regel.status) == ("chat", "onbekend", "geslaagd")


async def test_de_classificatie_is_verplicht_voor_de_chatgrafen():
    provision_langchain_model = _graphs_utils().provision_langchain_model

    with pytest.raises(TypeError):
        await provision_langchain_model("x", None, "chat")  # type: ignore[call-arg]


@pytest.mark.parametrize("veld,taak,json_verwacht", [
    ("default_extraction_model", "entity_extraction", True),
    ("default_chat_model", "chat", False),
    ("default_transformation_model", "summarization", False),
])
async def test_extractie_aanroeper_via_de_poort(poort, veld, taak, json_verwacht):
    from app_main.services.entity_extraction_service import make_default_llm_caller

    caller = await make_default_llm_caller(default_field=veld, classificatie="intern")
    tekst = await caller("systeem", "gebruiker", "model:oude-keuze")

    assert tekst
    _alles_naar_llama_swap(poort)
    assert ("response_format" in poort.payload()) is json_verwacht
    assert caller.served_provider == "llama-swap" and caller.served_model_id == "werkmodel"
    [regel] = poort.regels
    assert (regel.task, regel.classification) == (taak, "intern")


@pytest.mark.parametrize("classificatie", ["intern", None, "onbekend"])
async def test_geen_terugval_bij_een_model_dat_intern_niet_mag(classificatie):
    from app_main.services.entity_extraction_service import make_default_llm_caller

    poort = Poort()
    poort.configureer(_register(intern="nee"))
    try:
        caller = await make_default_llm_caller(
            default_field="default_extraction_model", classificatie=classificatie)
        with pytest.raises(ModelGeweigerd):
            await caller("s", "u")
    finally:
        aanroep.reset()
    assert poort.verzoeken == []
    [regel] = poort.regels
    assert regel.status == "geweigerd"


async def test_extern_mag_bij_een_model_dat_intern_niet_mag():
    from app_main.services.entity_extraction_service import make_default_llm_caller

    poort = Poort()
    poort.configureer(_register(intern="nee"))
    try:
        caller = await make_default_llm_caller(
            default_field="default_extraction_model", classificatie="extern")
        await caller("s", "u")
    finally:
        aanroep.reset()
    _alles_naar_llama_swap(poort)


async def test_samenvatten_via_de_poort(poort):
    from app_main.services.summarization_service import SummarizationService

    model = SummarizationService._poortmodel("extern")
    antwoord = await model.achat_complete([{"role": "user", "content": "vat samen"}])

    assert antwoord.content
    _alles_naar_llama_swap(poort)
    assert poort.payload()["model"] == "werkmodel"
    [regel] = poort.regels
    assert (regel.task, regel.classification) == ("summarization", "extern")


async def test_voorbewerking_via_de_poort(poort):
    from unittest.mock import AsyncMock

    from app_main.services.preprocessing_service import PreprocessingService

    svc = PreprocessingService(chunk_repo=AsyncMock(), met_model=True)
    samenvatting, _ = await svc._analyze_document("tekst", "intern")

    assert samenvatting == "s"
    _alles_naar_llama_swap(poort)
    [regel] = poort.regels
    assert (regel.task, regel.classification) == ("summarization", "intern")


async def test_filteringsmatcher_via_de_poort(poort):
    from app_main.services.entity_extraction_service import EntityExtractionService

    svc = EntityExtractionService.__new__(EntityExtractionService)
    svc._classificatie = "intern"
    svc._run_id = "processing_run:r"
    antwoord = await svc._llm_chat([{"role": "user", "content": "zelfde?"}])

    assert antwoord.model == "werkmodel"
    _alles_naar_llama_swap(poort)
    assert poort.payload()["response_format"] == {"type": "json_object"}
    [regel] = poort.regels
    assert (regel.task, regel.classification, regel.run) == (
        "entity_dedup", "intern", "processing_run:r")


async def test_extractie_en_samenvatten_geven_de_run_mee(poort):
    """F2-15 (review F2-08): de run gaat mee naar de poort, voor ``processing_run.models``."""
    from app_main.services.entity_extraction_service import EntityExtractionService
    from app_main.services.summarization_service import SummarizationService

    svc = EntityExtractionService.__new__(EntityExtractionService)
    svc._classificatie, svc._run_id = "intern", "processing_run:r"
    caller = await svc._make_routed_caller()
    await caller("systeem", "gebruiker")
    await SummarizationService._poortmodel("intern", "processing_run:r").achat_complete(
        [{"role": "user", "content": "vat samen"}])

    assert [(r.task, r.run) for r in poort.regels] == [
        ("entity_extraction", "processing_run:r"), ("summarization", "processing_run:r")]


async def test_zoeken_embedt_de_vraag_via_de_poort(poort):
    from unittest.mock import AsyncMock

    from app_main.services.search_service import SearchService

    retrieval = await SearchService(search_repo=AsyncMock())._get_retrieval()
    vector = await retrieval._embed("zoekvraag")

    assert len(vector) == 1024
    _alles_naar_llama_swap(poort, "/embeddings")
    assert poort.payload() == {"model": "embed", "input": ["zoekvraag"]}
    [regel] = poort.regels
    assert (regel.task, regel.classification, regel.run) == ("embedding", "onbekend", None)
