"""F2-08: modelaanroepen via ``ModelGate`` naar llama-swap (ADR 0006, 0007, 0009).

Er draait geen llama-swap: een ``httpx.MockTransport`` legt URL en payload vast.
Het register is synthetisch.
"""

from __future__ import annotations

import json
from typing import List

import httpx
import pytest
from llm_manager import aanroep
from llm_manager.aanroep import GeenPoort, OngeldigAntwoord, PoortAanroeper
from llm_manager.gate import ModelCallRegel, ModelGate, ModelGeweigerd, classificatie_van
from llm_manager.providers.factory import LLAMA_SWAP_URL, LlamaSwapClient, LlamaSwapFout
from llm_manager.register import ModelRegister, RegisterModel

GEHEIM = "INTERNE-TEKST-DIE-NIET-IN-EEN-FOUT-MAG"


def _register() -> ModelRegister:
    return ModelRegister(versie=1, modellen=[
        RegisterModel(naam="werkmodel", versie="v1", aanroepwijze="llama-swap",
                      locatie="lokaal", toegestaan_intern="ja",
                      taken=["chat", "summarization"]),
        RegisterModel(naam="embed", versie="v1", aanroepwijze="llama-swap",
                      locatie="lokaal", toegestaan_intern="ja", taken=["embedding"]),
        RegisterModel(naam="alleen-extern", versie="v1", aanroepwijze="llama-swap",
                      locatie="lokaal", taken=["chat"]),
        RegisterModel(naam="externe-dienst", versie="v1", aanroepwijze="llama-swap",
                      locatie="cloud", taken=["vertaling"]),
        RegisterModel(naam="cloudmodel", versie="v1", aanroepwijze="api",
                      locatie="cloud", taken=["chat"]),
    ])


class Audit:
    def __init__(self) -> None:
        self.regels: List[ModelCallRegel] = []

    async def schrijf(self, regel: ModelCallRegel) -> None:
        self.regels.append(regel)


class Llamaswap:
    """Nagebootst llama-swap: legt elke aanvraag vast en antwoordt volgens een lijst."""

    def __init__(self, statussen: List[int] | None = None, fouttype: str | None = None) -> None:
        self.verzoeken: List[httpx.Request] = []
        self.statussen = list(statussen or [])
        self.fouttype = fouttype

    def __call__(self, verzoek: httpx.Request) -> httpx.Response:
        self.verzoeken.append(verzoek)
        status = self.statussen.pop(0) if self.statussen else 200
        if status != 200:
            fout = {"message": GEHEIM, **({"type": self.fouttype} if self.fouttype else {})}
            return httpx.Response(status, json={"error": fout})
        body = json.loads(verzoek.content)
        if verzoek.url.path.endswith("/embeddings"):
            data = [{"index": i, "embedding": [float(i)] * 4} for i in range(len(body["input"]))]
            return httpx.Response(200, json={"data": list(reversed(data)),
                                             "usage": {"prompt_tokens": 7}})
        return httpx.Response(200, json={
            "choices": [{"message": {"role": "assistant", "content": "antwoord"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 3},
        })

    def payload(self, i: int = -1) -> dict:
        return json.loads(self.verzoeken[i].content)


def _poort(llamaswap: Llamaswap, audit: Audit) -> PoortAanroeper:
    gate = ModelGate(_register(), audit, pauze=0, slaap=lambda _s: _niets())
    return PoortAanroeper(gate, LlamaSwapClient(transport=httpx.MockTransport(llamaswap)))


async def _niets() -> None:
    return None


BERICHTEN = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def test_het_adres_ligt_vast():
    assert LLAMA_SWAP_URL == "http://llama-swap:8080/v1"


async def test_chat_gaat_naar_llama_swap_met_het_registermodel():
    llamaswap, audit = Llamaswap(), Audit()
    antwoord = await _poort(llamaswap, audit).chat(
        taak="chat", classificatie="intern", messages=BERICHTEN, run_id="processing_run:r1",
        json_mode=True, max_tokens=50)

    assert antwoord.tekst == "antwoord" and antwoord.model == "werkmodel"
    assert [str(v.url) for v in llamaswap.verzoeken] == [
        "http://llama-swap:8080/v1/chat/completions"]
    assert llamaswap.payload() == {
        "model": "werkmodel", "messages": BERICHTEN, "stream": False,
        "response_format": {"type": "json_object"}, "max_tokens": 50}
    [regel] = audit.regels
    assert (regel.status, regel.model, regel.task, regel.classification, regel.run,
            regel.input_tokens, regel.output_tokens, regel.origin) == (
        "geslaagd", "werkmodel", "chat", "intern", "processing_run:r1", 11, 3, "fork")


async def test_embeddings_gaan_naar_llama_swap_in_volgorde():
    llamaswap, audit = Llamaswap(), Audit()
    vectoren = await _poort(llamaswap, audit).embed(teksten=["a", "b", "c"],
                                                    classificatie="extern")

    assert vectoren == [[0.0] * 4, [1.0] * 4, [2.0] * 4]
    assert str(llamaswap.verzoeken[0].url) == "http://llama-swap:8080/v1/embeddings"
    assert llamaswap.payload() == {"model": "embed", "input": ["a", "b", "c"]}
    [regel] = audit.regels
    assert (regel.status, regel.task, regel.classification, regel.run) == (
        "geslaagd", "embedding", "extern", None)


@pytest.mark.parametrize("classificatie", ["intern", None, "onbekend", "iets"])
@pytest.mark.parametrize("model", ["alleen-extern", "cloudmodel", "externe-dienst"])
async def test_interne_invoer_bereikt_geen_niet_toegestaan_model(model, classificatie):
    llamaswap, audit = Llamaswap(), Audit()
    with pytest.raises(ModelGeweigerd):
        await _poort(llamaswap, audit).chat(taak="chat", classificatie=classificatie,
                                            messages=BERICHTEN, model=model)
    assert llamaswap.verzoeken == []
    [regel] = audit.regels
    assert regel.status == "geweigerd"


async def test_een_falend_lokaal_model_valt_niet_terug():
    llamaswap, audit = Llamaswap(statussen=[500]), Audit()
    with pytest.raises(LlamaSwapFout) as fout:
        await _poort(llamaswap, audit).chat(taak="chat", classificatie="intern",
                                            messages=BERICHTEN)
    assert len(llamaswap.verzoeken) == 1
    assert {v.url.host for v in llamaswap.verzoeken} == {"llama-swap"}
    assert GEHEIM not in str(fout.value)
    [regel] = audit.regels
    assert (regel.status, regel.error_code) == ("mislukt", "http_500")


async def test_502_geeft_een_nieuwe_poging():
    llamaswap, audit = Llamaswap(statussen=[502, 502]), Audit()
    antwoord = await _poort(llamaswap, audit).chat(taak="chat", classificatie="extern",
                                                   messages=BERICHTEN)
    assert antwoord.tekst == "antwoord"
    assert len(llamaswap.verzoeken) == 3
    [regel] = audit.regels
    assert (regel.status, regel.attempts) == ("geslaagd", 3)


async def test_een_taak_zonder_model_wordt_geweigerd():
    llamaswap, audit = Llamaswap(), Audit()
    with pytest.raises(ModelGeweigerd) as fout:
        await _poort(llamaswap, audit).chat(taak="onbekende_taak", classificatie="extern",
                                            messages=BERICHTEN)
    assert fout.value.code == "model_onbekend" and llamaswap.verzoeken == []
    [regel] = audit.regels
    assert (regel.status, regel.model) == ("geweigerd", "ongeldige_naam")


async def test_de_classificatie_is_verplicht():
    with pytest.raises(TypeError):
        await _poort(Llamaswap(), Audit()).chat(taak="chat", messages=BERICHTEN)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        await _poort(Llamaswap(), Audit()).embed(teksten=["a"])  # type: ignore[call-arg]


async def test_de_client_negeert_proxyvariabelen(monkeypatch):
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example:3128")
    llamaswap, audit = Llamaswap(), Audit()
    await _poort(llamaswap, audit).embed(teksten=["a"], classificatie="intern")
    assert llamaswap.verzoeken[0].url.host == "llama-swap"


def test_zonder_auditschrijver_geen_aanroep():
    aanroep.reset()
    with pytest.raises(GeenPoort):
        aanroep.get_aanroeper()
    poort = aanroep.configureer(Audit(), register=_register())
    assert aanroep.get_aanroeper() is poort
    aanroep.reset()


@pytest.mark.parametrize("waarden,verwacht", [
    (["extern", "extern"], "extern"),
    (["extern", "intern"], "intern"),
    (["extern", None], "onbekend"),
    ([], "onbekend"),
])
def test_classificatie_van_meer_delen(waarden, verwacht):
    assert classificatie_van(waarden) == verwacht


class Vast:
    """Nagebootst llama-swap dat altijd hetzelfde (afwijkende) antwoord geeft."""

    def __init__(self, antwoord) -> None:
        self.antwoord = antwoord
        self.verzoeken: List[httpx.Request] = []

    def __call__(self, verzoek: httpx.Request) -> httpx.Response:
        self.verzoeken.append(verzoek)
        return httpx.Response(200, json=self.antwoord)


@pytest.mark.parametrize("antwoord", [
    {"choices": []},
    {"choices": [{"message": {}}]},
    {"choices": [{"message": {"content": None}}]},
    {"geen": "choices"},
])
async def test_een_ongeldig_chatantwoord_is_mislukt_in_de_audit(antwoord):
    audit = Audit()
    poort = _poort(Vast(antwoord), audit)
    with pytest.raises(OngeldigAntwoord):
        await poort.chat(taak="chat", classificatie="intern", messages=BERICHTEN)
    [regel] = audit.regels
    assert (regel.status, regel.error_code) == ("mislukt", "ongeldigantwoord")


@pytest.mark.parametrize("antwoord", [
    {"data": [{"index": 0, "embedding": [0.1]}]},  # te weinig vectoren
    {"data": [{"index": 0, "embedding": [0.1]}, {"index": 0, "embedding": [0.2]}]},
    {"data": [{"index": 0, "embedding": []}, {"index": 1, "embedding": []}]},
    {"data": [{"index": 0, "embedding": [0.1]}, {"index": 1, "embedding": [0.1, 0.2]}]},
    {"data": [{"index": 0, "embedding": ["x"]}, {"index": 1, "embedding": [0.1]}]},
    {"geen": "data"},
])
async def test_een_ongeldig_embeddingantwoord_is_mislukt_in_de_audit(antwoord):
    audit = Audit()
    poort = _poort(Vast(antwoord), audit)
    with pytest.raises(OngeldigAntwoord):
        await poort.embed(teksten=["a", "b"], classificatie="intern")
    [regel] = audit.regels
    assert (regel.status, regel.error_code) == ("mislukt", "ongeldigantwoord")
