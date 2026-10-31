"""Stopconditie F2-07: de modelpoort (ADR 0006, 0007, 0009).

- een cloudmodel wordt geweigerd bij interne of onbekende invoer, ook als het
  lokale model faalt; er is geen terugval;
- een onbekend model wordt geweigerd;
- HTTP 502 leidt tot een nieuwe poging, tot het maximum;
- elke aanroep geeft precies één auditregel, met ``run`` als die is meegegeven,
  en nooit prompt- of antwoordtekst.

Het register in deze tests is synthetisch.
"""

from __future__ import annotations

from typing import List

import httpx
import pytest
from llm_manager.gate import (
    AuditNietGeschreven,
    ModelCallRegel,
    ModelGate,
    ModelGeweigerd,
)
from llm_manager.register import ModelRegister, RegisterModel

GEHEIM = "INTERNE-TEKST-DIE-NOOIT-IN-DE-AUDIT-MAG"


def _register() -> ModelRegister:
    return ModelRegister(versie=1, modellen=[
        RegisterModel(naam="lokaal-model", versie="v1", aanroepwijze="llama-swap",
                      locatie="lokaal", toegestaan_intern="ja", taken=["chat"]),
        RegisterModel(naam="lokaal-alleen-extern", versie="v1", aanroepwijze="llama-swap",
                      locatie="lokaal", taken=["chat"]),
        RegisterModel(naam="cloudmodel", versie="2026-10", aanroepwijze="api",
                      locatie="cloud", taken=["chat"]),
        RegisterModel(naam="externe-dienst", versie="v1", aanroepwijze="llama-swap",
                      locatie="cloud", taken=["document_extraction"]),
    ])


class Audit:
    def __init__(self, faalt: bool = False) -> None:
        self.regels: List[ModelCallRegel] = []
        self.faalt = faalt

    async def schrijf(self, regel: ModelCallRegel) -> None:
        if self.faalt:
            raise RuntimeError("database weg")
        self.regels.append(regel)


class Aanroeper:
    """Houdt bij welke modellen echt zijn aangeroepen."""

    def __init__(self, fouten: List[Exception] | None = None) -> None:
        self.fouten = list(fouten or [])
        self.aangeroepen: List[str] = []

    async def __call__(self, model: RegisterModel):
        self.aangeroepen.append(model.naam)
        if self.fouten:
            raise self.fouten.pop(0)
        return {"tekst": GEHEIM, "usage": {"prompt_tokens": 12, "completion_tokens": 3}}


def _http(status: int) -> httpx.HTTPStatusError:
    verzoek = httpx.Request("POST", "http://llama-swap:8080/v1/chat/completions")
    return httpx.HTTPStatusError(GEHEIM, request=verzoek,
                                 response=httpx.Response(status, request=verzoek))


def _gate(audit: Audit, **kw) -> ModelGate:
    kw.setdefault("slaap", _geen_slaap)
    return ModelGate(_register(), audit, **kw)


async def _geen_slaap(_: float) -> None:
    return None


@pytest.mark.parametrize("classificatie", ["intern", "onbekend", None, "INTERN", ""])
@pytest.mark.parametrize("herkomst", ["fork", "langflow"])
async def test_cloudmodel_geweigerd_bij_interne_invoer(classificatie, herkomst) -> None:
    audit, cloud = Audit(), Aanroeper()
    with pytest.raises(ModelGeweigerd):
        await _gate(audit, herkomst=herkomst).roep_aan(
            model="cloudmodel", classificatie=classificatie, taak="chat", uitvoer=cloud)
    assert cloud.aangeroepen == []
    [regel] = audit.regels
    assert regel.status == "geweigerd" and regel.attempts == 0
    verwacht = "intern" if classificatie == "intern" else "onbekend"
    assert regel.classification == verwacht


async def test_cloudmodel_geweigerd_ook_als_het_lokale_model_faalt() -> None:
    audit = Audit()
    gate = _gate(audit, herkomst="langflow")
    lokaal = Aanroeper(fouten=[_http(500)])
    with pytest.raises(httpx.HTTPStatusError):
        await gate.roep_aan(model="lokaal-model", classificatie="intern", taak="chat",
                            uitvoer=lokaal, run_id="processing_run:r1")
    # De poort valt niet zelf terug: alleen het gevraagde model is aangeroepen.
    assert lokaal.aangeroepen == ["lokaal-model"]

    # Een aanroeper die daarna zelf naar de cloud wil, wordt geweigerd.
    cloud = Aanroeper()
    with pytest.raises(ModelGeweigerd) as fout:
        await gate.roep_aan(model="cloudmodel", classificatie="intern", taak="chat",
                            uitvoer=cloud, run_id="processing_run:r1")
    assert fout.value.code == "niet_toegestaan_intern"
    assert cloud.aangeroepen == []
    assert [r.status for r in audit.regels] == ["mislukt", "geweigerd"]
    assert audit.regels[0].error_code == "http_500"


async def test_externe_verwerkingsdienst_geweigerd_bij_interne_invoer() -> None:
    # ADR 0019: een externe verwerkingsdienst valt onder dezelfde regel.
    audit, dienst = Audit(), Aanroeper()
    with pytest.raises(ModelGeweigerd):
        await _gate(audit).roep_aan(model="externe-dienst", classificatie="intern",
                                    taak="document_extraction", uitvoer=dienst)
    assert dienst.aangeroepen == []


async def test_lokaal_model_zonder_ja_geweigerd_bij_interne_invoer() -> None:
    # Standaard `nee`: ook een lokaal model krijgt geen interne invoer zonder `ja`.
    audit, lokaal = Audit(), Aanroeper()
    with pytest.raises(ModelGeweigerd):
        await _gate(audit).roep_aan(model="lokaal-alleen-extern", classificatie="intern",
                                    taak="chat", uitvoer=lokaal)
    assert lokaal.aangeroepen == []
    await _gate(audit).roep_aan(model="lokaal-alleen-extern", classificatie="extern",
                                taak="chat", uitvoer=lokaal)
    assert lokaal.aangeroepen == ["lokaal-alleen-extern"]


async def test_fork_roept_geen_cloudmodel_aan_ook_niet_bij_externe_invoer() -> None:
    audit, cloud = Audit(), Aanroeper()
    with pytest.raises(ModelGeweigerd) as fout:
        await _gate(audit, herkomst="fork").roep_aan(
            model="cloudmodel", classificatie="extern", taak="chat", uitvoer=cloud)
    assert fout.value.code == "aanroepwijze_niet_in_fork"
    assert cloud.aangeroepen == []
    # Vanuit Langflow mag een cloudmodel bij externe invoer (via de gateway, ADR 0022).
    await _gate(audit, herkomst="langflow").roep_aan(
        model="cloudmodel", classificatie="extern", taak="chat", uitvoer=cloud)
    assert cloud.aangeroepen == ["cloudmodel"]


@pytest.mark.parametrize("model", ["bestaat-niet", "vrije tekst met spaties " + GEHEIM])
async def test_onbekend_model_geweigerd(model) -> None:
    audit, aanroeper = Audit(), Aanroeper()
    with pytest.raises(ModelGeweigerd) as fout:
        await _gate(audit).roep_aan(model=model, classificatie="extern", taak="chat",
                                    uitvoer=aanroeper)
    assert fout.value.code == "model_onbekend"
    assert aanroeper.aangeroepen == []
    [regel] = audit.regels
    assert regel.status == "geweigerd" and regel.error_code == "model_onbekend"
    assert GEHEIM not in regel.model_dump_json()


async def test_502_nieuwe_poging_tot_het_maximum() -> None:
    pauzes: List[float] = []

    async def slaap(s: float) -> None:
        pauzes.append(s)

    audit = Audit()
    aanroeper = Aanroeper(fouten=[_http(502)] * 10)
    with pytest.raises(httpx.HTTPStatusError):
        await _gate(audit, max_pogingen=4, pauze=2.5, slaap=slaap).roep_aan(
            model="lokaal-model", classificatie="intern", taak="chat", uitvoer=aanroeper)
    assert aanroeper.aangeroepen == ["lokaal-model"] * 4
    assert pauzes == [2.5, 2.5, 2.5]
    [regel] = audit.regels
    assert (regel.status, regel.attempts, regel.error_code) == ("mislukt", 4, "http_502")


async def test_502_daarna_geslaagd() -> None:
    audit = Audit()
    aanroeper = Aanroeper(fouten=[_http(502), _http(502)])
    await _gate(audit, max_pogingen=4).roep_aan(
        model="lokaal-model", classificatie="intern", taak="chat", uitvoer=aanroeper)
    assert len(aanroeper.aangeroepen) == 3
    [regel] = audit.regels
    assert (regel.status, regel.attempts) == ("geslaagd", 3)


async def test_andere_fout_geen_nieuwe_poging() -> None:
    audit = Audit()
    aanroeper = Aanroeper(fouten=[_http(503)])
    with pytest.raises(httpx.HTTPStatusError):
        await _gate(audit).roep_aan(model="lokaal-model", classificatie="intern",
                                    taak="chat", uitvoer=aanroeper)
    assert len(aanroeper.aangeroepen) == 1
    assert audit.regels[0].error_code == "http_503"


async def test_een_auditregel_met_run_en_zonder_tekst() -> None:
    audit = Audit()
    gate = _gate(audit)
    resultaat = await gate.roep_aan(model="lokaal-model", classificatie="intern",
                                    taak="chat", uitvoer=Aanroeper(),
                                    run_id="processing_run:abc")
    assert resultaat["tekst"] == GEHEIM
    await gate.roep_aan(model="lokaal-model", classificatie="extern", taak="chat",
                        uitvoer=Aanroeper())
    eerste, tweede = audit.regels
    assert eerste.run == "processing_run:abc" and tweede.run is None
    assert eerste.model == "lokaal-model" and eerste.model_version == "v1"
    assert eerste.invocation == "llama-swap" and eerste.location == "lokaal"
    assert eerste.classification == "intern" and eerste.origin == "fork"
    assert (eerste.input_tokens, eerste.output_tokens) == (12, 3)
    assert eerste.status == "geslaagd" and eerste.duration_ms is not None
    for regel in audit.regels:
        assert GEHEIM not in regel.model_dump_json()


async def test_fouttekst_komt_niet_in_de_auditregel() -> None:
    audit = Audit()

    async def faalt(_: RegisterModel):
        raise ValueError(GEHEIM)

    with pytest.raises(ValueError):
        await _gate(audit).roep_aan(model="lokaal-model", classificatie="intern",
                                    taak="chat", uitvoer=faalt)
    [regel] = audit.regels
    assert regel.error_code == "valueerror"
    assert GEHEIM not in regel.model_dump_json()


async def test_geen_resultaat_zonder_auditregel() -> None:
    with pytest.raises(AuditNietGeschreven):
        await _gate(Audit(faalt=True)).roep_aan(
            model="lokaal-model", classificatie="intern", taak="chat", uitvoer=Aanroeper())


async def test_weigering_blijft_weigering_als_de_audit_faalt() -> None:
    cloud = Aanroeper()
    with pytest.raises(ModelGeweigerd):
        await _gate(Audit(faalt=True), herkomst="langflow").roep_aan(
            model="cloudmodel", classificatie="intern", taak="chat", uitvoer=cloud)
    assert cloud.aangeroepen == []


def test_toets_zonder_aanroep() -> None:
    gate = _gate(Audit(), herkomst="langflow")
    assert gate.toets("cloudmodel", "extern").naam == "cloudmodel"
    for klasse in ("intern", "onbekend", None):
        with pytest.raises(ModelGeweigerd):
            gate.toets("cloudmodel", klasse)


def test_auditregel_weigert_tekstvelden() -> None:
    with pytest.raises(ValueError):
        ModelCallRegel(model="m", invocation="llama-swap", location="lokaal", task="chat",
                       classification="intern", status="geslaagd", origin="langflow",
                       prompt=GEHEIM)
    with pytest.raises(ValueError):
        ModelCallRegel(model="m", invocation="llama-swap", location="lokaal", task="chat",
                       classification="intern", status="mislukt", origin="langflow",
                       error_code=GEHEIM)
