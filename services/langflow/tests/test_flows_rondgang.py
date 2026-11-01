"""Stopconditie F2-12: laden en daarna exporteren levert dezelfde flow-JSON op.

Op volgorde van sleutels na (ADR 0017 A en B). Twee delen:

* zonder Docker, tegen een nagebootste Langflow die doet wat 1.13.0 doet: velden
  toevoegen bij het opslaan (``user_id``, ``folder_id``, ``updated_at``,
  ``workspace_id``, ``last_modified_by``, ``version_token``) en bij de export
  alleen ``user_id``, ``folder_id`` en ``updated_at`` weer weghalen;
* live, in de draaiende container langflow: de flow uit
  ``fixtures/rondgang.json`` (een export van Langflow 1.13.0 met alleen typen
  van de lijst, waaronder een schema-trigger) laden met ``flows_laden.py``, de
  trigger staat daarna aan, en ``flows_exporteren.py`` geeft dezelfde flow terug.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent / "fixtures"

#: Wat Langflow 1.13.0 bij een export zelf weghaalt (normalize_flow_for_export).
LANGFLOW_VLUCHTIG = ("updated_at", "created_at", "user_id", "folder_id", "access_type", "gradient")


def _rondgang() -> dict:
    return json.loads((FIXTURES / "rondgang.json").read_text(encoding="utf-8"))


class NepLangflow:
    """Gedrag van de API van Langflow 1.13.0 dat de rondgang raakt."""

    def __init__(self) -> None:
        self.flows: dict[str, dict] = {}
        self.trigger_staat: dict[str, str] = {}

    def upload(self, flow: dict, bestandsnaam: str = "flow.json") -> list[dict]:
        opgeslagen = copy.deepcopy(flow)
        opgeslagen.update(user_id="u1", folder_id="f1", updated_at="2026-11-01T00:00:00",
                          workspace_id="w1", access_type="PRIVATE", last_modified_by="u1",
                          version_token=f"t{len(self.flows) + 1}")
        self.flows[flow["id"]] = opgeslagen
        for knoop in flow["data"]["nodes"]:
            if knoop["data"]["type"] == "ScheduleTrigger":
                self.trigger_staat.setdefault(f"{flow['id']}:{knoop['id']}", "pending")
        return [opgeslagen]

    def download(self, flow_id: str) -> dict:
        uit = copy.deepcopy(self.flows[flow_id])
        for sleutel in LANGFLOW_VLUCHTIG:
            uit.pop(sleutel, None)
        return uit

    def triggers(self, flow_id: str) -> list[dict]:
        return [{"id": t, "state": s} for t, s in self.trigger_staat.items()
                if t.startswith(f"{flow_id}:")]

    def zet_aan(self, trigger_id: str) -> dict:
        self.trigger_staat[trigger_id] = "active"
        return {"id": trigger_id, "state": "active"}


def test_rondgang_levert_dezelfde_flow(flows_laden, flows_exporteren, tmp_path: Path) -> None:
    map_ = tmp_path / "flows"
    map_.mkdir()
    (map_ / "rondgang.json").write_text(json.dumps(_rondgang(), indent=1), encoding="utf-8")
    api = NepLangflow()

    flows = flows_laden.lees_flows(map_)
    flows_laden.laad(api, flows)
    bekend = flows_exporteren.ids_in_map(map_)
    bundel = flows_exporteren.exporteer(api, list(bekend), bekend)

    assert list(bundel) == ["rondgang.json"]
    assert bundel["rondgang.json"] == _rondgang()
    assert flows_exporteren.vergelijk(bundel, map_) == []

    doel = tmp_path / "export"
    flows_exporteren.schrijf(bundel, doel)
    assert json.loads((doel / "rondgang.json").read_text(encoding="utf-8")) == _rondgang()


def test_laden_zet_de_triggers_aan(flows_laden, tmp_path: Path) -> None:
    api = NepLangflow()
    flows_laden.laad(api, {"rondgang.json": _rondgang()})
    assert api.trigger_staat and set(api.trigger_staat.values()) == {"active"}


def test_wijziging_in_de_editor_wordt_gezien(flows_laden, flows_exporteren, tmp_path: Path) -> None:
    map_ = tmp_path / "flows"
    map_.mkdir()
    (map_ / "rondgang.json").write_text(json.dumps(_rondgang()), encoding="utf-8")
    api = NepLangflow()
    flows_laden.laad(api, flows_laden.lees_flows(map_))
    api.flows[_rondgang()["id"]]["description"] = "gewijzigd in de editor"

    bekend = flows_exporteren.ids_in_map(map_)
    bundel = flows_exporteren.exporteer(api, list(bekend), bekend)
    assert flows_exporteren.vergelijk(bundel, map_) == ["rondgang.json: wijkt af"]


def test_nieuwe_flow_krijgt_naam_uit_de_flownaam(flows_exporteren) -> None:
    assert flows_exporteren.bestandsnaam({"id": "abc", "name": "Inname: Watchfolder!"}, {}) \
        == "inname-watchfolder.json"
    assert flows_exporteren.bestandsnaam({"id": "abc", "name": "x"}, {"abc": "bestaand.json"}) \
        == "bestaand.json"


def test_geschreven_bestand_heeft_gesorteerde_sleutels_en_lf(flows_exporteren, tmp_path: Path) -> None:
    flows_exporteren.schrijf({"a.json": {"z": 1, "a": {"y": 2, "b": 3}}}, tmp_path)
    inhoud = (tmp_path / "a.json").read_bytes()
    assert b"\r\n" not in inhoud and inhoud.endswith(b"\n")
    assert inhoud.index(b'"a"') < inhoud.index(b'"z"')


# --------------------------------------------------------------------------- live

def test_live_rondgang(live, flows_exporteren) -> None:
    flow = _rondgang()
    live.zet_map("/tmp/f212-rondgang", {"rondgang.json": flow})
    try:
        uit = live.exec("python", "/app/signaal/flows_laden.py", "--map", "/tmp/f212-rondgang")
        assert uit.returncode == 0, uit.stdout + uit.stderr
        assert "geladen: rondgang.json" in uit.stdout

        staat = live.python(
            "import os\n"
            "from langflow_api import LangflowApi\n"
            "from flows_laden import lees_lijst\n"
            "a = LangflowApi()\n"
            "toegestaan = lees_lijst(os.environ['SIGNAAL_COMPONENTENLIJST'])\n"
            "laag = {t.lower() for t in toegestaan}\n"
            "geblokkeerd = a.blokkeerset()\n"
            "print('beleid', a.componentnamen() <= toegestaan, len(geblokkeerd) > 300, "
            "sorted(b for b in geblokkeerd if b.lower() in laag))\n"
            f"print(sorted(t['state'] for t in a.triggers({flow['id']!r})))")
        assert staat.returncode == 0, staat.stderr
        regels = staat.stdout.strip().splitlines()
        assert regels[-1] == "['active']"
        # Catalogusbeleid (review F2-12, punt 2): in de editor zijn alleen de typen van
        # de lijst zichtbaar, de rest is geblokkeerd, en geen type van de lijst is dat.
        assert regels[-2] == "beleid True True []", regels[-2]

        uit = live.exec("python", "/app/signaal/flows_exporteren.py", "--map", "/tmp/f212-rondgang")
        assert uit.returncode == 0, uit.stderr
        bundel = json.loads(uit.stdout)
        assert list(bundel) == ["rondgang.json"]
        assert bundel["rondgang.json"] == flows_exporteren.normaliseer(flow)

        # Dezelfde vergelijking met het script zelf, in de container.
        uit = live.exec("python", "/app/signaal/flows_exporteren.py", "--vergelijk",
                        "/tmp/f212-rondgang", invoer=json.dumps(bundel))
        assert uit.returncode == 0, uit.stderr
    finally:
        live.python(f"from langflow_api import LangflowApi; LangflowApi().verwijder({flow['id']!r})")
        live.exec("rm", "-rf", "/tmp/f212-rondgang")


# --------------------------------------------------------------------------- review F2-12

class NepLangflowMetBeleid(NepLangflow):
    def __init__(self, namen: set[str], negeer_beleid: bool = False) -> None:
        super().__init__()
        self.namen = namen
        self.geblokkeerd: set[str] = set()
        self.negeer_beleid = negeer_beleid

    def componentnamen(self) -> set[str]:
        # Zoals Langflow 1.13.0: geblokkeerde componenten staan niet in /api/v1/all.
        return set(self.namen) - self.geblokkeerd

    def zet_blokkeerset(self, sleutels: set[str]) -> set[str]:
        if not self.negeer_beleid:
            self.geblokkeerd = set(sleutels)
        return self.geblokkeerd

    def blokkeerset(self) -> set[str]:
        return set(self.geblokkeerd)

    def flow_ids(self) -> list[str]:
        return list(self.flows)

    def zet_uit(self, trigger_id: str) -> dict:
        self.trigger_staat[trigger_id] = "disabled"
        return {"id": trigger_id, "state": "disabled"}


def test_catalogusbeleid_blokkeert_alles_buiten_de_lijst(flows_laden) -> None:
    api = NepLangflowMetBeleid({"TextInput", "TextOutput", "ScheduleTrigger", "APIRequest",
                                "ext:openai:OpenAIModelComponent@official"})
    regel = flows_laden.zet_catalogusbeleid(api, {"TextInput", "TextOutput", "ScheduleTrigger", "note"})
    assert api.geblokkeerd == {"APIRequest", "ext:openai:OpenAIModelComponent@official"}
    assert "2 van 5" in regel


def test_catalogusbeleid_blijft_staan_bij_een_tweede_start(flows_laden) -> None:
    """Live gevonden: /api/v1/all laat geblokkeerde componenten weg; een tweede start
    mocht het beleid niet terugzetten naar alleen wat nog zichtbaar is."""
    api = NepLangflowMetBeleid({"TextInput", "TextOutput", "ScheduleTrigger", "APIRequest", "Agent"})
    toegestaan = {"TextInput", "TextOutput", "ScheduleTrigger", "note"}
    flows_laden.zet_catalogusbeleid(api, toegestaan)
    flows_laden.zet_catalogusbeleid(api, toegestaan)
    assert api.geblokkeerd == {"APIRequest", "Agent"}


def test_catalogusbeleid_wordt_nagekeken(flows_laden) -> None:
    import pytest

    api = NepLangflowMetBeleid({"TextInput", "APIRequest"}, negeer_beleid=True)
    with pytest.raises(ValueError):
        flows_laden.zet_catalogusbeleid(api, {"TextInput"})


def test_triggers_van_flows_buiten_git_gaan_uit(flows_laden) -> None:
    api = NepLangflowMetBeleid(set())
    eigen = _rondgang()
    vreemd = copy.deepcopy(eigen)
    vreemd["id"] = "f2120000-0000-4000-8000-0000000000ff"
    flows_laden.laad(api, {"vreemd.json": vreemd, "rondgang.json": eigen})
    flows_laden.zet_andere_triggers_uit(api, [eigen["id"]])
    staat = {t.split(":")[0]: s for t, s in api.trigger_staat.items()}
    assert staat == {eigen["id"]: "active", vreemd["id"]: "disabled"}


def _schrijf(flows_exporteren, bundel: dict, doel: Path, monkeypatch) -> int:
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(bundel)))
    return flows_exporteren.main(["--schrijf", str(doel)])


def test_schrijven_toetst_aan_de_lijst(flows_exporteren, tmp_path: Path, monkeypatch) -> None:
    flow = _rondgang()
    flow["data"]["nodes"][0]["data"]["type"] = "APIRequest"
    assert _schrijf(flows_exporteren, {"x.json": flow}, tmp_path / "uit", monkeypatch) == 1
    assert not (tmp_path / "uit" / "x.json").exists()
    assert _schrijf(flows_exporteren, {"x.json": _rondgang()}, tmp_path / "uit", monkeypatch) == 0
    assert (tmp_path / "uit" / "x.json").exists()


def test_schrijven_weigert_waarden_die_op_een_sleutel_lijken(flows_exporteren, tmp_path: Path,
                                                             monkeypatch) -> None:
    sleutels = ["sk-ant-" + "a" * 30, "ak_" + "b" * 30, "ghp_" + "c" * 30,
                "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.x",
                "-----BEGIN RSA PRIVATE KEY-----"]
    for sleutel in sleutels:
        flow = _rondgang()
        flow["description"] = f"zie {sleutel}"
        assert flows_exporteren.geheimen(flow) == ["/description"], sleutel
        assert _schrijf(flows_exporteren, {"x.json": flow}, tmp_path / "uit", monkeypatch) == 1
        assert not (tmp_path / "uit").exists()
    # Een ingevuld wachtwoordveld, tenzij het de naam van een globale variabele is.
    flow = _rondgang()
    veld = {"password": True, "value": "geheim", "load_from_db": False}
    flow["data"]["nodes"][0]["data"]["node"]["template"]["api_key"] = veld
    assert flows_exporteren.geheimen(flow)
    veld["load_from_db"] = True
    assert flows_exporteren.geheimen(flow) == []
    assert flows_exporteren.geheimen(_rondgang()) == []
