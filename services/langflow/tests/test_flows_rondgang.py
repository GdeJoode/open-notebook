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
            "from langflow_api import LangflowApi\n"
            f"print(sorted(t['state'] for t in LangflowApi().triggers({flow['id']!r})))")
        assert staat.returncode == 0, staat.stderr
        assert staat.stdout.strip().splitlines()[-1] == "['active']"

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
