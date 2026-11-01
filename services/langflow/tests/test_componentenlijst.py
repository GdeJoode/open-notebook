"""Stopconditie F2-12: de componentenlijst (ADR 0022 D).

* Een flow met een type dat niet op ``config/langflow-componenten.yaml`` staat,
  wordt niet geladen en het script eindigt met een foutcode; ook live, in de
  container langflow: de flow staat daarna niet in Langflow.
* Elke flow in ``services/langflow/flows/`` gebruikt alleen typen van de lijst.
* Elke eigen component staat op de lijst, en de lijst noemt geen component met
  een netwerk- of codefunctie.
"""

from __future__ import annotations

import ast
import copy
import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

LANGFLOW = Path(__file__).resolve().parents[1]
ROOT = LANGFLOW.parents[1]
SCRIPTS = LANGFLOW / "scripts"
FIXTURES = Path(__file__).resolve().parent / "fixtures"
LIJST = ROOT / "config" / "langflow-componenten.yaml"
FLOWS = LANGFLOW / "flows"
COMPONENTEN = LANGFLOW / "components" / "signaalsysteem"

#: Typen met een netwerk- of codefunctie die nooit op de lijst mogen (ADR 0022 D).
NOOIT = {
    "APIRequest", "URLComponent", "URL", "WebSearchNoAPI", "WebSearch", "SearchAPI",
    "PythonREPLComponent", "PythonREPLTool", "PythonCodeStructuredTool", "PythonFunction",
    "CustomComponent", "MCPTools", "MCPToolsComponent", "Agent", "LanguageModelComponent",
    "OpenAIModel", "AnthropicModel", "OllamaModel", "LiteLLMProxy", "Webhook",
    "InboundWebhookTrigger", "Directory", "File", "SaveToFile",
}


def _rondgang() -> dict:
    return json.loads((FIXTURES / "rondgang.json").read_text(encoding="utf-8"))


def _met_type(flow: dict, soort: str) -> dict:
    flow = copy.deepcopy(flow)
    knoop = copy.deepcopy(flow["data"]["nodes"][0])
    knoop["id"] = knoop["data"]["id"] = f"{soort}-test"
    knoop["data"]["type"] = soort
    flow["data"]["nodes"].append(knoop)
    return flow


# --------------------------------------------------------------------------- lijst

def test_lijst_is_geldig_en_klein(flows_laden) -> None:
    typen = flows_laden.lees_lijst(LIJST)
    assert {"ScheduleTrigger", "TextInput", "TextOutput", "note"} <= typen
    assert not typen & NOOIT
    # Geen type van een aanbieder of een extensiebundel, behalve de eigen bundel.
    for soort in typen:
        assert not soort.startswith("ext:") or soort.startswith("ext:signaalsysteem:"), soort


def test_eigen_componenten_staan_op_de_lijst(flows_laden) -> None:
    namen = set()
    for pad in COMPONENTEN.glob("*.py"):
        for knoop in ast.walk(ast.parse(pad.read_text(encoding="utf-8"))):
            if isinstance(knoop, ast.ClassDef):
                for regel in knoop.body:
                    if (isinstance(regel, ast.Assign) and any(
                            isinstance(t, ast.Name) and t.id == "name" for t in regel.targets)
                            and isinstance(regel.value, ast.Constant)):
                        namen.add(regel.value.value)
    typen = flows_laden.lees_lijst(LIJST)
    eigen = {t for t in typen if t.startswith("ext:signaalsysteem:")}
    assert eigen == {f"ext:signaalsysteem:{n}@extra" for n in namen}


@pytest.mark.parametrize("fout", [
    "versie: 1\nkern: []\nnotities: []\n",                                   # eigen ontbreekt
    "versie: 1\nkern: []\nnotities: []\neigen: []\nextra: []\n",            # onbekende sleutel
    "versie: 1\nkern: [TextInput]\nnotities: []\neigen: []\n",              # zonder reden
    "versie: 1\nkern: []\nnotities: []\neigen: [{type: Foo, reden: x}]\n",  # verkeerde vorm
    "versie: 1\nkern: [{type: A, reden: x}, {type: A, reden: y}]\nnotities: []\neigen: []\n",
])
def test_ongeldige_lijst_geeft_exitcode_2(flows_laden, tmp_path: Path, fout: str) -> None:
    lijst = tmp_path / "lijst.yaml"
    lijst.write_text(fout, encoding="utf-8")
    with pytest.raises(flows_laden.LijstFout):
        flows_laden.lees_lijst(lijst)
    (tmp_path / "flows").mkdir()
    assert flows_laden.main(["--map", str(tmp_path / "flows"), "--lijst", str(lijst)]) == 2


# --------------------------------------------------------------------------- flows

def test_elke_flow_in_git_gebruikt_alleen_typen_van_de_lijst(flows_laden) -> None:
    flows = flows_laden.lees_flows(FLOWS)
    assert flows_laden.toets(flows, flows_laden.lees_lijst(LIJST)) == []


def test_fixture_van_de_rondgang_staat_op_de_lijst(flows_laden) -> None:
    flow = _rondgang()
    assert flows_laden.toets({"rondgang.json": flow}, flows_laden.lees_lijst(LIJST)) == []
    assert "ScheduleTrigger" in flows_laden.knooppunttypen(flow)


@pytest.mark.parametrize("soort", ["APIRequest", "PythonREPLComponent", "MCPTools",
                                   "ext:openai:OpenAIModelComponent@official", "textinput"])
def test_flow_met_type_buiten_de_lijst_wordt_geweigerd(flows_laden, soort: str) -> None:
    weigeringen = flows_laden.toets({"x.json": _met_type(_rondgang(), soort)},
                                    flows_laden.lees_lijst(LIJST))
    assert ("x.json", f"type niet op de lijst: {soort}") in weigeringen


def test_type_in_een_groep_telt_mee(flows_laden) -> None:
    flow = _rondgang()
    binnen = _met_type(_rondgang(), "APIRequest")
    groep = copy.deepcopy(flow["data"]["nodes"][0])
    groep["data"]["type"] = "TextInput"
    groep["data"]["node"]["flow"] = {"data": binnen["data"]}
    flow["data"]["nodes"].append(groep)
    weigeringen = flows_laden.toets({"x.json": flow}, flows_laden.lees_lijst(LIJST))
    assert ("x.json", "type niet op de lijst: APIRequest") in weigeringen


@pytest.mark.parametrize("wijziging,reden", [
    (lambda f: f["data"]["nodes"][0]["data"].pop("type"), "type niet op de lijst: <geen type>"),
    (lambda f: f.pop("id"), "flow zonder vaste id"),
    (lambda f: f["data"].pop("nodes"), "flow zonder lijst nodes"),
])
def test_onvolledige_flow_wordt_geweigerd(flows_laden, wijziging, reden: str) -> None:
    flow = _rondgang()
    wijziging(flow)
    assert ("x.json", reden) in flows_laden.toets({"x.json": flow}, flows_laden.lees_lijst(LIJST))


def test_dubbele_id_wordt_geweigerd(flows_laden) -> None:
    weigeringen = flows_laden.toets({"a.json": _rondgang(), "b.json": _rondgang()},
                                    flows_laden.lees_lijst(LIJST))
    assert ("b.json", "id ook in a.json") in weigeringen


def test_script_laadt_niets_en_eindigt_met_foutcode(flows_laden, tmp_path: Path,
                                                     monkeypatch) -> None:
    """Eén verboden flow: geen enkele flow geladen, geen aanroep van Langflow, exitcode 1."""
    map_ = tmp_path / "flows"
    map_.mkdir()
    (map_ / "goed.json").write_text(json.dumps(_rondgang()), encoding="utf-8")
    slecht = _met_type(_rondgang(), "APIRequest")
    slecht["id"] = str(uuid.uuid4())
    (map_ / "slecht.json").write_text(json.dumps(slecht), encoding="utf-8")

    import langflow_api

    class GeenLangflow:
        def __init__(self, *a, **k):
            raise AssertionError("Langflow aangeroepen na een weigering")

    monkeypatch.setattr(langflow_api, "LangflowApi", GeenLangflow)
    assert flows_laden.main(["--map", str(map_), "--lijst", str(LIJST)]) == 1

    uit = subprocess.run([sys.executable, str(SCRIPTS / "flows_laden.py"), "--map", str(map_),
                          "--lijst", str(LIJST), "--alleen-toetsen"], capture_output=True, text=True)
    assert uit.returncode == 1
    assert "slecht.json: type niet op de lijst: APIRequest" in uit.stderr
    assert "geen enkele flow geladen" in uit.stderr


# --------------------------------------------------------------------------- live

def test_live_verboden_flow_niet_geladen(live) -> None:
    """In de container: het laadscript weigert, de flow staat daarna niet in Langflow."""
    slecht = _met_type(_rondgang(), "APIRequest")
    slecht["id"] = str(uuid.uuid4())
    live.zet_map("/tmp/f212-verboden", {"verboden.json": slecht})
    uit = live.exec("python", "/app/signaal/flows_laden.py", "--map", "/tmp/f212-verboden")
    assert uit.returncode == 1, uit.stdout + uit.stderr
    assert "verboden.json: type niet op de lijst: APIRequest" in uit.stderr
    bestaat = live.python(f"from langflow_api import LangflowApi; print(LangflowApi().bestaat({slecht['id']!r}))")
    assert bestaat.returncode == 0, bestaat.stderr
    assert bestaat.stdout.strip().splitlines()[-1] == "False"
    live.exec("rm", "-rf", "/tmp/f212-verboden")


@pytest.mark.parametrize("waarde", ["PUBLIC", "public", None, ""])
def test_flow_die_niet_prive_is_wordt_geweigerd(flows_laden, waarde) -> None:
    """Herreview F2-12, punt 1: alleen access_type PRIVATE (of afwezig, dan privé)."""
    flow = _rondgang()
    flow["access_type"] = waarde
    weigeringen = flows_laden.toets({"x.json": flow}, flows_laden.lees_lijst(LIJST))
    assert ("x.json", "access_type is niet PRIVATE") in weigeringen
    flow["access_type"] = "PRIVATE"
    assert flows_laden.toets({"x.json": flow}, flows_laden.lees_lijst(LIJST)) == []


def test_live_langflow_weigert_zelf_een_geblokkeerde_component(live) -> None:
    """Catalogusbeleid (review F2-12, punt 2): ook een upload buiten het laadscript om
    met een component buiten de lijst weigert Langflow zelf."""
    slecht = _met_type(_rondgang(), "APIRequest")
    slecht["id"] = str(uuid.uuid4())
    uit = live.python(
        "import json\n"
        "from langflow_api import LangflowApi, LangflowFout\n"
        "a = LangflowApi()\n"
        f"f = json.loads({json.dumps(json.dumps(slecht))})\n"
        "try:\n"
        "    a.upload(f, 'v.json'); print('geladen'); a.verwijder(f['id'])\n"
        "except LangflowFout as e:\n"
        "    print('geweigerd', e.status)\n")
    assert uit.returncode == 0, uit.stderr
    assert uit.stdout.strip().splitlines()[-1] == "geweigerd 400"
