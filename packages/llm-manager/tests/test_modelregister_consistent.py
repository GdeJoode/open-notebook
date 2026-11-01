"""Stopconditie F2-11: modelregister en llama-swap-configuratie kloppen met elkaar.

- Elk model met aanroepwijze ``llama-swap`` in ``config/modelregister.yaml`` staat
  in ``services/llama-swap/config.yaml``, en omgekeerd.
- Uit de review van F2-07: elk model met ``toegestaan_intern: ja`` wijst in de
  configuratie naar een lokaal bestand (llama-server met een bestand uit
  ``/modellen``) of een lokale container op het netwerk ``verwerking``, zonder
  ``proxy`` naar een extern adres. De configuratie kent geen ``peers``.
- De versie in het register past bij wat llama-swap draait: revisie en bestand
  uit het downloadscript, contextlengte, en het image van docling-serve.

Leest alleen bestanden uit de repository; geen Docker nodig.
"""

from __future__ import annotations

import importlib.util
import re
import shlex
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml
from llm_manager.register import STANDAARD_PAD, lees_register

ROOT = STANDAARD_PAD.parents[1]
CONFIG = ROOT / "services" / "llama-swap" / "config.yaml"
COMPOSE = ROOT / "docker-compose.signaalsysteem.yml"
DOWNLOAD = ROOT / "scripts" / "signaalsysteem" / "modellen_downloaden.py"

LLAMA_SERVER = "/app/llama-server"
WACHT = "/app/wacht_op_dienst.sh"
MODELMAP = "/modellen/"
#: Opties van llama-server die een model van buiten halen.
OPHALEN = {"-hf", "--hf-repo", "-hff", "--hf-file", "-mu", "--model-url", "-hfr", "--hf-repo-v",
           "-dr", "--docker-repo"}
LOKAAL_NETWERK = "verwerking"


def _yaml(pad: Path) -> dict:
    return yaml.safe_load(pad.read_text(encoding="utf-8"))


def _manifest() -> list[dict]:
    spec = importlib.util.spec_from_file_location("modellen_downloaden", DOWNLOAD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GGUF


def _containers(compose: dict) -> dict[str, dict]:
    """Containernaam (en dienstnaam) naar dienst."""
    uit = {}
    for naam, dienst in (compose.get("services") or {}).items():
        uit[naam] = dienst
        if dienst.get("container_name"):
            uit[dienst["container_name"]] = dienst
    return uit


def _netwerken(dienst: dict) -> set[str]:
    netten = dienst.get("networks") or []
    return set(netten if isinstance(netten, list) else netten.keys())


def fouten_lokaal(naam: str, model: dict, compose: dict, bestanden: set[str]) -> list[str]:
    """Waarom een model in de llama-swap-configuratie niet lokaal is, of []."""
    fouten: list[str] = []
    cmd = shlex.split(str(model.get("cmd", "")))
    proxy = model.get("proxy")
    interne_netten = {n for n, w in (compose.get("networks") or {}).items()
                      if (w or {}).get("internal") is True}
    if proxy is None:
        if cmd[:1] != [LLAMA_SERVER]:
            fouten.append(f"{naam}: zonder proxy alleen {LLAMA_SERVER}")
        if "--host" not in cmd or cmd[cmd.index("--host") + 1] != "127.0.0.1":
            fouten.append(f"{naam}: llama-server luistert niet alleen op 127.0.0.1")
        if set(cmd) & OPHALEN or any("://" in d for d in cmd):
            fouten.append(f"{naam}: llama-server haalt een model van buiten")
        if "-m" not in cmd:
            fouten.append(f"{naam}: geen modelbestand (-m)")
        else:
            bestand = cmd[cmd.index("-m") + 1]
            if not bestand.startswith(MODELMAP) or bestand[len(MODELMAP):] not in bestanden:
                fouten.append(f"{naam}: {bestand} staat niet in het downloadscript onder {MODELMAP}")
        return fouten
    delen = urlsplit(str(proxy))
    host = delen.hostname or ""
    dienst = _containers(compose).get(host)
    if delen.scheme != "http" or dienst is None:
        fouten.append(f"{naam}: proxy {proxy} is geen container uit de Compose-opzet")
        return fouten
    netten = _netwerken(dienst)
    if LOKAAL_NETWERK not in netten or not netten <= interne_netten:
        fouten.append(f"{naam}: {host} hangt niet alleen aan interne netwerken met {LOKAAL_NETWERK}")
    if cmd[:3] != ["/bin/sh", WACHT, host] or not cmd[3].startswith(f"http://{host}:"):
        fouten.append(f"{naam}: cmd start niet de container {host} via {WACHT}")
    if model.get("cmdStop") != f"docker stop {host}":
        fouten.append(f"{naam}: cmdStop stopt niet de container {host}")
    if str(dienst.get("restart", "no")) != "no" or "door-llama-swap" not in (dienst.get("profiles") or []):
        fouten.append(f"{naam}: {host} wordt niet alleen door llama-swap gestart")
    return fouten


@pytest.fixture(scope="module")
def register():
    return lees_register(STANDAARD_PAD)


@pytest.fixture(scope="module")
def config() -> dict:
    return _yaml(CONFIG)


@pytest.fixture(scope="module")
def compose() -> dict:
    return _yaml(COMPOSE)


def test_register_en_configuratie_hebben_dezelfde_modellen(register, config) -> None:
    in_register = {m.naam for m in register.modellen if m.aanroepwijze == "llama-swap"}
    assert in_register == set(config["models"])


def test_geen_extern_adres_in_de_configuratie(config) -> None:
    assert not config.get("peers")
    assert not config.get("tailcat")
    for naam, model in config["models"].items():
        assert not model.get("aliases"), naam


def test_intern_toegestaan_wijst_naar_iets_lokaals(register, config, compose) -> None:
    bestanden = {item["bestand"] for item in _manifest()}
    intern = [m.naam for m in register.modellen
              if m.aanroepwijze == "llama-swap" and m.intern_toegestaan]
    assert intern
    fouten = []
    for naam in intern:
        fouten += fouten_lokaal(naam, config["models"][naam], compose, bestanden)
    assert not fouten, "\n".join(fouten)


def test_groepen_volgens_plan(config) -> None:
    groepen = config["groups"]
    alle = [m for g in groepen.values() for m in g["members"]]
    assert sorted(alle) == sorted(config["models"]), "elk model in precies één groep"
    zwaar, licht = groepen["zwaar"], groepen["licht"]
    assert zwaar["swap"] is True and zwaar["exclusive"] is True
    assert set(zwaar["members"]) == {"werkmodel", "docling"}
    assert licht == {"swap": False, "exclusive": False, "persistent": True, "members": ["embed"]}
    assert "ttl" not in config["models"]["embed"]
    assert config["models"]["docling"]["ttl"] == 300
    assert config["models"]["docling"]["checkEndpoint"] == "/health"


def _gguf_in_versie(versie: str, manifest: list[dict]) -> dict:
    """De versie begint met ``<repo>@<revisie>:`` uit het downloadscript."""
    for item in manifest:
        if versie.startswith(f"{item['repo']}@{item['revisie']}:"):
            return item
    raise AssertionError(f"versie {versie} niet in het downloadscript")


def test_versie_in_register_past_bij_wat_draait(register, config, compose) -> None:
    manifest = _manifest()
    for naam in ("werkmodel", "embed"):
        model = register.zoek(naam)
        item = _gguf_in_versie(model.versie, manifest)
        cmd = shlex.split(config["models"][naam]["cmd"])
        assert cmd[cmd.index("-m") + 1] == MODELMAP + item["bestand"], naam
        ctx = re.search(r":ctx-(\d+)$", model.versie)
        if ctx:
            assert cmd[cmd.index("-c") + 1] == ctx.group(1), naam
    docling = register.zoek("docling")
    image = compose["services"]["docling-serve"]["image"]
    assert image.startswith(docling.versie + "@sha256:")


# --------------------------------------------------------------------------
# De controle vindt wat ze zoekt
# --------------------------------------------------------------------------

COMPOSE_KLEIN = {
    "networks": {"verwerking": {"internal": True}, "uitgaand": {}},
    "services": {
        "dienst": {"container_name": "signaal-dienst", "networks": ["verwerking"],
                   "profiles": ["door-llama-swap"], "restart": "no"},
        "buiten": {"networks": ["verwerking", "uitgaand"], "profiles": ["door-llama-swap"]},
    },
}
GOED_CONTAINER = {"cmd": "/bin/sh /app/wacht_op_dienst.sh signaal-dienst http://signaal-dienst:1/h",
                  "cmdStop": "docker stop signaal-dienst", "proxy": "http://signaal-dienst:1"}
GOED_SERVER = {"cmd": "/app/llama-server --host 127.0.0.1 --port ${PORT} -m /modellen/a.gguf"}


@pytest.mark.parametrize("model", [GOED_CONTAINER, GOED_SERVER])
def test_lokaal_model_wordt_aanvaard(model) -> None:
    assert fouten_lokaal("m", model, COMPOSE_KLEIN, {"a.gguf"}) == []


@pytest.mark.parametrize("model", [
    dict(GOED_CONTAINER, proxy="https://api.example.org/v1"),           # extern adres
    dict(GOED_CONTAINER, proxy="http://10.0.0.5:8080"),                 # IP-adres
    dict(GOED_CONTAINER, proxy="http://buiten:1"),                      # container met internet
    dict(GOED_CONTAINER, cmdStop="docker stop iets-anders"),
    {"cmd": "/app/llama-server --host 127.0.0.1 --port ${PORT} -hf org/model"},
    {"cmd": "/app/llama-server --host 127.0.0.1 -m /modellen/a.gguf --model-url https://x/y"},
    {"cmd": "/app/llama-server --host 0.0.0.0 --port ${PORT} -m /modellen/a.gguf"},
    {"cmd": "/app/llama-server --host 127.0.0.1 --port ${PORT} -m /elders/a.gguf"},
    {"cmd": "/app/llama-server --host 127.0.0.1 --port ${PORT} -m /modellen/onbekend.gguf"},
    {"cmd": "curl http://ergens"},
])
def test_niet_lokaal_model_wordt_geweigerd(model) -> None:
    assert fouten_lokaal("m", model, COMPOSE_KLEIN, {"a.gguf"})
