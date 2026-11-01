"""Stopconditie F2-10: de Compose-basis van het signaalsysteem.

Toetst ``docker-compose.signaalsysteem.yml`` zoals Compose hem oplost
(``docker compose config --format json``), met tijdelijke mappen voor de
hostpaden uit ``.env``. Zonder Docker wordt de test overgeslagen.

Plan van aanpak F2-10; ADR 0002 F, 0010, 0016, 0020, 0021, 0022 A.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
COMPOSE = ROOT / "docker-compose.signaalsysteem.yml"

PROXY_NETWERK = {  # netwerk: (proxy, doeldienst)
    "werkbak-intern": ("werkbak-toegang", "open-notebook"),
    "editor-intern": ("editor-toegang", "langflow"),
    "mcp-intern": ("mcp-toegang", "open-notebook-mcp"),
}
PROXIES = {p for p, _ in PROXY_NETWERK.values()}
GEWOON = {"toegang", "uitgaand"}
MET_UITGAAND = {"ophalen", "cloud-gateway"}
SURREAL_VERSIE = "surrealdb/surrealdb:v2.7.0@sha256:"  # F2-26, ADR 0021


def _padlijsten():
    pad = ROOT / "scripts" / "signaalsysteem" / "controle" / "padlijsten.py"
    spec = importlib.util.spec_from_file_location("padlijsten", pad)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def hostpaden(tmp_path_factory) -> dict[str, str]:
    basis = tmp_path_factory.mktemp("signaal")
    paden = {}
    for naam in ("SIGNAAL_INNAMEMAP", "SIGNAAL_WATCHFOLDER", "SIGNAAL_VAULT", "SIGNAAL_EXPORTMAP"):
        map_ = basis / naam.lower()
        map_.mkdir()
        paden[naam] = str(map_)
    return paden


def _compose(hostpaden, *args) -> subprocess.CompletedProcess:
    if shutil.which("docker") is None:
        pytest.skip("docker niet beschikbaar")
    env = {**os.environ, **hostpaden}
    return subprocess.run(["docker", "compose", "-f", str(COMPOSE), *args],
                          cwd=ROOT, env=env, capture_output=True, text=True)


@pytest.fixture(scope="module")
def config(hostpaden) -> dict:
    uit = _compose(hostpaden, "config", "--format", "json")
    assert uit.returncode == 0, uit.stderr
    return json.loads(uit.stdout)


def _netwerken(dienst: dict) -> set[str]:
    return set((dienst.get("networks") or {}).keys())


def _leden(config: dict, netwerk: str) -> set[str]:
    return {n for n, d in config["services"].items() if netwerk in _netwerken(d)}


def _dockerfile(dienst: dict) -> Path:
    build = dienst["build"]
    return Path(build["context"]) / build.get("dockerfile", "Dockerfile")


# --------------------------------------------------------------------------
# Opbouw en versies
# --------------------------------------------------------------------------

def test_config_quiet_slaagt(hostpaden) -> None:
    uit = _compose(hostpaden, "config", "--quiet")
    assert uit.returncode == 0, uit.stderr


def test_config_faalt_zonder_innamemappen(hostpaden) -> None:
    """Geen stille koppeling van een niet-bestaande map: de hostpaden zijn verplicht."""
    zonder = {k: v for k, v in hostpaden.items() if k != "SIGNAAL_WATCHFOLDER"}
    env = {k: v for k, v in os.environ.items() if k != "SIGNAAL_WATCHFOLDER"}
    if shutil.which("docker") is None:
        pytest.skip("docker niet beschikbaar")
    uit = subprocess.run(["docker", "compose", "-f", str(COMPOSE), "--env-file", os.devnull,
                          "config", "--quiet"], cwd=ROOT, env={**env, **zonder},
                         capture_output=True, text=True)
    assert uit.returncode != 0
    assert "SIGNAAL_WATCHFOLDER" in uit.stderr


def test_elk_image_heeft_een_vaste_versie(config) -> None:
    for naam, dienst in config["services"].items():
        assert dienst.get("pull_policy") != "always", naam
        if "build" in dienst:
            vanaf = re.findall(r"^FROM\s+(\S+)", _dockerfile(dienst).read_text(encoding="utf-8"),
                               re.M)
            assert vanaf, naam
            for image in vanaf:
                assert "@sha256:" in image and ":latest" not in image, (naam, image)
            for bron in re.findall(r"COPY --from=(\S+)", _dockerfile(dienst).read_text(
                    encoding="utf-8")):
                if "/" in bron or ":" in bron:
                    assert "@sha256:" in bron, (naam, bron)
        else:
            assert "@sha256:" in dienst["image"], naam
    assert "pull_policy: always" not in COMPOSE.read_text(encoding="utf-8")


def test_surrealdb_versie_en_eigen_volume(config) -> None:
    surreal = config["services"]["surrealdb"]
    vanaf = _dockerfile(surreal).read_text(encoding="utf-8")
    assert re.search(r"^FROM " + re.escape(SURREAL_VERSIE), vanaf, re.M)
    [koppeling] = surreal["volumes"]
    assert koppeling["type"] == "volume" and koppeling["source"] == "signaal_surreal_data"
    assert config["volumes"]["signaal_surreal_data"]["name"] == "signaal_surreal_data"
    archief = (ROOT / "surreal_data").resolve()
    for naam, dienst in config["services"].items():
        for kop in dienst.get("volumes") or []:
            if kop["type"] == "bind":
                assert Path(kop["source"]).resolve() != archief, naam
                assert "surreal_data" not in Path(kop["source"]).name, naam
    assert "./surreal_data" not in _zonder_commentaar()


def _zonder_commentaar() -> str:
    return "\n".join(r.split("#", 1)[0] for r in COMPOSE.read_text(encoding="utf-8").splitlines())


def _ruwe_netwerken() -> dict:
    """Op het ruwe bestand: ``config`` laat netwerken zonder leden weg."""
    return yaml.safe_load(COMPOSE.read_text(encoding="utf-8"))["networks"]


# --------------------------------------------------------------------------
# Netwerken en poorten
# --------------------------------------------------------------------------

def test_netwerken_intern_en_gewoon() -> None:
    netwerken = _ruwe_netwerken()
    for naam, netwerk in netwerken.items():
        netwerk = netwerk or {}
        if naam in GEWOON:
            assert not netwerk.get("internal"), naam
        else:
            assert netwerk.get("internal") is True, naam
    assert set(netwerken) >= {
        "opslag", "verwerking", "beheer", "werkbak-intern", "editor-intern", "mcp-intern",
        "ophalen-intern", "gateway-intern", "cli-intern", "toegang", "uitgaand"}


def test_surrealdb_alleen_op_opslag(config) -> None:
    assert _netwerken(config["services"]["surrealdb"]) == {"opslag"}
    assert _leden(config, "opslag") == {"surrealdb", "open-notebook"}


def test_proxynetwerk_alleen_proxy_en_doeldienst(config) -> None:
    for netwerk, (proxy, doel) in PROXY_NETWERK.items():
        leden = _leden(config, netwerk)
        assert leden <= {proxy, doel}, (netwerk, leden)
        if leden:
            assert leden == {proxy, doel}, (netwerk, leden)
    assert _leden(config, "werkbak-intern") == {"werkbak-toegang", "open-notebook"}


def test_alleen_proxys_publiceren_en_alleen_op_localhost(config) -> None:
    for naam, dienst in config["services"].items():
        poorten = dienst.get("ports") or []
        if naam not in PROXIES:
            assert not poorten, naam
        for poort in poorten:
            assert poort.get("host_ip") == "127.0.0.1", (naam, poort)
    assert config["services"]["werkbak-toegang"]["ports"]


def test_toegang_en_uitgaand(config) -> None:
    assert _leden(config, "toegang") <= PROXIES
    assert _leden(config, "uitgaand") <= MET_UITGAAND


def test_geen_dienst_op_intern_en_gewoon_behalve_proxys(config) -> None:
    for naam, dienst in config["services"].items():
        netten = _netwerken(dienst)
        if netten & GEWOON and netten - GEWOON:
            assert naam in PROXIES | MET_UITGAAND, (naam, netten)


def test_proxyimage_zonder_shell() -> None:
    """Uitkomst van de proef masquerade (F2-10): niet geslaagd onder Docker Desktop,
    dus een geharde proxy: geen shell en geen pakketbeheer in het image."""
    tekst = (ROOT / "services" / "proxy" / "Dockerfile").read_text(encoding="utf-8")
    assert "rm -f /bin/busybox" in tekst
    assert "/sbin/apk" in tekst and "/docker-entrypoint.sh" in tekst
    assert re.search(r'^ENTRYPOINT \["/usr/sbin/nginx"', tekst, re.M)
    assert re.search(r"^USER 101$", tekst, re.M)


def test_proxy_is_gehard(config) -> None:
    for naam in PROXIES & set(config["services"]):
        dienst = config["services"][naam]
        assert dienst.get("read_only") is True, naam
        assert dienst.get("cap_drop") == ["ALL"], naam
        assert "no-new-privileges:true" in (dienst.get("security_opt") or []), naam
        assert str(dienst.get("user", "")).split(":")[0] not in ("", "0", "root"), naam
        assert not dienst.get("secrets"), naam
        assert all(k["type"] == "tmpfs" for k in dienst.get("volumes") or []), naam


# --------------------------------------------------------------------------
# Geheimen
# --------------------------------------------------------------------------

_GEHEIM = re.compile(
    r"(?i)(pass(word)?|passwd|secret|token|api[_-]?key|sleutel)[\"']?\s*[:=]\s*[\"']?"
    r"(?!\$\{)(?!/run/secrets)(?!\./secrets/)[^\s\"'#,}]+")


def test_geen_letterlijke_geheimen_in_compose() -> None:
    tekst = COMPOSE.read_text(encoding="utf-8")
    for regel in tekst.splitlines():
        code = regel.split("#", 1)[0]
        assert not _GEHEIM.search(code), regel
        assert "--pass" not in code and "--user" not in code, regel
        # Lange hexreeksen alleen als digest.
        for treffer in re.finditer(r"[0-9a-fA-F]{32,}", code):
            assert code[: treffer.start()].endswith("sha256:"), regel
    assert "root:root" not in tekst and "extra_hosts" not in tekst


def test_detector_vindt_een_geplant_geheim() -> None:
    for regel in ('      SURREAL_PASS: geheim123', '    command: start --pass x',
                  '      API_KEY=abc'):
        code = regel.split("#", 1)[0]
        assert _GEHEIM.search(code) or "--pass" in code, regel


def test_geheimen_alleen_uit_bestanden_in_secrets(config) -> None:
    for naam, geheim in (config.get("secrets") or {}).items():
        bestand = Path(geheim["file"])
        assert bestand.parent.name == "secrets", (naam, bestand)
        assert "environment" not in geheim, naam
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert re.search(r"^secrets/\s*$", gitignore, re.M)
    assert re.search(r"^secrets/\s*$", (ROOT / ".dockerignore").read_text(encoding="utf-8"),
                     re.M)


# --------------------------------------------------------------------------
# Archieven en padlijsten
# --------------------------------------------------------------------------

def _koppeling(dienst: dict, bron: str) -> list[dict]:
    return [k for k in dienst.get("volumes") or [] if k.get("source") == bron]


def test_archieven(config) -> None:
    fork = config["services"]["open-notebook"]
    env = fork["environment"]
    [bron] = _koppeling(fork, "bronarchief")
    assert bron["type"] == "volume" and bron.get("read_only") is True
    assert bron["target"] == env["SIGNAAL_BRONARCHIEF"]
    [eigen] = _koppeling(fork, "fork_archief")
    assert not eigen.get("read_only")
    assert eigen["target"] == env["SIGNAAL_FORK_ARCHIEF"]
    for naam, dienst in config["services"].items():
        if naam != "open-notebook":
            assert not _koppeling(dienst, "fork_archief"), naam
        if naam == "langflow":
            [lf] = _koppeling(dienst, "bronarchief")
            assert not lf.get("read_only")
        elif naam != "open-notebook":
            assert not _koppeling(dienst, "bronarchief"), naam


def test_innamemappen_en_watchfolder_alleen_lezen(config, hostpaden) -> None:
    fork = config["services"]["open-notebook"]
    lijst = fork["environment"]["SIGNAAL_INNAME_PADEN"].split(";")
    binds = {k["target"]: k for k in fork["volumes"] if k["type"] == "bind"}
    for doel, hostnaam in (("/inname", "SIGNAAL_INNAMEMAP"), ("/watchfolder", "SIGNAAL_WATCHFOLDER")):
        assert doel in lijst
        assert binds[doel].get("read_only") is True
        assert Path(binds[doel]["source"]) == Path(hostpaden[hostnaam])


def test_flows_en_git_alleen_lezen(config) -> None:
    fork = config["services"]["open-notebook"]
    binds = {k["target"]: k for k in fork["volumes"] if k["type"] == "bind"}
    flows = fork["environment"]["SIGNAAL_FLOWS_DIR"]
    assert binds[flows].get("read_only") is True
    assert Path(binds[flows]["source"]).resolve() == (ROOT / "services/langflow/flows").resolve()
    assert binds["/app/.git"].get("read_only") is True
    assert binds["/app/config"].get("read_only") is True


def test_padlijsten_kloppen_met_de_koppelingen(config, hostpaden) -> None:
    assert _padlijsten().controleer(config, hostpaden) == []


def _met(config: dict, naam: str, koppelingen: list[dict]) -> dict:
    uit = copy.deepcopy(config)
    uit["services"].setdefault(naam, {"networks": {"verwerking": None}, "volumes": []})
    uit["services"][naam].setdefault("volumes", []).extend(koppelingen)
    return uit


def test_startcontrole_faalt_bij_vault_buiten_de_lijst(config, hostpaden) -> None:
    controleer = _padlijsten().controleer
    vault = hostpaden["SIGNAAL_VAULT"]
    entiteiten = {"type": "bind", "source": f"{vault}/20 Atlas/28 Entities",
                  "target": "/vault/entities"}
    fouten = controleer(_met(config, "langflow", [entiteiten]), hostpaden)
    assert any("SIGNAAL_VAULT_PADEN" in f for f in fouten), fouten
    # Ook op een ander koppelpunt herkend aan de bron op de host.
    anders = dict(entiteiten, target="/data/notities")
    assert controleer(_met(config, "langflow", [anders]), hostpaden)
    # Staat het koppelpunt in de lijst, dan slaagt de controle.
    goed = copy.deepcopy(_met(config, "langflow", [entiteiten]))
    goed["services"]["open-notebook"]["environment"]["SIGNAAL_VAULT_PADEN"] = "/vault;/vault/entities"
    assert controleer(goed, hostpaden) == []


def test_startcontrole_faalt_bij_fout_in_inname_of_export(config, hostpaden) -> None:
    controleer = _padlijsten().controleer
    gevallen = [
        ("langflow", {"type": "bind", "source": hostpaden["SIGNAAL_INNAMEMAP"],
                      "target": "/inname"}),                          # niet alleen-lezen
        ("langflow", {"type": "bind", "source": hostpaden["SIGNAAL_WATCHFOLDER"],
                      "target": "/wf", "read_only": True}),           # niet in de lijst
        ("langflow", {"type": "bind", "source": hostpaden["SIGNAAL_EXPORTMAP"],
                      "target": "/uit"}),                             # export niet in de lijst
        ("langflow", {"type": "bind", "source": str(Path(hostpaden["SIGNAAL_VAULT"]).parent),
                      "target": "/alles", "read_only": True}),        # map boven de vault
        ("open-notebook", {"type": "bind", "source": hostpaden["SIGNAAL_VAULT"],
                           "target": "/vault", "read_only": True}),   # fork koppelt de vault
    ]
    for naam, koppeling in gevallen:
        assert controleer(_met(config, naam, [koppeling]), hostpaden), (naam, koppeling)
    # Een innamemap uit de lijst die niet alleen-lezen aan de fork hangt.
    zonder = copy.deepcopy(config)
    zonder["services"]["open-notebook"]["volumes"] = [
        k for k in zonder["services"]["open-notebook"]["volumes"] if k["target"] != "/watchfolder"]
    assert controleer(zonder, hostpaden)
