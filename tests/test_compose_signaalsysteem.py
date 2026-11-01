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
from pathlib import Path, PurePosixPath

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
    for naam in ("SIGNAAL_INNAMEMAP", "SIGNAAL_WATCHFOLDER", "SIGNAAL_VAULT", "SIGNAAL_EXPORTMAP",
                 "SIGNAAL_FLOWS_KLOON"):
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


def test_flows_uit_een_kloon_en_geen_git_van_de_host(config, hostpaden) -> None:
    """K1: alleen een minimale kloon van de flowmap, alleen-lezen; nooit .git."""
    fork = config["services"]["open-notebook"]
    binds = {k["target"]: k for k in fork["volumes"] if k["type"] == "bind"}
    assert fork["environment"]["SIGNAAL_FLOWS_DIR"].startswith("/app/flows-git/")
    assert binds["/app/flows-git"].get("read_only") is True
    assert Path(binds["/app/flows-git"]["source"]) == Path(hostpaden["SIGNAAL_FLOWS_KLOON"])
    assert binds["/app/config"].get("read_only") is True
    for naam, dienst in config["services"].items():
        for kop in dienst.get("volumes") or []:
            assert not str(kop.get("source", "")).rstrip("/\\").endswith(".git"), naam
            assert ".git" not in PurePosixPath(kop.get("target", "")).parts, naam


def test_commit_van_de_fork_als_build_argument(config) -> None:
    """K2: de build krijgt de commit mee; het image legt hem vast."""
    build = config["services"]["open-notebook"]["build"]
    assert "SIGNAAL_FORK_COMMIT" in (build.get("args") or {})
    tekst = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^ARG SIGNAAL_FORK_COMMIT", tekst, re.M)
    assert re.search(r"^LABEL org.opencontainers.image.revision=\$SIGNAAL_FORK_COMMIT", tekst, re.M)
    assert re.search(r"^ENV SIGNAAL_FORK_COMMIT=\$SIGNAAL_FORK_COMMIT", tekst, re.M)


def test_forkimage_zonder_upgrade_en_met_vaste_node() -> None:
    """K4 en B4: geen apt-get upgrade, Node met versie en SHA-256, tiktoken in het image."""
    tekst = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    code = "\n".join(r for r in tekst.splitlines() if not r.lstrip().startswith("#"))
    assert "apt-get upgrade" not in code and "nodesource" not in code.lower()
    assert re.search(r"^ARG NODE_SHA256=[0-9a-f]{64}$", tekst, re.M)
    assert "sha256sum -c" in tekst
    assert re.search(r"^ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken$", tekst, re.M)


def test_surrealdb_logs_zonder_anoniem_volume(config) -> None:
    surreal = config["services"]["surrealdb"]
    doelen = {k["target"]: k["type"] for k in surreal.get("volumes") or []}
    assert doelen.get("/logs") == "tmpfs" or "/logs" in (surreal.get("tmpfs") or [])


def test_toegang_zonder_icc() -> None:
    """B1: de proxy's op toegang bereiken elkaar niet."""
    opties = _ruwe_netwerken()["toegang"].get("driver_opts") or {}
    assert opties.get("com.docker.network.bridge.enable_icc") == "false"


def test_up_alleen_via_start_sh() -> None:
    kop = COMPOSE.read_text(encoding="utf-8").split("\nname:", 1)[0]
    assert "ALLEEN via" in kop and "start.sh" in kop


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


_VAULT = "D:/# Vault"


def _cfg(bron: str, doel: str = "/inname") -> dict:
    return {"services": {"open-notebook": {
        "environment": {"SIGNAAL_INNAME_PADEN": "/inname;/watchfolder",
                        "SIGNAAL_VAULT_PADEN": "/vault", "SIGNAAL_EXPORTMAP_PADEN": "/export"},
        "volumes": [{"type": "bind", "source": bron, "target": doel, "read_only": True},
                    {"type": "bind", "source": "C:/w", "target": "/watchfolder",
                     "read_only": True}]}}}


@pytest.mark.parametrize("bron,env", [
    (r"D:\# Vault", {"SIGNAAL_VAULT": _VAULT}),
    ("D:\\", {"SIGNAAL_VAULT": _VAULT}),
    (r"D:\# Vault\20 Atlas", {"SIGNAAL_VAULT": _VAULT}),
    (r"D:\# Vault", {}),
    ("/run/desktop/mnt/host/d/# Vault", {"SIGNAAL_VAULT": _VAULT}),
    ("/mnt/d/# Vault", {"SIGNAAL_VAULT": _VAULT}),
    (r"\\?\D:\# Vault", {"SIGNAAL_VAULT": _VAULT}),
    (r"D:\#VAULT~1", {"SIGNAAL_VAULT": _VAULT}),
    (r"D:\tmp\..\# Vault", {"SIGNAAL_VAULT": _VAULT}),
    (r"D:\# Vault ", {"SIGNAAL_VAULT": _VAULT}),
    (r"D:\# Vault.", {"SIGNAAL_VAULT": _VAULT}),
    (r"\\server\share", {"SIGNAAL_VAULT": _VAULT}),
])
def test_varianten_van_de_reviewer_worden_geweigerd(bron, env) -> None:
    """B3: omwegen naar de vault en een ontbrekende SIGNAAL_VAULT (fail-closed)."""
    assert _padlijsten().controleer(_cfg(bron), env)


def test_canoniek_pad_uit_start_sh_telt() -> None:
    """Een junction of subst lost start.sh op (canoniek.ps1); de controle gebruikt die vorm."""
    p = _padlijsten()
    env = {"SIGNAAL_VAULT": _VAULT}
    assert p.controleer(_cfg("C:/inname"), env) == []
    vast = {"C:/w": "C:/w", _VAULT: _VAULT}
    assert p.controleer(_cfg("C:/kop"), env, {**vast, "C:/kop": "D:/# Vault/20 Atlas"})
    assert p.controleer(_cfg("C:/weg"), env, {**vast, "C:/weg": None})


def test_andere_dienst_koppelt_de_vault() -> None:
    cfg = _cfg("C:/inname")
    cfg["services"]["x"] = {"volumes": [{"type": "bind", "source": r"D:\# Vault",
                                         "target": "/data", "read_only": True}]}
    assert _padlijsten().controleer(cfg, {"SIGNAAL_VAULT": _VAULT})


def test_koppelpunt_met_puntpunt() -> None:
    assert _padlijsten().controleer(_cfg("C:/inname", "/inname/../vault"),
                                    {"SIGNAAL_VAULT": _VAULT})


def test_ontbrekend_canoniek_pad_wordt_geweigerd() -> None:
    """Herreview 1: een pad dat niet in de canonieke map staat, telt niet als in orde."""
    p = _padlijsten()
    env = {"SIGNAAL_VAULT": _VAULT}
    alles = {"C:/inname": "C:/inname", "C:/w": "C:/w", _VAULT: _VAULT}
    assert p.controleer(_cfg("C:/inname"), env, alles) == []
    zonder = {k: v for k, v in alles.items() if k != "C:/inname"}
    assert p.controleer(_cfg("C:/inname"), env, zonder)
    assert p.controleer(_cfg("C:/inname"), env, {})


@pytest.mark.parametrize("waarde", [None, "", "{}"])
def test_main_weigert_zonder_canonieke_paden(monkeypatch, capsys, waarde) -> None:
    import io

    p = _padlijsten()
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(_cfg("C:/inname"))))
    monkeypatch.setenv("SIGNAAL_VAULT", _VAULT)
    if waarde is None:
        monkeypatch.delenv("SIGNAAL_CANONIEK", raising=False)
    else:
        monkeypatch.setenv("SIGNAAL_CANONIEK", waarde)
    assert p.main([]) == 1
    assert "SIGNAAL_CANONIEK" in capsys.readouterr().err


def _powershell() -> str | None:
    vast = "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
    return shutil.which("powershell.exe") or (vast if os.path.exists(vast) else None)


@pytest.mark.skipif(_powershell() is None, reason="PowerShell niet beschikbaar")
def test_canoniek_ps1_lost_paden_op_en_onderscheidt_hoofdletters() -> None:
    """Draait canoniek.ps1 zelf (vanuit WSL via powershell.exe)."""
    script = ROOT / "scripts" / "signaalsysteem" / "controle" / "canoniek.ps1"
    if str(script).startswith("/mnt/"):
        script_win = subprocess.run(["wslpath", "-w", str(script)], capture_output=True,
                                    text=True).stdout.strip()
    else:
        script_win = str(script)
    paden_in = [r"C:\Windows", r"c:\windows", r"C:\PROGRA~1", r"C:\bestaat-niet-f210",
                r"C:\Windows\..\Windows"]
    uit = subprocess.run([_powershell(), "-NoProfile", "-ExecutionPolicy", "Bypass",
                          "-File", script_win], input="\n".join(paden_in).encode("utf-8"),
                         capture_output=True, timeout=60)
    assert uit.returncode == 0, uit.stderr
    paden = json.loads(uit.stdout.decode("utf-8"))
    # Ordinaal: twee schrijfwijzen zijn twee sleutels.
    assert set(paden) == set(paden_in)
    assert paden[r"C:\Windows"].lower() == "c:/windows"
    assert paden[r"c:\windows"].lower() == "c:/windows"
    assert paden[r"C:\PROGRA~1"] == "C:/Program Files"
    assert paden[r"C:\bestaat-niet-f210"] is None
    assert paden[r"C:\Windows\..\Windows"] is None


# --------------------------------------------------------------------------
# GPU-diensten: llama-swap, socket-proxy en docling-serve (F2-11; ADR 0009, 0014)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def config_gpu(hostpaden) -> dict:
    """Met het profiel van de diensten die llama-swap start."""
    uit = _compose(hostpaden, "--profile", "door-llama-swap", "config", "--format", "json")
    assert uit.returncode == 0, uit.stderr
    return json.loads(uit.stdout)


def _gpu(dienst: dict) -> bool:
    apparaten = (((dienst.get("deploy") or {}).get("resources") or {}).get("reservations")
                 or {}).get("devices") or []
    return any("gpu" in (a.get("capabilities") or []) for a in apparaten)


def test_beheer_alleen_llama_swap_en_socket_proxy(config_gpu) -> None:
    assert _leden(config_gpu, "beheer") == {"llama-swap", "socket-proxy"}
    assert _netwerken(config_gpu["services"]["socket-proxy"]) == {"beheer"}
    assert _netwerken(config_gpu["services"]["llama-swap"]) == {"verwerking", "beheer"}
    assert _netwerken(config_gpu["services"]["docling-serve"]) == {"verwerking"}


def test_socket_proxy_configuratie_b(config_gpu) -> None:
    """P7, K2: alleen starten en stoppen; create, exec en inspect dicht."""
    env = config_gpu["services"]["socket-proxy"]["environment"]
    assert env == {"POST": "1", "ALLOW_START": "1", "ALLOW_STOP": "1", "CONTAINERS": "0"}


def test_docker_socket_alleen_in_de_socket_proxy(config_gpu) -> None:
    for naam, dienst in config_gpu["services"].items():
        socket = [k for k in dienst.get("volumes") or []
                  if "docker.sock" in str(k.get("source", "")) + str(k.get("target", ""))]
        if naam == "socket-proxy":
            assert [(k["source"], k["target"], k.get("read_only")) for k in socket] == [
                ("/var/run/docker.sock", "/var/run/docker.sock", True)]
        else:
            assert not socket, naam
    assert config_gpu["services"]["llama-swap"]["environment"]["DOCKER_HOST"] == \
        "tcp://socket-proxy:2375"


def test_startcontrole_weigert_socket_buiten_de_proxy(config_gpu, hostpaden) -> None:
    p = _padlijsten()
    # De socket is geen Windows-pad: hij gaat niet naar canoniek.ps1 en telt niet.
    assert "/var/run/docker.sock" not in p.bind_bronnen(config_gpu)
    socket = {"type": "bind", "source": "/var/run/docker.sock",
              "target": "/var/run/docker.sock", "read_only": True}
    assert p.controleer(_met(config_gpu, "llama-swap", [socket]), hostpaden)


def test_gpu_alleen_voor_diensten_van_llama_swap(config_gpu) -> None:
    assert {n for n, d in config_gpu["services"].items() if _gpu(d)} == {"llama-swap",
                                                                         "docling-serve"}


def test_cuda_omweg_zolang_de_driver_13_3_levert(config_gpu) -> None:
    """ADR 0006; vervalt in F2-25."""
    env = config_gpu["services"]["llama-swap"]["environment"]
    assert env["NVIDIA_DISABLE_REQUIRE"] == "1"
    assert env["LD_LIBRARY_PATH"].startswith("/usr/lib/x86_64-linux-gnu:")


def test_modellen_alleen_lezen_en_extern(config_gpu) -> None:
    for naam, volume, doel in (("llama-swap", "modellen", "/modellen"),
                               ("docling-serve", "docling_artifacts", "/artifacts")):
        [kop] = _koppeling(config_gpu["services"][naam], volume)
        assert kop["type"] == "volume" and kop["read_only"] is True and kop["target"] == doel
        assert config_gpu["volumes"][volume]["external"] is True
    assert config_gpu["volumes"]["modellen"]["name"] == "signaal_modellen"


def test_docling_serve_door_llama_swap(config_gpu, config) -> None:
    """Vooraf aangemaakt, geen restart; sleutel als bestand; synchroon eindpunt."""
    docling = config_gpu["services"]["docling-serve"]
    assert "docling-serve" not in config["services"]       # niet bij een kale `up`
    assert docling["container_name"] == "signaal-docling"
    assert docling["profiles"] == ["door-llama-swap"]
    assert docling.get("restart") in (None, "no")
    assert [s["source"] for s in docling["secrets"]] == ["docling_api_key"]
    assert Path(config_gpu["secrets"]["docling_api_key"]["file"]).parent.name == "secrets"
    env = docling["environment"]
    assert env["DOCLING_SERVE_MAX_SYNC_WAIT"] == "3600"
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["DOCLING_SERVE_ENABLE_REMOTE_SERVICES"] == "false"
    assert env["DOCLING_SERVE_LOAD_MODELS_AT_BOOT"] == "false"
    assert "/run/secrets/docling_api_key" in " ".join(docling["command"])
    assert "DOCLING_SERVE_API_KEY" not in env


def test_llama_swap_image_met_docker_client_en_configuratie() -> None:
    tekst = (ROOT / "services" / "llama-swap" / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"^FROM docker:29\.9\.2-cli@sha256:\S+ AS cli$", tekst, re.M)
    assert re.search(r"^FROM ghcr\.io/mostlygeek/llama-swap:v268-cuda13-b11977@sha256:", tekst, re.M)
    assert "COPY --from=cli /usr/local/bin/docker /usr/local/bin/docker" in tekst
    assert re.search(r"^COPY config\.yaml /app/config\.yaml$", tekst, re.M)
    assert "docker.sock" not in tekst
