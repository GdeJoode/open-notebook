"""Startcontrole van de container ``open-notebook`` (F2-10, ``services/open-notebook/start.py``).

De API start niet als een innamemap of het bronarchief schrijfbaar is, het
forkarchief alleen-lezen is, een koppeling ontbreekt of de fork de vault koppelt.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _start():
    spec = importlib.util.spec_from_file_location(
        "start_open_notebook", ROOT / "services" / "open-notebook" / "start.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ENV = {
    "SIGNAAL_INNAME_PADEN": "/inname;/watchfolder",
    "SIGNAAL_BRONARCHIEF": "/bronarchief",
    "SIGNAAL_FORK_ARCHIEF": "/fork_archief",
    "SIGNAAL_VAULT_PADEN": "/vault",
}


def _regel(nr: int, punt: str, opties: str) -> str:
    return f"{nr} 1 0:{nr} / {punt} {opties},relatime - ext4 /dev/sdx rw"


def _mountinfo(**anders: str) -> str:
    standaard = {"/": "rw", "/inname": "ro", "/watchfolder": "ro", "/bronarchief": "ro",
                 "/fork_archief": "rw", "/app/.git": "ro"}
    standaard.update(anders)
    return "\n".join(_regel(i, p, o) for i, (p, o) in enumerate(standaard.items(), 20)
                     if o)


def test_juiste_koppelingen_slagen() -> None:
    s = _start()
    assert s.controleer_koppelingen(s.koppelingen(_mountinfo()), ENV) == []


def test_fouten_in_de_koppelingen() -> None:
    s = _start()
    gevallen = {
        "/inname": "rw",            # innamemap schrijfbaar
        "/watchfolder": "",         # watchfolder niet gekoppeld
        "/bronarchief": "rw",       # bronarchief schrijfbaar
        "/fork_archief": "ro",      # forkarchief alleen-lezen
        "/vault/entities": "ro",    # de fork koppelt de vault
    }
    for punt, opties in gevallen.items():
        fouten = s.controleer_koppelingen(s.koppelingen(_mountinfo(**{punt: opties})), ENV)
        assert fouten, (punt, opties)


def test_windowspad_in_de_innamelijst_is_een_fout() -> None:
    s = _start()
    env = dict(ENV, SIGNAAL_INNAME_PADEN=r"D:\Data\inname")
    assert s.controleer_koppelingen(s.koppelingen(_mountinfo()), env)


def test_spatie_in_koppelpunt() -> None:
    s = _start()
    kop = s.koppelingen(_regel(1, "/in\\040naam", "ro"))
    assert kop == {"/in naam": {"ro", "relatime"}}


def test_geheimen_uit_bestanden(tmp_path) -> None:
    (tmp_path / "surreal_gebruiker").write_text("gebruiker\n", encoding="utf-8")
    (tmp_path / "surreal_wachtwoord").write_text("proefwaarde\n", encoding="utf-8")
    assert _start().lees_geheimen(tmp_path) == {
        "SURREAL_USER": "gebruiker", "SURREAL_PASSWORD": "proefwaarde"}


def test_koppeling_met_obsidian_wordt_geweigerd(tmp_path) -> None:
    """B3: een vault onder een andere naam herkent de fork aan ``.obsidian``."""
    s = _start()
    schoon = tmp_path / "inname"
    (schoon / "sub").mkdir(parents=True)
    vault = tmp_path / "andere-naam"
    (vault / "20 Atlas" / ".obsidian").mkdir(parents=True)
    assert s.obsidian_mappen([str(schoon)]) == []
    [fout] = s.obsidian_mappen([str(schoon), str(vault)])
    assert str(vault / "20 Atlas" / ".obsidian") in fout


def test_obsidian_alleen_tot_de_maximale_diepte(tmp_path) -> None:
    s = _start()
    diep = tmp_path / "inname"
    pad = diep
    for i in range(s.OBSIDIAN_DIEPTE + 2):
        pad = pad / f"n{i}"
    (pad / ".obsidian").mkdir(parents=True)
    assert s.obsidian_mappen([str(diep)]) == []
    ondiep = tmp_path / "watch" / "a" / "b"
    (ondiep / ".obsidian").mkdir(parents=True)
    assert s.obsidian_mappen([str(tmp_path / "watch")])


def test_fout_tijdens_de_doorloop_wordt_gemeld(tmp_path) -> None:
    s = _start()
    fouten = s.obsidian_mappen([str(tmp_path / "bestaat-niet")])
    assert fouten and "niet te lezen" in fouten[0]
