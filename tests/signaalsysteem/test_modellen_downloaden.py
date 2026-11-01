"""Controle van de modellen (F2-11, review punt 8): afwijking en ontbreken.

Toetst ``scripts/signaalsysteem/modellen_downloaden.py`` met kleine synthetische
bestanden, zonder netwerk, en dat ``start.sh`` de controle draait.
"""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "signaalsysteem" / "modellen_downloaden.py"


@pytest.fixture()
def mod(monkeypatch):
    spec = importlib.util.spec_from_file_location("modellen_downloaden", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    inhoud = b"synthetisch model"
    monkeypatch.setattr(m, "GGUF", [{"bestand": "a.gguf", "repo": "x/y", "revisie": "0" * 40,
                                     "sha256": hashlib.sha256(inhoud).hexdigest(),
                                     "grootte": len(inhoud)}])
    m.inhoud = inhoud
    return m


def test_juiste_hash_slaagt(mod, tmp_path) -> None:
    (tmp_path / "a.gguf").write_bytes(mod.inhoud)
    code, verslag = mod.gguf(tmp_path, None, mag_halen=False)
    assert code == 0 and verslag[0]["geslaagd"] is True


def test_afwijkende_hash_geeft_1_en_overschrijft_niets(mod, tmp_path) -> None:
    (tmp_path / "a.gguf").write_bytes(b"iets anders")
    code, verslag = mod.gguf(tmp_path, None, mag_halen=False)
    assert code == 1 and verslag[0]["geslaagd"] is False
    assert (tmp_path / "a.gguf").read_bytes() == b"iets anders"


def test_ontbrekend_bestand_geeft_2(mod, tmp_path) -> None:
    code, verslag = mod.gguf(tmp_path, None, mag_halen=False)
    assert code == 2 and verslag[0]["fout"] == "ontbreekt"


def test_docling_boom(mod, tmp_path, monkeypatch) -> None:
    assert mod.docling(tmp_path)[0] == 2                      # leeg
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "m.bin").write_bytes(b"x")
    som, aantal = mod.boom_sha256(tmp_path)
    monkeypatch.setattr(mod, "DOCLING", {"sha256_boom": som, "bestanden": aantal})
    assert mod.docling(tmp_path)[0] == 0
    (tmp_path / "sub" / "m.bin").write_bytes(b"y")
    assert mod.docling(tmp_path)[0] == 1                      # afwijkend


def test_start_sh_draait_de_controle_voor_het_starten() -> None:
    tekst = (ROOT / "scripts" / "signaalsysteem" / "start.sh").read_text(encoding="utf-8")
    controle = tekst.index("modellen_laden.sh\" --controle")
    assert controle < tekst.index('"${COMPOSE[@]}" up -d "$@"')
    assert "exit 1" in tekst[controle:controle + 300]
