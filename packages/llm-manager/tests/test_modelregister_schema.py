"""Stopconditie F2-07: het modelregister voldoet aan het schema (ADR 0007).

Valideert ``config/modelregister.yaml`` en toetst de schemaregels met
synthetische registers.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from llm_manager.register import (
    STANDAARD_PAD,
    RegisterFout,
    get_register,
    lees_register,
)


def _schrijf(tmp_path: Path, tekst: str) -> Path:
    pad = tmp_path / "register.yaml"
    pad.write_text(tekst, encoding="utf-8")
    return pad


MODEL = """
  - naam: {naam}
    versie: v1
    aanroepwijze: {wijze}
    locatie: {locatie}
    taken: [chat]
"""


def test_register_in_git_is_geldig() -> None:
    register = lees_register(STANDAARD_PAD)
    assert register.modellen
    for model in register.modellen:
        if model.locatie == "cloud":
            assert not model.intern_toegestaan


def test_werkmodel_volgens_b4() -> None:
    werkmodel = lees_register(STANDAARD_PAD).zoek("werkmodel")
    assert werkmodel is not None
    assert "Qwen3.5-9B-Q4_K_M" in werkmodel.versie and "8192" in werkmodel.versie
    assert werkmodel.aanroepwijze == "llama-swap" and werkmodel.locatie == "lokaal"


def test_omgevingsvariabele_kiest_het_bestand(tmp_path, monkeypatch) -> None:
    pad = _schrijf(tmp_path, "versie: 1\nmodellen:" + MODEL.format(
        naam="ander", wijze="llama-swap", locatie="lokaal"))
    monkeypatch.setenv("SIGNAAL_MODELREGISTER", str(pad))
    get_register.cache_clear()
    try:
        assert [m.naam for m in get_register().modellen] == ["ander"]
    finally:
        get_register.cache_clear()


def test_standaard_niet_toegestaan_voor_intern(tmp_path) -> None:
    pad = _schrijf(tmp_path, "versie: 1\nmodellen:" + MODEL.format(
        naam="m", wijze="llama-swap", locatie="lokaal"))
    assert lees_register(pad).zoek("m").toegestaan_intern == "nee"


@pytest.mark.parametrize("tekst", [
    # cloudmodel met ja
    "versie: 1\nmodellen:\n  - {naam: c, versie: v, aanroepwijze: api, locatie: cloud,"
    " toegestaan_intern: ja, taken: [chat]}",
    # externe verwerkingsdienst met ja
    "versie: 1\nmodellen:\n  - {naam: d, versie: v, aanroepwijze: llama-swap, locatie: cloud,"
    " toegestaan_intern: ja, taken: [document_extraction]}",
    # api of cli is altijd cloud
    "versie: 1\nmodellen:\n  - {naam: c, versie: v, aanroepwijze: cli, locatie: lokaal,"
    " taken: [chat]}",
    # YAML-bool in plaats van ja of nee
    "versie: 1\nmodellen:\n  - {naam: m, versie: v, aanroepwijze: llama-swap, locatie: lokaal,"
    " toegestaan_intern: yes, taken: [chat]}",
    # onbekende sleutel
    "versie: 1\nmodellen:\n  - {naam: m, versie: v, aanroepwijze: llama-swap, locatie: lokaal,"
    " taken: [chat], adres: 'https://example.org'}",
    # onbekende aanroepwijze of locatie
    "versie: 1\nmodellen:\n  - {naam: m, versie: v, aanroepwijze: ollama, locatie: lokaal,"
    " taken: [chat]}",
    # dubbele naam
    "versie: 1\nmodellen:\n  - {naam: m, versie: v, aanroepwijze: llama-swap, locatie: lokaal,"
    " taken: [chat]}\n  - {naam: m, versie: w, aanroepwijze: llama-swap, locatie: lokaal,"
    " taken: [chat]}",
    # geen taken
    "versie: 1\nmodellen:\n  - {naam: m, versie: v, aanroepwijze: llama-swap, locatie: lokaal,"
    " taken: []}",
    # geen versie van het register
    "modellen: []",
])
def test_ongeldig_register(tmp_path, tekst) -> None:
    with pytest.raises(RegisterFout):
        lees_register(_schrijf(tmp_path, tekst))


def test_ontbrekend_bestand(tmp_path) -> None:
    with pytest.raises(RegisterFout):
        lees_register(tmp_path / "bestaat-niet.yaml")
