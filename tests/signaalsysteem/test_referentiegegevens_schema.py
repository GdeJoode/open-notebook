"""Inventaris en laadscript van de referentiegegevens (F2-28).

Valideert ``config/referentiegegevens.yaml`` (elk item met tabel, bron,
SHA-256 en laadwijze) en toetst ``scripts/signaalsysteem/laad_referentiegegevens.py``
zonder database: een klein TOOI-register in de vorm van de openbare dump, een
nep-repository met dezelfde upsertsleutel als ``reference_entity``.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml
from shared.vocabulary import referentiegegevens as ref
from shared.vocabulary.tooi_bulk import DEFAULT_REGISTERS, register_json_url

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "config" / "referentiegegevens.yaml"
SCRIPT = ROOT / "scripts" / "signaalsysteem" / "laad_referentiegegevens.py"
ONTOLOGIEEN = ROOT / "packages" / "ontology-manager" / "ontologies"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
REPOSITORY = "https://repository.officiele-overheidspublicaties.nl/waardelijsten/"


def _laad_module():
    spec = importlib.util.spec_from_file_location("laad_referentiegegevens", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


laad = _laad_module()


@pytest.fixture(scope="module")
def config() -> Dict[str, Any]:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def _sets(config: Dict[str, Any], wijze: str) -> List[Dict[str, Any]]:
    return [s for s in config["inventaris"] if s["laadwijze"] == wijze]


# -- schema van de inventaris ---------------------------------------------------


def test_elk_item_heeft_tabel_bron_sha256_en_laadwijze(config):
    namen = [s["naam"] for s in config["inventaris"]]
    assert len(namen) == len(set(namen)), "naam dubbel"
    for s in config["inventaris"]:
        for veld in ("naam", "tabel", "bron", "laadwijze"):
            assert isinstance(s.get(veld), str) and s[veld].strip(), f"{s.get('naam')}: {veld}"
        assert s["laadwijze"] in laad.LAADWIJZEN, s["naam"]
        if s["laadwijze"] in {"tooi_registers", "image"}:
            assert s["bestanden"], s["naam"]
            for b in s["bestanden"]:
                assert HEX64.match(b["sha256"]), f"{s['naam']}: {b}"
        else:
            # Geen bestand om te hashen; dan wel zeggen waarom of wanneer.
            assert "sha256" in s and s["sha256"] is None, s["naam"]
            assert s.get("taak") if s["laadwijze"] == "gepland" else s.get("reden"), s["naam"]


def test_tooi_registers_zijn_openbare_vaste_versies(config):
    sets = _sets(config, "tooi_registers")
    assert [s["naam"] for s in sets] == ["tooi_overheidsorganisaties"]
    (tooi,) = sets
    assert tooi["tabel"] == "reference_entity"
    assert tooi["filter"] == {"source_vocabulary": "tooi"}
    urls = [b["url"] for b in tooi["bestanden"]]
    assert len(urls) == len(set(urls)) == 8
    for b in tooi["bestanden"]:
        assert b["url"].startswith(REPOSITORY) and b["url"].endswith("/" + b["bestand"])
        # <set>/<versie>/json/<set>_<versie>.json: een vaste, onveranderlijke versie.
        assert re.search(r"/([a-z_]+)/(\d+)/json/\1_\2\.json$", b["url"]), b["url"]
        assert b["default_soort"]
    assert any("gemeenten" in b["bestand"] for b in tooi["bestanden"])


def test_reference_entity_openbaar_alleen_beheerder_niet_via_mcp(config):
    (tooi,) = _sets(config, "tooi_registers")
    assert tooi["classificatie"].startswith("geen")
    assert "beheerder" in tooi["toegang"]
    assert tooi["mcp"].startswith("niet") and "F3-10" in tooi["mcp"]


def test_geen_heeft_de_gevraagde_items(config):
    geen = {s["naam"]: s for s in _sets(config, "geen")}
    assert "ONB_ALIAS_OVERRIDES_PATH" in geen["aliasoverrides"]["tabel"]
    assert geen["zotero"]["tabel"] == "zotero_config"


def test_default_registers_gelijk_aan_de_inventaris(config):
    """Eén bron van waarheid: ``DEFAULT_REGISTERS`` (TOOIBulkFetcher) volgt de inventaris."""
    (tooi,) = _sets(config, "tooi_registers")
    assert [(b["url"], b["default_soort"]) for b in tooi["bestanden"]] == [
        (register_json_url(r["set"], r["version"]), r["default_soort"]) for r in DEFAULT_REGISTERS]


def test_ontologieen_compleet_en_sha256_klopt(config):
    (onto,) = [s for s in config["inventaris"] if s["naam"] == "ontologieen"]
    assert onto["laadwijze"] == "image"
    gelijst = {Path(b["pad"]).name for b in onto["bestanden"]}
    assert gelijst == {p.name for p in ONTOLOGIEEN.glob("*.yaml")}
    for b in onto["bestanden"]:
        assert laad.sha256(ROOT / b["pad"], lf=True) == b["sha256"], (
            f"{b['pad']} gewijzigd: werk de SHA-256 in config/referentiegegevens.yaml bij")


def test_iv3_en_bwb_staan_gepland_in_f3_03(config):
    gepland = {s["naam"]: s for s in _sets(config, "gepland")}
    assert gepland["iv3_taakvelden_2025"]["taak"] == "F3-03"
    assert gepland["iv3_taakvelden_2025"]["tabel"] == "ontology_version"
    assert gepland["bwb_identificaties"]["taak"] == "F3-03"
    assert gepland["bwb_identificaties"]["tabel"] == "reference_entity"


def test_downloadlijst_noemt_alle_registers(config, capsys):
    assert laad.main(["--config", str(CONFIG), "--downloadlijst"]) == 0
    regels = capsys.readouterr().out.strip().splitlines()
    assert len(regels) == 8
    for regel in regels:
        bestand, url, sha = regel.split("\t")
        assert url.endswith(bestand) and HEX64.match(sha)


# -- laadscript zonder database ---------------------------------------------------

_ONT = "https://identifier.overheid.nl/tooi/def/ont/"


def _org(uri: str, code: str, naam: str, afk: str, **extra: Any) -> Dict[str, Any]:
    node = {
        "@id": uri,
        "@type": [_ONT + "Ministerie"],
        _ONT + "organisatiecode": [{"@value": code}],
        _ONT + "officieleNaamExclSoort": [{"@value": naam}],
        _ONT + "officieleNaamInclSoort": [{"@value": "ministerie van " + naam}],
        _ONT + "afkorting": [{"@value": afk}],
    }
    node.update(extra)
    return node


def _register(tmp_path: Path) -> Dict[str, Any]:
    """Klein register in de vorm van de openbare TOOI-dump: twee ministeries, één
    met een historische naam (die als alias telt, niet als eigen organisatie)."""
    jv = "https://identifier.overheid.nl/tooi/id/ministerie/mnre1058"
    payload = [
        {"@id": "kop", "@type": ["https://identifier.overheid.nl/tooi/def/wl/RegisterwaardelijstCompleet"]},
        _org("https://identifier.overheid.nl/tooi/id/ministerie/mnre1034", "mnre1034",
             "Binnenlandse Zaken en Koninkrijksrelaties", "BZK"),
        _org(jv, "mnre1058", "Justitie en Veiligheid", "JenV"),
        _org(jv + "/hv1", "mnre1058", "Veiligheid en Justitie", "VenJ",
             **{"http://www.w3.org/ns/prov#specializationOf": [{"@id": jv}],
                _ONT + "einddatumHV": [{"@value": "2017-10-26"}]}),
    ]
    pad = tmp_path / "rwc_ministeries_compleet_1.json"
    pad.write_text(json.dumps(payload), encoding="utf-8")
    return {
        "naam": "tooi_proef", "tabel": "reference_entity", "laadwijze": "tooi_registers",
        "filter": {"source_vocabulary": "tooi"},
        "bestanden": [{"bestand": pad.name, "url": REPOSITORY + pad.name,
                       "sha256": hashlib.sha256(pad.read_bytes()).hexdigest(),
                       "default_soort": "ministerie"}],
    }


class _NepRepo:
    """Upsert op ``(external_id, source_vocabulary)``, zoals migratie 57."""

    def __init__(self) -> None:
        self.rijen: Dict[tuple, Dict[str, Any]] = {}
        self.volgnummer = 0

    async def bulk_load(self, records: List[Dict[str, Any]]) -> int:
        for r in records:
            sleutel = (r["external_id"], r["source_vocabulary"])
            if sleutel not in self.rijen:
                self.volgnummer += 1
                self.rijen[sleutel] = {"id": f"reference_entity:{self.volgnummer}"}
            self.rijen[sleutel].update(r)
        return len(records)

    async def external_ids(self, bron: str) -> List[str]:
        return [eid for (eid, b) in self.rijen if b == bron]

    async def rijen_van(self, bron: str) -> List[Dict[str, Any]]:
        return [dict(r) for (_, b), r in self.rijen.items() if b == bron]


def test_laden_is_idempotent(tmp_path):
    set_ = _register(tmp_path)
    repo = _NepRepo()
    eerste = asyncio.run(laad.laad_tooi_set(set_, tmp_path, repo, repo.rijen_van))
    tweede = asyncio.run(laad.laad_tooi_set(set_, tmp_path, repo, repo.rijen_van))
    assert eerste["geslaagd"] and tweede["geslaagd"]
    assert eerste["bron"] == eerste["in_db"] == 2
    assert {k: v for k, v in eerste.items()} == {k: v for k, v in tweede.items()}
    jv = repo.rijen[("mnre1058", "tooi")]
    assert jv["canonical_name"] == "Justitie en Veiligheid"
    assert "Veiligheid en Justitie" in jv["aliases"]


def test_nooit_minder_gegevens_dan_er_staan(tmp_path):
    """Een geladen organisatie die in de bron ontbreekt: weigeren, niets schrijven."""
    set_ = _register(tmp_path)
    repo = _NepRepo()
    asyncio.run(repo.bulk_load([{"external_id": "mnre9999", "source_vocabulary": "tooi"}]))
    voor = {k: dict(v) for k, v in repo.rijen.items()}
    with pytest.raises(ref.MinderGegevens, match="mnre9999"):
        asyncio.run(laad.laad_tooi_set(set_, tmp_path, repo, repo.rijen_van))
    assert repo.rijen == voor


def test_afwijkende_sha256_laadt_niets(tmp_path):
    set_ = _register(tmp_path)
    set_["bestanden"][0]["sha256"] = "0" * 64
    repo = _NepRepo()
    with pytest.raises(ref.BronOntbreekt, match="SHA-256"):
        asyncio.run(laad.laad_tooi_set(set_, tmp_path, repo, repo.rijen_van))
    assert repo.rijen == {}


def test_ontbrekend_bronbestand(tmp_path):
    set_ = _register(tmp_path)
    (tmp_path / set_["bestanden"][0]["bestand"]).unlink()
    with pytest.raises(ref.BronOntbreekt, match="ontbreekt"):
        ref.tooi_records(set_, tmp_path)


class _NepManager:
    def __init__(self, namen: List[str]) -> None:
        self.namen = namen

    async def list_ontologies(self) -> List[str]:
        return list(self.namen)

    async def get_ontology(self, naam: str):
        return object() if naam in self.namen else None


def _actief(namen: List[str]):
    async def actief() -> List[str]:
        return list(namen)
    return actief


def _onto_set(tmp_path: Path) -> Dict[str, Any]:
    map_ = tmp_path / "ontologies"
    map_.mkdir()
    bestanden = []
    for naam in ("base", "general"):
        pad = map_ / f"{naam}.yaml"
        pad.write_bytes(b"metadata:\r\n  name: " + naam.encode() + b"\r\n")
        bestanden.append({"pad": f"ontologies/{naam}.yaml",
                          "sha256": laad.sha256(pad, lf=True)})
    return {"naam": "ontologieen", "tabel": "ontology_version", "laadwijze": "image",
            "bestanden": bestanden}


def test_ontologieen_gevonden_zoals_de_extractie_ze_opzoekt(tmp_path):
    set_ = _onto_set(tmp_path)
    goed = asyncio.run(laad.controleer_image(set_, tmp_path, _NepManager(["base", "general"]),
                                             _actief([])))
    assert goed["geslaagd"] and goed["gevonden"] == 2 and goed["actief_in_db"] == []
    mist = asyncio.run(laad.controleer_image(set_, tmp_path, _NepManager(["base"]), _actief([])))
    assert not mist["geslaagd"]


def test_actieve_versie_in_de_database_wordt_gemeld(tmp_path):
    """Met een actieve versie gebruikt de extractie die, niet de YAML-hash."""
    set_ = _onto_set(tmp_path)
    verslag = asyncio.run(laad.controleer_image(
        set_, tmp_path, _NepManager(["base", "general"]), _actief(["general"])))
    assert verslag["actief_in_db"] == ["general"]


def test_ontologie_niet_in_de_inventaris_valt_op(tmp_path):
    set_ = _onto_set(tmp_path)
    (tmp_path / "ontologies" / "extra.yaml").write_text("x: 1\n", encoding="utf-8")
    verslag = asyncio.run(laad.controleer_image(
        set_, tmp_path, _NepManager(["base", "general", "extra"]), _actief([])))
    assert not verslag["geslaagd"]
