"""`POST /vocabulary/refresh` — one source of truth for TOOI (F2-28 review).

The registers come from ``config/referentiegegevens.yaml`` and the cache the admin
step fills; there is no seed and no internet. The reviewer's scenario: after a
refresh that cannot reach the registers, mnre1150 keeps its aliases and end date.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import pytest
import yaml
from app_main.api.routers import vocabulary
from app_main.dependencies import get_reference_entity_repo
from fastapi import FastAPI
from fastapi.testclient import TestClient

_ONT = "https://identifier.overheid.nl/tooi/def/ont/"


class _Repo:
    """Upsert on ``(external_id, source_vocabulary)``, like migration 57."""

    def __init__(self) -> None:
        self.rows: Dict[tuple, Dict[str, Any]] = {}

    async def bulk_load(self, records: List[Dict[str, Any]]) -> int:
        for r in records:
            self.rows[(r["external_id"], r["source_vocabulary"])] = json.loads(json.dumps(r))
        return len(records)

    async def external_ids(self, source: str) -> List[str]:
        return [eid for (eid, src) in self.rows if src == source]


def _node(code: str, naam: str, afk: str, **extra: Any) -> Dict[str, Any]:
    node = {"@id": f"https://identifier.overheid.nl/tooi/id/ministerie/{code}",
            "@type": [_ONT + "Ministerie"],
            _ONT + "organisatiecode": [{"@value": code}],
            _ONT + "officieleNaamExclSoort": [{"@value": naam}],
            _ONT + "officieleNaamInclSoort": [{"@value": "ministerie van " + naam}],
            _ONT + "afkorting": [{"@value": afk}]}
    node.update(extra)
    return node


_LNV = _node("mnre1150", "Landbouw, Natuur en Voedselkwaliteit", "LNV",
             **{_ONT + "einddatum": [{"@value": "2010-10-13"}]})
_BZK = _node("mnre1034", "Binnenlandse Zaken en Koninkrijksrelaties", "BZK")


def _opzet(tmp_path: Path, monkeypatch, nodes: List[Dict[str, Any]]) -> Path:
    """An inventory with one register; the cache holds it with that SHA-256."""
    cache = tmp_path / "cache"
    cache.mkdir()
    data = json.dumps(nodes).encode()
    (cache / "rwc_ministeries_compleet_6.json").write_bytes(data)
    inventaris = {"inventaris": [{
        "naam": "tooi_overheidsorganisaties", "tabel": "reference_entity",
        "laadwijze": "tooi_registers", "filter": {"source_vocabulary": "tooi"},
        "bestanden": [{"bestand": "rwc_ministeries_compleet_6.json",
                       "sha256": hashlib.sha256(data).hexdigest(),
                       "default_soort": "ministerie"}]}]}
    pad = tmp_path / "referentiegegevens.yaml"
    pad.write_text(yaml.safe_dump(inventaris), encoding="utf-8")
    monkeypatch.setenv("SIGNAAL_REFERENTIEGEGEVENS", str(pad))
    monkeypatch.setenv("SIGNAAL_REFERENTIE_CACHE", str(cache))
    return cache


def _client(repo: _Repo) -> TestClient:
    app = FastAPI()
    app.include_router(vocabulary.router)
    app.dependency_overrides[get_reference_entity_repo] = lambda: repo
    return TestClient(app)


def _refresh(repo: _Repo):
    return _client(repo).post("/vocabulary/refresh", json={"providers": ["tooi"]})


def test_refresh_loads_the_inventory_registers(tmp_path, monkeypatch):
    _opzet(tmp_path, monkeypatch, [_LNV, _BZK])
    repo = _Repo()
    r = _refresh(repo)
    assert r.status_code == 200, r.text
    assert r.json()["results"] == [{"name": "tooi", "loaded": 2, "on_demand": False}]
    assert repo.rows[("mnre1150", "tooi")]["properties"]["einddatum"] == "2010-10-13"


@pytest.mark.parametrize("bederf", ["weg", "ander"])
def test_refresh_without_the_cache_keeps_mnre1150(tmp_path, monkeypatch, bederf):
    """Without internet and without (a valid) cache: 503, and nothing changes."""
    cache = _opzet(tmp_path, monkeypatch, [_LNV, _BZK])
    repo = _Repo()
    assert _refresh(repo).status_code == 200
    voor = copy.deepcopy(repo.rows)

    bestand = cache / "rwc_ministeries_compleet_6.json"
    if bederf == "weg":
        bestand.unlink()
    else:
        bestand.write_bytes(json.dumps([_BZK]).encode())   # other bytes, other SHA-256
    r = _refresh(repo)

    assert r.status_code == 503
    assert "laad_referentiegegevens.sh" in r.json()["detail"]
    assert repo.rows == voor
    lnv = repo.rows[("mnre1150", "tooi")]
    assert lnv["properties"]["einddatum"] == "2010-10-13"
    assert "LNV" in lnv["aliases"]


def test_refresh_never_loads_less_than_is_loaded(tmp_path, monkeypatch):
    """A register set without mnre1150 would leave it stale: 409, nothing written."""
    _opzet(tmp_path, monkeypatch, [_LNV, _BZK])
    repo = _Repo()
    assert _refresh(repo).status_code == 200
    voor = copy.deepcopy(repo.rows)
    (tmp_path / "kleiner").mkdir()
    _opzet(tmp_path / "kleiner", monkeypatch, [_BZK])
    r = _refresh(repo)
    assert r.status_code == 409
    assert "mnre1150" in r.json()["detail"]
    assert repo.rows == voor
