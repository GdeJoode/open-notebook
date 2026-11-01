"""TOOI-registers uit de inventaris van de referentiegegevens (F2-28).

Eén bron van waarheid voor het laden van ``reference_entity`` (TOOI): de
registers en hun SHA-256 staan in ``config/referentiegegevens.yaml``, de bestanden
in een cachemap die de beheerstap ``scripts/signaalsysteem/laad_referentiegegevens.sh``
vult (de container heeft geen internet). Het laadscript en
``POST /api/vocabulary/refresh`` gebruiken allebei :func:`laad_tooi`.

Er is geen terugval: ontbreekt een bestand of wijkt een SHA-256 af, dan
:class:`BronOntbreekt`; zou een lading organisaties weglaten die al geladen zijn,
dan :class:`MinderGegevens`. In beide gevallen wordt niets geschreven.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from shared.vocabulary.tooi_bulk import parse_register


class BronOntbreekt(Exception):
    """Een bronbestand ontbreekt, is onleesbaar of heeft een andere SHA-256."""


class MinderGegevens(Exception):
    """De lading zou organisaties weglaten die al in de database staan."""


def lees_inventaris(pad: Path) -> Dict[str, Any]:
    try:
        return yaml.safe_load(Path(pad).read_text(encoding="utf-8"))
    except OSError as exc:
        raise BronOntbreekt(f"inventaris onleesbaar: {pad} ({exc})") from exc


def tooi_set(inventaris: Dict[str, Any]) -> Dict[str, Any]:
    """De ene set met laadwijze ``tooi_registers``."""
    sets = [s for s in inventaris["inventaris"] if s["laadwijze"] == "tooi_registers"]
    if len(sets) != 1:
        raise BronOntbreekt(f"verwacht één TOOI-set in de inventaris, gevonden {len(sets)}")
    return sets[0]


def _lees_gecontroleerd(pad: Path, verwacht: str) -> bytes:
    """De bytes van een bronbestand, alleen als hun SHA-256 klopt.

    Hash en ontleden gebruiken dezelfde bytes: het bestand wordt één keer gelezen.
    """
    try:
        data = pad.read_bytes()
    except OSError as exc:
        raise BronOntbreekt(f"bronbestand ontbreekt: {pad}") from exc
    werkelijk = hashlib.sha256(data).hexdigest()
    if werkelijk != verwacht:
        raise BronOntbreekt(f"SHA-256 van {pad.name} is {werkelijk}, verwacht {verwacht}")
    return data


def tooi_records(set_: Dict[str, Any], cachemap: Path) -> List[Dict[str, Any]]:
    """De organisaties uit de registers van een set, uniek per organisatiecode.

    Controleert eerst elk bestand; pas als alle SHA-256's kloppen wordt er ontleed.
    """
    inhoud = [(b, _lees_gecontroleerd(Path(cachemap) / b["bestand"], b["sha256"]))
              for b in set_["bestanden"]]
    gezien: set[str] = set()
    records: List[Dict[str, Any]] = []
    for b, data in inhoud:
        for rec in parse_register(json.loads(data), default_soort=b["default_soort"]):
            if rec["organisatiecode"] not in gezien:
                gezien.add(rec["organisatiecode"])
                records.append(rec)
    if not records:
        raise BronOntbreekt(f"{set_['naam']}: geen organisaties in de registers")
    return records


class _VasteRecords:
    """Levert vooraf gelezen records aan ``TOOIProvider``."""

    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self._records = records

    async def fetch_records(self) -> List[Dict[str, Any]]:
        return list(self._records)


async def laad_tooi(repo: Any, set_: Dict[str, Any], cachemap: Path) -> Dict[str, int]:
    """Laadt de TOOI-set in ``reference_entity``; ``{"bron", "geladen"}``.

    ``repo`` heeft ``bulk_load`` en ``external_ids(source)``. Weigert (zonder te
    schrijven) als een bronbestand niet klopt of als een organisatie die al geladen
    is, in de bron ontbreekt.
    """
    from shared.vocabulary.tooi_provider import TOOIProvider

    records = tooi_records(set_, cachemap)
    bron = {r["organisatiecode"] for r in records}
    bestaand = set(await repo.external_ids(set_["filter"]["source_vocabulary"]))
    weg = bestaand - bron
    if weg:
        raise MinderGegevens(
            f"{len(weg)} geladen organisaties ontbreken in de bron "
            f"(bijv. {sorted(weg)[:3]}); niets geladen")
    provider = TOOIProvider(repo, bulk_fetcher=_VasteRecords(records))  # type: ignore[arg-type]
    geladen = await provider.refresh()
    return {"bron": len(bron), "geladen": geladen}


__all__ = ["BronOntbreekt", "MinderGegevens", "lees_inventaris", "tooi_set",
           "tooi_records", "laad_tooi"]
