"""Laadt de referentiegegevens van het signaalsysteem (F2-28; ADR 0021, 0013).

Leest de inventaris in ``config/referentiegegevens.yaml`` en verwerkt elke set
volgens zijn laadwijze:

* ``tooi_registers``: controleert de SHA-256 van elk gedownload register in
  ``--bronmap``, ontleedt het met ``tooi_bulk.parse_register`` en laadt de
  organisaties met ``TOOIProvider.refresh`` in ``reference_entity`` (upsert op
  ``(external_id, source_vocabulary)``, dus idempotent);
* ``image``: controleert de SHA-256 van elk bestand en of de code het vindt (voor
  de ontologieën: ``OntologyManager.list_ontologies`` en ``get_ontology``, zoals
  de extractie ze opzoekt);
* ``gepland`` en ``geen``: niets.

Draait in de container ``open-notebook`` met de verbinding van de migraties:

    docker exec -i <container> /app/.venv/bin/python /app/services/open-notebook/start.py \\
        /app/.venv/bin/python - --bronmap /tmp/referentiegegevens < laad_referentiegegevens.py

Schrijft per set een verslag als één JSON-regel op stdout.

Exitcodes: 0 elke set klopt (geladen = in de database = aantal in de bron);
1 een aantal klopt niet; 2 bron ontbreekt of SHA-256 wijkt af; 3 scriptfout.
``--downloadlijst`` drukt alleen ``bestand<TAB>url<TAB>sha256`` per te downloaden
bestand af (de container heeft geen internet; de host downloadt).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional

import yaml

CONFIG = Path("/app/config/referentiegegevens.yaml")
REPO = Path("/app")
LAADWIJZEN = {"tooi_registers", "image", "gepland", "geen"}


class BronFout(Exception):
    """Een bronbestand ontbreekt of heeft een andere SHA-256 dan de inventaris."""


def lees_config(pad: Path) -> Dict[str, Any]:
    return yaml.safe_load(pad.read_text(encoding="utf-8"))


def sha256(pad: Path, lf: bool = False) -> str:
    """SHA-256 van een bestand; met ``lf`` over de inhoud met LF-regeleinden."""
    data = pad.read_bytes()
    if lf:
        data = data.replace(b"\r\n", b"\n")
    return hashlib.sha256(data).hexdigest()


def downloadlijst(config: Dict[str, Any]) -> List[Dict[str, str]]:
    return [b for s in config["inventaris"] if s["laadwijze"] == "tooi_registers"
            for b in s["bestanden"]]


def controleer_bestand(pad: Path, verwacht: str, lf: bool = False) -> None:
    if not pad.is_file():
        raise BronFout(f"bronbestand ontbreekt: {pad}")
    werkelijk = sha256(pad, lf)
    if werkelijk != verwacht:
        raise BronFout(f"SHA-256 van {pad.name} is {werkelijk}, verwacht {verwacht}")


def tooi_records(set_: Dict[str, Any], bronmap: Path) -> List[Dict[str, Any]]:
    """De organisaties uit de registers van een set, uniek per organisatiecode
    (zoals ``TOOIBulkFetcher.fetch_records``). Controleert eerst elke SHA-256."""
    from shared.vocabulary.tooi_bulk import parse_register

    for b in set_["bestanden"]:
        controleer_bestand(bronmap / b["bestand"], b["sha256"])
    gezien: set[str] = set()
    records: List[Dict[str, Any]] = []
    for b in set_["bestanden"]:
        payload = json.loads((bronmap / b["bestand"]).read_text(encoding="utf-8"))
        for rec in parse_register(payload, default_soort=b["default_soort"]):
            if rec["organisatiecode"] not in gezien:
                gezien.add(rec["organisatiecode"])
                records.append(rec)
    if not records:
        raise BronFout(f"{set_['naam']}: geen organisaties in de registers")
    return records


class _VasteRecords:
    """Levert vooraf gelezen records aan ``TOOIProvider`` (in plaats van het web)."""

    def __init__(self, records: List[Dict[str, Any]]) -> None:
        self._records = records

    async def fetch_records(self) -> List[Dict[str, Any]]:
        return list(self._records)


Rijen = Callable[[str], Awaitable[List[Dict[str, Any]]]]


async def laad_tooi(set_: Dict[str, Any], bronmap: Path, repo: Any, rijen: Rijen) -> Dict[str, Any]:
    from shared.vocabulary.tooi_provider import TOOIProvider

    records = tooi_records(set_, bronmap)
    provider = TOOIProvider(repo, bulk_fetcher=_VasteRecords(records))  # type: ignore[arg-type]
    geladen = await provider.refresh()
    bron_codes = {r["organisatiecode"] for r in records}
    rows = await rijen(set_["filter"]["source_vocabulary"])
    db_codes = [str(r.get("external_id")) for r in rows]
    ids = sorted(str(r["id"]) for r in rows)
    return {
        "naam": set_["naam"],
        "tabel": set_["tabel"],
        "bron": len(bron_codes),
        "geladen": geladen,
        "in_db": len(rows),
        "ontbreekt": len(bron_codes - set(db_codes)),
        "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "geslaagd": geladen == len(bron_codes) == len(rows) and set(db_codes) == bron_codes,
    }


async def controleer_image(set_: Dict[str, Any], repo_root: Path, manager: Any) -> Dict[str, Any]:
    """Elk bestand met de juiste SHA-256, en de registry vindt elke ontologie."""
    for b in set_["bestanden"]:
        controleer_bestand(repo_root / b["pad"], b["sha256"], lf=True)
    namen = {Path(b["pad"]).stem for b in set_["bestanden"]}
    map_ = {(repo_root / b["pad"]).parent for b in set_["bestanden"]}
    in_map = {p.stem for m in map_ for p in m.glob("*.yaml")}
    gelijst = set(await manager.list_ontologies())
    gevonden = {n for n in namen if await manager.get_ontology(n) is not None}
    return {
        "naam": set_["naam"],
        "tabel": set_["tabel"],
        "bron": len(namen),
        "in_map": len(in_map),
        "gelijst": len(gelijst),
        "gevonden": len(gevonden),
        "geslaagd": namen == in_map == gelijst == gevonden,
    }


async def verwerk(config: Dict[str, Any], bronmap: Path, repo_root: Path,
                  repo: Any, rijen: Rijen, manager: Any) -> List[Dict[str, Any]]:
    verslag = []
    for set_ in config["inventaris"]:
        wijze = set_["laadwijze"]
        if wijze not in LAADWIJZEN:
            raise BronFout(f"{set_['naam']}: onbekende laadwijze {wijze!r}")
        if wijze == "tooi_registers":
            verslag.append(await laad_tooi(set_, bronmap, repo, rijen))
        elif wijze == "image":
            verslag.append(await controleer_image(set_, repo_root, manager))
    return verslag


async def _hoofd(args: argparse.Namespace, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    from ontology_manager.manager import get_ontology_manager
    from surrealdb_service.connection import execute_query
    from surrealdb_service.repositories.reference_entity import ReferenceEntityRepository

    async def rijen(bron: str) -> List[Dict[str, Any]]:
        return await execute_query(
            "SELECT id, external_id FROM reference_entity WHERE source_vocabulary = $b",
            {"b": bron},
        ) or []

    return await verwerk(config, Path(args.bronmap), Path(args.repo), ReferenceEntityRepository(),
                         rijen, get_ontology_manager())


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Laadt de referentiegegevens (F2-28).")
    p.add_argument("--config", default=str(CONFIG))
    p.add_argument("--bronmap", default="/tmp/referentiegegevens")
    p.add_argument("--repo", default=str(REPO))
    p.add_argument("--downloadlijst", action="store_true")
    args = p.parse_args(argv)
    config = lees_config(Path(args.config))
    if args.downloadlijst:
        for b in downloadlijst(config):
            print(f"{b['bestand']}\t{b['url']}\t{b['sha256']}")
        return 0

    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    try:
        verslag = asyncio.run(_hoofd(args, config))
    except BronFout as exc:
        print(f"bron: {exc}", file=sys.stderr)
        return 2
    for regel in verslag:
        print(json.dumps(regel, sort_keys=True))
    return 0 if all(r["geslaagd"] for r in verslag) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - scriptfout
        print(f"scriptfout: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(3)
