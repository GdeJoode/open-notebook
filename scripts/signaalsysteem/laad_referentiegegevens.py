"""Laadt de referentiegegevens van het signaalsysteem (F2-28; ADR 0021, 0013).

Leest de inventaris in ``config/referentiegegevens.yaml`` en verwerkt elke set
volgens zijn laadwijze:

* ``tooi_registers``: ``shared.vocabulary.referentiegegevens.laad_tooi``, dezelfde
  code als ``POST /api/vocabulary/refresh``. Controleert de SHA-256 van elk
  register in ``--bronmap``, ontleedt dezelfde bytes en laadt de organisaties in
  ``reference_entity`` (upsert op ``(external_id, source_vocabulary)``, dus
  idempotent). Weigert zonder te schrijven als een geladen organisatie in de bron
  ontbreekt. Het verslag geeft een SHA-256 over de record-id's en een over de
  inhoud zonder ``last_validated``: dat veld zet elke lading opnieuw (het moment
  van de laatste controle tegen de bron); de inhoud verandert bij dezelfde bron niet;
* ``image``: controleert de SHA-256 van elk bestand en of de code het vindt (voor
  de ontologieën: ``OntologyManager.list_ontologies`` en ``get_ontology``, zoals
  de extractie ze opzoekt), en meldt welke ontologieën een actieve versie in
  ``ontology_version`` hebben: die gebruikt de extractie in plaats van het bestand;
* ``gepland`` en ``geen``: niets.

Draait in de container ``open-notebook`` met de verbinding van de migraties. De
beheerstap ``scripts/signaalsysteem/laad_referentiegegevens.sh`` zet de registers
in ``/app/data/referentiegegevens`` en roept dit script via stdin aan:

    docker exec -i <container> /app/.venv/bin/python /app/services/open-notebook/start.py
        /app/.venv/bin/python - --bronmap /app/data/referentiegegevens < laad_referentiegegevens.py

Schrijft per set een verslag als één JSON-regel op stdout.

Exitcodes: 0 elke set klopt (geladen = in de database = aantal in de bron);
1 een aantal klopt niet; 2 bron ontbreekt, SHA-256 wijkt af of de lading zou
geladen organisaties weglaten; 3 scriptfout. ``--downloadlijst`` drukt alleen
``bestand<TAB>url<TAB>sha256`` per te downloaden bestand af (de container heeft
geen internet; de host downloadt).
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
from shared.vocabulary.referentiegegevens import (
    BronOntbreekt,
    MinderGegevens,
    laad_tooi,
)

CONFIG = Path("/app/config/referentiegegevens.yaml")
BRONMAP = Path("/app/data/referentiegegevens")
REPO = Path("/app")
LAADWIJZEN = {"tooi_registers", "image", "gepland", "geen"}
#: De inhoud van een rij voor R4: alles behalve ``id`` en ``last_validated``.
INHOUD = ("canonical_name", "entity_type", "source_vocabulary", "external_uri",
          "external_id", "aliases", "lookup_keys", "properties", "parent_name")


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
        raise BronOntbreekt(f"bronbestand ontbreekt: {pad}")
    werkelijk = sha256(pad, lf)
    if werkelijk != verwacht:
        raise BronOntbreekt(f"SHA-256 van {pad.name} is {werkelijk}, verwacht {verwacht}")


def _sha(regels: List[str]) -> str:
    return hashlib.sha256("\n".join(sorted(regels)).encode()).hexdigest()


Rijen = Callable[[str], Awaitable[List[Dict[str, Any]]]]
Actief = Callable[[], Awaitable[List[str]]]


async def laad_tooi_set(set_: Dict[str, Any], bronmap: Path, repo: Any, rijen: Rijen) -> Dict[str, Any]:
    geladen = await laad_tooi(repo, set_, bronmap)
    rows = await rijen(set_["filter"]["source_vocabulary"])
    codes = {str(r.get("external_id")) for r in rows}
    inhoud = [json.dumps({k: r.get(k) for k in INHOUD}, sort_keys=True, default=str)
              for r in rows]
    return {
        "naam": set_["naam"],
        "tabel": set_["tabel"],
        "bron": geladen["bron"],
        "geladen": geladen["geladen"],
        "in_db": len(rows),
        "ids_sha256": _sha([str(r["id"]) for r in rows]),
        "inhoud_sha256": _sha(inhoud),
        "geslaagd": geladen["geladen"] == geladen["bron"] == len(rows) == len(codes),
    }


async def controleer_image(set_: Dict[str, Any], repo_root: Path, manager: Any,
                           actief: Actief) -> Dict[str, Any]:
    """Elk bestand met de juiste SHA-256, en de registry vindt elke ontologie.

    ``actief`` geeft de namen met een actieve versie in ``ontology_version``: voor
    die ontologieën gebruikt de extractie de opgeslagen versie, niet het bestand;
    de SHA-256 van het bestand zegt dan niets over wat de extractie gebruikt.
    """
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
        "actief_in_db": sorted(set(await actief())),
        "geslaagd": namen == in_map == gelijst == gevonden,
    }


async def verwerk(config: Dict[str, Any], bronmap: Path, repo_root: Path, repo: Any,
                  rijen: Rijen, manager: Any, actief: Actief) -> List[Dict[str, Any]]:
    verslag = []
    for set_ in config["inventaris"]:
        wijze = set_["laadwijze"]
        if wijze not in LAADWIJZEN:
            raise BronOntbreekt(f"{set_['naam']}: onbekende laadwijze {wijze!r}")
        if wijze == "tooi_registers":
            verslag.append(await laad_tooi_set(set_, bronmap, repo, rijen))
        elif wijze == "image":
            verslag.append(await controleer_image(set_, repo_root, manager, actief))
    return verslag


async def _hoofd(args: argparse.Namespace, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    from ontology_manager.manager import get_ontology_manager
    from surrealdb_service.connection import execute_query
    from surrealdb_service.repositories.reference_entity import (
        ReferenceEntityRepository,
    )

    async def rijen(bron: str) -> List[Dict[str, Any]]:
        return await execute_query(
            "SELECT * FROM reference_entity WHERE source_vocabulary = $b", {"b": bron}) or []

    async def actief() -> List[str]:
        rows = await execute_query(
            "SELECT name FROM ontology_version WHERE is_active = true", {}) or []
        return [str(r["name"]) for r in rows]

    return await verwerk(config, Path(args.bronmap), Path(args.repo), ReferenceEntityRepository(),
                         rijen, get_ontology_manager(), actief)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Laadt de referentiegegevens (F2-28).")
    p.add_argument("--config", default=str(CONFIG))
    p.add_argument("--bronmap", default=str(BRONMAP))
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
    except (BronOntbreekt, MinderGegevens) as exc:
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
