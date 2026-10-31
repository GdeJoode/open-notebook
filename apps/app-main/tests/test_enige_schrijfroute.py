"""Eén inname: alleen het importeindpunt schrijft bronnen en fragmenten (F2-14, ADR 0010).

Eerste versie; F2-29 breidt de test uit naar wijzigen en verwijderen en naar
``source_embedding``, ``doc_node`` en ``source_version``.

De test leest de syntaxisboom van alle code in ``app_main`` en zoekt plekken
die een ``source``, ``source_version`` of ``chunk`` aanmaken: aanroepen van de
schrijfmethoden van de repositories (``bulk_create``, ``schrijf_import``,
``register`` van een versie, ``create`` van een bronrepository of -dienst) en
SurrealQL met ``CREATE``/``INSERT INTO``/``UPSERT`` op die tabellen. Elke
plek moet op de lijst hieronder staan: het importeindpunt, de handmatige run
uit F2-06, en de ingebouwde inname die met F2-19 vervalt (ADR 0010). Een
nieuwe plek laat de test falen, net als een plek op de lijst die niet meer
bestaat (dan gaat hij van de lijst).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from app_main.api.app import create_app
from fastapi.routing import APIRoute

ROOT = Path(__file__).resolve().parents[1] / "src" / "app_main"

#: (bestand, functie) -> waarom de plek mag schrijven.
TOEGESTAAN = {
    ("services/import_service.py", "ImportService._nieuw"): "het importeindpunt (F2-14)",
    ("services/chunking/chunk_mutator.py", "<module>"): "handmatige run (F2-06, ADR 0005)",
    # De ingebouwde inname; vervalt met F2-19 (ADR 0010).
    ("api/routers/sources_upload.py", "_create_source_impl"): "vervalt in F2-19",
    ("services/source_service.py", "SourceService.create"): "vervalt in F2-19",
    ("services/source_processor.py", "SourceProcessor.process_source"): "vervalt in F2-19",
    ("services/okf_import_service.py", "OkfImportService._import_source"): "vervalt in F2-19",
    ("services/okf_import_service.py", "OkfImportService._create_import_source"):
        "vervalt in F2-19",
}

_SQL = re.compile(r"\b(?:CREATE(?:\s+ONLY)?|INSERT\s+INTO|UPSERT)\s+(?:source|source_version|chunk)\b",
                  re.IGNORECASE)
_BRON_SCHRIJVERS = re.compile(r"^_?source_(?:svc|service|repo)$")


def _is_schrijfaanroep(node: ast.Call) -> bool:
    doel = node.func
    if not isinstance(doel, ast.Attribute):
        return False
    if doel.attr in ("bulk_create", "schrijf_import"):
        return True
    ontvanger = doel.value
    naam = ontvanger.id if isinstance(ontvanger, ast.Name) else (
        ontvanger.attr if isinstance(ontvanger, ast.Attribute) else None)
    if isinstance(ontvanger, ast.Call) and isinstance(ontvanger.func, ast.Name):
        naam = ontvanger.func.id
    if doel.attr == "register" and naam and "version" in naam.lower():
        return True
    return doel.attr == "create" and bool(naam) and (
        bool(_BRON_SCHRIJVERS.match(naam)) or naam in ("SourceRepository", "SourceService"))


def schrijfplekken(bron: str) -> set[str]:
    """De functies (``Klasse.methode`` of ``<module>``) die een bron of fragment maken."""
    boom = ast.parse(bron)
    plekken: set[str] = set()

    def loop(node: ast.AST, pad: list[str]) -> None:
        for kind in ast.iter_child_nodes(node):
            nieuw = pad
            if isinstance(kind, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                nieuw = [*pad, kind.name]
            plek = ".".join(nieuw) or "<module>"
            if isinstance(kind, ast.Call) and _is_schrijfaanroep(kind):
                plekken.add(plek)
            elif isinstance(kind, ast.Constant) and isinstance(kind.value, str) and _SQL.search(
                kind.value
            ):
                plekken.add(plek)
            loop(kind, nieuw)

    loop(boom, [])
    return plekken


def _alle_plekken() -> set[tuple[str, str]]:
    uit = set()
    for pad in sorted(ROOT.rglob("*.py")):
        rel = pad.relative_to(ROOT).as_posix()
        for plek in schrijfplekken(pad.read_text(encoding="utf-8")):
            uit.add((rel, plek))
    return uit


def test_alleen_toegestane_plekken_schrijven_bronnen_en_fragmenten() -> None:
    gevonden = _alle_plekken()
    nieuw = sorted(gevonden - TOEGESTAAN.keys())
    assert not nieuw, "Schrijfplek buiten het importeindpunt:\n" + "\n".join(map(str, nieuw))


def test_lijst_bevat_geen_vervallen_plekken() -> None:
    weg = sorted(TOEGESTAAN.keys() - _alle_plekken())
    assert not weg, "Niet meer gevonden; haal van de lijst:\n" + "\n".join(map(str, weg))


def test_importeindpunt_is_de_route_naar_de_importdienst() -> None:
    routes = [r for r in create_app().routes if isinstance(r, APIRoute)
              and r.endpoint.__module__ == "app_main.api.routers.imports"]
    assert [(r.path, sorted(r.methods)) for r in routes] == [("/api/v1/import", ["POST"])]
    assert "service.importeer(" in Path(
        ROOT / "api" / "routers" / "imports.py").read_text(encoding="utf-8")


@pytest.mark.parametrize("bron, verwacht", [
    ("async def f(r):\n    await r.chunk_repo.bulk_create([])", {"f"}),
    ("class A:\n    async def g(self):\n        await self.source_repo.create({})", {"A.g"}),
    ("Q = 'INSERT INTO chunk $x'", {"<module>"}),
    ("def h():\n    return 'CREATE ONLY source CONTENT $d'", {"h"}),
    ("def i():\n    return SourceRepository().create({})", {"i"}),
    ("def j(v):\n    return v.version_repo.register('s', 'h', 'r')", {"j"}),
    ("def k():\n    return 'CREATE source_map_rule CONTENT $d; CREATE chunk_edit'", set()),
    ("def m(s):\n    return s.notebook_repo.create({})", set()),
])
def test_de_test_vindt_een_schrijfplek(bron, verwacht) -> None:
    assert schrijfplekken(bron) == verwacht
