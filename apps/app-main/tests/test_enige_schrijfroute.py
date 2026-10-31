"""Eén inname: alleen het importeindpunt schrijft bronnen en fragmenten (F2-14, ADR 0010).

Eerste versie, aangescherpt na de review van F2-14; F2-29 breidt de test uit
naar wijzigen en verwijderen.

De test leest de syntaxisboom van alle code in ``app_main`` en zoekt plekken
die een ``source``, ``source_version``, ``chunk``, ``doc_node`` of
``source_embedding`` aanmaken:

* een aanroep ``create``, ``upsert``, ``insert`` of ``bulk_create`` op een
  ontvanger met ``chunk`` of ``source`` in de naam;
* ``schrijf_import``, ``add_embedding``, ``register`` van een versie, en
  ``DocGraphBuilder`` (aanmaken of ``build``);
* SurrealQL met ``CREATE``/``INSERT INTO``/``INSERT RELATION INTO``/``UPSERT``
  op die tabellen, en ``CREATE $`` op een record-variabele.

Elke plek moet op de lijst hieronder staan, met de taak die haar weghaalt of
ombouwt. Een nieuwe plek laat de test falen, net als een plek op de lijst die
niet meer bestaat (dan gaat hij van de lijst).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest
from app_main.api.app import create_app
from fastapi.routing import APIRoute

REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "apps" / "app-main" / "src" / "app_main"
#: De code van de fork die schrijft: de app en de pipelines die zij aanroept.
BRONMAPPEN = (ROOT, *sorted((REPO / "pipelines").glob("*/src")))

#: (bestand, functie) -> waarom de plek mag schrijven.
_APP = "apps/app-main/src/app_main/"
TOEGESTAAN = {
    (_APP + "services/import_service.py", "ImportService._nieuw"): "het importeindpunt (F2-14)",
    (_APP + "services/chunking/chunk_mutator.py", "<module>"):
        "handmatige run (F2-06, ADR 0005)",
}
#: Tijdelijke uitzonderingen, elk met de taak die de plek weghaalt of ombouwt.
TIJDELIJK = {
    # De ingebouwde inname (ADR 0010).
    (_APP + "api/routers/sources_upload.py", "_create_source_impl"): "F2-19",
    (_APP + "services/source_service.py", "SourceService.create"): "F2-19",
    (_APP + "services/source_processor.py", "SourceProcessor.__init__"): "F2-19",
    (_APP + "services/source_processor.py", "SourceProcessor.process_source"): "F2-19",
    (_APP + "services/source_processor.py", "SourceProcessor._build_structure_graph"): "F2-19",
    (_APP + "services/okf_import_service.py", "OkfImportService._import_source"): "F2-19",
    (_APP + "services/okf_import_service.py", "OkfImportService._create_import_source"):
        "F2-19",
    # De structuurgraaf buiten de import: bouwer en zijn dependency (ADR 0005).
    (_APP + "services/graph/doc_graph_builder.py", "DocGraphBuilder._insert_nodes"): "F2-29",
    (_APP + "dependencies.py", "get_doc_graph_builder"): "F2-29",
    # Embeddings: de stap ``embed`` met run en model (ADR 0011).
    (_APP + "services/source_service.py", "SourceService.add_embedding"): "F2-16",
    ("pipelines/embeddings/src/embeddings/service.py", "EmbeddingService._embed_chunks"): "F2-16",
}

_TABELLEN = r"(?:source|source_version|chunk|doc_node|source_embedding)\b"
_SQL = re.compile(
    r"\b(?:(?:CREATE(?:\s+ONLY)?|INSERT\s+(?:RELATION\s+)?INTO|UPSERT)\s+" + _TABELLEN
    + r"|CREATE(?:\s+ONLY)?\s+\$)",
    re.IGNORECASE,
)
_SCHRIJVEND = {"create", "upsert", "insert", "bulk_create"}


def _naam(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _naam(node.func)
    return ""


def _is_schrijfaanroep(node: ast.Call) -> bool:
    doel = node.func
    if isinstance(doel, ast.Name):
        return doel.id == "DocGraphBuilder"
    if not isinstance(doel, ast.Attribute):
        return False
    if doel.attr in ("schrijf_import", "add_embedding"):
        return True
    ontvanger = _naam(doel.value).lower()
    if doel.attr == "register" and "version" in ontvanger:
        return True
    if doel.attr == "build" and ("graph_builder" in ontvanger or "docgraphbuilder" in ontvanger):
        return True
    return doel.attr in _SCHRIJVEND and ("chunk" in ontvanger or "source" in ontvanger)


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
    for map_ in BRONMAPPEN:
        for pad in sorted(map_.rglob("*.py")):
            rel = pad.relative_to(REPO).as_posix()
            for plek in schrijfplekken(pad.read_text(encoding="utf-8")):
                uit.add((rel, plek))
    return uit


def test_alleen_toegestane_plekken_schrijven_bronnen_en_fragmenten() -> None:
    gevonden = _alle_plekken()
    nieuw = sorted(gevonden - TOEGESTAAN.keys() - TIJDELIJK.keys())
    assert not nieuw, "Schrijfplek buiten het importeindpunt:\n" + "\n".join(map(str, nieuw))


def test_lijst_bevat_geen_vervallen_plekken() -> None:
    weg = sorted((TOEGESTAAN.keys() | TIJDELIJK.keys()) - _alle_plekken())
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
    ("def n(s):\n    return s.repo.add_embedding(1)", {"n"}),
    ("def o():\n    return DocGraphBuilder()", {"o"}),
    ("def q(s):\n    return s.graph_builder.build('x')", {"q"}),
    ("def r(s):\n    return s.chunks.upsert({})", {"r"}),
    ("def t(s):\n    return s.source_store.insert([])", {"t"}),
    ("Q = 'CREATE $bron CONTENT $d'", {"<module>"}),
    ("Q = 'INSERT INTO doc_node $n'", {"<module>"}),
    ("Q = 'INSERT RELATION INTO derived_from $r; CREATE source_embedding'", {"<module>"}),
])
def test_de_test_vindt_een_schrijfplek(bron, verwacht) -> None:
    assert schrijfplekken(bron) == verwacht
