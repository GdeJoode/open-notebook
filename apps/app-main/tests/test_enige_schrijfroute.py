"""Eén inname: alleen het importeindpunt schrijft bronnen en fragmenten (F2-14, ADR 0010).

Eerste versie, aangescherpt na de review en herreview van F2-14; F2-29 breidt
de test uit naar wijzigen en verwijderen. Gescand: ``app_main``, ``pipelines/*/src`` en
``packages/*/src``.

De test leest de syntaxisboom van alle code in die mappen en zoekt plekken
die een ``source``, ``source_version``, ``chunk``, ``doc_node`` of
``source_embedding``, ``chunk_extern`` of een structuurrelatie aanmaken:

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
#: De code van de fork die schrijft: de app, de pipelines en de pakketten
#: (repositories) die zij aanroepen.
BRONMAPPEN = (
    ROOT,
    *sorted((REPO / "pipelines").glob("*/src")),
    *sorted((REPO / "packages").glob("*/src")),
)

#: (bestand, functie) -> waarom de plek mag schrijven.
_APP = "apps/app-main/src/app_main/"
_REPO = "packages/surrealdb-service/src/surrealdb_service/repositories/"
TOEGESTAAN = {
    (_APP + "services/import_service.py", "ImportService._nieuw"): "het importeindpunt (F2-14)",
    (_REPO + "source.py", "SourceRepository.schrijf_import"): "alleen voor ImportService",
    (_REPO + "source_version.py", "<module>"): "NIEUWE_VERSIE: import en register",
    (_APP + "services/chunking/chunk_mutator.py", "<module>"):
        "handmatige run (F2-06, ADR 0005)",
    # Algemene schrijvers voor elke tabel: een aanroep voor een bron of
    # fragment vindt de test bij de aanroeper (naam van de ontvanger).
    (_REPO + "base.py", "BaseRepository.create"): "algemeen; aanroeper apart getoetst",
    (_REPO + "base.py", "BaseRepository.upsert"): "algemeen; aanroeper apart getoetst",
    (_REPO + "base.py", "RecordRepository.update"): "algemeen; aanroeper apart getoetst",
    (_REPO + "base.py", "wijzig_met_rev"): "algemeen; ``aanmaken`` apart getoetst",
    (_REPO + "source_map.py", "maak_met_log"): "bronkaartregel of batch met logregel",
    ("packages/surrealdb-service/src/surrealdb_service/testing/herkomst.py", "herkomst"):
        "alleen voor tests",
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
    (_REPO + "source.py", "ChunkRepository.bulk_create"): "F2-19",
    # De structuurgraaf buiten de import: bouwer en zijn dependency (ADR 0005).
    (_APP + "services/graph/doc_graph_builder.py", "DocGraphBuilder._insert_nodes"): "F2-29",
    (_APP + "dependencies.py", "get_doc_graph_builder"): "F2-29",
    # Embeddings: de stap ``embed`` met run en model (ADR 0011).
    (_APP + "services/source_service.py", "SourceService.add_embedding"): "F2-16",
    (_REPO + "source.py", "SourceRepository.add_embedding"): "F2-16",
    ("pipelines/embeddings/src/embeddings/service.py", "EmbeddingService._embed_chunks"): "F2-16",
}

_TABELLEN = (r"(?:source|source_version|chunk|chunk_extern|doc_node|source_embedding"
             r"|derived_from|parent_of|next_node)\b")
#: Het doel van een schrijvend werkwoord: een van de tabellen, een record-
#: variabele, ``type::thing``/``type::table``/``type::record`` met een
#: tabelparameter of een van de tabellen, of een ingevuld deel van een
#: f-string of aaneengeschakelde tekst (``{}``).
_DOEL = (r"(?:" + _TABELLEN + r"|\$|\{\}|type::(?:thing|table|record)\s*\(\s*"
         r"(?:\$|\{\}|['\"]" + _TABELLEN + r"))")
_SQL = re.compile(
    r"\b(?:CREATE|UPSERT)(?:\s+ONLY)?\s+" + _DOEL
    + r"|\bINSERT(?:\s+(?:IGNORE|RELATION))*\s+INTO\s+" + _DOEL
    + r"|\bRELATE\b[^;]*->\s*(?:derived_from|parent_of|next_node|\{\})\b",
    re.IGNORECASE,
)
_SCHRIJVEND = {"create", "upsert", "insert", "bulk_create"}
_SCHRIJFTABELLEN = {"source", "source_version", "chunk", "chunk_extern", "doc_node",
                    "source_embedding", "derived_from", "parent_of", "next_node"}


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
        if doel.id == "DocGraphBuilder":
            return True
    elif isinstance(doel, ast.Attribute):
        if doel.attr in ("schrijf_import", "add_embedding"):
            return True
        ontvanger = _naam(doel.value).lower()
        if doel.attr == "register" and "version" in ontvanger:
            return True
        if doel.attr == "build" and ("graph_builder" in ontvanger
                                     or "docgraphbuilder" in ontvanger):
            return True
        if doel.attr in _SCHRIJVEND and ("chunk" in ontvanger or "source" in ontvanger):
            return True
    # ``wijzig_met_rev(..., aanmaken=[("chunk", {...})])``
    for kw in node.keywords:
        if kw.arg == "aanmaken":
            for sub in ast.walk(kw.value):
                if (isinstance(sub, ast.Tuple) and sub.elts
                        and isinstance(sub.elts[0], ast.Constant)
                        and sub.elts[0].value in _SCHRIJFTABELLEN):
                    return True
    return False


def _tekst_van(node: ast.AST) -> str | None:
    """De SQL-tekst van een constante, f-string of aaneengeschakelde tekst; een
    ingevuld deel wordt ``{}``."""
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.JoinedStr):
        return "".join(
            v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}"
            for v in node.values)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        links, rechts = _tekst_van(node.left), _tekst_van(node.right)
        if links is None and rechts is None:
            return None
        return (links if links is not None else "{}") + (rechts if rechts is not None else "{}")
    return None


def schrijfplekken(bron: str) -> set[str]:
    """De functies (``Klasse.methode`` of ``<module>``) die een bron, fragment,
    structuur of embedding maken."""
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
            tekst = _tekst_van(kind)
            if tekst is not None and _SQL.search(tekst):
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
    # Omzeilvormen uit de herreview van F2-14.
    ("Q = 'INSERT IGNORE INTO chunk $x'", {"<module>"}),
    ("Q = \"CREATE type::thing('chunk', $id) CONTENT $d\"", {"<module>"}),
    ("Q = \"CREATE type::table($t) CONTENT $d\"", {"<module>"}),
    ("def g(t):\n    return f'CREATE {t} CONTENT $d'", {"g"}),
    ("Q = 'CREATE ' + 'chunk CONTENT $d'", {"<module>"}),
    ("async def h():\n    await wijzig_met_rev([], door='x', aanmaken=[('chunk', {})])", {"h"}),
    ("Q = 'UPSERT ONLY chunk:1 CONTENT $d'", {"<module>"}),
    ("Q = 'INSERT RELATION IGNORE INTO derived_from $r'", {"<module>"}),
    ("Q = 'RELATE $a->derived_from->$b'", {"<module>"}),
    ("Q = 'INSERT INTO chunk_extern $x'", {"<module>"}),
    # Een repository-methode met een neutrale naam: de SQL erin wordt gevonden
    # (pakketten worden ook gescand), dus ze moet op de lijst.
    ("class R:\n    async def bewaar_fragmenten(self, x):\n"
     "        await execute_query('INSERT INTO chunk $x', {'x': x})", {"R.bewaar_fragmenten"}),
    ("Q = \"UPSERT type::thing('ontology_draft', $n) SET x = 1\"", set()),
    ("def u(a, b):\n    return wijzig_met_rev([], door='x', aanmaken=[('classification_log', {})])",
     set()),
])
def test_de_test_vindt_een_schrijfplek(bron, verwacht) -> None:
    assert schrijfplekken(bron) == verwacht
