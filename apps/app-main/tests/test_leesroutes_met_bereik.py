"""Routes voor ``lezen_extern`` lezen alleen met classificatiebereik (F2-05, ADR 0002 C).

Een handler van zo'n route, en elke functie of dienstmethode die hij aanroept,
mag geen lezer zonder bereik gebruiken: ``get``, ``get_all``,
``get_or_raise``, ``query`` of ``execute_query`` (de algemene leesmethoden van
de basisrepository en de ruwe query). Alleen de leesmethoden met een verplicht
``scope`` horen daar. De test leest de syntaxisboom van de handler en, één
niveau dieper, van de functies en dienstmethoden die hij aanroept.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
import typing

import pytest
from fastapi.routing import APIRoute
from shared.models.agents import Right

from app_main.api.app import create_app

VERBODEN = {"get", "get_all", "get_or_raise", "query", "execute_query"}


def _routes_voor_lezen_extern(app) -> list[APIRoute]:
    uit = []
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        rechten = []

        def loop(dependant) -> None:
            for dep in dependant.dependencies:
                if getattr(dep.call, "required_right", None) is not None:
                    rechten.append(dep.call.required_right)
                loop(dep)

        loop(route.dependant)
        if rechten == [Right.LEZEN_EXTERN]:
            uit.append(route)
    return uit


def _boom(func) -> ast.AST:
    return ast.parse(textwrap.dedent(inspect.getsource(func)))


def _verboden_aanroepen(boom: ast.AST) -> list[str]:
    """Elke ``await``-aanroep van een lezer zonder bereik."""
    fout = []
    for node in ast.walk(boom):
        if isinstance(node, ast.Await) and isinstance(node.value, ast.Call):
            doel = node.value.func
            naam = doel.attr if isinstance(doel, ast.Attribute) else getattr(doel, "id", None)
            if naam in VERBODEN:
                fout.append(f"{naam} (regel {node.lineno})")
    return fout


def _aangeroepen(func, boom: ast.AST) -> list:
    """De functies en dienstmethoden die ``func`` met ``await`` aanroept."""
    namen = func.__globals__
    hints = typing.get_type_hints(func) if hasattr(func, "__annotations__") else {}
    uit = []
    for node in ast.walk(boom):
        if not (isinstance(node, ast.Await) and isinstance(node.value, ast.Call)):
            continue
        doel = node.value.func
        if isinstance(doel, ast.Name) and inspect.isfunction(namen.get(doel.id)):
            uit.append(namen[doel.id])
        elif isinstance(doel, ast.Attribute) and isinstance(doel.value, ast.Name):
            klasse = hints.get(doel.value.id)
            methode = getattr(klasse, doel.attr, None) if inspect.isclass(klasse) else None
            if inspect.isfunction(methode):
                uit.append(methode)
    return uit


def test_er_zijn_routes_voor_lezen_extern() -> None:
    assert len(_routes_voor_lezen_extern(create_app())) >= 9


@pytest.mark.parametrize(
    "route", _routes_voor_lezen_extern(create_app()), ids=lambda r: f"{sorted(r.methods)[0]} {r.path}"
)
def test_handler_leest_alleen_met_bereik(route: APIRoute) -> None:
    handler = route.endpoint
    boom = _boom(handler)
    fouten = [f"{handler.__qualname__}: {f}" for f in _verboden_aanroepen(boom)]
    for func in _aangeroepen(handler, boom):
        fouten += [f"{func.__qualname__}: {f}" for f in _verboden_aanroepen(_boom(func))]
    assert not fouten, "Lezer zonder bereik op een route voor lezen_extern:\n" + "\n".join(fouten)


def test_de_test_vindt_een_lezer_zonder_bereik() -> None:
    bron = """
async def handler(svc):
    a = await svc.get_scoped("x", scope=s)
    b = await svc.get("x")
    c = await execute_query("SELECT 1")
    d = {}.get("x")
"""
    assert _verboden_aanroepen(ast.parse(bron)) == ["get (regel 4)", "execute_query (regel 5)"]


def test_de_test_volgt_dienstmethoden_en_functies() -> None:
    route = next(r for r in _routes_voor_lezen_extern(create_app())
                 if r.path == "/api/sources/" and "GET" in r.methods)
    namen = {f.__qualname__ for f in _aangeroepen(route.endpoint, _boom(route.endpoint))}
    assert "stopped_sources" in namen
    route = next(r for r in _routes_voor_lezen_extern(create_app())
                 if r.path == "/api/sources/{source_id}")
    namen = {f.__qualname__ for f in _aangeroepen(route.endpoint, _boom(route.endpoint))}
    assert {"SourceService.get_scoped", "SourceService.get_text"} <= namen
