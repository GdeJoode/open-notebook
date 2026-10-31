"""Geen uitzonderingstekst in een foutmelding van de API (F2-05, ADR 0002 C).

Een uitzondering kan tekst uit een intern record bevatten. De routers en de
foutafhandelaars in ``apps/app-main/src/app_main/api/`` zetten daarom nooit de
uitzondering zelf in ``detail``: niet als ``str(e)``, niet als ``repr(e)``,
niet in een f-string en niet als ``{"detail": ...}`` in een ``JSONResponse``.
In de plaats komt een vaste melding met een fout-id (``api/fouten.py``).

De test zoekt in de syntaxisboom naar een verwijzing naar de uitzondering
binnen ``detail``: de naam uit ``except ... as e``, of de uitzondering die een
``@app.exception_handler`` als argument krijgt.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
API = ROOT / "apps" / "app-main" / "src" / "app_main" / "api"


def _namen_in(node: ast.AST) -> set[str]:
    """De namen in ``node``, behalve als ``<naam>.detail``.

    ``exc.detail`` is de vaste melding die de code zelf in een
    ``HTTPException`` of een domeinfout zette (zoals ``OntologyEditError``),
    geen tekst van de uitzondering: dat mag door.
    """
    toegestaan = {
        id(n.value)
        for n in ast.walk(node)
        if isinstance(n, ast.Attribute) and n.attr == "detail" and isinstance(n.value, ast.Name)
    }
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and id(n) not in toegestaan}


def _detail_waarden(node: ast.AST):
    """Elke uitdrukking die als ``detail`` in een antwoord terechtkomt."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            for kw in sub.keywords:
                if kw.arg == "detail":
                    yield sub.lineno, kw.value
        elif isinstance(sub, ast.Dict):
            for key, value in zip(sub.keys, sub.values):
                if isinstance(key, ast.Constant) and key.value == "detail":
                    yield sub.lineno, value


def _is_exception_handler(func: ast.AST) -> bool:
    for dec in getattr(func, "decorator_list", []):
        doel = dec.func if isinstance(dec, ast.Call) else dec
        if isinstance(doel, ast.Attribute) and doel.attr == "exception_handler":
            return True
    return False


def _overtredingen(pad: Path) -> list[str]:
    boom = ast.parse(pad.read_text(encoding="utf-8"))
    rel = pad.relative_to(ROOT).as_posix()
    fouten: list[str] = []

    # 1. `except ... as e:` met `e` in detail.
    for handler in ast.walk(boom):
        if isinstance(handler, ast.ExceptHandler) and handler.name:
            for stmt in handler.body:
                for regel, waarde in _detail_waarden(stmt):
                    if handler.name in _namen_in(waarde):
                        fouten.append(f"{rel}:{regel}: '{handler.name}' in detail")

    # 2. Een foutafhandelaar van de app met de uitzondering in detail.
    for func in ast.walk(boom):
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_exception_handler(func):
            uitzondering = {a.arg for a in func.args.args[1:]}
            for regel, waarde in _detail_waarden(func):
                if uitzondering & _namen_in(waarde):
                    fouten.append(f"{rel}:{regel}: uitzondering in detail van {func.name}")
    return fouten


def test_geen_uitzonderingstekst_in_detail() -> None:
    bestanden = sorted(API.rglob("*.py"))
    assert bestanden, f"geen bestanden onder {API}"
    fouten = [f for pad in bestanden for f in _overtredingen(pad)]
    assert not fouten, "Uitzonderingstekst in een foutmelding:\n" + "\n".join(fouten)


def test_de_zoektest_vindt_de_bekende_vormen(tmp_path: Path) -> None:
    """De test zelf: elke verboden vorm wordt gevonden, een vaste melding niet."""
    bron = '''
from fastapi import HTTPException
from fastapi.responses import JSONResponse

def a():
    try:
        pass
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

def b():
    try:
        pass
    except Exception as exc:
        raise HTTPException(500, detail=f"mislukt: {exc}")

def c():
    try:
        pass
    except Exception as err:
        raise HTTPException(500, detail=repr(err))

def d():
    try:
        pass
    except Exception as e:
        return JSONResponse(status_code=500, content={"detail": str(e)})

@app.exception_handler(ValueError)
async def h(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})

def goed():
    try:
        pass
    except Exception as e:
        raise HTTPException(500, detail="Interne fout (fout-id x)")

def goed_eigen_melding():
    try:
        pass
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
'''
    pad = ROOT / "tests" / "_tmp_zoektest_detail.py"
    try:
        pad.write_text(bron, encoding="utf-8")
        fouten = _overtredingen(pad)
    finally:
        pad.unlink(missing_ok=True)
    assert len(fouten) == 5, fouten
