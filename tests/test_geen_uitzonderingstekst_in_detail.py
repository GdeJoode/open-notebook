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


#: Sleutels en trefwoorden waaronder tekst in een antwoord terechtkomt: in
#: ``detail``, en in ``error`` en ``message`` (statusantwoorden, SSE).
ANTWOORDSLEUTELS = {"detail", "error", "message"}

#: Aanroepen waarvan de uitkomst geen uitzonderingstekst is, ook al krijgen ze
#: de uitzondering mee: het fout-id, en de telling die ``_write_to_vault`` aan
#: zijn uitzondering hangt.
ZONDER_TEKST = {"log_fout", "nieuw_fout_id", "fout_zonder_inhoud", "_extract_partial_state"}


def _detail_waarden(node: ast.AST):
    """Elke uitdrukking die als ``detail``, ``error`` of ``message`` in een
    antwoord terechtkomt: een trefwoord of een sleutel van een dict."""
    for sub in ast.walk(node):
        if isinstance(sub, ast.Call):
            for kw in sub.keywords:
                if kw.arg in ANTWOORDSLEUTELS:
                    yield sub.lineno, kw.value
        elif isinstance(sub, ast.Dict):
            for key, value in zip(sub.keys, sub.values):
                if isinstance(key, ast.Constant) and key.value in ANTWOORDSLEUTELS:
                    yield sub.lineno, value


def _besmet(stmts: list[ast.stmt], bron: set[str]) -> set[str]:
    """``bron`` plus de lokale namen waaraan iets met die namen is toegewezen,
    tot er niets meer bijkomt (``body = {"error": str(exc)}``, daarna
    ``detail=body``)."""
    namen = set(bron)
    while True:
        erbij = set()
        for stmt in stmts:
            for node in ast.walk(stmt):
                if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and node.value is not None:
                    waarde = node.value
                    if (isinstance(waarde, ast.Call) and isinstance(waarde.func, ast.Name)
                            and waarde.func.id in ZONDER_TEKST):
                        continue
                    if namen & _namen_in(waarde):
                        doelen = node.targets if isinstance(node, ast.Assign) else [node.target]
                        erbij |= {d.id for d in doelen if isinstance(d, ast.Name)}
        if erbij <= namen:
            return namen
        namen |= erbij


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

    # 1. `except ... as e:` met `e`, of een naam die ervan afgeleid is, in een antwoord.
    for handler in ast.walk(boom):
        if isinstance(handler, ast.ExceptHandler) and handler.name:
            besmet = _besmet(handler.body, {handler.name})
            for stmt in handler.body:
                for regel, waarde in _detail_waarden(stmt):
                    if besmet & _namen_in(waarde):
                        fouten.append(f"{rel}:{regel}: '{handler.name}' in een antwoord")

    # 2. Een foutafhandelaar van de app met de uitzondering in een antwoord.
    for func in ast.walk(boom):
        if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)) and _is_exception_handler(func):
            besmet = _besmet(func.body, {a.arg for a in func.args.args[1:]})
            for regel, waarde in _detail_waarden(func):
                if besmet & _namen_in(waarde):
                    fouten.append(f"{rel}:{regel}: uitzondering in een antwoord van {func.name}")
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

def e_via_variabele():
    try:
        pass
    except OSError as exc:
        body = {"error": str(exc)}
        tekst = body
        raise HTTPException(status_code=500, detail=tekst)

def f_status():
    try:
        pass
    except Exception as e:
        return {"status": "offline", "error": str(e)}

async def g_sse():
    try:
        pass
    except Exception as e:
        yield json.dumps({"type": "error", "message": str(e)})

def h_trefwoord():
    try:
        pass
    except Exception as exc:
        return Antwoord(healthy=False, error=str(exc))

def goed_fout_id():
    try:
        pass
    except Exception as e:
        fout_id = log_fout(e, "Mislukt")
        return {"error": melding_met_id("Mislukt", fout_id)}
'''
    pad = ROOT / "tests" / "_tmp_zoektest_detail.py"
    try:
        pad.write_text(bron, encoding="utf-8")
        fouten = _overtredingen(pad)
    finally:
        pad.unlink(missing_ok=True)
    # e_via_variabele telt twee keer: de dict en `detail=tekst`.
    assert len(fouten) == 10, fouten
