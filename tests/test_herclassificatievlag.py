"""De herclassificatievlag staat alleen in de migratie en in ``SourceRepository.reclassify`` (F2-03).

Migratie 89 staat 'intern' naar 'extern' alleen toe als ``$herclassificatie``
waar is. ``execute_query`` en ``execute_transaction`` weigeren een query die
het woord noemt; alleen ``reclassify`` en de migratieloader (die migratie 89
uitvoert) gebruiken de interne route ``_herclassificatie=True``.
Deze test faalt zodra de vlag ergens anders in de code opduikt. Tests zelf
zijn uitgezonderd: die toetsen de vlag juist.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAPPEN = ("apps", "packages", "pipelines", "services", "scripts", "migrations")
VLAG = re.compile(r"\$herclassificatie\b|\b_herclassificatie\b|[\"']herclassificatie[\"']", re.I)

CONNECTION = "packages/surrealdb-service/src/surrealdb_service/connection.py"
SOURCE_REPO = "packages/surrealdb-service/src/surrealdb_service/repositories/source.py"
MIGRATIE = "migrations/89.surrealql"
MIGRATIELOADER = "packages/surrealdb-service/src/surrealdb_service/migrations.py"


def _bestanden():
    for map_ in MAPPEN:
        for pad in (ROOT / map_).rglob("*"):
            if pad.suffix not in (".py", ".surrealql") or not pad.is_file():
                continue
            rel = pad.relative_to(ROOT).as_posix()
            if "/tests/" in f"/{rel}" or "node_modules" in rel or "/.venv/" in f"/{rel}":
                continue
            yield rel, pad


def test_de_vlag_staat_alleen_op_de_toegestane_plekken() -> None:
    gevonden = {
        rel for rel, pad in _bestanden()
        if VLAG.search(pad.read_text(encoding="utf-8", errors="ignore"))
    }
    assert gevonden == {CONNECTION, SOURCE_REPO, MIGRATIE, MIGRATIELOADER}


def test_de_migratieloader_gebruikt_de_route_alleen_voor_de_up_migratie() -> None:
    bron = (ROOT / MIGRATIELOADER).read_text(encoding="utf-8")
    regels = [r for r in bron.splitlines() if VLAG.search(r) and not r.lstrip().startswith("#")]
    assert regels == ["                self.up[version].sql, config=self.config, _herclassificatie=True"]


def test_in_de_repository_alleen_in_reclassify() -> None:
    bron = (ROOT / SOURCE_REPO).read_text(encoding="utf-8")
    boom = ast.parse(bron)
    toegestaan: set[int] = set()
    for knoop in ast.walk(boom):
        if isinstance(knoop, ast.Assign) and any(
            isinstance(d, ast.Name) and d.id == "_HERCLASSIFICATIE" for d in knoop.targets
        ):
            toegestaan |= set(range(knoop.lineno, knoop.end_lineno + 1))
        if isinstance(knoop, ast.AsyncFunctionDef) and knoop.name == "reclassify":
            toegestaan |= set(range(knoop.lineno, knoop.end_lineno + 1))
    regels = [
        n for n, regel in enumerate(bron.splitlines(), start=1)
        if VLAG.search(regel) and not regel.lstrip().startswith("#")
    ]
    assert regels and set(regels) <= toegestaan, regels
    # De vlag als query-tekst staat alleen in _HERCLASSIFICATIE, en die query
    # wordt alleen in reclassify gebruikt.
    gebruik = [
        knoop.lineno for knoop in ast.walk(boom)
        if isinstance(knoop, ast.Name) and knoop.id == "_HERCLASSIFICATIE"
        and isinstance(knoop.ctx, ast.Load)
    ]
    assert gebruik and set(gebruik) <= toegestaan
