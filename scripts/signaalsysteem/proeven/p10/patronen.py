"""P10, 10a: statische telling van SurrealQL-patronen uit de migratiegids 2.x naar 3.x.

Gebruik: python patronen.py <kopie> <uit.csv>  (schrijft een samenvatting als JSON naar stdout)

Telt per patroon de treffers in migrations/*.surrealql en in de Python-code onder
apps/, packages/ en pipelines/. Bron van de patronen:
https://surrealdb.com/docs/build/migrating/from-old-surrealdb-versions/2x-to-3x
(geraadpleegd 30-10-2026). "mechanisch" zijn de vervangingen die vervang.py doet
(ADR 0021: functienamen, SEARCH ANALYZER, LET); de rest vraagt beoordeling.

Patronen met "heuristisch" in de soort tellen in Python alleen op regels die op
SurrealQL lijken (SELECT, FROM, WHERE, RETURN, UPDATE, ->, <-); ze kunnen ruis bevatten.
"""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

# naam, soort, regex, alleen_op_surrealql_regels_in_python
PATRONEN: list[tuple[str, str, str, bool]] = [
    ("search_analyzer", "mechanisch", r"\bSEARCH\s+ANALYZER\b", False),
    ("type_thing", "mechanisch", r"\btype::thing\s*\(", False),
    ("type_is", "mechanisch", r"\btype::is::\w+", False),
    ("string_is", "mechanisch", r"\bstring::is::\w+", False),
    ("time_from_is", "mechanisch", r"\btime::(?:from|is)::\w+", False),
    ("duration_from", "mechanisch", r"\bduration::from::\w+", False),
    ("geo_is", "mechanisch", r"\bgeo::is::\w+", False),
    ("osa_distance", "mechanisch", r"\bstring::distance::osa_distance\b", False),
    ("rand_guid", "mechanisch", r"\brand::guid\s*\(", False),
    ("let_ontbreekt", "mechanisch", r"(?:^|[;{])\s*\$\w+\s*=(?![=~])", False),
    ("mtree", "niet-mechanisch", r"\bMTREE\b", False),
    ("future", "niet-mechanisch", r"<future>", False),
    ("like_operator", "niet-mechanisch (heuristisch)", r"\s(?:!~|\?~|\*~|~)\s", True),
    ("group_met_split", "niet-mechanisch", r"\bSPLIT\b", True),
    ("all_idioom", "niet-mechanisch (heuristisch)", r"[\w\])]\.\*(?=[.\s,;)]|$)", True),
    ("optionele_keten", "niet-mechanisch (heuristisch)", r"[\w\])]\?\.(?=\w)", True),
    ("array_range", "niet-mechanisch", r"\barray::range\s*\(", False),
    ("analyze_statement", "niet-mechanisch", r"^\s*ANALYZE\b", True),
    ("set_type", "niet-mechanisch", r"\bTYPE\s+(?:option<)?set\b", False),
]

SURREALQL_REGEL = re.compile(r"\b(SELECT|FROM|WHERE|RETURN|UPDATE|DEFINE)\b|->|<-")


def migratie_tekst(pad: Path) -> list[str]:
    """Regels zonder commentaarregels, zoals AsyncMigration.from_file ze leest."""
    regels = []
    for regel in pad.read_text(encoding="utf-8").split("\n"):
        s = regel.strip()
        regels.append("" if s.startswith("--") else regel)
    return regels


def soort_bestand(rel: str) -> str:
    if rel.startswith("migrations/"):
        return "migratie"
    naam = rel.rsplit("/", 1)[-1]
    if "/tests/" in rel or naam.startswith("test_") or naam == "conftest.py":
        return "test"
    return "broncode"


def bestanden(kopie: Path):
    for p in sorted((kopie / "migrations").glob("*.surrealql")):
        yield p, migratie_tekst(p), True
    for map_ in ("apps", "packages", "pipelines"):
        for p in sorted((kopie / map_).rglob("*.py")):
            if any(d in p.parts for d in (".venv", "node_modules", "__pycache__")):
                continue
            yield p, p.read_text(encoding="utf-8", errors="replace").split("\n"), False


def main() -> int:
    kopie, uit = Path(sys.argv[1]), Path(sys.argv[2])
    gecompileerd = [(n, s, re.compile(r, re.MULTILINE), alleen) for n, s, r, alleen in PATRONEN]
    rijen = []
    for pad, regels, is_migratie in bestanden(kopie):
        rel = pad.relative_to(kopie).as_posix()
        for naam, soort, rx, alleen_sql in gecompileerd:
            treffers, eerste = 0, None
            for nr, regel in enumerate(regels, 1):
                if not is_migratie and alleen_sql and not SURREALQL_REGEL.search(regel):
                    continue
                n = len(rx.findall(regel))
                if n:
                    treffers += n
                    eerste = eerste or nr
            if treffers:
                rijen.append([naam, soort, soort_bestand(rel), rel, treffers, eerste])
    with uit.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["patroon", "soort", "bestandssoort", "bestand", "treffers", "eerste_regel"])
        w.writerows(rijen)
    samenvatting: dict[str, dict] = {}
    for naam, soort, *_ in PATRONEN:
        r = [x for x in rijen if x[0] == naam]
        per = {}
        for bs in ("migratie", "broncode", "test"):
            rb = [x for x in r if x[2] == bs]
            per[bs] = {"bestanden": len(rb), "treffers": sum(x[4] for x in rb)}
        samenvatting[naam] = {"soort": soort, **per}
    # Telling zoals ADR 0021: type::thing, type::is::*, string::is::* in Python.
    adr = [x for x in rijen if x[0] in ("type_thing", "type_is", "string_is") and x[2] != "migratie"]
    samenvatting["adr0021_python"] = {
        "treffers": sum(x[4] for x in adr),
        "bestanden": len({x[3] for x in adr}),
        "broncodebestanden": len({x[3] for x in adr if x[2] == "broncode"}),
    }
    json.dump(samenvatting, sys.stdout, ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
