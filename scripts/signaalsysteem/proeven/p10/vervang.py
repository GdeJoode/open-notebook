"""P10, 10d: alleen de mechanische vervangingen uit de migratiegids 2.x naar 3.x.

Gebruik: python vervang.py <kopie> <vervangingen.csv> [--extra]

Past in de kopie (nooit in de repository) toe, in migrations/*.surrealql en in
*.py onder apps/, packages/ en pipelines/:
- de hernoemde functies uit de tabel van de migratiegids (type::thing -> type::record,
  type::is::x -> type::is_x, string::is::x -> string::is_x, time::from::x ->
  time::from_x, time::is::x, duration::from::x, geo::is::x, rand::guid -> rand::id,
  string::distance::osa_distance -> string::distance::osa);
- SEARCH ANALYZER -> FULLTEXT ANALYZER;
- LET vóór een toekenning aan een parameter aan het begin van een opdracht.
Elke vervanging komt in de CSV: bestand, regel, vervanging, oud, nieuw.

Met --extra (stap 10g, alleen ter informatie; telt niet voor K2-K4) daarnaast de
volgorde FLEXIBLE TYPE x -> TYPE x FLEXIBLE. De parser van v3.3.5 eist die volgorde
("FLEXIBLE must be specified after TYPE"); de migratiegids noemt haar niet.
"""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

VERVANGINGEN: list[tuple[str, re.Pattern[str], str]] = [
    ("search_analyzer", re.compile(r"\bSEARCH(\s+)ANALYZER\b"), r"FULLTEXT\1ANALYZER"),
    ("type_thing", re.compile(r"\btype::thing(\s*\()"), r"type::record\1"),
    ("type_is", re.compile(r"\btype::is::(\w+)"), r"type::is_\1"),
    ("string_is", re.compile(r"\bstring::is::(\w+)"), r"string::is_\1"),
    ("time_from_is", re.compile(r"\btime::(from|is)::(\w+)"), r"time::\1_\2"),
    ("duration_from", re.compile(r"\bduration::from::(\w+)"), r"duration::from_\1"),
    ("geo_is", re.compile(r"\bgeo::is::(\w+)"), r"geo::is_\1"),
    ("osa_distance", re.compile(r"\bstring::distance::osa_distance\b"), "string::distance::osa"),
    ("rand_guid", re.compile(r"\brand::guid(\s*\()"), r"rand::id\1"),
    # Toekenning zonder LET aan het begin van een regel of na ; of {.
    ("let_ontbreekt", re.compile(r"(^|[;{])(\s*)(\$\w+\s*=)(?![=~])"), r"\1\2LET \3"),
]

# Type met hooguit drie niveaus <...>, zoals option<array<object>>.
_TYPE = r"\w+(?:<(?:[^<>]|<(?:[^<>]|<[^<>]*>)*>)*>)?"
EXTRA: list[tuple[str, re.Pattern[str], str]] = [
    ("flexible_volgorde", re.compile(r"\bFLEXIBLE(\s+)TYPE\s+(" + _TYPE + r")"), r"TYPE \2\1FLEXIBLE"),
]


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "--extra"]
    kopie, uit = Path(args[0]), Path(args[1])
    vervangingen = VERVANGINGEN + (EXTRA if "--extra" in sys.argv else [])
    paden = sorted((kopie / "migrations").glob("*.surrealql"))
    for map_ in ("apps", "packages", "pipelines"):
        paden += sorted(
            p for p in (kopie / map_).rglob("*.py")
            if not any(d in p.parts for d in (".venv", "node_modules", "__pycache__"))
        )
    rijen = []
    for pad in paden:
        tekst = pad.read_text(encoding="utf-8")
        regels = tekst.split("\n")
        gewijzigd = False
        for i, regel in enumerate(regels):
            if regel.strip().startswith("--") or regel.strip().startswith("#"):
                continue
            nieuw = regel
            for naam, rx, door in vervangingen:
                for m in rx.finditer(nieuw):
                    rijen.append([pad.relative_to(kopie).as_posix(), i + 1, naam,
                                  m.group(0).strip(), m.expand(door).strip()])
                nieuw = rx.sub(door, nieuw)
            if nieuw != regel:
                regels[i] = nieuw
                gewijzigd = True
        if gewijzigd:
            pad.write_text("\n".join(regels), encoding="utf-8")
    with uit.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bestand", "regel", "vervanging", "oud", "nieuw"])
        w.writerows(rijen)
    print(f"{len(rijen)} vervangingen in {len({r[0] for r in rijen})} bestanden")
    return 0


if __name__ == "__main__":
    sys.exit(main())
