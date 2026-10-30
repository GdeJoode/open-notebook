"""P10, 10b/10c/10d en 10f: migraties heen, terug en weer heen, en de schemacontrole.

Gebruik (PYTHONPATH met de src-mappen van de kopie, zodat de migratiecode van de
fork uit de kopie komt):
  python migraties.py migraties <url> <migratiemap> <naam>   -> JSON naar stdout
  python migraties.py schema <url> <naam>                    -> JSON naar stdout
  python migraties.py verbind <url>                          -> JSON naar stdout

Wachtwoord van root in de omgevingsvariabele SP_SURREAL_PASS. <naam> wordt de
namespace en database: elke stap begint op een lege database.

De migratiecode is die van de fork (surrealdb_service.migrations):
discover_migrations en AsyncMigration.from_file voor het lezen,
execute_transaction (per opdracht gecontroleerd) voor het uitvoeren, zoals
AsyncMigrationRunner._apply_up, en _bump_version/_lower_version voor de versie.
Afwijkingen van de runner, om alle fouten te zien:
- na een mislukte migratie gaat de proef door met de volgende (de runner stopt);
- de _down-varianten gaan ook via execute_transaction. run_one_down gebruikt
  execute_query, dat een fout in een latere opdracht niet meldt.
Een fout met "already exists" telt, net als in de runner, als geslaagd
(stand "al_aanwezig").
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path


def _config(url: str, naam: str):
    from surrealdb_service.config import SurrealDBConfig

    return SurrealDBConfig(
        url=url, username="root", password=os.environ["SP_SURREAL_PASS"],
        namespace=naam, database=naam, pool_size=2,
    )


async def _stap(sql: str, config) -> tuple[str, str | None]:
    from surrealdb_service.connection import execute_transaction

    try:
        await execute_transaction(sql, config=config)
        return "geslaagd", None
    except Exception as exc:  # noqa: BLE001 - elke fout is een meetwaarde
        tekst = f"{type(exc).__name__}: {exc}"
        if "already exists" in str(exc).lower():
            return "al_aanwezig", tekst[:2000]
        return "mislukt", tekst[:2000]


async def _versie_stap(functie, versie: int, config) -> str | None:
    """_bump_version of _lower_version van de fork; een fout is een meetwaarde.

    Na een mislukte _down-variant blijft de versie staan, en faalt het opnieuw
    registreren bij de tweede keer heen ("already exists"). Dat volgt uit de
    eerdere fout en telt niet apart.
    """
    try:
        await functie(versie, config)
        return None
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {exc}"[:500]


async def migraties(url: str, map_: str, naam: str) -> dict:
    from surrealdb_service import connection
    from surrealdb_service.migrations import _bump_version, _lower_version, discover_migrations

    connection._pool = None
    config = _config(url, naam)
    op, neer = discover_migrations(Path(map_))
    uitkomst: dict = {"naam": naam, "aantal_op": len(op), "aantal_neer": len(neer), "stappen": {}}

    async def heen(label: str) -> None:
        r = []
        for v in sorted(op):
            stand, fout = await _stap(op[v].sql, config)
            versie = await _versie_stap(_bump_version, v, config) if stand != "mislukt" else None
            r.append({"bestand": f"{v}.surrealql", "stand": stand, "fout": fout, "versiefout": versie})
        uitkomst["stappen"][label] = r

    async def terug() -> None:
        r = []
        for v in sorted(op, reverse=True):
            if v not in neer:
                r.append({"bestand": f"{v}_down.surrealql", "stand": "geen_down", "fout": None})
                continue
            stand, fout = await _stap(neer[v].sql, config)
            versie = await _versie_stap(_lower_version, v, config) if stand != "mislukt" else None
            r.append({"bestand": f"{v}_down.surrealql", "stand": stand, "fout": fout, "versiefout": versie})
        uitkomst["stappen"]["terug"] = r

    await heen("heen_1")
    await terug()
    await heen("heen_2")
    await connection.close_pool()
    alle = [s for stap in uitkomst["stappen"].values() for s in stap]
    uitkomst["telling"] = {
        k: sum(1 for s in alle if s["stand"] == k)
        for k in ("geslaagd", "al_aanwezig", "mislukt", "geen_down")
    }
    uitkomst["mislukte_bestanden"] = sorted(
        {s["bestand"] for s in alle if s["stand"] in ("mislukt", "geen_down")},
        key=lambda b: (int(b.split(".")[0].split("_")[0]), b),
    )
    return uitkomst


async def schema(url: str, naam: str) -> dict:
    """10f: een veld schrijven dat een SCHEMAFULL-tabel niet kent (het geval uit P4)."""
    from surrealdb_service import connection
    from surrealdb_service.connection import execute_query

    connection._pool = None
    config = _config(url, naam)
    await execute_query(
        "DEFINE TABLE p10_schema SCHEMAFULL; DEFINE FIELD naam ON p10_schema TYPE string;",
        config=config,
    )
    uit = {}
    for label, rid, sql in (
        ("create_set", "a", "CREATE p10_schema:a SET naam = 'a', classification = 'intern';"),
        ("create_content", "b", "CREATE p10_schema:b CONTENT { naam: 'b', classification: 'intern' };"),
        ("update_set", "c", "CREATE p10_schema:c SET naam = 'c'; UPDATE p10_schema:c SET classification = 'intern';"),
    ):
        stand, fout = await _stap(sql, config)
        try:
            rij = await execute_query(f"SELECT * FROM p10_schema:{rid};", config=config)
        except Exception as exc:  # noqa: BLE001
            rij = f"{type(exc).__name__}: {exc}"
        uit[label] = {"stand": stand, "fout": fout, "record": json.loads(json.dumps(rij, default=str))}
    await connection.close_pool()
    weggelaten = all(
        v["stand"] == "geslaagd" and isinstance(v["record"], list) and v["record"]
        and "classification" not in v["record"][0]
        for v in uit.values()
    )
    fout = all(v["stand"] == "mislukt" for v in uit.values())
    uit["gedrag"] = "stil_weggelaten" if weggelaten else ("fout" if fout else "gemengd")
    return uit


async def verbind(url: str) -> dict:
    from surrealdb_service import connection
    from surrealdb_service.connection import execute_query

    connection._pool = None
    try:
        r = await execute_query("RETURN 1;", config=_config(url, "p10_verbind"))
        await connection.close_pool()
        return {"verbinding": True, "antwoord": r}
    except Exception as exc:  # noqa: BLE001
        return {"verbinding": False, "fout": f"{type(exc).__name__}: {exc}"[:2000]}


def main() -> int:
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    import surrealdb

    soort = sys.argv[1]
    if soort == "migraties":
        r = asyncio.run(migraties(sys.argv[2], sys.argv[3], sys.argv[4]))
    elif soort == "schema":
        r = asyncio.run(schema(sys.argv[2], sys.argv[3]))
    elif soort == "verbind":
        r = asyncio.run(verbind(sys.argv[2]))
    else:
        raise SystemExit(f"onbekende opdracht {soort}")
    import surrealdb_service

    r["sdk"] = getattr(surrealdb, "__version__", None) or _versie("surrealdb")
    r["sdk_pad"] = surrealdb.__file__
    r["fork_pad"] = surrealdb_service.__file__
    json.dump(r, sys.stdout, ensure_ascii=False, indent=1, default=str)
    return 0


def _versie(pakket: str) -> str:
    import importlib.metadata as md

    try:
        return md.version(pakket)
    except md.PackageNotFoundError:
        return "onbekend"


if __name__ == "__main__":
    sys.exit(main())
