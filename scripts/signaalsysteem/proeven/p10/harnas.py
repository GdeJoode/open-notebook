"""P10, 10e: laat de testfixture live_surrealdb een bestaande proefdatabase gebruiken.

Gebruik: python harnas.py <kopie>

De fixture van de fork (packages/surrealdb-service/src/surrealdb_service/testing/
fixtures.py) start met testcontainers een eigen container met de zwevende tag
surrealdb/surrealdb:v2, zonder projectlabel en met een Ryuk-container. Dat past
niet in de isolatie van de proeven (proeven.md 1.2) en kiest niet de te meten
versie. Deze aanpassing, alleen in de kopie en gelijk voor v2.7.0 en v3.3.5:
als P10_SURREAL_URL is gezet, gebruikt de fixture die server met een nieuwe,
lege namespace en database per pytest-sessie, en past daar alle migraties toe met
dezelfde code (AsyncMigrationManager en _apply_with_diagnostics). Zonder de
variabele verandert er niets.
"""

from __future__ import annotations

import sys
from pathlib import Path

ANKER = '''    if not docker_available():
        pytest.skip("Docker daemon not reachable — skipping requires_docker tests")
'''

INVOEG = '''    import os as _p10_os

    if _p10_os.environ.get("P10_SURREAL_URL"):
        yield from _p10_extern(_p10_os.environ["P10_SURREAL_URL"])
        return
'''

FUNCTIE = '''

def _p10_extern(url: str) -> Iterator[SurrealDBConfig]:
    """P10: bestaande proefserver, nieuwe lege namespace en database per sessie."""
    import asyncio
    import os
    import uuid

    naam = f"p10_{uuid.uuid4().hex[:12]}"
    config = SurrealDBConfig(
        url=url,
        username="root",
        password=os.environ["SP_SURREAL_PASS"],
        namespace=naam,
        database=naam,
    )
    connection_module._pool = None
    manager = AsyncMigrationManager(migrations_dir=_MIGRATIONS_DIR, config=config)
    try:
        asyncio.run(_apply_with_diagnostics(manager))
    except Exception as exc:
        raise RuntimeError(
            f"Failed to apply migrations against P10 SurrealDB: {exc}"
        ) from exc
    connection_module._pool = None
    try:
        yield config
    finally:
        try:
            if connection_module._pool is not None:
                asyncio.run(connection_module.close_pool())
        except Exception:
            pass
        connection_module._pool = None
'''


def main() -> int:
    pad = Path(sys.argv[1]) / "packages/surrealdb-service/src/surrealdb_service/testing/fixtures.py"
    tekst = pad.read_text(encoding="utf-8")
    if ANKER not in tekst:
        print(f"anker niet gevonden in {pad}", file=sys.stderr)
        return 3
    tekst = tekst.replace(ANKER, INVOEG + ANKER, 1) + FUNCTIE
    pad.write_text(tekst, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
