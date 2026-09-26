"""`reference_entity.lookup_keys` — the D2 lookup, against a real SurrealDB.

The provider tests use an in-memory repository that implements `$key IN
lookup_keys` in Python. This file is the same contract executed by SurrealDB,
through the real repository, on a container with every migration applied —
the fake is only as good as its agreement with this.
"""

from __future__ import annotations

import uuid

import pytest
from shared.utils.nl_normalization import normalize_alias_key
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.reference_entity import ReferenceEntityRepository


def _record(code: str, name: str, aliases: list[str]) -> dict:
    return {
        "canonical_name": name,
        "entity_type": "organization",
        "source_vocabulary": "tooi",
        "external_uri": f"https://identifier.overheid.nl/tooi/id/ministerie/{code}",
        "external_id": code,
        "aliases": aliases,
        "lookup_keys": sorted({normalize_alias_key(n) for n in [name, *aliases]}),
        "properties": {},
    }


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_a_row_is_found_by_its_key_whatever_the_casing(
    live_surrealdb: SurrealDBConfig,
) -> None:
    code = f"t{uuid.uuid4().hex[:8]}"
    repo = ReferenceEntityRepository(live_surrealdb)
    await repo.upsert(_record(code, "Binnenlandse Zaken en Koninkrijksrelaties",
                              ["BZK", "ministerie van Binnenlandse Zaken en Koninkrijksrelaties"]))

    for surface in ("binnenlandse zaken en koninkrijksrelaties",
                    "Ministerie van Binnenlandse Zaken en Koninkrijksrelaties",
                    # the corpus spelling, canonicalised by the key
                    "Binnenlandse Zaken en Koninkrijkrelaties",
                    "bzk"):
        rows = await repo.lookup_by_key(normalize_alias_key(surface), source="tooi")
        assert [r["external_id"] for r in rows] == [code], surface


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_a_row_written_without_keys_is_found_by_nothing(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Rows from before migration 81 carry the field's default and stay invisible
    until `refresh()` rewrites them — stated in the migration, checked here."""
    code = f"t{uuid.uuid4().hex[:8]}"
    await execute_query(
        "CREATE reference_entity CONTENT $d",
        {"d": {"canonical_name": f"Oud {code}", "entity_type": "organization",
               "source_vocabulary": "tooi", "external_id": code, "aliases": []}},
        live_surrealdb,
    )
    rows = await execute_query("SELECT lookup_keys FROM reference_entity WHERE external_id = $c",
                               {"c": code}, live_surrealdb)
    assert rows[0]["lookup_keys"] == []
    repo = ReferenceEntityRepository(live_surrealdb)
    assert await repo.lookup_by_key(normalize_alias_key(f"Oud {code}"), source="tooi") == []
