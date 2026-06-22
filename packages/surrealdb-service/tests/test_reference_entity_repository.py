"""Integration roundtrip for ReferenceEntityRepository (testcontainers, K.4).

Seeds the ``reference_entity`` vocabulary table against a real SurrealDB
container and asserts the K.4 contract:

* ``bulk_load`` then ``lookup_by_name`` / ``lookup_by_alias`` round-trip;
* a TOOI org loaded via the provider's row shape resolves by name AND
  abbreviation, with a non-null ``external_uri`` (AC1);
* re-loading the same records is idempotent — no duplicate rows (AC6).

Gated behind ``@requires_docker`` like the other roundtrip suites.
"""

from __future__ import annotations

import uuid

import pytest

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.repositories.reference_entity import (
    ReferenceEntityRepository,
)


def _u(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _bzk_record(name: str, code: str) -> dict:
    return {
        "canonical_name": name,
        "entity_type": "organization",
        "source_vocabulary": "tooi",
        "external_uri": f"https://identifier.overheid.nl/tooi/id/ministerie/{code}",
        "external_id": code,
        "aliases": ["BZK", "ministerie van " + name],
        "properties": {"soort": "ministerie"},
    }


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_bulk_load_and_lookup_by_name(live_surrealdb: SurrealDBConfig) -> None:
    repo = ReferenceEntityRepository(config=live_surrealdb)
    name = _u("Binnenlandse Zaken")
    code = _u("mnre")
    loaded = await repo.bulk_load([_bzk_record(name, code)])
    assert loaded == 1

    rows = await repo.lookup_by_name(name, source="tooi")
    assert len(rows) == 1
    assert rows[0]["external_uri"].endswith(code)
    assert rows[0]["external_uri"]  # non-null (AC1)
    assert rows[0]["source_vocabulary"] == "tooi"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_lookup_by_alias(live_surrealdb: SurrealDBConfig) -> None:
    repo = ReferenceEntityRepository(config=live_surrealdb)
    name = _u("Onderwijs Cultuur")
    code = _u("mnre")
    rec = _bzk_record(name, code)
    rec["aliases"] = ["OCW-" + code, name]
    await repo.bulk_load([rec])

    rows = await repo.lookup_by_alias("OCW-" + code, source="tooi")
    assert len(rows) == 1
    assert rows[0]["canonical_name"] == name


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_refresh_idempotent_no_duplicates(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC6: loading the same record twice yields a single row."""
    repo = ReferenceEntityRepository(config=live_surrealdb)
    name = _u("Economische Zaken")
    code = _u("mnre")
    rec = _bzk_record(name, code)

    await repo.bulk_load([rec])
    await repo.bulk_load([rec])  # second refresh

    rows = await repo.lookup_by_name(name, source="tooi")
    assert len(rows) == 1  # no duplicate from the re-load


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_last_validated_stamped(live_surrealdb: SurrealDBConfig) -> None:
    repo = ReferenceEntityRepository(config=live_surrealdb)
    src = _u("vocab").replace("-", "")
    rec = _bzk_record(_u("Defensie"), _u("mnre"))
    rec["source_vocabulary"] = src
    await repo.bulk_load([rec])

    stamp = await repo.last_validated(src)
    assert stamp is not None
