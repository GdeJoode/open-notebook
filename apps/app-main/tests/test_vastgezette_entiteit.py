"""Een automatisch proces voegt een vastgezette entiteit niet samen (F2-06, voorstel B28).

Tegen een echte SurrealDB: een entiteit met ``manual_override = true`` wordt
door een automatische samenvoeging niet aangeraakt, als winnaar of als
verliezer, ook niet via de OKF-import met ``AUTO_MERGE``. Het overslaan komt
als ``beheer_melding`` (er is geen run). Een samenvoeging door de beheerder
(met ``door``) gaat wel door.

Alle testdata is synthetisch.
"""

from __future__ import annotations

import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import surrealdb_service.config as surreal_config
import surrealdb_service.connection as conn
from app_main.services.entity_resolution import merge_audit
from app_main.services.entity_resolution.recanonicalization_service import (
    MergeCluster,
    RecanonicalizationService,
)
from app_main.services.okf_import_service import OkfImportService

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]


@pytest.fixture
async def env(live_surrealdb, monkeypatch):
    monkeypatch.setattr(surreal_config, "_config", live_surrealdb)
    original = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    yield
    await conn._pool.close_all()
    conn._pool = original or conn.ConnectionPool(config=live_surrealdb)


async def _q(sql: str, params: dict | None = None):
    return await conn.execute_query(sql, params or {})


async def _entiteit(naam: str, **velden) -> str:
    sleutel = f"{naam.lower()} {uuid.uuid4().hex[:8]}"
    [rij] = await _q("CREATE entity CONTENT $d RETURN id", {"d": {
        "canonical_name": naam, "name": naam, "name_key": sleutel,
        "entity_type": "organization", "hash_id": uuid.uuid4().hex,
        "embedding": [], "status": "active", **velden,
    }})
    return str(rij["id"])


def _cluster(winner: str, loser: str) -> MergeCluster:
    return MergeCluster(new_canonical="Openbare organisatie", entity_type="organization",
                        winner_id=winner, loser_ids=[loser],
                        member_surface_forms=[], total_source_docs=0)


async def _status(eid: str) -> str:
    [rij] = await _q("SELECT status FROM $id", {"id": conn.ensure_record_id(eid)})
    return rij["status"]


async def _meldingen(eid: str) -> list:
    return await _q("SELECT melding FROM beheer_melding WHERE entity = $e",
                    {"e": conn.ensure_record_id(eid)})


@pytest.mark.parametrize("vast_is", ["winnaar", "verliezer"])
async def test_automatisch_samenvoegen_slaat_een_vastgezette_entiteit_over(env, vast_is) -> None:
    vast = await _entiteit("Openbare organisatie", manual_override=True)
    vrij = await _entiteit("openbare organisatie")
    winner, loser = (vast, vrij) if vast_is == "winnaar" else (vrij, vast)

    result = await RecanonicalizationService().apply_merge(_cluster(winner, loser))

    assert result.skipped and result.merged_loser_ids == []
    assert await _status(loser) == "active"
    assert [m["melding"] for m in await _meldingen(vast)] == ["samenvoegen overgeslagen: manual_override"]


async def test_okf_auto_merge_voegt_een_vastgezette_entiteit_niet_samen(env, monkeypatch) -> None:
    logged = []

    async def _record(**kw):
        logged.append(kw)
        return 1

    monkeypatch.setattr(merge_audit, "record_merges", _record)
    vast = await _entiteit("Openbare dienst", manual_override=True)
    vrij = await _entiteit("openbare dienst")
    kandidaat = SimpleNamespace(
        to_merge_cluster=lambda: _cluster(vrij, vast), id_a=vrij, id_b=vast,
        name_a="Openbare dienst", name_b="openbare dienst", entity_type="organization",
        entity_type_b="", winner_id=vrij, method="fold_equal", score=1.0, evidence="",
    )
    dedup = MagicMock()
    dedup.propose_candidates = AsyncMock(return_value=SimpleNamespace(auto_merge=[kandidaat]))
    svc = OkfImportService(
        entity_repository=MagicMock(), persistence_service=MagicMock(),
        note_repository=MagicMock(), source_repository=MagicMock(),
        candidate_dedup_service=dedup, recanonicalization_service=RecanonicalizationService(),
    )

    assert await svc._run_dedup(None) == 0
    assert await _status(vast) == "active"
    assert logged == []
    assert await _meldingen(vast)


async def test_de_beheerder_mag_een_vastgezette_entiteit_samenvoegen(env) -> None:
    vast = await _entiteit("Openbaar bureau", manual_override=True)
    vrij = await _entiteit("openbaar bureau")

    result = await RecanonicalizationService().apply_merge(_cluster(vrij, vast), door="Testpersoon")

    assert result.merged_loser_ids == [vast]
    assert await _status(vast) == "merged"
    assert await _meldingen(vast) == []


async def test_vastzetting_na_de_voorcontrole_stopt_de_samenvoeging(env) -> None:
    """De pin valt na de voorcontrole: de verliezer wordt toch niet samengevoegd."""
    from surrealdb_service.repositories.entity import EntityRepository

    vast = await _entiteit("Openbaar instituut", manual_override=True)
    vrij = await _entiteit("openbaar instituut")
    repo = EntityRepository()
    echt = repo.bewaak_vastgezet
    aanroepen = []

    async def _te_laat(ids, **kw):
        aanroepen.append(kw["actie"])
        return set() if len(aanroepen) == 1 else await echt(ids, **kw)

    repo.bewaak_vastgezet = _te_laat
    result = await RecanonicalizationService(entity_repo=repo).apply_merge(_cluster(vrij, vast))

    assert result.skipped and result.merged_loser_ids == []
    assert await _status(vast) == "active"
    assert not await _q("SELECT id FROM entity_alias WHERE canonical_entity = $e",
                        {"e": conn.ensure_record_id(vrij)})
    assert await _meldingen(vast)
