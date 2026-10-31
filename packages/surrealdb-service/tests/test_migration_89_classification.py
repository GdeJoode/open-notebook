"""Migratie 89: classificatie intern/extern (F2-03, ADR 0001).

Onder SurrealDB v2.7.0 laat een SCHEMAFULL-tabel een onbekend veld stil weg
(P4, P10 K6). Daarom schrijft elke test een waarde én leest die terug.

De fixture is per sessie: een test die `_down` draait, zet 89 daarna terug.
"""

from __future__ import annotations

import uuid

import pytest
from surrealdb import RecordID
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import (
    ClassificationConflictError,
    ClassificationValueError,
    db_connection,
    execute_query,
    execute_transaction,
)
from surrealdb_service.testing import fixtures as fx

pytestmark = [pytest.mark.requires_docker, pytest.mark.asyncio]

#: De tabellen met een eigen waarde, standaard 'intern'.
EIGEN = ["source", "chunk", "source_embedding", "doc_node", "note", "source_insight", "notebook"]
#: De tabellen met een afgeleide waarde.
AFGELEID = ["entity", "entity_alias", "relation", "triage_queue"]


def _migratie(versie: int, *, down: bool = False) -> str:
    from surrealdb_service.migrations import AsyncMigration

    suffix = "_down" if down else ""
    return AsyncMigration.from_file(fx._MIGRATIONS_DIR / f"{versie}{suffix}.surrealql").sql


@pytest.fixture(autouse=True, scope="module")
def _migratie_89_bovenop(live_surrealdb: SurrealDBConfig) -> None:
    """De database is per sessie gedeeld, en eerdere tests voeren oude migraties
    opnieuw uit (zoals 62, die `relation` opnieuw definieert). Daarom eerst 89
    opnieuw: die is idempotent."""
    import asyncio

    import surrealdb_service.connection as conn

    asyncio.run(execute_transaction(_migratie(89), config=live_surrealdb))
    # De pool hoort nu bij de afgesloten lus van asyncio.run.
    conn._pool = None


def _id(tabel: str) -> str:
    return f"{tabel}:t{uuid.uuid4().hex[:12]}"


async def _q(config: SurrealDBConfig, sql: str, params: dict | None = None):
    return await execute_query(sql, params, config)


async def _lees(config: SurrealDBConfig, rid: str) -> dict:
    rows = await _q(config, "SELECT * FROM type::thing($id)", {"id": rid})
    return rows[0]


async def _bron(config: SurrealDBConfig, classification: str | None = None) -> str:
    rid = _id("source")
    data = {"title": "Openbare testbron"}
    if classification:
        data["classification"] = classification
    await _q(config, "CREATE type::thing($id) CONTENT $data", {"id": rid, "data": data})
    return rid


_KIND = {
    "chunk": "{source: $bron, text: 'x', order: 0, physical_page: 0, element_type: 'paragraph',"
    " metadata: {}, positions: []}",
    "doc_node": "{source: $bron, element_type: 'paragraph', self_ref: '#/texts/0', sequence: 0}",
    "source_insight": "{source: $bron, content: 'x', insight_type: 'summary', embedding: []}",
    "source_embedding": "{source: $bron, content: 'x', order: 0, embedding: []}",
}


async def _maak(config: SurrealDBConfig, tabel: str, bron: str | None, extra: dict | None = None) -> str:
    rid = _id(tabel)
    if tabel in _KIND:
        await _q(
            config,
            f"CREATE type::thing($id) CONTENT {_KIND[tabel]}",
            {"id": rid, "bron": _rec(bron) if bron else None},
        )
    elif tabel == "note":
        await _q(config, "CREATE type::thing($id) CONTENT {title: 'x', embedding: []}", {"id": rid})
    else:
        await _q(config, "CREATE type::thing($id) CONTENT {name: 'x'}", {"id": rid})
    if extra:
        await _q(config, "UPDATE type::thing($id) MERGE $data", {"id": rid, "data": extra})
    return rid


def _rec(rid: str) -> RecordID:
    tabel, _, sleutel = rid.partition(":")
    return RecordID(tabel, sleutel)


# --------------------------------------------------------------------------
# Velden en standaardwaarde
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tabel", EIGEN + AFGELEID)
async def test_veld_is_gedefinieerd(live_surrealdb: SurrealDBConfig, tabel: str) -> None:
    info = await _q(live_surrealdb, f"INFO FOR TABLE {tabel}")
    veld = (info[0] if isinstance(info, list) else info)["fields"]["classification"]
    assert "ASSERT" in veld and "'intern'" in veld and "'extern'" in veld
    if tabel in ("source", "note", "notebook"):
        assert "DEFAULT ALWAYS 'intern'" in veld
    elif tabel in EIGEN:
        assert "DEFAULT ALWAYS" in veld
    else:
        assert "DEFAULT ALWAYS 'intern'" in veld and "VALUE" in veld


@pytest.mark.parametrize("tabel", EIGEN)
async def test_record_zonder_waarde_is_intern(live_surrealdb: SurrealDBConfig, tabel: str) -> None:
    bron = await _bron(live_surrealdb)
    rid = bron if tabel == "source" else await _maak(live_surrealdb, tabel, bron)
    assert (await _lees(live_surrealdb, rid))["classification"] == "intern"


@pytest.mark.parametrize("tabel", ["source", "note", "notebook"])
async def test_extern_schrijven_en_teruglezen(live_surrealdb: SurrealDBConfig, tabel: str) -> None:
    rid = _id(tabel)
    await _q(
        live_surrealdb,
        "CREATE type::thing($id) CONTENT {classification: 'extern', embedding: []}",
        {"id": rid},
    )
    assert (await _lees(live_surrealdb, rid))["classification"] == "extern"


@pytest.mark.parametrize("tabel", ["chunk", "source_embedding", "doc_node", "source_insight"])
async def test_kind_neemt_de_waarde_van_de_bron_over(
    live_surrealdb: SurrealDBConfig, tabel: str
) -> None:
    bron = await _bron(live_surrealdb, "extern")
    rid = await _maak(live_surrealdb, tabel, bron)
    assert (await _lees(live_surrealdb, rid))["classification"] == "extern"
    # Strenger dan de bron mag: schrijven en teruglezen.
    await _q(live_surrealdb, "UPDATE type::thing($id) SET classification = 'intern'", {"id": rid})
    assert (await _lees(live_surrealdb, rid))["classification"] == "intern"


@pytest.mark.parametrize("tabel", ["chunk", "source_embedding", "doc_node", "source_insight"])
async def test_kind_losser_dan_de_bron_wordt_geweigerd(
    live_surrealdb: SurrealDBConfig, tabel: str
) -> None:
    bron = await _bron(live_surrealdb)
    rid = await _maak(live_surrealdb, tabel, bron)
    with pytest.raises(ClassificationConflictError):
        await _q(live_surrealdb, "UPDATE type::thing($id) SET classification = 'extern'", {"id": rid})
    assert (await _lees(live_surrealdb, rid))["classification"] == "intern"


async def test_embedding_niet_losser_dan_zijn_fragment(live_surrealdb: SurrealDBConfig) -> None:
    bron = await _bron(live_surrealdb, "extern")
    fragment = await _maak(live_surrealdb, "chunk", bron)
    await _q(live_surrealdb, "UPDATE type::thing($id) SET classification = 'intern'", {"id": fragment})
    rid = _id("source_embedding")
    await _q(
        live_surrealdb,
        "CREATE type::thing($id) CONTENT {source: type::thing($b), chunk: type::thing($c),"
        " content: 'x', order: 0, embedding: []}",
        {"id": rid, "b": bron, "c": fragment},
    )
    assert (await _lees(live_surrealdb, rid))["classification"] == "intern"
    with pytest.raises(ClassificationConflictError):
        await _q(live_surrealdb, "UPDATE type::thing($id) SET classification = 'extern'", {"id": rid})


@pytest.mark.parametrize("tabel", EIGEN)
async def test_onbekende_waarde_wordt_geweigerd(live_surrealdb: SurrealDBConfig, tabel: str) -> None:
    bron = await _bron(live_surrealdb, "extern")
    rid = bron if tabel == "source" else await _maak(live_surrealdb, tabel, bron)
    # Een ongeldige waarde is geen regelschending: 422, geen 409.
    with pytest.raises(ClassificationValueError):
        await _q(live_surrealdb, "UPDATE type::thing($id) SET classification = 'openbaar'", {"id": rid})


@pytest.mark.parametrize("tabel", ["source", "note", "notebook", "chunk"])
async def test_intern_naar_extern_alleen_bij_herclassificatie(
    live_surrealdb: SurrealDBConfig, tabel: str
) -> None:
    bron = await _bron(live_surrealdb, "extern")
    rid = bron if tabel == "source" else await _maak(live_surrealdb, tabel, bron)
    await _q(live_surrealdb, "UPDATE type::thing($id) SET classification = 'intern'", {"id": rid})
    with pytest.raises(ClassificationConflictError):
        await _q(live_surrealdb, "UPDATE type::thing($id) SET classification = 'extern'", {"id": rid})
    assert (await _lees(live_surrealdb, rid))["classification"] == "intern"
    # Met de vlag mag het in de database. De fork zet de vlag alleen in
    # SourceRepository.reclassify; hier rechtstreeks via de SDK.
    async with db_connection(live_surrealdb) as db:
        await db.query(
            "BEGIN; LET $herclassificatie = true;"
            " UPDATE type::thing($id) SET classification = 'extern'; COMMIT;",
            {"id": rid},
        )
    assert (await _lees(live_surrealdb, rid))["classification"] == "extern"


@pytest.mark.parametrize(
    "sql, params",
    [
        ("UPDATE type::thing($id) SET classification = 'extern'", {"herclassificatie": True}),
        ("UPDATE type::thing($id) SET classification = 'extern'", {"$herclassificatie": True}),
        ("LET $herclassificatie = true; UPDATE type::thing($id) SET classification = 'extern'", {}),
        ("let  $Herclassificatie = true; UPDATE type::thing($id) SET classification = 'extern'", {}),
        ("DEFINE PARAM OVERWRITE $herclassificatie VALUE true", {}),
    ],
)
async def test_de_vlag_kan_niet_van_buiten_worden_gezet(
    live_surrealdb: SurrealDBConfig, sql: str, params: dict
) -> None:
    bron = await _bron(live_surrealdb)
    for uitvoeren in (execute_query, execute_transaction):
        with pytest.raises(ValueError, match="herclassificatievlag"):
            await uitvoeren(sql, {"id": bron, **params}, live_surrealdb)
    assert (await _lees(live_surrealdb, bron))["classification"] == "intern"


async def test_strengste_wint_bij_een_bron_die_intern_wordt(live_surrealdb: SurrealDBConfig) -> None:
    bron = await _bron(live_surrealdb, "extern")
    kinderen = [await _maak(live_surrealdb, t, bron) for t in _KIND]
    assert {(await _lees(live_surrealdb, k))["classification"] for k in kinderen} == {"extern"}
    await _q(live_surrealdb, "UPDATE type::thing($id) SET classification = 'intern'", {"id": bron})
    assert {(await _lees(live_surrealdb, k))["classification"] for k in kinderen} == {"intern"}


@pytest.mark.parametrize("tabel", AFGELEID)
async def test_afgeleid_record_zonder_vermelding_is_intern(
    live_surrealdb: SurrealDBConfig, tabel: str
) -> None:
    entiteit = _id("entity")
    await _q(live_surrealdb, "CREATE type::thing($id) SET canonical_name = $id, entity_type = 'thing', name_key = $id,"
        " embedding = [], classification = 'extern'",
        {"id": entiteit},
    )
    if tabel == "entity":
        rid = entiteit
    elif tabel == "entity_alias":
        rid = _id("entity_alias")
        await _q(
            live_surrealdb,
            "CREATE type::thing($id) SET canonical_entity = type::thing($e), alias_text = 'x',"
            " match_type = 'exact', classification = 'extern'",
            {"id": rid, "e": entiteit},
        )
    elif tabel == "relation":
        rows = await _q(
            live_surrealdb,
            "RELATE (type::thing($e))->relation->(type::thing($e))"
            " SET relation_type = 'related_to', classification = 'extern'",
            {"e": entiteit},
        )
        rid = str(rows[0]["id"])
    else:
        rid = _id("triage_queue")
        await _q(
            live_surrealdb,
            "CREATE type::thing($id) SET entity = type::thing($e), batch_id = 'b', name = 'x',"
            " reason = 'r', classification = 'extern'",
            {"id": rid, "e": entiteit},
        )
    # Een geschreven 'extern' telt niet: de database berekent de waarde.
    assert (await _lees(live_surrealdb, rid))["classification"] == "intern"


# --------------------------------------------------------------------------
# Logboek
# --------------------------------------------------------------------------


async def _logregel(config: SurrealDBConfig, **over) -> list:
    bron = await _bron(config)
    data = {
        "from_classification": "intern",
        "to_classification": "extern",
        "reason": "openbaar gepubliceerd",
        "agent_key": "agent_keys:test",
        "person": "Testpersoon",
    } | over
    data = {k: v for k, v in data.items() if v is not None}
    return await _q(
        config, "CREATE classification_log CONTENT $data", {"data": data | {"source": _rec(bron)}}
    )


async def test_logregel_schrijven_en_teruglezen(live_surrealdb: SurrealDBConfig) -> None:
    rows = await _logregel(live_surrealdb)
    regel = await _lees(live_surrealdb, str(rows[0]["id"]))
    assert regel["person"] == "Testpersoon"
    assert regel["reason"] == "openbaar gepubliceerd"
    assert regel["agent_key"] == "agent_keys:test"
    assert regel["created_at"]


@pytest.mark.parametrize("veld", ["person", "reason", "agent_key"])
@pytest.mark.parametrize("waarde", [None, "", "  "])
async def test_logregel_zonder_persoon_of_reden_wordt_geweigerd(
    live_surrealdb: SurrealDBConfig, veld: str, waarde: str | None
) -> None:
    over = {veld: waarde}
    with pytest.raises(Exception):
        await _logregel(live_surrealdb, **over)


async def test_logregel_wijzigen_of_verwijderen_wordt_geweigerd(live_surrealdb: SurrealDBConfig) -> None:
    rows = await _logregel(live_surrealdb)
    rid = str(rows[0]["id"])
    with pytest.raises(Exception):
        await _q(live_surrealdb, "UPDATE type::thing($id) SET reason = 'anders'", {"id": rid})
    with pytest.raises(Exception):
        await _q(live_surrealdb, "DELETE type::thing($id)", {"id": rid})
    with pytest.raises(Exception):
        await _q(live_surrealdb, "DELETE classification_log")
    assert (await _lees(live_surrealdb, rid))["reason"] == "openbaar gepubliceerd"


# --------------------------------------------------------------------------
# Heen en terug
# --------------------------------------------------------------------------


async def test_migratie_twee_keer_na_elkaar(live_surrealdb: SurrealDBConfig) -> None:
    await execute_transaction(_migratie(89), config=live_surrealdb)
    await execute_transaction(_migratie(89), config=live_surrealdb)
    bron = await _bron(live_surrealdb)
    assert (await _lees(live_surrealdb, bron))["classification"] == "intern"


async def test_5_down_is_uitvoerbaar(live_surrealdb: SurrealDBConfig) -> None:
    # Gevonden in P10: `REMOVE TABLE ... SCHEMAFULL` is ook op v2.7.0 een
    # parsefout. In een afgebroken transactie: de tabel blijft staan.
    # Een parsefout geeft een andere melding dan de afgebroken transactie.
    with pytest.raises(RuntimeError, match="cancelled transaction"):
        await execute_transaction(
            f"BEGIN TRANSACTION; {_migratie(5, down=True)} CANCEL TRANSACTION;",
            config=live_surrealdb,
        )
    info = await _q(live_surrealdb, "INFO FOR DB")
    assert "transformation" in (info[0] if isinstance(info, list) else info)["tables"]


async def test_down_draait_terug_en_zet_private_op_alle_bronnen(
    live_surrealdb: SurrealDBConfig,
) -> None:
    extern = await _bron(live_surrealdb, "extern")
    await _q(live_surrealdb, "UPDATE type::thing($id) SET private = false", {"id": extern})
    logregel = str((await _logregel(live_surrealdb))[0]["id"])
    try:
        await execute_transaction(_migratie(89, down=True), config=live_surrealdb)
        # Twee keer na elkaar: idempotent.
        await execute_transaction(_migratie(89, down=True), config=live_surrealdb)

        for tabel in EIGEN + AFGELEID:
            info = await _q(live_surrealdb, f"INFO FOR TABLE {tabel}")
            assert "classification" not in (info[0] if isinstance(info, list) else info)["fields"]
        db = await _q(live_surrealdb, "INFO FOR DB")
        db = db[0] if isinstance(db, list) else db
        # Het logboek blijft staan, met zijn regels; alleen het event en de
        # ASSERTs vervallen.
        assert "classification_log" in db["tables"]
        assert await _q(live_surrealdb, "SELECT * FROM type::thing($id)", {"id": logregel})
        log = await _q(live_surrealdb, "INFO FOR TABLE classification_log")
        log = log[0] if isinstance(log, list) else log
        assert log["events"] == {}
        assert not [f for f in log["fields"].values() if "ASSERT" in f]
        assert not [f for f in db["functions"] if "classificatie" in f]
        rows = await _q(live_surrealdb, "SELECT VALUE private FROM source")
        assert rows and all(p is True for p in rows)
    finally:
        await execute_transaction(_migratie(89), config=live_surrealdb)

    bron = await _bron(live_surrealdb)
    assert (await _lees(live_surrealdb, bron))["classification"] == "intern"
