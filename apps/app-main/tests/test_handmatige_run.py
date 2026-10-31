"""Samenvoegen en splitsen als handmatige run; bescherming van die run (F2-06).

ADR 0005 en ADR 0003 ("Handmatige correcties en optimistische vergrendeling"),
tegen een echte SurrealDB en via de routes met een sleutel ``beheerder``:

* samenvoegen geeft een run met ``trigger = handmatig``, de persoon van de
  sleutel als ``actor`` en een nieuwe fragmentenset; het aantal fragmenten van
  de vorige run blijft gelijk, hun tekst ook;
* splitsen werkt ook als de vorige run al is afgesloten (review F2-04);
* een fragment uit een intern fragment blijft intern;
* een automatische run op dezelfde bronversie verzet ``current_run`` niet en
  zet een melding in de wachtrij van de beheerder;
* ``resolution_log`` heeft een auteur.

Alle testdata is synthetisch.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import surrealdb_service.config as surreal_config
import surrealdb_service.connection as conn
from app_main.api.app import create_app
from app_main.services.agents.audit_service import AgentAuditService
from app_main.services.entity_resolution.merge_audit import record_merges
from app_main.services.source_extractor import ExtractionResult
from app_main.services.source_processor import SourceProcessor
from surrealdb_service.repositories import (
    ChunkRepository,
    ContentSettingsRepository,
    SourceRepository,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]

TEKSTEN = ["Eerste openbare alinea.", "Tweede openbare alinea.", "Derde openbare alinea."]


@pytest.fixture(scope="module")
def app():
    return create_app()


@pytest.fixture
async def env(live_surrealdb, monkeypatch):
    monkeypatch.setattr(surreal_config, "_config", live_surrealdb)
    monkeypatch.setattr(AgentAuditService, "record", AsyncMock())
    original = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    yield
    await conn._pool.close_all()
    conn._pool = original or conn.ConnectionPool(config=live_surrealdb)


def _client(app) -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://t",
        headers={"Authorization": "Bearer ak_test"},
    )


async def _q(sql: str, params: dict | None = None):
    return await conn.execute_query(sql, params or {})


def _rid(waarde) -> object:
    return conn.ensure_record_id(str(waarde))


class _Extractor:
    """Levert steeds dezelfde drie fragmenten: dezelfde inhoud, dezelfde hash."""

    async def extract(self, content_state, content_settings, *, source_id=None):
        return ExtractionResult(
            full_text="\n".join(TEKSTEN),
            chunks=[{"text": t, "order": i, "physical_page": 0, "positions": [[0, 0.1, 0.9, 0.1 * i, 0.1 * i + 0.05]]}
                    for i, t in enumerate(TEKSTEN)],
            metadata={"parser_engine_used": "test"},
        )


def _processor() -> SourceProcessor:
    return SourceProcessor(
        source_repo=SourceRepository(),
        chunk_repo=ChunkRepository(),
        settings_repo=ContentSettingsRepository(),
        extractor=_Extractor(),
        graph_builder=SimpleNamespace(build=AsyncMock(return_value=SimpleNamespace(nodes=[]))),
    )


async def _verwerkte_bron(classification: str = "extern") -> str:
    [rij] = await _q("CREATE source SET title = 'Openbare testbron', classification = $c RETURN id",
                     {"c": classification})
    bron = str(rij["id"])
    await _processor().process_source(bron, {"content": "\n".join(TEKSTEN)})
    return bron


async def _stand(bron: str) -> dict:
    [rij] = await _q(
        "SELECT current_version AS versie, current_version.current_run AS run FROM $b", {"b": _rid(bron)}
    )
    return rij


async def _fragmenten(run) -> list[dict]:
    return await _q("SELECT * FROM chunk WHERE run = $r ORDER BY order ASC", {"r": _rid(run)})


async def _run(run) -> dict:
    [rij] = await _q("SELECT * FROM $r", {"r": _rid(run)})
    return rij


async def test_samenvoegen_geeft_een_handmatige_run_met_nieuwe_fragmenten(env, app) -> None:
    bron = await _verwerkte_bron()
    oud = await _stand(bron)
    oude_run = oud["run"]
    oude = await _fragmenten(oude_run)
    assert [c["text"] for c in oude] == TEKSTEN
    assert (await _run(oude_run))["status"] == "geslaagd"  # al afgesloten

    async with _client(app) as client:
        resp = await client.post(f"/api/sources/{bron}/chunks/{oude[0]['id']}/merge", json={})
    assert resp.status_code == 200, resp.text
    samengevoegd = resp.json()

    nieuw = await _stand(bron)
    assert nieuw["versie"] == oud["versie"]
    assert nieuw["run"] != oude_run
    run = await _run(nieuw["run"])
    assert (run["trigger"], run["actor"], run["status"]) == ("handmatig", "Testpersoon", "geslaagd")
    assert run["flow_vastgelegd"] is True

    nieuwe = await _fragmenten(nieuw["run"])
    assert [c["text"] for c in nieuwe] == [f"{TEKSTEN[0]}\n\n{TEKSTEN[1]}", TEKSTEN[2]]
    assert str(nieuwe[0]["id"]) == samengevoegd["id"]
    assert all(c["extractor"] == "handmatig" and c["source_version"] == oud["versie"] for c in nieuwe)
    assert all(c["classification"] == "extern" for c in nieuwe)
    assert nieuwe[0]["metadata"]["afgeleid_van"] == [str(oude[0]["id"]), str(oude[1]["id"])]
    assert len(nieuwe[0]["positions"]) == 2

    # De vorige run is onaangeroerd.
    assert [(str(c["id"]), c["text"]) for c in await _fragmenten(oude_run)] == [
        (str(c["id"]), c["text"]) for c in oude
    ]


async def test_splitsen_na_een_afgesloten_run(env, app) -> None:
    bron = await _verwerkte_bron()
    oude_run = (await _stand(bron))["run"]
    oude = await _fragmenten(oude_run)

    async with _client(app) as client:
        resp = await client.post(f"/api/sources/{bron}/chunks/{oude[1]['id']}/split",
                                 json={"cursorOffset": 6})
    assert resp.status_code == 200, resp.text
    [eerste, tweede] = resp.json()["chunks"]
    assert (eerste["text"], tweede["text"]) == (TEKSTEN[1][:6], TEKSTEN[1][6:])

    nieuwe_run = (await _stand(bron))["run"]
    assert (await _run(nieuwe_run))["trigger"] == "handmatig"
    assert [c["text"] for c in await _fragmenten(nieuwe_run)] == [
        TEKSTEN[0], TEKSTEN[1][:6], TEKSTEN[1][6:], TEKSTEN[2]
    ]
    assert len(await _fragmenten(oude_run)) == 3


async def test_een_fragment_van_een_vorige_run_is_niet_te_bewerken(env, app) -> None:
    bron = await _verwerkte_bron()
    oude = await _fragmenten((await _stand(bron))["run"])
    async with _client(app) as client:
        assert (await client.post(f"/api/sources/{bron}/chunks/{oude[0]['id']}/merge", json={})).status_code == 200
        # Het fragment hoort nu bij de vorige run.
        resp = await client.post(f"/api/sources/{bron}/chunks/{oude[0]['id']}/split", json={"cursorOffset": 3})
    assert resp.status_code == 400


async def test_een_intern_fragment_blijft_intern(env, app) -> None:
    bron = await _verwerkte_bron()
    oude = await _fragmenten((await _stand(bron))["run"])
    # Eén fragment van de externe bron is intern (strenger mag).
    await _q("UPDATE $c SET classification = 'intern'", {"c": _rid(oude[1]["id"])})

    async with _client(app) as client:
        resp = await client.post(f"/api/sources/{bron}/chunks/{oude[0]['id']}/merge", json={})
    assert resp.status_code == 200, resp.text
    nieuwe = await _fragmenten((await _stand(bron))["run"])
    assert [c["classification"] for c in nieuwe] == ["intern", "extern"]
    # Het interne fragment komt niet in de externe zoekindex.
    assert not await _q("SELECT * FROM chunk_extern WHERE chunk = $c", {"c": _rid(nieuwe[0]["id"])})


async def test_automatische_run_verzet_de_handmatige_run_niet(env, app) -> None:
    bron = await _verwerkte_bron()
    oude = await _fragmenten((await _stand(bron))["run"])
    async with _client(app) as client:
        assert (await client.post(f"/api/sources/{bron}/chunks/{oude[0]['id']}/merge", json={})).status_code == 200
    handmatig = await _stand(bron)

    # Herverwerking met dezelfde inhoud: dezelfde bronversie, een automatische run.
    await _processor().process_source(bron, {"content": "\n".join(TEKSTEN)})

    na = await _stand(bron)
    assert na == handmatig
    [auto] = await _q(
        "SELECT * FROM processing_run WHERE trigger = 'herverwerking' AND id IN"
        " (SELECT VALUE run FROM chunk WHERE source = $b)", {"b": _rid(bron)}
    )
    # De automatische run maakte wel nieuwe fragmenten, en logde de bescherming.
    assert len(await _fragmenten(auto["id"])) == 3
    assert ("huidige_run", "beschermd") in [(r["step"], r["status"]) for r in auto["log"]]
    [melding] = await _q("SELECT * FROM beheer_melding WHERE source = $b", {"b": _rid(bron)})
    assert melding["soort"] == "handmatige_run_beschermd" and melding["status"] == "open"
    assert str(melding["run"]) == str(auto["id"])
    assert str(melding["current_run"]) == str(handmatig["run"])
    assert str(melding["source_version"]) == str(handmatig["versie"])


async def test_mislukte_transactie_laat_alles_staan(env, monkeypatch) -> None:
    import app_main.services.chunking.chunk_mutator as mutator_mod
    from app_main.services.chunking.chunk_mutator import ChunkMutator

    bron = await _verwerkte_bron()
    voor = await _stand(bron)
    oude = await _fragmenten(voor["run"])

    async def _faalt(*_a, **_k):
        raise RuntimeError("geforceerde fout")

    monkeypatch.setattr(mutator_mod, "execute_transaction", _faalt)
    with pytest.raises(RuntimeError):
        await ChunkMutator().merge(bron, str(oude[0]["id"]), actor="Testpersoon")
    assert await _stand(bron) == voor
    [run] = await _q("SELECT * FROM processing_run WHERE trigger = 'handmatig' AND actor = 'Testpersoon'"
                     " AND status = 'mislukt' AND id NOT IN (SELECT VALUE run FROM chunk)"
                     " ORDER BY started_at DESC LIMIT 1")
    assert run["error"] == "RuntimeError"
    assert len(await _fragmenten(voor["run"])) == 3


async def test_bewerking_komt_in_chunk_edit_met_actor(env, app) -> None:
    bron = await _verwerkte_bron()
    oude = await _fragmenten((await _stand(bron))["run"])
    async with _client(app) as client:
        assert (await client.post(f"/api/sources/{bron}/chunks/{oude[0]['id']}/merge", json={})).status_code == 200
    [rij] = await _q("SELECT op, actor FROM chunk_edit WHERE source = $b", {"b": _rid(bron)})
    assert (rij["op"], rij["actor"]) == ("merge", "Testpersoon")


async def test_resolution_log_heeft_een_auteur(env) -> None:
    [a] = await _q("CREATE entity CONTENT {canonical_name: 'A', name: 'A', name_key: 'a-f206',"
                   " entity_type: 'organization', hash_id: 'f206a', embedding: []} RETURN id")
    [b] = await _q("CREATE entity CONTENT {canonical_name: 'B', name: 'B', name_key: 'b-f206',"
                   " entity_type: 'organization', hash_id: 'f206b', embedding: []} RETURN id")
    n = await record_merges(
        winner_id=str(a["id"]), merged_loser_ids=[str(b["id"])], names={}, types={},
        origin="curator_approved", method="test", author="Testpersoon",
    )
    assert n == 1
    [rij] = await _q("SELECT author FROM resolution_log WHERE winner_id = $w", {"w": str(a["id"])})
    assert rij["author"] == "Testpersoon"


# --------------------------------------------------------------------------
# Randgevallen van samenvoegen en splitsen (terug uit test_chunk_mutator.py)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("basis, doel", [(1, 0), (1, 2)])
async def test_samenvoegen_met_een_aangrenzend_doel(env, app, basis, doel) -> None:
    bron = await _verwerkte_bron()
    oude = await _fragmenten((await _stand(bron))["run"])
    async with _client(app) as client:
        resp = await client.post(f"/api/sources/{bron}/chunks/{oude[basis]['id']}/merge",
                                 json={"target_chunk_id": str(oude[doel]["id"])})
    assert resp.status_code == 200, resp.text
    eerste, tweede = sorted((basis, doel))
    assert resp.json()["text"] == f"{TEKSTEN[eerste]}\n\n{TEKSTEN[tweede]}"
    assert len(await _fragmenten((await _stand(bron))["run"])) == 2


async def _geweigerd(app, bron: str, pad: str, body: dict) -> None:
    voor = await _stand(bron)
    async with _client(app) as client:
        resp = await client.post(f"/api/sources/{bron}/chunks/{pad}", json=body)
    assert resp.status_code == 400, resp.text
    assert await _stand(bron) == voor
    # Een mislukte bewerking schrijft geen auditregel.
    assert not await _q("SELECT id FROM chunk_edit WHERE source = $b", {"b": _rid(bron)})


async def test_niet_aangrenzend_doel_wordt_geweigerd(env, app) -> None:
    bron = await _verwerkte_bron()
    oude = await _fragmenten((await _stand(bron))["run"])
    await _geweigerd(app, bron, f"{oude[0]['id']}/merge", {"target_chunk_id": str(oude[2]["id"])})


async def test_het_laatste_fragment_heeft_geen_opvolger(env, app) -> None:
    bron = await _verwerkte_bron()
    oude = await _fragmenten((await _stand(bron))["run"])
    await _geweigerd(app, bron, f"{oude[2]['id']}/merge", {})


@pytest.mark.parametrize("positie", [0, len(TEKSTEN[0]), len(TEKSTEN[0]) + 5])
async def test_cursoroffset_buiten_bereik(env, app, positie) -> None:
    bron = await _verwerkte_bron()
    oude = await _fragmenten((await _stand(bron))["run"])
    await _geweigerd(app, bron, f"{oude[0]['id']}/split", {"cursorOffset": positie})


async def test_mislukte_transactie_schrijft_geen_auditregel(env, monkeypatch) -> None:
    import app_main.services.chunking.chunk_mutator as mutator_mod
    from app_main.services.audit.chunk_audit import ChunkAuditService
    from app_main.services.chunking.chunk_mutator import ChunkMutator

    bron = await _verwerkte_bron()
    oude = await _fragmenten((await _stand(bron))["run"])

    async def _faalt(*_a, **_k):
        raise RuntimeError("geforceerde fout")

    monkeypatch.setattr(mutator_mod, "execute_transaction", _faalt)
    with pytest.raises(RuntimeError):
        await ChunkMutator(audit=ChunkAuditService()).merge(
            bron, str(oude[0]["id"]), actor="Testpersoon")
    assert not await _q("SELECT id FROM chunk_edit WHERE source = $b", {"b": _rid(bron)})


# --------------------------------------------------------------------------
# Twee bewerkingen op dezelfde run (optimistische vergrendeling)
# --------------------------------------------------------------------------


async def test_tweede_bewerking_op_dezelfde_run_geeft_409(env, app, monkeypatch) -> None:
    from app_main.services.chunking.chunk_mutator import ChunkMutator

    bron = await _verwerkte_bron()
    voor = await _stand(bron)
    oude = await _fragmenten(voor["run"])
    # Beide bewerkingen lezen dezelfde stand; de eerste verzet de run.
    gelezen = await ChunkMutator()._huidige_set(bron)

    async with _client(app) as client:
        assert (await client.post(f"/api/sources/{bron}/chunks/{oude[0]['id']}/merge",
                                  json={})).status_code == 200
        na_eerste = await _stand(bron)

        async def _oude_stand(self, source_id):
            return gelezen

        monkeypatch.setattr(ChunkMutator, "_huidige_set", _oude_stand)
        resp = await client.post(f"/api/sources/{bron}/chunks/{oude[1]['id']}/split",
                                 json={"cursorOffset": 4})
    assert resp.status_code == 409, resp.text
    # De tweede bewerking wijzigt niets: de huidige run blijft die van de eerste.
    assert await _stand(bron) == na_eerste
    [mislukt] = await _q(
        "SELECT * FROM processing_run WHERE trigger = 'handmatig' AND status = 'mislukt'"
        " AND error = 'verouderd' ORDER BY started_at DESC LIMIT 1")
    assert not await _fragmenten(mislukt["id"])
