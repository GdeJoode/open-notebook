"""Contracttest ``lezen_extern``: geen intern id of interne tekst (F2-05, ADR 0002).

De lektest uit ADR 0002 (Gevolgen) tegen een echte SurrealDB en de echte app,
met een sleutel ``lezen_extern`` (de sleutel van open-notebook-mcp):

* synthetische interne en externe gegevens voor bronnen, versies, fragmenten,
  embeddings, ``doc_node``, inzichten, notities, notebooknamen en
  -beschrijvingen, entiteiten, aliassen, relaties en ``mentions``, en
  bronkaartregels, batches en een wachtrij-item (F2-13);
* elke route voor ``lezen_extern`` wordt aangeroepen, ook met interne,
  ongeldige en vreemde id's, ongeldige invoer, ``history=true`` en een
  geforceerde fout in de repositorylaag;
* in geen antwoord en geen foutmelding staat een intern kenmerk of een intern
  id; een intern id geeft 404; een externe bron met ``source_insight`` geeft
  200; geen ``mentions`` van een interne bron.

Alle testdata is synthetisch. Interne teksten dragen een uniek kenmerk van
alleen letters (de BM25-analyzer splitst cijfers af).
"""

from __future__ import annotations

import hashlib
import math
import random
import re
import string
import uuid
from dataclasses import dataclass, field
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
import surrealdb_service.config as surreal_config
import surrealdb_service.connection as conn
from app_main import dependencies
from app_main.api import agent_auth
from app_main.api.app import create_app
from app_main.services.agents.audit_service import AgentAuditService
from app_main.services.search_service import SearchService
from fastapi.routing import APIRoute
from shared.models.agents import Right
from shared.models.classification import ClassificationScope
from surrealdb_service.repositories.entity import EntityRepository
from surrealdb_service.repositories.notebook import NotebookRepository, NoteRepository
from surrealdb_service.repositories.processing_run import ProcessingRunRepository
from surrealdb_service.repositories.search import SearchRepository
from surrealdb_service.repositories.source import SourceRepository
from surrealdb_service.repositories.source_version import SourceVersionRepository

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker, pytest.mark.echte_sleutels]

DIM = 7

#: Alle routes voor ``lezen_extern`` (ADR 0002 B). Komt er een bij, dan faalt
#: :func:`test_de_contracttest_dekt_elke_route_voor_lezen_extern` tot deze test
#: haar aanroept.
GEDEKT = {
    ("GET", "/"),
    ("GET", "/api/v1/auth/scope"),
    ("GET", "/api/notebooks"),
    ("GET", "/api/notebooks/{notebook_id}"),
    ("GET", "/api/sources/"),
    ("GET", "/api/sources/{source_id}"),
    ("GET", "/api/notes"),
    ("GET", "/api/notes/{note_id}"),
    ("POST", "/api/search"),
}


def _letters(n: int = 12) -> str:
    return "".join(random.choices(string.ascii_lowercase, k=n))


def _richting() -> list[float]:
    v = [random.uniform(-1, 1) for _ in range(DIM)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


@dataclass
class Gegevens:
    woord: str
    vector: list[float]
    kenmerk: str
    intern_ids: set[str] = field(default_factory=set)
    extern: dict[str, str] = field(default_factory=dict)
    intern: dict[str, str] = field(default_factory=dict)

    def lekken(self, tekst: str) -> list[str]:
        """De interne kenmerken en id's die in ``tekst`` staan."""
        gevonden = [self.kenmerk] if self.kenmerk in tekst.lower() else []
        gevonden += [i for i in self.intern_ids if re.search(re.escape(i) + r"(?![\w])", tekst)]
        kaal = {i.split(":", 1)[1] for i in self.intern_ids}
        gevonden += [k for k in kaal if k in tekst]
        return gevonden


async def _q(sql: str, params: dict | None = None):
    return await conn.execute_query(sql, params or {})


def _rid(waarde: str):
    return conn.ensure_record_id(waarde)


async def _bron_met_alles(g: Gegevens, classification: str) -> dict[str, str]:
    """Een bron met twee versies (een oude en de huidige), fragmenten,
    embeddings, ``doc_node``, een inzicht, een notebook en een notitie."""
    intern = classification == "intern"
    tekst = f"{g.woord} {g.kenmerk}" if intern else f"{g.woord} openbare tekst"
    titel = f"Titel {g.kenmerk}" if intern else "Openbare titel"
    [bron] = await _q(
        "CREATE source CONTENT {title: $t, classification: $c} RETURN id",
        {"t": titel, "c": classification},
    )
    ids = {"source": str(bron["id"])}
    for naam in ("oud", "huidig"):
        run = await ProcessingRunRepository().create(
            trigger="inname", pipeline_id="test", flow_vastgelegd=True
        )
        versie, _ = await SourceVersionRepository().register(
            ids["source"], hashlib.sha256(uuid.uuid4().bytes).hexdigest(), str(run.id)
        )
        herkomst = {"source": _rid(ids["source"]), "source_version": _rid(str(versie.id)),
                    "run": _rid(str(run.id))}
        [chunk] = await _q("CREATE chunk CONTENT $d RETURN id", {"d": {
            **herkomst, "text": f"{tekst} {naam}", "order": 0, "physical_page": 0,
            "element_type": "paragraph", "positions": [], "metadata": {}}})
        [emb] = await _q("CREATE source_embedding CONTENT $d RETURN id", {"d": {
            "source": herkomst["source"], "chunk": _rid(str(chunk["id"])), "run": herkomst["run"],
            "content": f"{tekst} {naam}", "order": 0, "embedding": g.vector,
            "embedding_model": "test"}})
        [node] = await _q("CREATE doc_node CONTENT $d RETURN id", {"d": {
            **herkomst, "element_type": "paragraph", "self_ref": f"#/texts/{_letters(6)}",
            "sequence": 0, "text": tekst}})
        ids.update({f"version_{naam}": str(versie.id), f"run_{naam}": str(run.id),
                    f"chunk_{naam}": str(chunk["id"]), f"embedding_{naam}": str(emb["id"]),
                    f"doc_node_{naam}": str(node["id"])})
    [inzicht] = await _q(
        "CREATE source_insight CONTENT {source: type::thing($b), insight_type: 'samenvatting',"
        " content: $t, embedding: $v} RETURN id",
        {"b": ids["source"], "t": tekst, "v": g.vector},
    )
    ids["insight"] = str(inzicht["id"])
    [notebook] = await _q(
        "CREATE notebook CONTENT {name: $n, description: $d, classification: $c} RETURN id",
        {"n": f"Notebook {g.kenmerk}" if intern else "Openbaar notebook",
         "d": f"Beschrijving {g.kenmerk}" if intern else "Openbare beschrijving",
         "c": classification},
    )
    ids["notebook"] = str(notebook["id"])
    await _q("RELATE (type::thing($b))->reference->(type::thing($n))",
             {"b": ids["source"], "n": ids["notebook"]})
    [notitie] = await _q(
        "CREATE note CONTENT {title: $t, content: $t, embedding: $v, classification: $c} RETURN id",
        {"t": tekst, "v": g.vector, "c": classification},
    )
    ids["note"] = str(notitie["id"])
    await _q("RELATE (type::thing($n))->artifact->(type::thing($b))",
             {"n": ids["note"], "b": ids["notebook"]})
    return ids


async def _graaf(g: Gegevens) -> None:
    """Entiteiten, alias, relatie en ``mentions``: één externe en één interne."""
    from shared.models.entity import Entity

    for soort, bron in (("extern", g.extern["source"]), ("intern", g.intern["source"])):
        naam = f"Entiteit {g.kenmerk}" if soort == "intern" else f"Entiteit openbaar {_letters(6)}"
        entiteit = str(await EntityRepository().upsert_entity(Entity(
            canonical_name=naam, entity_type="organization", source_documents=[bron])))
        [alias] = await _q(
            "CREATE entity_alias SET canonical_entity = type::thing($e), alias_text = $a,"
            " match_type = 'exact' RETURN id", {"e": entiteit, "a": naam})
        [relatie] = await _q(
            "RELATE (type::thing($a))->relation->(type::thing($a))"
            " SET relation_type = 'related_to', source_documents = [$s] RETURN id",
            {"a": entiteit, "s": bron})
        [vermelding] = await _q(
            "RELATE (type::thing($b))->mentions->(type::thing($e))"
            " SET weight = 0.5, concept_name = $n RETURN id",
            {"b": bron, "e": entiteit, "n": naam})
        doel = g.intern if soort == "intern" else g.extern
        doel.update({"entity": entiteit, "alias": str(alias["id"]),
                     "relation": str(relatie["id"]), "mention": str(vermelding["id"])})


async def _randgevallen(g: Gegevens) -> None:
    """Review F2-05: gevallen op routeniveau rond de externe bron.

    * de externe bron staat ook in het interne notebook;
    * de externe bron heeft een intern fragment in haar huidige versie en run;
    * de externe bron heeft een job met een foutmelding met interne tekst;
    * ``extraction_result`` met een relatietelling voor de externe bron.
    """
    bron = g.extern["source"]
    await _q("RELATE (type::thing($b))->reference->(type::thing($n))",
             {"b": bron, "n": g.intern["notebook"]})
    [chunk] = await _q("CREATE chunk CONTENT $d RETURN id", {"d": {
        "source": _rid(bron), "source_version": _rid(g.extern["version_huidig"]),
        "run": _rid(g.extern["run_huidig"]), "text": f"{g.woord} {g.kenmerk} fragment",
        "order": 1, "physical_page": 1, "element_type": "paragraph", "positions": [],
        "metadata": {}, "classification": "intern"}})
    g.intern["chunk_in_externe_bron"] = str(chunk["id"])
    [job] = await _q(
        "CREATE job CONTENT {job_type: 'process_source', status: 'failed',"
        " error_message: $m, source_id: $b} RETURN id",
        {"m": f"verwerking mislukt bij {g.kenmerk}", "b": bron},
    )
    g.extern["job"] = str(job["id"])  # de job hoort bij de externe bron
    await _q("UPDATE type::thing($b) SET command = type::thing($j)", {"b": bron, "j": str(job["id"])})
    await _q(
        "CREATE extraction_result CONTENT {source_id: $b, metadata: {persisted:"
        " {relations_created: 4, relations_merged: 1}}}",
        {"b": bron},
    )


async def _bronkaart(g: Gegevens) -> None:
    """F2-13: bronkaartregels, een batch en een wachtrij-item, intern en extern
    (bevestigd), met het kenmerk in naam, locatie en pad. Geen antwoord voor
    ``lezen_extern`` mag die tonen (ADR 0020)."""
    for soort in ("intern", "extern"):
        doel = g.intern if soort == "intern" else g.extern
        regel = {"name": f"Regel {g.kenmerk}", "route_type": "api", "source_type": "api",
                 "location": f"https://example.org/{g.kenmerk}/{soort}",
                 "pipeline_start": "automatisch", "route": "inname_api",
                 "fetch_content": "volledig_document", "classification": soort}
        if soort == "extern":
            regel["classification_confirmed_by"] = "Beheerder Test"
        [rij] = await _q("CREATE source_map_rule CONTENT $d RETURN id", {"d": regel})
        doel["source_map_rule"] = str(rij["id"])
        [batch] = await _q("CREATE intake_batch CONTENT $d RETURN id", {"d": {
            "route": "inname_pdf", "created_by": "Beheerder Test", "classification": soort,
            "files": [f"/data/batch/{g.kenmerk}.pdf"]}})
        doel["intake_batch"] = str(batch["id"])
    [item] = await _q("CREATE intake_item CONTENT $d RETURN id", {"d": {
        "origin": "watchfolder", "path": f"/data/inbox/{g.kenmerk}.pdf",
        "content_hash": "sha256:" + "d" * 64, "registered_by": "langflow"}})
    g.intern["intake_item"] = str(item["id"])


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


@pytest.fixture
async def gegevens(env) -> Gegevens:
    g = Gegevens(woord=_letters(), vector=_richting(), kenmerk="geheimkenmerk" + _letters(8))
    g.extern = await _bron_met_alles(g, "extern")
    g.intern = await _bron_met_alles(g, "intern")
    await _graaf(g)
    await _randgevallen(g)
    await _bronkaart(g)
    g.intern_ids = set(g.intern.values())
    # Controle op de testdata zelf: interne records zijn echt intern.
    rijen = await _q("SELECT VALUE classification FROM $ids",
                     {"ids": [_rid(i) for i in g.intern_ids
                              if not i.startswith(("processing_run:", "intake_item:"))]})
    assert set(rijen) == {"intern"}, rijen
    # Controle op de lekdetectie zelf.
    assert g.lekken(f'{{"id": "{g.intern["note"]}"}}') and g.lekken(g.kenmerk.upper())
    assert not g.lekken(f'{{"id": "{g.extern["note"]}", "t": "{g.woord}"}}')
    return g


@pytest.fixture
def lezen_extern(app, monkeypatch, gegevens):
    """Een client met een sleutel ``lezen_extern`` en een vast embeddingmodel."""
    monkeypatch.setattr(
        agent_auth.AgentKeyService, "authenticate",
        AsyncMock(return_value={"id": "agent_keys:mcp", "agent_id": "open-notebook-mcp",
                                "permission": "lezen_extern"}),
    )
    embedding = SimpleNamespace(aembed=AsyncMock(return_value=[gegevens.vector]))
    manager = SimpleNamespace(get_defaults=lambda: SimpleNamespace(default_embedding_model="model:test"))
    app.dependency_overrides[dependencies.get_search_service] = lambda: SearchService(
        search_repo=SearchRepository(), model_manager=manager, embedding_model=embedding)
    app.dependency_overrides[dependencies.get_model_service] = lambda: SimpleNamespace(
        model_manager=manager)
    client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
        base_url="http://t",
        headers={"Authorization": "Bearer ak_mcp"},
    )
    yield client
    app.dependency_overrides.clear()


def _geen_lek(g: Gegevens, r: httpx.Response, wat: str) -> None:
    lek = g.lekken(r.text)
    assert not lek, f"{wat} -> {r.status_code}: lekt {lek}"


def _routes_voor_lezen_extern(app) -> set[tuple[str, str]]:
    uit = set()
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        rechten = []

        def loop(dependant) -> None:
            for dep in dependant.dependencies:
                if getattr(dep.call, "required_right", None) is not None:
                    rechten.append(dep.call.required_right)
                loop(dep)

        loop(route.dependant)
        if rechten == [Right.LEZEN_EXTERN]:
            uit |= {(m, route.path) for m in route.methods - {"HEAD", "OPTIONS"}}
    return uit


def _ongeldige_ids(tabel: str) -> list[str]:
    return ["x", "%3B%20SELECT%20*%20FROM%20source", f"{tabel}:", "note:abc", "source:abc",
            "notebook:abc", "geen:geldig:id", "a" * 300]


# --------------------------------------------------------------------------
# Dekking
# --------------------------------------------------------------------------


async def test_de_contracttest_dekt_elke_route_voor_lezen_extern(app) -> None:
    assert _routes_voor_lezen_extern(app) == GEDEKT


# --------------------------------------------------------------------------
# Elke route, gewone en foute paden
# --------------------------------------------------------------------------


async def test_geen_intern_id_of_interne_tekst_in_een_antwoord(lezen_extern, gegevens) -> None:
    g, c = gegevens, lezen_extern
    verzoeken: list[tuple[str, str, dict | None]] = [
        ("GET", "/", None),
        ("GET", "/api/v1/auth/scope", None),
        ("GET", "/api/notebooks", None),
        ("GET", "/api/notebooks?archived=false&limit=100", None),
        ("GET", "/api/notebooks?order_by=name%20asc&limit=100", None),
        ("GET", f"/api/notebooks/{g.extern['notebook']}", None),
        ("GET", "/api/notes?limit=100", None),
        ("GET", f"/api/notes?notebook_id={g.extern['notebook']}", None),
        ("GET", f"/api/notes/{g.extern['note']}", None),
        ("GET", "/api/sources/?limit=100", None),
        ("GET", "/api/sources/?limit=100&sort_by=attention", None),
        ("GET", "/api/sources/?limit=100&stalled=true", None),
        ("GET", f"/api/sources/?notebook_id={g.extern['notebook']}", None),
        ("GET", f"/api/sources/{g.extern['source']}", None),
    ]
    for soort in ("text", "vector", "hybrid"):
        for history in (False, True):
            for vraag in (g.woord, g.kenmerk, f"{g.woord} {g.kenmerk}"):
                verzoeken.append(("POST", "/api/search", {
                    "query": vraag, "type": soort, "history": history, "limit": 1000,
                    "search_notes": True, "search_sources": True}))
    for methode, pad, body in verzoeken:
        r = await c.request(methode, pad, json=body)
        assert r.status_code == 200, f"{methode} {pad} {body}: {r.status_code} {r.text[:200]}"
        _geen_lek(g, r, f"{methode} {pad} {body}")


async def test_extern_is_zichtbaar(lezen_extern, gegevens) -> None:
    """De positieve controle: zonder die zou een leeg antwoord ook slagen."""
    g, c = gegevens, lezen_extern
    bron = await c.get(f"/api/sources/{g.extern['source']}")
    assert bron.status_code == 200  # een bron met `source_insight` (P4 K3)
    assert bron.json()["insights_count"] == 1
    assert bron.json()["classification"] == "extern"
    assert g.woord in (bron.json()["full_text"] or "")
    assert "huidig" in bron.json()["full_text"] and "oud" not in bron.json()["full_text"]
    assert bron.json()["notebooks"] == [g.extern["notebook"]]

    lijst = (await c.get("/api/sources/?limit=100")).json()
    assert g.extern["source"] in {s["id"] for s in lijst}
    notebooks = (await c.get("/api/notebooks?limit=100")).json()
    nb = next(n for n in notebooks if n["id"] == g.extern["notebook"])
    assert (nb["source_count"], nb["note_count"]) == (1, 1)
    notities = (await c.get(f"/api/notes?notebook_id={g.extern['notebook']}")).json()
    assert [n["id"] for n in notities] == [g.extern["note"]]

    for soort in ("text", "vector", "hybrid"):
        r = await c.post("/api/search", json={"query": g.woord, "type": soort})
        ids = {h["id"] for h in r.json()["results"]}
        assert g.extern["source"] in ids, soort
        for hit in r.json()["results"]:
            if hit["id"] == g.extern["source"]:
                assert g.extern["chunk_huidig"] in hit["chunks"], soort
                assert g.extern["chunk_oud"] not in hit["chunks"], soort
    r = await c.post("/api/search", json={"query": g.woord, "type": "text", "history": True})
    hit = next(h for h in r.json()["results"] if h["id"] == g.extern["source"])
    assert {g.extern["chunk_oud"], g.extern["chunk_huidig"]} <= set(hit["chunks"])


async def test_bronkaart_niet_leesbaar(lezen_extern, gegevens) -> None:
    """F2-13 (ADR 0020): ook een externe, bevestigde regel blijft onzichtbaar."""
    g, c = gegevens, lezen_extern
    paden = ["/api/v1/source-map-rules", "/api/v1/intake/batches",
             "/api/v1/intake/items?status=besloten"]
    for soort in (g.intern, g.extern):
        paden += [f"/api/v1/source-map-rules/{soort['source_map_rule']}",
                  f"/api/v1/intake/batches/{soort['intake_batch']}"]
    for pad in paden:
        r = await c.get(pad)
        assert r.status_code == 403, (pad, r.status_code)
        _geen_lek(g, r, pad)
        assert g.extern["source_map_rule"] not in r.text


@pytest.mark.parametrize("sleutel", ["source", "note", "notebook"])
async def test_een_intern_id_geeft_404(lezen_extern, gegevens, sleutel) -> None:
    g, c = gegevens, lezen_extern
    pad = {"source": "/api/sources/", "note": "/api/notes/", "notebook": "/api/notebooks/"}[sleutel]
    for waarde in (g.intern[sleutel], g.intern[sleutel].split(":", 1)[1]):
        r = await c.get(pad + waarde)
        assert r.status_code == 404, (waarde, r.status_code)
        _geen_lek(g, r, pad + waarde)
    # Hetzelfde antwoord als voor een id dat niet bestaat.
    bestaat_niet = await c.get(pad + f"{sleutel}:{_letters(20)}")
    assert bestaat_niet.status_code == 404
    assert (await c.get(pad + g.intern[sleutel])).json() == bestaat_niet.json()
    # Een filter op een intern notebook geeft 404, niet de inhoud.
    for lijst in ("/api/notes", "/api/sources/"):
        r = await c.get(f"{lijst}?notebook_id={g.intern['notebook']}")
        assert r.status_code == 404
        _geen_lek(g, r, lijst)


async def test_ongeldige_ids_en_invoer(lezen_extern, gegevens) -> None:
    g, c = gegevens, lezen_extern
    for pad, tabel in (("/api/sources/", "source"), ("/api/notes/", "note"),
                       ("/api/notebooks/", "notebook")):
        for waarde in _ongeldige_ids(tabel):
            r = await c.get(pad + waarde)
            assert r.status_code in (404, 405, 422), (pad, waarde, r.status_code)
            _geen_lek(g, r, pad + waarde)
    for pad in ("/api/notebooks?order_by=name;%20SELECT%20*%20FROM%20source",
                "/api/notebooks?order_by=onzin", "/api/notebooks?limit=0",
                "/api/sources/?sort_by=onzin", "/api/sources/?stage=onzin",
                "/api/sources/?sort_order=sideways", "/api/notes?limit=1000",
                "/api/notes?notebook_id=%3B%20DROP"):
        r = await c.get(pad)
        assert 400 <= r.status_code < 500, (pad, r.status_code)
        _geen_lek(g, r, pad)
    for body in ({"query": g.woord, "limit": 5000}, {"query": g.woord, "limit": 0},
                 {"query": g.woord, "type": "onzin"},
                 {"type": "text"}, {"query": g.kenmerk, "minimum_score": 3},
                 {"query": g.woord, "history": "misschien"}):
        r = await c.post("/api/search", json=body)
        assert r.status_code == 422, (body, r.status_code)
        _geen_lek(g, r, str(body))
    r = await c.post("/api/search", content=b"{niet: json", headers={"Content-Type": "application/json"})
    assert r.status_code == 422


async def test_geforceerde_fout_in_de_repositorylaag(lezen_extern, gegevens, monkeypatch) -> None:
    """Een fout met interne tekst in de uitzondering komt niet in het antwoord."""
    g, c = gegevens, lezen_extern
    melding = f"mislukt bij {g.intern['source']} met {g.kenmerk}"

    async def fout(*_args, **_kwargs):
        raise RuntimeError(melding)

    for klasse, naam in ((SourceRepository, "get_scoped"), (SourceRepository, "list_with_metadata"),
                         (SourceRepository, "get_text"), (SourceRepository, "count_related"),
                         (NoteRepository, "get_scoped"), (NoteRepository, "list_scoped"),
                         (NotebookRepository, "get_scoped"), (NotebookRepository, "list_scoped"),
                         (NotebookRepository, "get_notes_scoped"),
                         (SearchRepository, "text_search"), (SearchRepository, "vector_search")):
        monkeypatch.setattr(klasse, naam, fout)
    verzoeken = [
        ("GET", f"/api/sources/{g.extern['source']}", None),
        ("GET", "/api/sources/", None),
        ("GET", f"/api/notes/{g.extern['note']}", None),
        ("GET", "/api/notes", None),
        ("GET", f"/api/notebooks/{g.extern['notebook']}", None),
        ("GET", "/api/notebooks", None),
        ("POST", "/api/search", {"query": g.woord, "type": "text"}),
        ("POST", "/api/search", {"query": g.woord, "type": "hybrid"}),
    ]
    for methode, pad, body in verzoeken:
        r = await c.request(methode, pad, json=body)
        assert r.status_code == 500, (pad, r.status_code)
        _geen_lek(g, r, pad)
        assert "fout-id" in r.json()["detail"], r.text


async def test_geen_mentions_van_een_interne_bron(gegevens) -> None:
    g = gegevens
    extern = await EntityRepository().load_mentions_edges(scope=ClassificationScope.EXTERN)
    ids = {str(e["id"]) for e in extern}
    assert g.intern["mention"] not in ids
    assert g.extern["mention"] in ids
    assert not any(str(e["source"]) == g.intern["source"] for e in extern)
    alles = await EntityRepository().load_mentions_edges(scope=ClassificationScope.ALLES)
    assert g.intern["mention"] in {str(e["id"]) for e in alles}
    leeg = await EntityRepository().load_mentions_edges(scope=ClassificationScope.LEEG)
    assert leeg == []


async def test_randgevallen_op_routeniveau(lezen_extern, gegevens) -> None:
    """Review F2-05: gedeeld notebook, intern fragment, job-fout, relaties."""
    g, c = gegevens, lezen_extern
    r = await c.get(f"/api/sources/{g.extern['source']}")
    assert r.status_code == 200
    _geen_lek(g, r, "bron")
    bron = r.json()
    # In een intern én een extern notebook: alleen het externe is zichtbaar.
    assert bron["notebooks"] == [g.extern["notebook"]]
    # Een intern fragment in de huidige versie: niet in de tekst.
    assert g.woord in bron["full_text"] and g.kenmerk not in bron["full_text"].lower()
    # De job: status wel, foutmelding niet.
    assert bron["status"] == "failed"
    assert bron["processing_info"]["error"] is None
    # `extraction_result` kent geen classificatie: 0 voor `lezen_extern`.
    assert bron["relation_count"] == 0
    lijst = await c.get("/api/sources/?limit=100")
    _geen_lek(g, lijst, "bronlijst")
    rij = next(s for s in lijst.json() if s["id"] == g.extern["source"])
    assert rij["processing_info"]["error"] is None
    for soort in ("text", "vector", "hybrid"):
        r = await c.post("/api/search", json={"query": g.kenmerk, "type": soort})
        _geen_lek(g, r, f"zoeken {soort}")
        assert g.intern["chunk_in_externe_bron"] not in r.text
