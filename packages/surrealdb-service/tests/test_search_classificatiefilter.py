"""Classificatiefilter en nieuwste versie in zoeken (F2-05, ADR 0002 C, 0003, 0004).

Tegen een echte SurrealDB met migratie 91:

* met bereik ``EXTERN`` 0 interne records bij tekst-, vector- en hybride
  zoeken, ook met ``history=True``;
* standaard alleen de fragmenten van de huidige bronversie en run;
* notities en inzichten komen nooit terug; elk resultaat verwijst naar
  fragmenten;
* een zoek- of leesmethode zonder ``scope`` geeft ``TypeError``;
* volledigheid: het filter staat vóór de rangschikking, dus interne treffers
  nemen geen plaats in binnen ``results`` (gemeten en vastgelegd).

Alle testdata is synthetisch. Elke test maakt eigen, unieke zoekwoorden en een
eigen richting in de vectorruimte, zodat records uit andere tests niet meetellen.
"""

from __future__ import annotations

import hashlib
import inspect
import math
import random
import string
import uuid

import pytest
from loguru import logger
from shared.models.classification import ClassificationScope
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query, execute_transaction
from surrealdb_service.repositories.entity import EntityRepository
from surrealdb_service.repositories.notebook import NotebookRepository, NoteRepository
from surrealdb_service.repositories.processing_run import ProcessingRunRepository
from surrealdb_service.repositories.search import SearchRepository
from surrealdb_service.repositories.source import ChunkRepository, SourceRepository
from surrealdb_service.repositories.source_version import SourceVersionRepository
from surrealdb_service.testing import fixtures as fx

pytestmark = [pytest.mark.requires_docker, pytest.mark.asyncio]

EXTERN = ClassificationScope.EXTERN
ALLES = ClassificationScope.ALLES
DIM = 7


@pytest.fixture(autouse=True, scope="module")
def _migraties_bovenop(live_surrealdb: SurrealDBConfig) -> None:
    """De database is per sessie gedeeld; andere tests voeren oude migraties
    opnieuw uit. 89 tot en met 91 zijn idempotent."""
    import asyncio

    import surrealdb_service.connection as conn
    from surrealdb_service.migrations import AsyncMigration

    async def _op() -> None:
        for versie in (89, 90, 91):
            sql = AsyncMigration.from_file(fx._MIGRATIONS_DIR / f"{versie}.surrealql").sql
            await execute_transaction(sql, config=live_surrealdb, _herclassificatie=True)

    asyncio.run(_op())
    conn._pool = None


def _woord() -> str:
    """Een uniek zoekwoord van alleen letters (de analyzer splitst cijfers af)."""
    return "".join(random.choices(string.ascii_lowercase, k=14))


def _richting() -> list[float]:
    v = [random.uniform(-1, 1) for _ in range(DIM)]
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


class Data:
    """Bronnen met fragmenten en embeddings, per test vers."""

    def __init__(self, config: SurrealDBConfig) -> None:
        self.config = config

    async def q(self, sql: str, params: dict | None = None):
        return await execute_query(sql, params, self.config)

    async def bron(self, classification: str) -> str:
        [rij] = await self.q(
            "CREATE source CONTENT {title: 'Openbare testbron', classification: $c} RETURN id",
            {"c": classification},
        )
        return str(rij["id"])

    async def versie(self, bron: str) -> tuple[str, str]:
        run = await ProcessingRunRepository(self.config).create(
            trigger="inname", pipeline_id="test", flow_vastgelegd=True
        )
        versie, _ = await SourceVersionRepository(self.config).register(
            bron, hashlib.sha256(uuid.uuid4().bytes).hexdigest(), str(run.id)
        )
        return str(versie.id), str(run.id)

    async def fragment(
        self, bron: str, versie: str, run: str, tekst: str, vector: list[float] | None
    ) -> str:
        [chunk] = await self.q("CREATE chunk CONTENT $d RETURN id", {"d": {
            "source": ensure_record_id(bron), "source_version": ensure_record_id(versie),
            "run": ensure_record_id(run), "text": tekst, "order": 0, "physical_page": 0,
            "element_type": "paragraph", "positions": [], "metadata": {}}})
        if vector is not None:
            await self.q("CREATE source_embedding CONTENT $d", {"d": {
                "source": ensure_record_id(bron), "chunk": ensure_record_id(str(chunk["id"])), "run": ensure_record_id(run),
                "content": tekst, "order": 0, "embedding": vector, "embedding_model": "test"}})
        return str(chunk["id"])

    async def bron_met_fragment(self, classification: str, tekst: str, vector: list[float]) -> str:
        bron = await self.bron(classification)
        versie, run = await self.versie(bron)
        await self.fragment(bron, versie, run, tekst, vector)
        return bron


@pytest.fixture
def data(live_surrealdb) -> Data:
    return Data(live_surrealdb)


def _ids(hits) -> set[str]:
    return {str(h["id"]) for h in hits}


async def _zoek_alle_soorten(repo: SearchRepository, woord: str, vector, *, scope, history=False):
    return {
        "tekst": await repo.text_search(woord, 50, scope=scope, history=history),
        "vector": await repo.vector_search(vector, 50, 0.99, scope=scope, history=history),
        "hybride": await repo.hybrid_search(woord, vector, 50, 0.99, scope=scope, history=history),
    }


# --------------------------------------------------------------------------
# Classificatiefilter
# --------------------------------------------------------------------------


@pytest.mark.parametrize("history", [False, True])
async def test_extern_ziet_geen_interne_treffers(live_surrealdb, data, history) -> None:
    woord, vector = _woord(), _richting()
    extern = await data.bron_met_fragment("extern", f"openbaar {woord}", vector)
    intern = await data.bron_met_fragment("intern", f"vertrouwelijk {woord}", vector)
    repo = SearchRepository(live_surrealdb)

    for soort, hits in (await _zoek_alle_soorten(repo, woord, vector, scope=EXTERN, history=history)).items():
        assert intern not in _ids(hits), soort
        assert extern in _ids(hits), soort
        for hit in hits:
            assert str(hit["id"]).startswith("source:")
            chunks = await data.q(
                "SELECT VALUE classification FROM $c", {"c": [ensure_record_id(str(c)) for c in hit["chunks"]]}
            )
            assert set(chunks) == {"extern"}, soort

    for soort, hits in (await _zoek_alle_soorten(repo, woord, vector, scope=ALLES, history=history)).items():
        assert {extern, intern} <= _ids(hits), soort


async def test_leeg_bereik_ziet_niets(live_surrealdb, data) -> None:
    woord, vector = _woord(), _richting()
    await data.bron_met_fragment("extern", woord, vector)
    hits = await _zoek_alle_soorten(
        SearchRepository(live_surrealdb), woord, vector, scope=ClassificationScope.LEEG
    )
    assert all(not h for h in hits.values())


async def test_intern_fragment_in_een_externe_bron_blijft_verborgen(live_surrealdb, data) -> None:
    """Een fragment mag strenger zijn dan zijn bron; zoeken filtert op het fragment."""
    woord, vector = _woord(), _richting()
    bron = await data.bron("extern")
    versie, run = await data.versie(bron)
    chunk = await data.fragment(bron, versie, run, woord, None)
    await data.q("UPDATE type::thing($c) SET classification = 'intern'", {"c": chunk})
    await data.q("CREATE source_embedding CONTENT $d", {"d": {
        "source": ensure_record_id(bron), "chunk": ensure_record_id(chunk), "run": ensure_record_id(run),
        "content": woord, "order": 0, "embedding": vector, "embedding_model": "test"}})
    hits = await _zoek_alle_soorten(SearchRepository(live_surrealdb), woord, vector, scope=EXTERN)
    assert all(bron not in _ids(h) for h in hits.values())


# --------------------------------------------------------------------------
# Nieuwste versie
# --------------------------------------------------------------------------


async def test_standaard_alleen_de_huidige_versie_en_run(live_surrealdb, data) -> None:
    oud_woord, nieuw_woord = _woord(), _woord()
    oud_vector, nieuw_vector = _richting(), _richting()
    bron = await data.bron("extern")
    v1, r1 = await data.versie(bron)
    await data.fragment(bron, v1, r1, oud_woord, oud_vector)
    v2, r2 = await data.versie(bron)
    await data.fragment(bron, v2, r2, nieuw_woord, nieuw_vector)
    repo = SearchRepository(live_surrealdb)

    assert bron in _ids(await repo.text_search(nieuw_woord, 10, scope=EXTERN))
    assert bron not in _ids(await repo.text_search(oud_woord, 10, scope=EXTERN))
    assert bron not in _ids(await repo.vector_search(oud_vector, 10, 0.99, scope=EXTERN))
    assert bron in _ids(await repo.text_search(oud_woord, 10, scope=EXTERN, history=True))
    assert bron in _ids(await repo.vector_search(oud_vector, 10, 0.99, scope=EXTERN, history=True))

    # Een herverwerking van v2 (nieuwe run) verdringt de oude run.
    r3 = await ProcessingRunRepository(live_surrealdb).create(
        trigger="herverwerking", pipeline_id="test", flow_vastgelegd=True
    )
    await SourceVersionRepository(live_surrealdb).set_current_run(v2, str(r3.id))
    assert bron not in _ids(await repo.text_search(nieuw_woord, 10, scope=EXTERN))


# --------------------------------------------------------------------------
# Alleen fragmenten (ADR 0004)
# --------------------------------------------------------------------------


async def test_notities_en_inzichten_komen_niet_terug(live_surrealdb, data) -> None:
    woord, vector = _woord(), _richting()
    bron = await data.bron("extern")
    [notitie] = await data.q(
        "CREATE note CONTENT {title: $w, content: $w, embedding: $v, classification: 'extern'} RETURN id",
        {"w": woord, "v": vector},
    )
    [inzicht] = await data.q(
        "CREATE source_insight CONTENT {source: type::thing($b), insight_type: 't', content: $w,"
        " embedding: $v} RETURN id",
        {"b": bron, "w": woord, "v": vector},
    )
    hits = await _zoek_alle_soorten(SearchRepository(live_surrealdb), woord, vector, scope=ALLES, history=True)
    for soort, rijen in hits.items():
        assert not rijen, soort
    assert notitie and inzicht


# --------------------------------------------------------------------------
# Verplicht bereik
# --------------------------------------------------------------------------


async def test_zoekmethode_zonder_scope_geeft_typeerror(live_surrealdb) -> None:
    repo = SearchRepository(live_surrealdb)
    with pytest.raises(TypeError):
        await repo.text_search("x", 10)
    with pytest.raises(TypeError):
        await repo.vector_search([0.1] * DIM, 10)
    with pytest.raises(TypeError):
        await repo.hybrid_search("x", [0.1] * DIM, 10)


#: De leesmethoden voor de API (F2-05, ADR 0002 C): `scope` is verplicht,
#: zonder standaard, en alleen als trefwoord.
LEESMETHODEN = [
    (SearchRepository, "text_search"),
    (SearchRepository, "vector_search"),
    (SearchRepository, "hybrid_search"),
    (SearchRepository, "hydrate_provenance"),
    (SourceRepository, "get_scoped"),
    (SourceRepository, "list_with_metadata"),
    (SourceRepository, "get_text"),
    (SourceRepository, "get_notebook_ids_scoped"),
    (ChunkRepository, "get_texts"),
    (SourceRepository, "count_related"),
    (NoteRepository, "get_scoped"),
    (NoteRepository, "list_scoped"),
    (NotebookRepository, "get_scoped"),
    (NotebookRepository, "list_scoped"),
    (NotebookRepository, "get_notes_scoped"),
    (EntityRepository, "load_mentions_edges"),
]


@pytest.mark.parametrize(("klasse", "naam"), LEESMETHODEN, ids=lambda x: getattr(x, "__name__", x))
def test_leesmethode_vraagt_scope_zonder_standaard(klasse, naam) -> None:
    param = inspect.signature(getattr(klasse, naam)).parameters["scope"]
    assert param.kind is inspect.Parameter.KEYWORD_ONLY
    assert param.default is inspect.Parameter.empty


async def test_leesmethode_zonder_scope_faalt(live_surrealdb) -> None:
    with pytest.raises(TypeError):
        await SourceRepository(live_surrealdb).get_scoped("source:x")
    with pytest.raises(TypeError):
        await NoteRepository(live_surrealdb).get_scoped("note:x")
    with pytest.raises(TypeError):
        await NotebookRepository(live_surrealdb).get_scoped("notebook:x")
    with pytest.raises(TypeError):
        await SourceRepository(live_surrealdb).get_scoped("source:x", scope="ALLES")


# --------------------------------------------------------------------------
# Volledigheid (ADR 0002 C, risico R12)
# --------------------------------------------------------------------------


async def test_volledigheid_filter_voor_de_rangschikking(live_surrealdb, data) -> None:
    """Interne treffers die beter scoren, verdringen geen externe treffers.

    Tien interne bronnen staan dichter bij de zoekvector (en hebben het
    zoekwoord vaker) dan vijf externe. Met ``results=5`` en bereik ``EXTERN``
    moeten alle vijf externe bronnen terugkomen.
    """
    woord, vector = _woord(), _richting()
    extern, intern = set(), set()
    for i in range(10):
        intern.add(await data.bron_met_fragment("intern", f"{woord} {woord} {woord}", vector))
    schuin = [x + 0.02 * (1 if i % 2 else -1) for i, x in enumerate(vector)]
    n = math.sqrt(sum(x * x for x in schuin))
    schuin = [x / n for x in schuin]
    for i in range(5):
        extern.add(await data.bron_met_fragment("extern", f"{woord} tekst", schuin))
    repo = SearchRepository(live_surrealdb)

    meting = {}
    for soort, hits in {
        "tekst": await repo.text_search(woord, 5, scope=EXTERN),
        "vector": await repo.vector_search(vector, 5, 0.9, scope=EXTERN),
        "hybride": await repo.hybrid_search(woord, vector, 5, 0.9, scope=EXTERN),
    }.items():
        assert _ids(hits) == extern, soort
        meting[soort] = len(_ids(hits) & extern)
    alles = {
        "tekst": _ids(await repo.text_search(woord, 50, scope=ALLES)),
        "vector": _ids(await repo.vector_search(vector, 50, 0.9, scope=ALLES)),
    }
    assert all(ids == extern | intern for ids in alles.values())
    logger.info(
        "Volledigheid zoeken met filter (results=5, 5 extern, 10 intern beter "
        f"gerangschikt): extern gevonden {meting}; zonder filter tekst "
        f"{len(alles['tekst'])}, vector {len(alles['vector'])} van 15"
    )


# --------------------------------------------------------------------------
# BM25 verraadt geen interne documentfrequentie (review F2-05)
# --------------------------------------------------------------------------


async def test_score_en_volgorde_extern_onafhankelijk_van_interne_fragmenten(
    live_surrealdb, data
) -> None:
    """Met bereik ``EXTERN`` telt de BM25-index alleen externe fragmenten mee.

    Interne fragmenten met hetzelfde woord veranderen score en volgorde van
    externe treffers niet, ook niet in het tekstdeel van hybride zoeken.
    """
    woord, vector = _woord(), _richting()
    await data.bron_met_fragment("extern", f"{woord} openbare tekst een twee drie", vector)
    await data.bron_met_fragment("extern", f"{woord} {woord} tweede openbare bron", _richting())
    repo = SearchRepository(live_surrealdb)

    async def meting():
        tekst = await repo.text_search(woord, 10, scope=EXTERN)
        hybride = await repo.hybrid_search(woord, vector, 10, 0.99, scope=EXTERN)
        return (
            [(h["id"], h["relevance"]) for h in tekst],
            [(h["id"], h["_combined_score"], h["_text_rank"]) for h in hybride],
        )

    voor = await meting()
    for _ in range(15):
        await data.bron_met_fragment("intern", f"{woord} geheim", _richting())
    na = await meting()
    assert len(voor[0]) == 2
    assert voor == na


async def test_externe_index_volgt_herclassificatie(live_surrealdb, data) -> None:
    woord = _woord()
    bron = await data.bron_met_fragment("extern", f"{woord} openbaar", _richting())
    repo = SearchRepository(live_surrealdb)
    bronnen = SourceRepository(live_surrealdb)

    async def extern_gevonden() -> bool:
        return bron in _ids(await repo.text_search(woord, 10, scope=EXTERN))

    async def in_externe_index() -> set:
        return set(await data.q(
            "SELECT VALUE text FROM chunk_extern WHERE chunk.source = type::thing($b)", {"b": bron}))

    assert await extern_gevonden() and await in_externe_index() == {f"{woord} openbaar"}
    await bronnen.reclassify(bron, from_classification="extern", to_classification="intern",
                             reason="Synthetische test", agent_key="agent_keys:t", person="Test")
    assert await in_externe_index() == set()
    assert not await extern_gevonden()
    assert bron in _ids(await repo.text_search(woord, 10, scope=ALLES))
    await bronnen.reclassify(bron, from_classification="intern", to_classification="extern",
                             reason="Synthetische test", agent_key="agent_keys:t", person="Test")
    assert await in_externe_index() == {f"{woord} openbaar"}
    assert await extern_gevonden()


async def test_fragmentteksten_en_huidige_fragmenten(live_surrealdb, data) -> None:
    """``ChunkRepository.get_texts`` filtert op bereik; ``get_by_source(huidig=True)``
    geeft alleen de fragmenten van de huidige versie en run (review F2-05)."""
    bron = await data.bron("extern")
    v1, r1 = await data.versie(bron)
    oud = await data.fragment(bron, v1, r1, "oude tekst", None)
    v2, r2 = await data.versie(bron)
    nieuw = await data.fragment(bron, v2, r2, "nieuwe tekst", None)
    intern = await data.fragment(bron, v2, r2, "strenger fragment", None)
    await data.q("UPDATE type::thing($c) SET classification = 'intern'", {"c": intern})
    repo = ChunkRepository(live_surrealdb)

    assert await repo.get_texts([oud, nieuw, intern, "bogus"], scope=EXTERN) == {
        oud: "oude tekst", nieuw: "nieuwe tekst"}
    assert set(await repo.get_texts([nieuw, intern], scope=ALLES)) == {nieuw, intern}
    assert await repo.get_texts([nieuw], scope=ClassificationScope.LEEG) == {}
    huidig = {str(c.id) for c in await repo.get_by_source(bron, huidig=True)}
    assert huidig == {nieuw, intern}
    assert {str(c.id) for c in await repo.get_by_source(bron)} == {oud, nieuw, intern}
