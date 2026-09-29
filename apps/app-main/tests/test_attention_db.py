"""WB.1: ``GET /api/attention`` (the Werkbak) against a real SurrealDB.

* Every group's count equals the endpoint its page reads. The groups are taken
  from the response, and each needs a comparator here: a new group without one
  fails the test.
* A source added through the API and a job that fails while the app is closed
  both show on the next read, linked to where they are resolved.
* A failing group reports an error; the others still answer.
* ``?stage=`` on ``GET /api/sources/`` filters by stage.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List
from urllib.parse import quote

import httpx
import pytest
import surrealdb_service.config as surreal_config
import surrealdb_service.connection as conn
from fastapi import FastAPI
from job_queue.repository import JobRepository
from shared.models.notebook_schema import NotebookSchema
from shared.types.enums import JobStatus
from surrealdb_service.repositories.notebook_schema import NotebookSchemaRepository
from surrealdb_service.repositories.source import SourceRepository

from app_main.api.rate_limit import limiter
from app_main.api.routers import (
    attention,
    commands,
    entity_resolution,
    notebooks,
    ontologies,
    schemas,
    sources,
    triage,
)
from app_main.dependencies import get_attention_service
from app_main.services.attention_service import AttentionReport, AttentionService
from app_main.services.notebook_service import NotebookService
from app_main.services.triage.triage_queue import TriageQueueRepository

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]

_TABLES = (
    "source", "job", "notebook", "reference", "notebook_schema", "triage_queue",
    "entity", "source_embedding", "source_insight", "chunk",
)
_CLEAN = " ".join(f"DELETE {t};" for t in _TABLES)


@pytest.fixture
async def env(live_surrealdb, monkeypatch):
    # ``BaseRepository.create`` opens its own connection from the global config.
    monkeypatch.setattr(surreal_config, "_config", live_surrealdb)
    original = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    await conn.execute_query(_CLEAN, {})
    yield
    await conn.execute_query(_CLEAN, {})
    await conn._pool.close_all()
    conn._pool = original


def _app() -> FastAPI:
    app = FastAPI()
    app.state.limiter = limiter
    for r in (attention, commands, entity_resolution, notebooks, schemas, sources,
              triage, ontologies):
        app.include_router(r.router, prefix="/api")
    return app


def _client(app: FastAPI) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")


async def _q(sql: str, params: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
    return await conn.execute_query(sql, params or {})


async def _source(title: str, stage: str, notebook: str | None = None) -> str:
    [row] = await _q("CREATE source SET title = $t, processing_stage = $s RETURN id;",
                     {"t": title, "s": stage})
    sid = str(row["id"])
    if notebook:
        await _q(f"RELATE {sid}->reference->{notebook};")
    return sid


async def _age(source_id: str, minutes: int = 60) -> None:
    """Backdate ``updated``, which the schema sets to now on every write: the
    field is redefined for the one write, then restored (migration 1)."""
    await _q(
        "DEFINE FIELD OVERWRITE updated ON source DEFAULT time::now() "
        "VALUE $value OR time::now();"
        "UPDATE type::thing($id) SET updated = $t;"
        "DEFINE FIELD OVERWRITE updated ON source DEFAULT time::now() VALUE time::now();",
        {"id": source_id,
         "t": datetime.now(timezone.utc) - timedelta(minutes=minutes)},
    )


async def _chunk(source_id: str) -> None:
    await _q("CREATE chunk CONTENT $c;", {"c": {
        "source": conn.ensure_record_id(source_id), "text": "t", "order": 0,
        "physical_page": 1, "element_type": "text", "positions": [], "metadata": {}}})


async def _notebook(name: str) -> str:
    [row] = await _q("CREATE notebook SET name = $n, description = '' RETURN id;",
                     {"n": name})
    return str(row["id"])


async def _job(job_type: str, status: str, source_id: str | None = None,
               error: str | None = None, completed: bool = False,
               payload: Dict[str, Any] | None = None) -> str:
    [row] = await _q(
        "CREATE job CONTENT $c RETURN id;",
        {"c": {"job_type": job_type, "status": status, "priority": "normal",
               "payload": payload or {}, "retry_count": 0, "max_retries": 2,
               "source_id": source_id, "error_message": error,
               "created": datetime.now(timezone.utc),
               "updated": datetime.now(timezone.utc),
               **({"completed_at": datetime.now(timezone.utc)} if completed else {})}},
    )
    return str(row["id"])


async def _entity(name: str, etype: str, label: str) -> str:
    [row] = await _q(
        "CREATE entity CONTENT $c RETURN id;",
        {"c": {"canonical_name": name, "name_key": name.lower(), "entity_type": etype,
               "status": "active", "primary_type": label, "properties": {},
               "embedding": []}},
    )
    return str(row["id"])


async def _seed() -> Dict[str, str]:
    """One or more rows for every group, plus rows each group must leave out."""
    nb1, nb2 = await _notebook("Alpha"), await _notebook("Beta")
    ids = {"nb1": nb1, "nb2": nb2}
    ids["failed"] = await _source("Broken", "failed", nb1)
    ids["gated"] = await _source("Gated", "awaiting_schema_review", nb1)
    ids["gated2"] = await _source("Gated too", "awaiting_schema_review", nb2)
    # Stopped: short of complete, no live job, last write over 15 minutes ago.
    # Its parse job completed: a finished job does not make a source busy.
    ids["stalled"] = await _source("Stopped", "embedded", nb2)
    await _chunk(ids["stalled"])
    ids["empty"] = await _source("Empty", "ingested", nb2)  # parsed, no text
    ids["busy"] = await _source("Busy", "embedded", nb2)  # a job is queued for it
    ids["fresh"] = await _source("Fresh", "embedded", nb2)  # written just now
    for sid in (ids["stalled"], ids["empty"], ids["busy"]):
        await _age(sid)
    ids["done"] = await _source("Done", "complete", nb2)

    await _job("document_parse", "queued", ids["busy"])
    await _job("document_parse", "completed", ids["stalled"], completed=True)
    await _job("entity_extract", "processing", None)
    await _job("entity_extract", "paused_for_review", ids["gated"], completed=True)
    ids["failed_job"] = await _job("document_parse", "failed", ids["failed"], "boom",
                                   completed=True)
    # Resolved failures: re-run since, and a failure whose source is gone.
    await _job("entity_extract", "failed", ids["done"], "old", completed=True)
    await asyncio.sleep(0.01)
    await _job("entity_extract", "completed", ids["done"], completed=True)
    await _job("document_parse", "failed", "source:gone", "not found", completed=True)
    # A failure on a complete source: no source group owns it.
    ids["insight_job"] = await _job("insight_extract", "failed", ids["done"], "llm down",
                                    completed=True)

    schema_repo = NotebookSchemaRepository()
    await schema_repo.upsert(NotebookSchema(notebook=nb1, base_ontology="",
        pending_extensions=[{"type_name": "RegioDeal"}, {"type_name": "Convenant"}]))
    await schema_repo.upsert(NotebookSchema(notebook=nb2, base_ontology="",
        pending_extensions=[{"type_name": "Streekfonds"}]))

    queue = TriageQueueRepository()
    kept = await _entity("Provincie Utrecht", "organization", "Organization")
    decided = await _entity("Gemeente Delft", "organization", "Organization")
    await queue.upsert(kept, name="Provincie Utrecht", type="organization",
                       structural_degree=2, doc_count=1, reason="unsure", batch_id="b1")
    row = await queue.upsert(decided, name="Gemeente Delft", type="organization",
                             structural_degree=1, doc_count=1, reason="unsure",
                             batch_id="b1")
    await queue.set_decision(row, "active")

    # An OCR-mangled pair: a merge candidate.
    await _entity("Koninkrijksrelaties", "organization", "Organization")
    await _entity("Koninkrijksreiaties", "organization", "Organization")
    # One name twice under the current normalization: a plan cluster.
    await _entity("Waterschap Rivierenland.", "organization", "Organization")
    await _entity("Waterschap  Rivierenland", "organization", "Organization")
    # A row typed off its label: the retype moves it.
    await _entity("Jan Jansen", "concept", "Person")
    return ids


async def _paged_sources(c: httpx.AsyncClient, params: List[tuple]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    while True:
        r = await c.get("/api/sources/", params=params + [("limit", 100),
                                                           ("offset", len(out))])
        assert r.status_code == 200, r.text
        page = r.json()
        out += page
        if len(page) < 100:
            return out


async def _jobs(c: httpx.AsyncClient, **params: Any) -> List[Dict[str, Any]]:
    r = await c.get("/api/commands/jobs", params={"limit": 100000, **params})
    assert r.status_code == 200, r.text
    return r.json()


def _endpoint_counts(c: httpx.AsyncClient, since: datetime) -> Dict[str, Callable[[], Awaitable[int]]]:
    """Each group's count as the page it links to reads it."""

    async def notebook_ids() -> List[str]:
        r = await c.get("/api/notebooks")
        return [n["id"] for n in r.json()]

    async def stage(name: str) -> int:
        return len(await _paged_sources(c, [("stage", name)]))

    async def stalled() -> int:
        return len(await _paged_sources(c, [("stalled", "true")]))

    async def paused() -> int:
        return sum([(await c.get(f"/api/notebooks/{quote(n, safe='')}/extraction/paused"))
                    .json()["paused_count"] for n in await notebook_ids()])

    async def proposals() -> int:
        return sum([len((await c.get(f"/api/notebooks/{quote(n, safe='')}/schema"))
                        .json()["pending_extensions"]) for n in await notebook_ids()])

    async def triage_count() -> int:
        return (await c.get("/api/triage/queue")).json()["count"]

    async def merges() -> int:
        body = (await c.get("/api/entity-resolution/candidates")).json()
        plan = (await c.post("/api/entity-resolution/plan", json={})).json()
        return plan["cluster_count"] + body["auto_merge_count"] + body["review_count"]

    async def retype() -> int:
        counts = (await c.get("/api/ontologies/retype/dry-run")).json()["counts"]
        return counts["changes"] + counts["flag_only"]

    async def failed_jobs() -> int:
        return len(await _jobs(c, unresolved="true"))

    async def running() -> int:
        return sum([len(await _jobs(c, status_filter=s))
                    for s in ("queued", "processing", "retrying")])

    async def new_sources() -> int:
        rows = await _paged_sources(c, [("sort_by", "created")])
        return len([r for r in rows if datetime.fromisoformat(r["created"]) > since])

    async def finished() -> int:
        return len([j for j in await _jobs(c)
                    if j["status"] in ("completed", "failed", "cancelled")
                    and j["completed_at"]
                    and datetime.fromisoformat(j["completed_at"]) > since])

    return {
        "sources_failed": lambda: stage("failed"),
        "sources_awaiting_review": lambda: stage("awaiting_schema_review"),
        "sources_stalled": stalled,
        "extraction_paused": paused,
        "schema_proposals": proposals,
        "triage": triage_count,
        "merge_candidates": merges,
        "retype": retype,
        "jobs_failed": failed_jobs,
        "jobs_running": running,
        "new_sources": new_sources,
        "finished_jobs": finished,
    }


async def test_every_count_equals_its_endpoint(env) -> None:
    since = datetime.now(timezone.utc) - timedelta(hours=1)
    await _seed()
    async with _client(_app()) as c:
        r = await c.get("/api/attention", params={"since": since.isoformat()})
        assert r.status_code == 200, r.text
        body = r.json()
        groups = body["groups"] + body["arrived"]
        comparators = _endpoint_counts(c, since)
        assert {g["key"] for g in groups} == set(comparators)
        for g in groups:
            assert g["error"] is None, g
            assert g["count"] > 0, f"{g['key']}: the seed must reach every group"
            assert g["count"] == await comparators[g["key"]](), g["key"]
            # A group spread over several notebooks has no card-level link.
            assert g["link"].startswith("/") or (
                g["link"] == g["action"] == "" and len(g["items"]) > 1), g
            assert all(i["link"].startswith("/") for i in g["items"]), g
        assert body["total"] == sum(g["counted"] for g in body["groups"])


async def test_seeded_values(env) -> None:
    """The comparators above could agree on a wrong rule; pin the seed's answer."""
    ids = await _seed()
    async with _client(_app()) as c:
        body = (await c.get("/api/attention")).json()
    by_key = {g["key"]: g for g in body["groups"]}
    # No mark sent: arrivals over the default window, and the report says so.
    assert body["since_default"] is True
    since = datetime.fromisoformat(body["since"])
    assert datetime.fromisoformat(body["generated_at"]) - since == timedelta(days=7)
    arrived = {g["key"]: g["count"] for g in body["arrived"]}
    assert arrived["new_sources"] == 8
    assert by_key["sources_failed"]["count"] == 1
    # "Busy" has a queued job; "Fresh" was written less than 15 minutes ago.
    stalled = by_key["sources_stalled"]
    assert stalled["link"] == "/sources?stalled=true"
    reasons = {i["id"]: i["detail"] for i in stalled["items"]}
    assert reasons == {
        ids["stalled"]: "Stopped at embedded",
        ids["empty"]: "No text extracted (at ingested): reprocess or delete",
    }
    assert by_key["extraction_paused"]["count"] == 1
    assert by_key["schema_proposals"]["count"] == 3
    assert by_key["triage"]["count"] == 1
    assert by_key["retype"]["count"] == 1
    # The re-run and the deleted source left.
    assert by_key["jobs_failed"]["count"] == 2
    assert {i["id"] for i in by_key["jobs_failed"]["items"]} == {
        ids["failed_job"], ids["insight_job"]}
    assert by_key["jobs_running"]["count"] == 2
    assert by_key["jobs_running"]["needs_action"] is False
    # Counted once: the paused job belongs to the gated source, the parse
    # failure to the failed source; proposals are decisions of their own.
    counted = {k: g["counted"] for k, g in by_key.items()}
    assert counted["extraction_paused"] == 0
    assert counted["jobs_failed"] == 1
    assert counted["schema_proposals"] == 3
    assert counted["jobs_running"] == 0
    assert body["total"] == sum(counted.values()) == 1 + 2 + 2 + 0 + 3 + 1 + by_key[
        "merge_candidates"]["count"] + 1 + 1
    # One item per notebook, each linked to its Schema tab. Two notebooks: no
    # card-level action, since no one page covers both.
    gated = by_key["sources_awaiting_review"]
    schema = {nb: f"/notebooks/{quote(ids[nb], safe='')}/schema" for nb in ("nb1", "nb2")}
    assert {i["id"]: i["link"] for i in gated["items"]} == {
        ids["nb1"]: schema["nb1"], ids["nb2"]: schema["nb2"]}
    assert (gated["link"], gated["action"]) == ("", "")
    assert gated["detail"] == "In 2 places: open each below."
    proposals = by_key["schema_proposals"]
    assert {i["link"] for i in proposals["items"]} == set(schema.values())
    assert (proposals["link"], proposals["action"]) == ("", "")
    # One notebook: the action goes to its page.
    paused = by_key["extraction_paused"]
    assert paused["link"] == f"/notebooks/{quote(ids['nb1'], safe='')}"
    assert paused["action"] == "Resume 1 paused extraction"
    # "N more" counts items: per notebook, all listed; elsewhere the count.
    assert (proposals["count"], proposals["item_total"], len(proposals["items"])) == (3, 2, 2)
    assert {i["detail"].split(":")[0] for i in proposals["items"]} == {
        "2 proposals", "1 proposal"}
    assert gated["item_total"] == len(gated["items"]) == 2
    assert by_key["triage"]["item_total"] == by_key["triage"]["count"]
    for g in body["groups"] + body["arrived"]:
        assert g["item_total"] >= len(g["items"]), g["key"]
    assert by_key["merge_candidates"]["detail"].startswith("1 cluster, ")


async def test_stalled_list_reads_no_chunk_counts(env, monkeypatch) -> None:
    """The list needs the predicate only; the reason text is the Werkbak's."""
    ids = await _seed()

    async def no_chunks(self, source_ids):
        raise AssertionError("the list endpoint read chunk counts")

    monkeypatch.setattr(SourceRepository, "chunk_counts", no_chunks)
    async with _client(_app()) as c:
        listed = await _paged_sources(c, [("stalled", "true")])
    assert {s["id"] for s in listed} == {ids["stalled"], ids["empty"]}


async def test_stalled_filter_on_sources(env) -> None:
    """The group's link lists exactly the counted set, also within a notebook."""
    ids = await _seed()
    async with _client(_app()) as c:
        listed = await _paged_sources(c, [("stalled", "true")])
        in_nb1 = await _paged_sources(c, [("stalled", "true"), ("notebook_id", ids["nb1"])])
        in_nb2 = await _paged_sources(c, [("stalled", "true"), ("notebook_id", ids["nb2"])])
    assert {s["id"] for s in listed} == {ids["stalled"], ids["empty"]}
    # nb2 holds both stopped sources, plus a busy, a fresh and a complete one.
    assert {s["id"] for s in in_nb2} == {ids["stalled"], ids["empty"]}
    assert in_nb1 == []


async def test_api_source_and_failed_job_show_on_next_open(env) -> None:
    """AC: added through the API and failed while the app was closed; the next
    read lists both, each linked to the source page where it is resolved."""
    nb = await _notebook("Inbox")
    async with _client(_app()) as c:
        last_visit = (await c.get("/api/attention")).json()["generated_at"]
        await asyncio.sleep(0.01)

        created = await c.post("/api/sources/json", json={
            "type": "text", "content": "Added by an agent.", "title": "From the API",
            "notebooks": [nb], "async_processing": True})
        assert created.status_code == 200, created.text
        source_id = created.json()["id"]
        [job] = await JobRepository().get_by_source(source_id)
        # The worker's own write when the job fails.
        await JobRepository().update_status(job.id, JobStatus.FAILED, error_message="parse failed")

        body = (await c.get("/api/attention", params={"since": last_visit})).json()
        assert body["since_default"] is False
        arrived = {g["key"]: g for g in body["arrived"]}
        groups = {g["key"]: g for g in body["groups"]}
        link = f"/sources/{quote(source_id, safe='')}"

        [new] = arrived["new_sources"]["items"]
        assert (new["id"], new["label"], new["link"]) == (source_id, "From the API", link)
        [finished] = arrived["finished_jobs"]["items"]
        assert (finished["id"], finished["link"]) == (str(job.id), link)
        assert "parse failed" in finished["detail"]
        assert [i["link"] for i in groups["jobs_failed"]["items"]] == [link]

        # Marked as seen: nothing has arrived since.
        seen = body["generated_at"]
        again = (await c.get("/api/attention", params={"since": seen})).json()
        assert [g["count"] for g in again["arrived"]] == [0, 0]
        # The failure still needs the owner, seen or not.
        assert next(g for g in again["groups"] if g["key"] == "jobs_failed")["count"] == 1


async def test_a_failing_group_degrades_alone(env, monkeypatch) -> None:
    await _seed()

    async def boom(self):
        raise RuntimeError("triage table unreadable")

    monkeypatch.setattr(TriageQueueRepository, "list_open", boom)
    async with _client(_app()) as c:
        r = await c.get("/api/attention")
    assert r.status_code == 200
    by_key = {g["key"]: g for g in r.json()["groups"]}
    assert by_key["triage"]["error"] == "triage table unreadable"
    assert by_key["triage"]["count"] == 0 and by_key["triage"]["items"] == []
    assert by_key["sources_failed"]["count"] == 1 and by_key["sources_failed"]["error"] is None
    assert by_key["triage"]["counted"] == 0
    assert r.json()["total"] == sum(g["counted"] for g in by_key.values())


async def test_a_slow_group_times_out_alone(env, monkeypatch) -> None:
    await _seed()

    async def slow(self):
        await asyncio.sleep(5)

    monkeypatch.setattr(TriageQueueRepository, "list_open", slow)
    app = _app()
    app.dependency_overrides[get_attention_service] = lambda: AttentionService(
        group_timeout_s=0.5)
    async with _client(app) as c:
        started = asyncio.get_running_loop().time()
        body = (await c.get("/api/attention")).json()
        elapsed = asyncio.get_running_loop().time() - started
    by_key = {g["key"]: g for g in body["groups"]}
    assert by_key["triage"]["error"] == "timed out after 0.5 s"
    assert by_key["sources_failed"]["count"] == 1
    assert elapsed < 4


async def test_sources_stage_filter(env) -> None:
    ids = await _seed()
    async with _client(_app()) as c:
        one = await _paged_sources(c, [("stage", "failed")])
        many = await _paged_sources(c, [("stage", "failed"), ("stage", "complete")])
        in_nb = await _paged_sources(c, [("stage", "embedded"), ("notebook_id", ids["nb2"])])
        everything = await _paged_sources(c, [])
        bad = await c.get("/api/sources/", params={"stage": "nonsense"})
    assert [s["id"] for s in one] == [ids["failed"]]
    assert {s["id"] for s in many} == {ids["failed"], ids["done"]}
    assert {s["id"] for s in in_nb} == {ids["stalled"], ids["busy"], ids["fresh"]}
    assert len(everything) == 8
    assert bad.status_code == 400


async def test_a_slow_shared_read_times_out_its_groups(env, monkeypatch) -> None:
    """Three groups wait on a slow shared notebooks read: each times out on its
    own and the rest of the report answers. (The shield on the shared read is
    pinned in test_attention_service.py.)"""
    await _seed()

    async def slow(self, **kwargs):
        await asyncio.sleep(5)
        return []

    monkeypatch.setattr(NotebookService, "get_all_with_counts", slow)
    app = _app()
    app.dependency_overrides[get_attention_service] = lambda: AttentionService(
        group_timeout_s=0.5)
    async with _client(app) as c:
        r = await c.get("/api/attention")
    assert r.status_code == 200
    by_key = {g["key"]: g for g in r.json()["groups"]}
    for key in ("sources_awaiting_review", "extraction_paused", "schema_proposals"):
        assert by_key[key]["error"] == "timed out after 0.5 s", key
    assert by_key["sources_failed"]["count"] == 1


_FIXTURE = (Path(__file__).resolve().parents[3]
            / "frontend" / "e2e" / "track-wb" / "attention.fixture.json")


async def test_e2e_fixture_follows_the_contract(env) -> None:
    """The e2e and vitest double: the model's exact shape (no field missing or
    extra), and the groups the service returns, in its order, with their
    titles, badges' nav, needs_action, and each fixed link."""
    fixture = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    assert AttentionReport.model_validate(fixture).model_dump(mode="json") == fixture
    real = (await AttentionService()
            .report(since=datetime.now(timezone.utc))).model_dump(mode="json")
    for part in ("groups", "arrived"):
        assert [g["key"] for g in fixture[part]] == [g["key"] for g in real[part]]
        for f, r in zip(fixture[part], real[part]):
            assert (f["title"], f["nav"], f["needs_action"]) == (
                r["title"], r["nav"], r["needs_action"]), f["key"]
            if r["link"]:
                assert f["link"] == r["link"], f["key"]
    assert fixture["total"] == sum(g["counted"] for g in fixture["groups"])


async def test_a_sourceless_failure_is_resolved_per_notebook(env) -> None:
    """A failed export of one notebook stays until that notebook is exported
    again; an export of another notebook does not resolve it."""
    nb1, nb2 = await _notebook("One"), await _notebook("Two")
    open_failure = await _job("export_okf", "failed", error="disk full", completed=True,
                              payload={"notebook_id": nb1})
    await _job("export_okf", "failed", error="disk full", completed=True,
               payload={"notebook_id": nb2})
    await asyncio.sleep(0.01)
    await _job("export_okf", "completed", completed=True, payload={"notebook_id": nb2})
    [job] = await JobRepository().list_unresolved_failures()
    assert str(job.id) == open_failure
    async with _client(_app()) as c:
        body = (await c.get("/api/attention")).json()
    failed = next(g for g in body["groups"] if g["key"] == "jobs_failed")
    assert [i["link"] for i in failed["items"]] == [f"/notebooks/{quote(nb1, safe='')}"]


async def test_a_failed_source_group_owns_nothing(env, monkeypatch) -> None:
    """The gated group reads its sources, then fails looking up their
    notebooks. It lists nothing, so the gated source's paused extraction
    counts in the total again."""
    await _seed()

    async def failing(self, source_ids):
        raise RuntimeError("reference table unreadable")

    monkeypatch.setattr(SourceRepository, "get_notebook_id_many", failing)
    async with _client(_app()) as c:
        body = (await c.get("/api/attention")).json()
    by_key = {g["key"]: g for g in body["groups"]}
    assert by_key["sources_awaiting_review"]["error"] == "reference table unreadable"
    assert by_key["extraction_paused"]["counted"] == 1


async def test_batched_notebook_lookup_matches_the_single_one(env) -> None:
    """One query for all gated sources, and the same answer as get_notebook_id
    for a source in no, one and two notebooks."""
    nb1, nb2 = await _notebook("One"), await _notebook("Two")
    none = await _source("Loose", "awaiting_schema_review")
    one = await _source("In one", "awaiting_schema_review", nb1)
    two = await _source("In two", "awaiting_schema_review", nb2)
    await _q(f"RELATE {two}->reference->{nb1};")
    repo = SourceRepository()
    many = await repo.get_notebook_id_many([none, one, two, "source:missing"])
    assert many == {s: await repo.get_notebook_id(s) for s in (none, one, two)}
    assert many[none] is None and many[one] == nb1
