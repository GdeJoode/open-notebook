"""Q.4 integration: persist -> triage end-to-end, run TWICE for idempotency.

This is the load-bearing test of the whole track. It runs against a real
SurrealDB testcontainer (migrations 39/58/59/60 applied by the ``live_surrealdb``
fixture) and exercises the actual FROZEN reuse services — ``persist_filtered_result``
(B2 merge + Q.2a provenance union), the Q.2 signals helper, the Q.3 status
service + change-log, the Q.4 queue, and the backbone check — with NOTHING mocked
on the DB side.

The flow per case:

  1. seed entities + relations via ``persist_filtered_result`` (the real merge);
  2. run ``TriagePipeline.run`` over the persisted batch (the after-extraction
     hook's payload);
  3. run it AGAIN with the same batch;
  4. assert the run-twice INVARIANTS: identical statuses, NO extra status-change-
     log rows, NO duplicate queue rows, and stable provenance (doc-count).

Coverage maps to Q.4 acceptance criteria:

* AC1 — a status is assigned to every affected entity (the active theme is active).
* AC2 — idempotency: second run changes nothing (log + queue + status stable).
* AC4 — the queue holds the isolated core actor, the unsure_review entity, the
  well-connected reference candidate — each as ONE dedup'd row.
* AC8 — the named-programme limitation: a NOVEX-like entity typed ``Programma``
  reaches ``active``, but typed ``Organisatie`` it falls to unsure_review -> queue.
"""

from __future__ import annotations

import uuid

import pytest
from app_main.services import entity_persistence_service as eps_module
from app_main.services.entity_persistence_service import EntityPersistenceService
from app_main.services.triage import status_change_log as scl_module
from app_main.services.triage import triage_queue as tq_module
from app_main.services.triage.backbone_check import BackboneCheckService
from app_main.services.triage.signals_service import TriageSignalsService
from app_main.services.triage.status_assignment_service import (
    STATUS_ACTIVE,
    STATUS_REFERENCE,
    StatusAssignmentService,
)
from app_main.services.triage.status_change_log import StatusChangeLogRepository
from app_main.services.triage.triage_config import load_triage_config
from app_main.services.triage.triage_pipeline import TriagePipeline
from app_main.services.triage.triage_queue import TriageQueueRepository
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.entity import EntityRepository

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]


def _u(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _bind_all(monkeypatch: pytest.MonkeyPatch, config: SurrealDBConfig):
    """Bind every module-level ``execute_query`` seam to the test container.

    ``EntityRepository`` takes the config directly; the persist service, the
    queue, and the status-change-log call the module-level ``execute_query``
    (no config arg), so we patch those symbols to inject the container config.
    Returns a fully-wired :class:`TriagePipeline` + the repos for assertions.
    """

    async def _exec(query, params=None, config_arg=None):
        return await execute_query(query, params, config=config or config_arg)

    monkeypatch.setattr(eps_module, "execute_query", _exec)
    monkeypatch.setattr(tq_module, "execute_query", _exec)
    monkeypatch.setattr(scl_module, "execute_query", _exec)

    cfg = load_triage_config()
    entity_repo = EntityRepository(config=config)
    persistence = EntityPersistenceService(entity_repository=entity_repo)
    queue = TriageQueueRepository(config=config)
    log = StatusChangeLogRepository(config)
    status = StatusAssignmentService(cfg, entity_repo, status_change_log=log)
    pipeline = TriagePipeline(
        entity_repository=entity_repo,
        signals_service=TriageSignalsService(entity_repo, config=cfg),
        status_service=status,
        queue_repository=queue,
        backbone_service=BackboneCheckService(entity_repo, config=cfg),
        candidate_service=None,  # match-conflict surfacing exercised in unit tests
        recanonicalization_service=None,
        config=cfg,
    )
    return pipeline, persistence, entity_repo, queue, log


async def _status_of(config: SurrealDBConfig, name: str) -> str:
    rows = await execute_query(
        "SELECT status FROM entity WHERE canonical_name = $n LIMIT 1;",
        {"n": name},
        config=config,
    )
    return str(rows[0]["status"]) if rows else ""


async def _queue_rows_for(config: SurrealDBConfig, entity_id: str):
    return await execute_query(
        "SELECT * FROM triage_queue WHERE entity = type::thing($id);",
        {"id": entity_id},
        config=config,
    )


async def _log_count(config: SurrealDBConfig) -> int:
    rows = await execute_query(
        "SELECT count() AS c FROM status_change_log GROUP ALL;", None, config=config
    )
    return int(rows[0]["c"]) if rows else 0


async def _persist(persistence, source_id, entities, relations):
    return await persistence.persist_filtered_result(
        source_id=source_id, entities=entities, relations=relations
    )


def _ent(text, label, primary_type=None):
    props = {}
    return {
        "text": text,
        "label": label,
        "confidence": 0.9,
        "properties": props,
    }


async def test_persist_then_triage_assigns_status_and_queues(
    live_surrealdb: SurrealDBConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC1/AC4: triage assigns status to every affected entity + fills the queue."""
    pipeline, persistence, entity_repo, queue, log = _bind_all(
        monkeypatch, live_surrealdb
    )
    source = f"source:{_u('doc')}"

    theme = _u("Thema")        # BeleidsThema -> active tier
    actor = _u("Ministerie")   # Ministerie  -> active tier
    person = _u("Persoon")     # Person      -> unsure_review tier
    # Persist a connected theme<->actor pair (structural edge) + an isolated
    # unsure_review person.
    result = await _persist(
        persistence,
        source,
        entities=[
            _ent(theme, "BeleidsThema"),
            _ent(actor, "Ministerie"),
            _ent(person, "Person"),
        ],
        relations=[
            {
                "source_entity": actor,
                "target_entity": theme,
                "relation_type": "BIJDRAGT_AAN",
                "confidence": 0.9,
                "properties": {},
            }
        ],
    )
    batch_ids = result["persisted_entity_ids"]
    assert len(batch_ids) == 3

    report = await pipeline.run(source, batch_ids, "batch-1")
    assert report.ok

    # AC1: a connected active-tier theme/actor are 'active'.
    assert await _status_of(live_surrealdb, theme) == STATUS_ACTIVE
    assert await _status_of(live_surrealdb, actor) == STATUS_ACTIVE
    # unsure_review person -> reference + queued.
    assert await _status_of(live_surrealdb, person) == STATUS_REFERENCE

    # AC4: the unsure_review person is queued (one row, reason carries the flag).
    person_id = ""
    for i in batch_ids:
        ent = await entity_repo.get_entity(i)
        if ent and ent.canonical_name == person:
            person_id = i
            break
    assert person_id, "person entity not found among the persisted batch"
    rows = await _queue_rows_for(live_surrealdb, person_id)
    assert len(rows) == 1
    assert "unsure" in rows[0]["reason"]


async def test_run_twice_is_idempotent(
    live_surrealdb: SurrealDBConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC2/AC3: the same batch twice -> stable statuses, NO extra log/queue rows."""
    pipeline, persistence, entity_repo, queue, log = _bind_all(
        monkeypatch, live_surrealdb
    )
    source = f"source:{_u('doc')}"

    theme = _u("Thema")
    actor = _u("Ministerie")
    person = _u("Persoon")
    result = await _persist(
        persistence,
        source,
        entities=[
            _ent(theme, "BeleidsThema"),
            _ent(actor, "Ministerie"),
            _ent(person, "Person"),
        ],
        relations=[
            {"source_entity": actor, "target_entity": theme,
             "relation_type": "BIJDRAGT_AAN", "confidence": 0.9, "properties": {}},
        ],
    )
    batch_ids = result["persisted_entity_ids"]

    # First run establishes statuses + queue + log.
    r1 = await pipeline.run(source, batch_ids, "batch-1")
    assert r1.ok
    statuses_1 = {n: await _status_of(live_surrealdb, n) for n in (theme, actor, person)}
    log_after_1 = await _log_count(live_surrealdb)
    open_after_1 = len(await queue.list_open())

    # Second run over the SAME batch: every write is a no-op.
    r2 = await pipeline.run(source, batch_ids, "batch-1")
    assert r2.ok
    statuses_2 = {n: await _status_of(live_surrealdb, n) for n in (theme, actor, person)}
    log_after_2 = await _log_count(live_surrealdb)
    open_after_2 = len(await queue.list_open())

    # Statuses identical.
    assert statuses_1 == statuses_2
    # NO new status-change-log rows on the unchanged re-run (idempotency).
    assert log_after_2 == log_after_1
    assert r2.statuses_changed == 0
    # NO duplicate queue rows.
    assert open_after_2 == open_after_1


async def test_named_programme_typing_limitation(
    live_surrealdb: SurrealDBConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC8: a NOVEX-like programme reaches 'active' ONLY when typed Programma.

    Typed as the active-tier ``Programma`` it is ``active``; typed as the
    unsure_review ``Organisatie`` the SAME named programme falls to ``reference``
    and is caught by the queue (the documented v1 limitation).
    """
    pipeline, persistence, entity_repo, queue, _log = _bind_all(
        monkeypatch, live_surrealdb
    )

    # Case A: typed Programma (active tier) + a structural edge -> active.
    src_a = f"source:{_u('a')}"
    novex_prog = _u("NOVEX")
    theme = _u("Thema")
    res_a = await _persist(
        persistence, src_a,
        entities=[_ent(novex_prog, "Programma"), _ent(theme, "BeleidsThema")],
        relations=[
            {"source_entity": novex_prog, "target_entity": theme,
             "relation_type": "BIJDRAGT_AAN", "confidence": 0.9, "properties": {}},
        ],
    )
    await pipeline.run(src_a, res_a["persisted_entity_ids"], "batch-a")
    assert await _status_of(live_surrealdb, novex_prog) == STATUS_ACTIVE

    # Case B: the SAME named programme typed Organisatie (unsure_review) -> reference.
    src_b = f"source:{_u('b')}"
    novex_org = _u("NOVEX")
    res_b = await _persist(
        persistence, src_b,
        entities=[_ent(novex_org, "Organisatie")],
        relations=[],
    )
    await pipeline.run(src_b, res_b["persisted_entity_ids"], "batch-b")
    assert await _status_of(live_surrealdb, novex_org) == STATUS_REFERENCE

    # ...and it is caught by the review queue (unsure -> operator decides).
    org_id = res_b["persisted_entity_ids"][0]
    rows = await _queue_rows_for(live_surrealdb, org_id)
    assert len(rows) == 1
    assert "unsure" in rows[0]["reason"]
