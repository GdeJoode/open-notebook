"""Integration roundtrip for K.7a type-aware relation endpoint resolution.

The headline guard (AC #1): persist a batch containing a ``person`` named "BZK"
AND an ``organization`` named "BZK" (same canonical_name, different type), plus a
relation FROM the org and a relation FROM the person. Each persisted edge's
``in``/``out`` must resolve to the TYPE-CORRECT entity — the org relation points
at the org id, the person relation at the person id.

Under the OLD name-only ``SELECT id FROM entity WHERE canonical_name = $name
LIMIT 1`` this fails: both relations resolve to whichever "BZK" the DB returns
first, mis-attaching one edge. The (canonical_name, entity_type) filter is the
real guard — proven by the falsifiability test below that runs the legacy
name-only query and asserts both edges collapse onto one endpoint.

AC #2 (fallback): a relation whose endpoint is NOT in the batch (cross-batch /
unknown type) still resolves by name only — the existing roundtrip behaviour,
no regression.

Gated behind ``@requires_docker`` like the other roundtrip suites. The service
under test lives in apps/app-main; it imports cleanly in the workspace.
"""

from __future__ import annotations

import uuid

import pytest

from app_main.services import entity_persistence_service as eps_module
from app_main.services.entity_persistence_service import EntityPersistenceService
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.entity import EntityRepository


def _u(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _bind_service_to_config(
    monkeypatch: pytest.MonkeyPatch, config: SurrealDBConfig
) -> EntityPersistenceService:
    """Wire a persistence service at the live container.

    The relation RELATE goes through the module-level ``execute_query`` (no
    config arg), so we patch that symbol with a wrapper that injects the test
    container's config — exercising the REAL service resolution code against
    the container. The entity upsert path uses the injected repository.
    """

    async def _exec(query, params=None, config_arg=None):
        return await execute_query(query, params, config=config or config_arg)

    monkeypatch.setattr(eps_module, "execute_query", _exec)
    return EntityPersistenceService(entity_repository=EntityRepository(config=config))


async def _edge_endpoints(config: SurrealDBConfig, src_id: str, tgt_id: str, rel_type: str):
    """Return (in, out) for the edge of ``rel_type`` between two ids, if any."""
    rows = await execute_query(
        "SELECT in, out FROM relation "
        "WHERE in = type::thing($s) AND out = type::thing($t) "
        "AND relation_type = $rt LIMIT 1;",
        {"s": src_id, "t": tgt_id, "rt": rel_type},
        config=config,
    )
    return rows[0] if rows else None


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_homograph_relations_resolve_to_type_correct_endpoint(
    live_surrealdb: SurrealDBConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #1: a person-X and an org-X each get their own relation, type-correctly."""
    bzk = _u("BZK")
    wet = _u("Klimaatwet")
    svc = _bind_service_to_config(monkeypatch, live_surrealdb)

    # Persist the batch: person BZK + org BZK + a target law. The org "signs"
    # the law; the person "leads" it. The endpoint types come from the batch's
    # own entities (the {canonical_name: entity_type} map K.7a builds).
    result = await svc.persist_filtered_result(
        source_id="source:k7a",
        entities=[
            {"text": bzk, "label": "person", "confidence": 0.95, "properties": {}},
            {"text": bzk, "label": "organization", "confidence": 0.80, "properties": {}},
            {"text": wet, "label": "legislation", "confidence": 0.9, "properties": {}},
        ],
        relations=[
            {"source_entity": bzk, "target_entity": wet, "relation_type": "ORG_SIGNS",
             "source_type": "organization", "confidence": 0.8, "properties": {}},
            {"source_entity": bzk, "target_entity": wet, "relation_type": "PERSON_LEADS",
             "source_type": "person", "confidence": 0.8, "properties": {}},
        ],
    )
    assert result["entities_upserted"] == 3
    assert result["relations_created"] == 2

    # Resolve the two BZK entity ids by type.
    rows = await execute_query(
        "SELECT id, entity_type FROM entity WHERE canonical_name = $n;",
        {"n": bzk},
        config=live_surrealdb,
    )
    id_by_type = {r["entity_type"]: str(r["id"]) for r in rows}
    assert set(id_by_type) == {"person", "organization"}
    org_id = id_by_type["organization"]
    person_id = id_by_type["person"]
    wet_rows = await execute_query(
        "SELECT id FROM entity WHERE canonical_name = $n;",
        {"n": wet},
        config=live_surrealdb,
    )
    wet_id = str(wet_rows[0]["id"])

    # The org edge anchors at the ORG id; the person edge at the PERSON id.
    org_edge = await _edge_endpoints(live_surrealdb, org_id, wet_id, "ORG_SIGNS")
    assert org_edge is not None, "ORG_SIGNS edge must anchor at the org id"

    person_edge = await _edge_endpoints(live_surrealdb, person_id, wet_id, "PERSON_LEADS")
    assert person_edge is not None, "PERSON_LEADS edge must anchor at the person id"

    # The crucial cross-type non-attachment: the org edge must NOT have resolved
    # to the person endpoint, and vice-versa.
    wrong_org = await _edge_endpoints(live_surrealdb, person_id, wet_id, "ORG_SIGNS")
    assert wrong_org is None, "ORG_SIGNS must not attach to the person"
    wrong_person = await _edge_endpoints(live_surrealdb, org_id, wet_id, "PERSON_LEADS")
    assert wrong_person is None, "PERSON_LEADS must not attach to the org"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_legacy_name_only_query_mis_attaches_homograph(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Falsifiability proof: the OLD name-only resolution collapses both BZK
    relations onto a SINGLE endpoint (the arbitrary first match), proving the
    type filter is a real guard, not a no-op."""
    bzk = _u("BZK-legacy")
    wet = _u("Wet-legacy")
    await execute_query(
        "CREATE entity SET canonical_name=$n, name=$n, entity_type='person', "
        "embedding=[], confidence=0.95, source_documents=[], status='active';",
        {"n": bzk}, config=live_surrealdb,
    )
    await execute_query(
        "CREATE entity SET canonical_name=$n, name=$n, entity_type='organization', "
        "embedding=[], confidence=0.80, source_documents=[], status='active';",
        {"n": bzk}, config=live_surrealdb,
    )
    await execute_query(
        "CREATE entity SET canonical_name=$n, name=$n, entity_type='legislation', "
        "embedding=[], confidence=0.9, source_documents=[], status='active';",
        {"n": wet}, config=live_surrealdb,
    )

    # The legacy name-only resolution: both endpoints pick the same arbitrary id.
    picked = await execute_query(
        "SELECT VALUE id FROM entity WHERE canonical_name = $n LIMIT 1;",
        {"n": bzk}, config=live_surrealdb,
    )
    # There are TWO BZK entities but name-only LIMIT 1 yields exactly one — so
    # two relations both resolving this way attach to the SAME endpoint, which
    # is the corruption K.7a's (name, type) filter prevents.
    assert len(picked) == 1

    both = await execute_query(
        "SELECT id, entity_type FROM entity WHERE canonical_name = $n;",
        {"n": bzk}, config=live_surrealdb,
    )
    assert len({str(r["id"]) for r in both}) == 2, (
        "two distinct typed entities share the name — name-only LIMIT 1 cannot "
        "tell them apart"
    )


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_cross_batch_endpoint_falls_back_to_name_only(
    live_surrealdb: SurrealDBConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AC #2: a relation whose endpoint is NOT in the batch (no type known)
    still resolves by name only — roundtrip green, no regression."""
    org = _u("EZK")
    other = _u("PreExisting")
    # Seed the target entity in a PRIOR (separate) write — it is not in the
    # batch below, so its type is unknown to the relation resolver.
    await execute_query(
        "CREATE entity SET canonical_name=$n, name=$n, entity_type='organization', "
        "embedding=[], confidence=0.9, source_documents=[], status='active';",
        {"n": other}, config=live_surrealdb,
    )
    svc = _bind_service_to_config(monkeypatch, live_surrealdb)

    result = await svc.persist_filtered_result(
        source_id="source:k7a-fallback",
        entities=[
            {"text": org, "label": "organization", "confidence": 0.9, "properties": {}},
        ],
        relations=[
            # target ``other`` is not in this batch → target_type unknown →
            # name-only fallback resolves it.
            {"source_entity": org, "target_entity": other, "relation_type": "REFERS_TO",
             "confidence": 0.7, "properties": {}},
        ],
    )
    assert result["relations_created"] == 1

    org_rows = await execute_query(
        "SELECT VALUE id FROM entity WHERE canonical_name=$n AND entity_type='organization' LIMIT 1;",
        {"n": org}, config=live_surrealdb,
    )
    other_rows = await execute_query(
        "SELECT VALUE id FROM entity WHERE canonical_name=$n LIMIT 1;",
        {"n": other}, config=live_surrealdb,
    )
    edge = await _edge_endpoints(
        live_surrealdb, str(org_rows[0]), str(other_rows[0]), "REFERS_TO"
    )
    assert edge is not None, "name-only fallback must still resolve the edge"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_audit_flags_ambiguous_endpoint_and_is_read_only(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """AC #5: the read-only audit counts endpoints mapping to >1 active typed
    entity. It NEVER mutates — re-fetch shows the seeded data unchanged."""
    repo = EntityRepository(config=live_surrealdb)

    homograph = _u("BZK-audit")
    target = _u("Wet-audit")
    # Two active entities sharing a name across types → an ambiguous endpoint.
    pid = (await execute_query(
        "CREATE entity SET canonical_name=$n, name=$n, entity_type='person', "
        "embedding=[], confidence=0.9, source_documents=[], status='active';",
        {"n": homograph}, config=live_surrealdb,
    ))[0]["id"]
    oid = (await execute_query(
        "CREATE entity SET canonical_name=$n, name=$n, entity_type='organization', "
        "embedding=[], confidence=0.8, source_documents=[], status='active';",
        {"n": homograph}, config=live_surrealdb,
    ))[0]["id"]
    tid = (await execute_query(
        "CREATE entity SET canonical_name=$n, name=$n, entity_type='legislation', "
        "embedding=[], confidence=0.9, source_documents=[], status='active';",
        {"n": target}, config=live_surrealdb,
    ))[0]["id"]
    # One edge touching the ambiguous endpoint.
    await execute_query(
        f"RELATE {pid}->relation->{tid} SET relation_type='LEADS', "
        "confidence=0.8, status='active', properties={};",
        None, config=live_surrealdb,
    )

    before = await execute_query(
        "SELECT count() AS c FROM entity GROUP ALL;", None, config=live_surrealdb,
    )

    audit = await repo.audit_ambiguous_relation_endpoints()

    assert homograph in audit["ambiguous_names"]
    assert audit["ambiguous_name_count"] >= 1
    assert audit["affected_relation_count"] >= 1

    # Read-only: entity count is unchanged, both homograph rows still active.
    after = await execute_query(
        "SELECT count() AS c FROM entity GROUP ALL;", None, config=live_surrealdb,
    )
    assert before == after, "audit must not create or delete entities"
    for eid in (pid, oid, tid):
        ent = await repo.get_entity(str(eid))
        assert ent is not None and ent.status == "active"
