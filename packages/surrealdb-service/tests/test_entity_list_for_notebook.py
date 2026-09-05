"""Roundtrip tests for ``EntityRepository.list_entities_for_notebook`` (D.0).

Boots the testcontainers SurrealDB harness (``live_surrealdb`` fixture
from ``surrealdb_service.testing``), applies every migration up to and
including #48 (orphan-prune lifecycle, B.5b), and exercises the Track D
notebook-scoped projection across six filter combinations.

Acceptance criteria mapped to test cases (from Track D plan §D.0):

* All entities returned with permissive filter --
  ``test_returns_all_entities_with_permissive_filter``.
* ``include_orphans=False`` excludes pending_reconnect entities --
  ``test_excludes_pending_reconnect_when_include_orphans_false``.
* ``include_archived=False`` excludes archived entities --
  ``test_excludes_archived_when_include_archived_false``.
* ``entity_types`` allow-list narrows the result --
  ``test_entity_types_allowlist_filters_correctly``.
* ``min_confidence=0.95`` filters low-confidence entities --
  ``test_min_confidence_filters_low_confidence_entities``.
* ``embedding`` field NOT in the SELECT projection --
  ``test_embedding_is_omitted_from_projection``.

Plus two integration cross-checks:

* ``list_relations_for_notebook`` returns only edges among filtered
  entities -- ``test_relations_limited_to_filtered_entities``.
* ``min_relation_confidence`` falls back to ``min_confidence`` when
  None -- ``test_relation_confidence_fallback``.
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from shared.models.export import ExportFilter
from shared.utils.external_ids import resolve_external_ids
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories.entity import EntityRepository

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


async def _create_notebook(config: SurrealDBConfig, name: str) -> str:
    rows = await execute_query(
        "CREATE notebook SET name = $name;", {"name": name}, config=config
    )
    assert rows, "CREATE notebook returned no rows"
    return str(rows[0]["id"])


async def _create_source(config: SurrealDBConfig, title: str) -> str:
    rows = await execute_query(
        "CREATE source SET title = $title;", {"title": title}, config=config
    )
    assert rows, "CREATE source returned no rows"
    return str(rows[0]["id"])


async def _link_source_to_notebook(
    config: SurrealDBConfig, source_id: str, notebook_id: str
) -> None:
    """Stand up the ``reference`` RELATE edge (migration 1 line 54-56).

    The query mirrors ``NotebookService.add_source`` -- ``type::thing()``
    cannot appear inside RELATE source/target positions, so we bind
    RecordIDs directly.
    """
    await execute_query(
        "RELATE $src->reference->$nb;",
        {
            "src": ensure_record_id(source_id),
            "nb": ensure_record_id(notebook_id),
        },
        config=config,
    )


async def _create_entity(
    config: SurrealDBConfig,
    *,
    canonical_name: str,
    entity_type: str,
    source_documents: List[str],
    confidence: float = 1.0,
    primary_type: str | None = None,
    orphan_status: str | None = None,
    embedding: List[float] | None = None,
    external_ids: List[str] | None = None,
    aliases: List[str] | None = None,
) -> str:
    payload: Dict[str, Any] = {
        "canonical_name": canonical_name,
        "entity_type": entity_type,
        "source_documents": list(source_documents),
        "extraction_method": "test",
        "confidence": confidence,
        "provenance_chain": [],
        "properties": {},
        "embedding": embedding if embedding is not None else [],
        "status": "active",
        "type_tags": [entity_type],
        "primary_type": primary_type or entity_type,
        "external_ids": list(external_ids) if external_ids else [],
        "aliases": list(aliases) if aliases else [],
    }
    rows = await execute_query(
        """
        CREATE entity SET
            canonical_name = $canonical_name, name_key = string::lowercase(string::trim($canonical_name)),
            entity_type = $entity_type,
            description = NONE,
            source_documents = $source_documents,
            extraction_method = $extraction_method,
            confidence = $confidence,
            provenance_chain = $provenance_chain,
            properties = $properties,
            embedding = $embedding,
            status = $status,
            type_tags = $type_tags,
            primary_type = $primary_type,
            external_ids = $external_ids,
            aliases = $aliases;
        """,
        payload,
        config=config,
    )
    assert rows, "CREATE entity returned no rows"
    eid = str(rows[0]["id"])

    if orphan_status is not None:
        repo = EntityRepository(config=config)
        ok = await repo.update_orphan_status(
            eid, orphan_status, set_first_orphaned_at=True
        )
        assert ok, "update_orphan_status failed during fixture setup"
    return eid


async def _build_notebook_with_mixed_entities(
    config: SurrealDBConfig, prefix: str
) -> Dict[str, Any]:
    """Build the standard fixture used by every filter test.

    Layout: 1 notebook + 3 sources, each carrying 5 entities. Across
    the 15 entities we vary entity_type / confidence / orphan_status so
    one fixture suffices for every assertion in this module.

    The unique ``prefix`` keeps the fixtures isolated when tests share
    the same DB instance (the harness ships one container for the
    module).
    """
    notebook_id = await _create_notebook(config, f"{prefix}-notebook")

    source_ids: List[str] = []
    for i in range(3):
        sid = await _create_source(config, f"{prefix}-src-{i}")
        await _link_source_to_notebook(config, sid, notebook_id)
        source_ids.append(sid)

    # 15 entities total. We label them deterministically so the asserts
    # can speak about ``person_high_0``, ``org_archived_1``, etc.
    entity_ids: Dict[str, str] = {}

    # Source 0: 3 Persons, 1 Organization, 1 Concept -- all healthy, all high confidence.
    for i in range(3):
        eid = await _create_entity(
            config,
            canonical_name=f"{prefix}-Person-{i}",
            entity_type="Person",
            source_documents=[source_ids[0]],
            confidence=0.95,
        )
        entity_ids[f"person_healthy_{i}"] = eid

    for i in range(1):
        eid = await _create_entity(
            config,
            canonical_name=f"{prefix}-Org-{i}",
            entity_type="Organization",
            source_documents=[source_ids[0]],
            confidence=0.95,
        )
        entity_ids[f"org_healthy_{i}"] = eid

    for i in range(1):
        eid = await _create_entity(
            config,
            canonical_name=f"{prefix}-Concept-0",
            entity_type="Concept",
            source_documents=[source_ids[0]],
            confidence=0.95,
        )
        entity_ids["concept_healthy_0"] = eid

    # Source 1: 2 low-confidence Persons + 2 pending_reconnect orphans + 1 archived.
    for i in range(2):
        eid = await _create_entity(
            config,
            canonical_name=f"{prefix}-LowPerson-{i}",
            entity_type="Person",
            source_documents=[source_ids[1]],
            confidence=0.5,
        )
        entity_ids[f"person_low_{i}"] = eid

    for i in range(2):
        eid = await _create_entity(
            config,
            canonical_name=f"{prefix}-Pending-{i}",
            entity_type="Person",
            source_documents=[source_ids[1]],
            confidence=0.95,
            orphan_status="pending_reconnect",
        )
        entity_ids[f"person_pending_{i}"] = eid

    eid = await _create_entity(
        config,
        canonical_name=f"{prefix}-Archived-0",
        entity_type="Organization",
        source_documents=[source_ids[1]],
        confidence=0.95,
        orphan_status="archived",
    )
    entity_ids["org_archived_0"] = eid

    # Source 2: 5 healthy Concepts.
    for i in range(5):
        eid = await _create_entity(
            config,
            canonical_name=f"{prefix}-S2Concept-{i}",
            entity_type="Concept",
            source_documents=[source_ids[2]],
            confidence=0.95,
        )
        entity_ids[f"s2_concept_{i}"] = eid

    return {
        "notebook_id": notebook_id,
        "source_ids": source_ids,
        "entity_ids": entity_ids,
    }


# ---------------------------------------------------------------------------
# Acceptance Criterion 1: permissive filter returns all 15 entities
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_returns_all_entities_with_permissive_filter(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """``min_connections=0, min_confidence=0.0`` returns every entity.

    Even the orphan + archived rows come through when ``include_orphans``
    and ``include_archived`` are both True.
    """
    fixture = await _build_notebook_with_mixed_entities(
        live_surrealdb, "all-perm"
    )
    repo = EntityRepository(config=live_surrealdb)

    entities = await repo.list_entities_for_notebook(
        fixture["notebook_id"],
        ExportFilter(
            min_connections=0,
            min_confidence=0.0,
            include_orphans=True,
            include_archived=True,
        ),
    )

    returned_ids = {e.id for e in entities}
    expected_ids = set(fixture["entity_ids"].values())
    assert returned_ids == expected_ids, (
        f"expected all 15 entities; got {len(returned_ids)} returned: "
        f"missing={expected_ids - returned_ids!r}"
    )


# ---------------------------------------------------------------------------
# Acceptance Criterion 2: include_orphans=False filters pending_reconnect
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_excludes_pending_reconnect_when_include_orphans_false(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The two ``person_pending_*`` entities must NOT come through."""
    fixture = await _build_notebook_with_mixed_entities(
        live_surrealdb, "no-orph"
    )
    repo = EntityRepository(config=live_surrealdb)

    entities = await repo.list_entities_for_notebook(
        fixture["notebook_id"],
        ExportFilter(
            min_connections=0,
            min_confidence=0.0,
            include_orphans=False,
            include_archived=True,  # leave archived alone for this test
        ),
    )

    returned_ids = {e.id for e in entities}
    pending_ids = {
        fixture["entity_ids"]["person_pending_0"],
        fixture["entity_ids"]["person_pending_1"],
    }
    assert pending_ids.isdisjoint(returned_ids), (
        f"pending_reconnect entities leaked through: "
        f"{pending_ids & returned_ids!r}"
    )


# ---------------------------------------------------------------------------
# Acceptance Criterion 3: include_archived=False filters archived
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_excludes_archived_when_include_archived_false(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The single ``org_archived_0`` row must NOT come through."""
    fixture = await _build_notebook_with_mixed_entities(
        live_surrealdb, "no-arch"
    )
    repo = EntityRepository(config=live_surrealdb)

    entities = await repo.list_entities_for_notebook(
        fixture["notebook_id"],
        ExportFilter(
            min_connections=0,
            min_confidence=0.0,
            include_orphans=True,  # leave orphans alone for this test
            include_archived=False,
        ),
    )

    returned_ids = {e.id for e in entities}
    archived_id = fixture["entity_ids"]["org_archived_0"]
    assert archived_id not in returned_ids, (
        f"archived entity {archived_id!r} leaked through"
    )


# ---------------------------------------------------------------------------
# Acceptance Criterion 4: entity_types allow-list
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_entity_types_allowlist_filters_correctly(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """``entity_types=['Person']`` returns Persons only.

    Sanity check on the SurrealQL ``primary_type INSIDE`` clause -- the
    allow-list filter must be applied DB-side, not in Python (the export
    pipeline relies on this for cost).
    """
    fixture = await _build_notebook_with_mixed_entities(
        live_surrealdb, "allowlist"
    )
    repo = EntityRepository(config=live_surrealdb)

    entities = await repo.list_entities_for_notebook(
        fixture["notebook_id"],
        ExportFilter(
            min_connections=0,
            min_confidence=0.0,
            entity_types=["Person"],
            include_orphans=True,
            include_archived=True,
        ),
    )

    assert entities, "expected at least one Person"
    assert all(e.primary_type == "Person" for e in entities), (
        f"non-Person leaked through: "
        f"{[e.primary_type for e in entities if e.primary_type != 'Person']!r}"
    )


# ---------------------------------------------------------------------------
# Acceptance Criterion 5: min_confidence filter
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_min_confidence_filters_low_confidence_entities(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """``min_confidence=0.95`` excludes the 0.5-confidence rows."""
    fixture = await _build_notebook_with_mixed_entities(
        live_surrealdb, "conf"
    )
    repo = EntityRepository(config=live_surrealdb)

    entities = await repo.list_entities_for_notebook(
        fixture["notebook_id"],
        ExportFilter(
            min_connections=0,
            min_confidence=0.95,
            include_orphans=True,
            include_archived=True,
        ),
    )

    returned_ids = {e.id for e in entities}
    low_ids = {
        fixture["entity_ids"]["person_low_0"],
        fixture["entity_ids"]["person_low_1"],
    }
    assert low_ids.isdisjoint(returned_ids), (
        f"low-confidence entities leaked through: "
        f"{low_ids & returned_ids!r}"
    )
    # Confirm we kept at least the high-confidence rows.
    assert all(e.confidence >= 0.95 for e in entities), (
        f"under-threshold confidence in result: "
        f"{[e.confidence for e in entities if e.confidence < 0.95]!r}"
    )


# ---------------------------------------------------------------------------
# Acceptance Criterion 6: embedding field NOT in projection (Q-D-1)
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_embedding_is_omitted_from_projection(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The 60MB-at-10K-entities embedding column must NOT be projected.

    Stores a sentinel embedding on a row, then asserts the rehydrated
    Entity has the model default ``[]`` rather than the stored vector
    -- proves the column never crossed the wire.

    Also reads the projection field constant directly to ensure no future
    refactor adds the field back.
    """
    fixture = await _build_notebook_with_mixed_entities(
        live_surrealdb, "no-embed"
    )

    # Tag one specific entity with a sentinel 8-dim vector.
    sentinel_vec = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88]
    target_id = fixture["entity_ids"]["person_healthy_0"]
    await execute_query(
        "UPDATE type::thing($id) SET embedding = $vec;",
        {"id": target_id, "vec": sentinel_vec},
        config=live_surrealdb,
    )

    # Sanity: the row really has the vector now.
    rows = await execute_query(
        "SELECT embedding FROM entity WHERE id = type::thing($id) LIMIT 1;",
        {"id": target_id},
        config=live_surrealdb,
    )
    assert rows and rows[0]["embedding"] == sentinel_vec, (
        "fixture sanity check failed -- row did not receive the sentinel "
        "embedding"
    )

    # Now exercise the projection.
    repo = EntityRepository(config=live_surrealdb)
    entities = await repo.list_entities_for_notebook(
        fixture["notebook_id"],
        ExportFilter(
            min_connections=0,
            min_confidence=0.0,
            include_orphans=True,
            include_archived=True,
        ),
    )

    by_id = {e.id: e for e in entities}
    assert target_id in by_id, "fixture entity missing from export projection"
    # The Entity model defaults ``embedding=[]`` and the projection
    # omits the column, so the rehydrated model carries [].
    assert by_id[target_id].embedding == [], (
        f"embedding was projected by list_entities_for_notebook -- got "
        f"{by_id[target_id].embedding!r}; expected [] (omitted from SELECT)"
    )

    # Belt-and-braces: the projection constant itself must not mention
    # embedding. Catches a future refactor that adds it back inline.
    assert "embedding" not in EntityRepository._ENTITY_EXPORT_FIELDS, (
        "EntityRepository._ENTITY_EXPORT_FIELDS regressed -- 'embedding' "
        "must stay omitted per Q-D-1"
    )


# ---------------------------------------------------------------------------
# Bonus: list_relations_for_notebook respects entity filter (Q-D-4)
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_relations_limited_to_filtered_entities(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Relations into a filtered-out entity are silently dropped (Q-D-4).

    Setup:
      * Two healthy Persons (P1, P2) -- both will pass the filter.
      * One low-confidence Person (LowP) -- filtered out.
      * Relations: P1→P2 (keeps), P1→LowP (drops).
    """
    fixture = await _build_notebook_with_mixed_entities(
        live_surrealdb, "rels"
    )

    # Pick three known entity ids from the fixture.
    p1 = fixture["entity_ids"]["person_healthy_0"]
    p2 = fixture["entity_ids"]["person_healthy_1"]
    lowp = fixture["entity_ids"]["person_low_0"]

    # Relate P1 -> P2 (kept) and P1 -> LowP (dropped after entity filter).
    await execute_query(
        "RELATE $a->relation->$b "
        "SET relation_type = 'KNOWS', confidence = 0.99, "
        "extraction_method = 'test', source_documents = [], "
        "provenance_chain = [], properties = {}, status = 'active';",
        {"a": ensure_record_id(p1), "b": ensure_record_id(p2)},
        config=live_surrealdb,
    )
    await execute_query(
        "RELATE $a->relation->$b "
        "SET relation_type = 'KNOWS', confidence = 0.99, "
        "extraction_method = 'test', source_documents = [], "
        "provenance_chain = [], properties = {}, status = 'active';",
        {"a": ensure_record_id(p1), "b": ensure_record_id(lowp)},
        config=live_surrealdb,
    )

    repo = EntityRepository(config=live_surrealdb)

    relations = await repo.list_relations_for_notebook(
        fixture["notebook_id"],
        ExportFilter(
            min_connections=0,
            min_confidence=0.95,  # filters out LowP (conf 0.5)
            include_orphans=True,
            include_archived=True,
        ),
    )

    # The (P1, P2) edge should still be there; (P1, LowP) must be gone.
    pairs = {(r.in_entity, r.out_entity) for r in relations}
    assert (p1, p2) in pairs, f"expected (P1, P2) edge; got {pairs!r}"
    assert (p1, lowp) not in pairs, (
        f"edge into filtered LowP leaked through: {pairs!r}"
    )


# ---------------------------------------------------------------------------
# Bonus: min_relation_confidence fallback to min_confidence
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_relation_confidence_fallback(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """``min_relation_confidence=None`` inherits from ``min_confidence``.

    Pin the documented single-knob ergonomics: a caller that sets
    ``min_confidence=0.95`` and leaves the relation knob unset must NOT
    accidentally see a 0.5-confidence edge.
    """
    fixture = await _build_notebook_with_mixed_entities(
        live_surrealdb, "rel-fb"
    )

    p1 = fixture["entity_ids"]["person_healthy_0"]
    p2 = fixture["entity_ids"]["person_healthy_1"]

    # Create a low-confidence edge between two high-confidence entities.
    await execute_query(
        "RELATE $a->relation->$b "
        "SET relation_type = 'KNOWS', confidence = 0.5, "
        "extraction_method = 'test', source_documents = [], "
        "provenance_chain = [], properties = {}, status = 'active';",
        {"a": ensure_record_id(p1), "b": ensure_record_id(p2)},
        config=live_surrealdb,
    )

    repo = EntityRepository(config=live_surrealdb)

    # min_relation_confidence stays None -> inherits 0.95 -> edge drops.
    relations = await repo.list_relations_for_notebook(
        fixture["notebook_id"],
        ExportFilter(
            min_connections=0,
            min_confidence=0.95,
            min_relation_confidence=None,
            include_orphans=True,
            include_archived=True,
        ),
    )

    pairs = {(r.in_entity, r.out_entity) for r in relations}
    assert (p1, p2) not in pairs, (
        "low-confidence (0.5) edge leaked through under min_confidence=0.95 "
        "with no relation-confidence override -- fallback broken"
    )

    # Override -> edge surfaces.
    relations = await repo.list_relations_for_notebook(
        fixture["notebook_id"],
        ExportFilter(
            min_connections=0,
            min_confidence=0.95,
            min_relation_confidence=0.1,  # explicitly low
            include_orphans=True,
            include_archived=True,
        ),
    )
    pairs = {(r.in_entity, r.out_entity) for r in relations}
    assert (p1, p2) in pairs, (
        "explicit min_relation_confidence=0.1 should allow the 0.5 edge"
    )


# ---------------------------------------------------------------------------
# Empty-input / failure-path guards
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_empty_notebook_id_returns_empty(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Empty input short-circuits without a DB call."""
    repo = EntityRepository(config=live_surrealdb)
    result = await repo.list_entities_for_notebook("", ExportFilter())
    assert result == []


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_notebook_with_no_sources_returns_empty(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A notebook without any ``reference`` edges yields zero entities."""
    nb_id = await _create_notebook(live_surrealdb, "lonely-nb")
    repo = EntityRepository(config=live_surrealdb)
    result = await repo.list_entities_for_notebook(nb_id, ExportFilter())
    assert result == []


# ---------------------------------------------------------------------------
# K.4 rev2: reconciled external_ids/aliases survive DB -> projection -> exporter
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_external_ids_and_aliases_reach_exporter_through_projection(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """The reconciled URIs must survive the real DB->export read path.

    K.4 review MAJOR-1: ``_ENTITY_EXPORT_FIELDS`` previously omitted
    ``external_ids`` and ``aliases``, so ``list_entities_for_notebook``
    (the *only* loader the Track D exporters use) rehydrated those fields
    to the model default ``[]`` and every reconciled URI was silently
    dropped — the YAML frontmatter always emitted ``external_ids: []``.

    This test exercises the exact path the prior AC5 unit test skipped:
    persist an entity *with* reconciled ``external_ids``/``aliases`` set on
    the DB row, load it back through ``list_entities_for_notebook``, then
    call the exporter swap-point ``resolve_external_ids(loaded_entity)`` and
    assert it returns the populated URIs (NOT ``[]``).
    """
    notebook_id = await _create_notebook(live_surrealdb, "extids-nb")
    source_id = await _create_source(live_surrealdb, "extids-src")
    await _link_source_to_notebook(live_surrealdb, source_id, notebook_id)

    reconciled_uris = [
        "https://identifier.overheid.nl/tooi/id/ministerie/mnre1034",
        "https://doi.org/10.1145/example",
    ]
    reconciled_aliases = ["Min. BZK", "Binnenlandse Zaken"]

    target_id = await _create_entity(
        live_surrealdb,
        canonical_name="Ministerie van BZK",
        entity_type="Organization",
        source_documents=[source_id],
        confidence=0.99,
        external_ids=reconciled_uris,
        aliases=reconciled_aliases,
    )

    repo = EntityRepository(config=live_surrealdb)
    entities = await repo.list_entities_for_notebook(
        notebook_id,
        ExportFilter(
            min_connections=0,
            min_confidence=0.0,
            include_orphans=True,
            include_archived=True,
        ),
    )

    by_id = {e.id: e for e in entities}
    assert target_id in by_id, "reconciled entity missing from export projection"
    loaded = by_id[target_id]

    # The projection must carry the real DB values, not the model default [].
    assert loaded.external_ids == reconciled_uris, (
        f"external_ids dropped by projection -- got {loaded.external_ids!r}; "
        f"expected {reconciled_uris!r}"
    )
    assert loaded.aliases == reconciled_aliases, (
        f"aliases dropped by projection -- got {loaded.aliases!r}; "
        f"expected {reconciled_aliases!r}"
    )

    # The exporter swap-point must surface the reconciled URIs, not [].
    resolved = resolve_external_ids(loaded)
    assert resolved == reconciled_uris, (
        f"resolve_external_ids returned {resolved!r} after the DB->projection "
        f"round-trip; expected the reconciled URIs {reconciled_uris!r}. A [] "
        f"here means the K.4 feature is inert end-to-end (MAJOR-1 regression)."
    )

    # Belt-and-braces: the projection constant itself must keep both fields.
    assert "external_ids" in EntityRepository._ENTITY_EXPORT_FIELDS, (
        "external_ids regressed out of _ENTITY_EXPORT_FIELDS -- exporter path "
        "would silently drop reconciled URIs again"
    )
    assert "aliases" in EntityRepository._ENTITY_EXPORT_FIELDS, (
        "aliases regressed out of _ENTITY_EXPORT_FIELDS"
    )
