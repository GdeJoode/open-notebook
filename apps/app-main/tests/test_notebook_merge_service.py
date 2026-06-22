"""Tests for the B.6 ``NotebookMergeService``.

The fixtures cover the acceptance criteria:

1. ``test_single_source_merges_into_target`` — N=1 source with 5 entities
   + 3 relations writes them all into the target.
2. ``test_multi_source_merges_with_dedup`` — same canonical name AND same
   type across two notebooks merges with union ``type_tags`` and max
   confidence (the legitimate cross-notebook dedup case).
3. ``test_same_name_different_type_stays_distinct`` — K.1 rev2 regression:
   same normalized name + different type in the SAME notebook stay two
   distinct entities (no silent cross-type over-merge).
4. ``test_idempotent_re_run`` — a second run is a no-op (counters zero).
5. ``test_cross_type_homographs_stay_distinct_when_required`` /
   ``test_cross_type_homographs_stay_distinct_when_not_required`` — a
   name shared across notebooks with disjoint types is kept as separate
   entities (no conflict, no union) under both ``type_match_required``
   values, because the ``(name, type)`` bucket key never collides them.
6. ``test_dry_run_returns_counts_without_writing`` — counts populated
   but ``upsert_entity`` is never invoked.

All tests use AsyncMock-only seams — no SurrealDB required. The
``execute_query`` helper is patched at the service module so the
notebook-source resolve + relation queries return deterministic rows.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app_main.services.notebook_merge_service import NotebookMergeService
from shared.utils.name_normalizer import normalize_entity_name

# ---------------------------------------------------------------------------
# Helpers — a programmable fake of the per-notebook query layer.
# ---------------------------------------------------------------------------


class _FakeExecuteQuery:
    """Records calls + returns scripted rows by query-substring matching.

    The merge service makes 5 distinct queries:

    1. ``SELECT VALUE in FROM reference`` — resolve notebook → source ids
    2. ``SELECT id, canonical_name, entity_type, ... FROM entity WHERE
       source_documents ANYINSIDE`` — list entities for a notebook
    3. ``SELECT id, in AS source_id, ... FROM relation WHERE
       source_documents ANYINSIDE`` — list relations for a notebook
    4. ``SELECT id, canonical_name FROM entity WHERE id INSIDE`` —
       resolve endpoint canonical_name for relations
    5. ``SELECT id, canonical_name, entity_type, ... LIMIT 1`` — target
       entity probe + post-write probe
    6. The RELATE / probe LET-blocks — return [] (no existing relation)

    We dispatch by substring (cheaper than parsing SurrealQL).
    """

    def __init__(
        self,
        sources_by_notebook: Dict[str, List[str]],
        entities_by_source: Dict[str, List[Dict[str, Any]]],
        relations_by_source: Dict[str, List[Dict[str, Any]]],
        target_existing: Dict[tuple, Dict[str, Any]] | None = None,
        relation_exists_keys: set[tuple] | None = None,
    ) -> None:
        self.sources_by_notebook = sources_by_notebook
        self.entities_by_source = entities_by_source
        self.relations_by_source = relations_by_source
        # Map (canonical_name, entity_type) → row stub.
        self.target_existing = target_existing or {}
        # After a write, the row's updated_at is bumped.
        self.target_after_write: Dict[tuple, Dict[str, Any]] = {}
        # (src_text, tgt_text, rel_type) tuples that the DB already has.
        self.relation_exists_keys = relation_exists_keys or set()
        # Track when upsert was called so the post-write probe returns
        # a refreshed updated_at.
        self.upsert_calls: List[tuple] = []
        # Counter used by the strict-advance mock to produce a monotonically
        # increasing ``updated_at`` on every upsert — mirrors production
        # ``time::now()`` semantics so the semantic-content idempotency
        # path is the only thing that can drive the counter to zero.
        self.upsert_call_count: int = 0
        # Record raw query history for assertion convenience.
        self.calls: List[tuple] = []

    async def __call__(
        self, sql: str, params: Dict[str, Any] | None = None, *args, **kwargs
    ):
        self.calls.append((sql, params))
        s = sql.strip()

        # 1. notebook → source ids resolve
        if "FROM reference" in s and "out = type::thing" in s:
            nb_id = (params or {}).get("notebook_id", "")
            return list(self.sources_by_notebook.get(nb_id, []))

        # 2. list entities for notebook
        if "FROM entity" in s and "source_documents ANYINSIDE" in s and "canonical_name" in s and "primary_type" in s:
            source_ids = (params or {}).get("source_ids", [])
            out: List[Dict[str, Any]] = []
            for sid in source_ids:
                out.extend(self.entities_by_source.get(sid, []))
            return out

        # 3. list relations for notebook
        if "FROM relation" in s and "source_documents ANYINSIDE" in s:
            source_ids = (params or {}).get("source_ids", [])
            out = []
            for sid in source_ids:
                out.extend(self.relations_by_source.get(sid, []))
            return out

        # 4. resolve endpoint canonical_name (+ entity_type for K.7a)
        if "FROM entity" in s and "id INSIDE" in s:
            wanted_ids = (params or {}).get("ids", [])
            rows: List[Dict[str, Any]] = []
            seen = set()
            for src_id, ents in self.entities_by_source.items():
                for e in ents:
                    eid = e.get("id")
                    if eid in wanted_ids and eid not in seen:
                        rows.append(
                            {
                                "id": eid,
                                "canonical_name": e.get("canonical_name", ""),
                                "entity_type": e.get("entity_type", ""),
                            }
                        )
                        seen.add(eid)
            return rows

        # 5. target entity probe (LIMIT 1)
        if "FROM entity" in s and "canonical_name = $canonical_name" in s and "entity_type = $entity_type" in s:
            cn = (params or {}).get("canonical_name", "")
            et = (params or {}).get("entity_type", "")
            row = self.target_existing.get((cn, et))
            # If a write happened for this key, return the after-write row.
            after = self.target_after_write.get((cn, et))
            if after is not None:
                return [after]
            if row is None:
                return []
            return [row]

        # 6. RELATE / existing-edge probe — return [] (no existing)
        if "RELATE" in s or "FROM relation" in s:
            rn = (params or {}).get("src_name", "")
            tn = (params or {}).get("tgt_name", "")
            rt = (params or {}).get("rel_type", "")
            already_exists = (rn, tn, rt) in self.relation_exists_keys
            if "RELATE" in s:
                # _create_relation: returns the inserted edge row when
                # a fresh edge was written, [] when the IF-block found
                # an existing one. Service uses bool(rows) as the
                # idempotency signal.
                if already_exists:
                    return []
                # Mark "now exists" so subsequent calls in the same
                # test recognise the second call as a no-op.
                self.relation_exists_keys.add((rn, tn, rt))
                return [{"id": f"relation:{rn}-{tn}-{rt}"}]
            # Plain probe (_relation_exists, used by dry-run).
            if already_exists:
                return [{"id": "relation:exists"}]
            return []

        # Default: no rows.
        return []


def _entity_row(
    id_: str,
    canonical_name: str,
    entity_type: str,
    confidence: float = 0.9,
    source_documents: List[str] | None = None,
    type_tags: List[str] | None = None,
    properties: Dict[str, Any] | None = None,
    updated_at: str | None = None,
) -> Dict[str, Any]:
    return {
        "id": id_,
        "canonical_name": canonical_name,
        "entity_type": entity_type,
        "confidence": confidence,
        "source_documents": source_documents or [],
        "type_tags": type_tags or [],
        "properties": properties or {},
        "provenance_chain": [],
        "primary_type": entity_type,
        "updated_at": updated_at or "2026-06-01T00:00:00Z",
    }


def _relation_row(
    src_id: str,
    tgt_id: str,
    relation_type: str,
    confidence: float = 0.8,
) -> Dict[str, Any]:
    return {
        "id": f"relation:{src_id}-{tgt_id}-{relation_type}",
        "source_id": src_id,
        "target_id": tgt_id,
        "relation_type": relation_type,
        "confidence": confidence,
        "properties": {},
        "source_documents": [],
    }


def _make_service_with_fake_db(fake: _FakeExecuteQuery) -> tuple[NotebookMergeService, AsyncMock]:
    """Return (service, mock_upsert_entity) with the fake query layer wired in.

    The mock ``_upsert`` faithfully mimics ``EntityRepository.upsert_entity``:

    - ALWAYS strictly advances ``updated_at`` on every call (production
      runs ``SET updated_at = time::now()`` unconditionally — the mock
      must reflect this so a naive timestamp-based idempotency check
      would fail; this is exactly what review blocker B1 surfaced).
    - Persists the FULL merged row (canonical_name, entity_type,
      type_tags, primary_type, confidence, source_documents,
      properties) so the service's semantic-content comparison sees a
      faithful "after" view on re-runs.
    """
    mock_repo = MagicMock()

    async def _upsert(entity):
        fake.upsert_calls.append((entity.canonical_name, entity.entity_type))
        # Strictly-advancing counter — production repo bumps
        # ``updated_at`` via ``time::now()`` on every UPDATE, so any
        # idempotency signal that compares ``updated_at`` would fire
        # every time. We honour the same contract here.
        fake.upsert_call_count += 1
        ts = f"2026-06-12T00:00:{fake.upsert_call_count:02d}Z"
        key = (entity.canonical_name, entity.entity_type)
        # Compose the post-write row from the upserted Entity model
        # rather than the pre-write row — this is what the repo would
        # produce after the UPDATE (or what a CREATE would emit).
        fake.target_after_write[key] = {
            "id": f"entity:{entity.canonical_name}-{entity.entity_type}",
            "canonical_name": entity.canonical_name,
            "entity_type": entity.entity_type,
            "type_tags": list(entity.type_tags),
            "primary_type": entity.primary_type,
            "confidence": entity.confidence,
            "source_documents": list(entity.source_documents),
            "properties": dict(entity.properties),
            "updated_at": ts,
        }
        return f"entity:{entity.canonical_name}-{entity.entity_type}"

    mock_repo.upsert_entity = AsyncMock(side_effect=_upsert)
    svc = NotebookMergeService(entity_repository=mock_repo)
    return svc, mock_repo.upsert_entity


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_single_source_merges_into_target():
    """N=1 source with 5 entities + 3 relations → all 5 + 3 land in target."""
    entities = [
        _entity_row(f"entity:src-e{i}", f"Ent{i}", "ORG", 0.8)
        for i in range(5)
    ]
    relations = [
        _relation_row(f"entity:src-e{i}", f"entity:src-e{i+1}", "RELATED")
        for i in range(3)
    ]
    fake = _FakeExecuteQuery(
        sources_by_notebook={
            "notebook:src": ["source:s1"],
            "notebook:target": ["source:t1"],
        },
        entities_by_source={"source:s1": entities, "source:t1": []},
        relations_by_source={"source:s1": relations, "source:t1": []},
    )
    svc, mock_upsert = _make_service_with_fake_db(fake)

    with patch(
        "app_main.services.notebook_merge_service.execute_query",
        new=fake,
    ):
        report = await svc.merge_notebooks(
            source_notebook_ids=["notebook:src"],
            target_notebook_id="notebook:target",
        )

    assert report.entities_merged == 5
    assert report.relations_created == 3
    assert report.conflicts == []
    assert report.source_notebook_ids == ["notebook:src"]
    assert report.target_notebook_id == "notebook:target"
    assert mock_upsert.call_count == 5


@pytest.mark.asyncio
async def test_multi_source_merges_with_dedup():
    """Same canonical name AND same type across two notebooks → one entity.

    The legitimate cross-notebook dedup case: two notebooks each carry an
    ``OECD`` Organization. The (name, type) bucket key collapses them into
    a single entity, max-confidence label/text winning. Different-typed
    homographs are covered by
    ``test_same_name_different_type_stays_distinct`` instead — they must
    NOT collapse.
    """
    src_a_entities = [
        _entity_row(
            "entity:a-oecd",
            "OECD",
            "Organization",
            0.9,
            type_tags=["Organization", "IGO"],
        )
    ]
    src_b_entities = [
        _entity_row(
            "entity:b-oecd",
            "OECD",
            "Organization",
            0.7,
            type_tags=["Organization", "Institution"],
        )
    ]
    fake = _FakeExecuteQuery(
        sources_by_notebook={
            "notebook:a": ["source:a"],
            "notebook:b": ["source:b"],
            "notebook:target": [],
        },
        entities_by_source={
            "source:a": src_a_entities,
            "source:b": src_b_entities,
        },
        relations_by_source={"source:a": [], "source:b": []},
    )
    svc, mock_upsert = _make_service_with_fake_db(fake)

    with patch(
        "app_main.services.notebook_merge_service.execute_query",
        new=fake,
    ):
        report = await svc.merge_notebooks(
            source_notebook_ids=["notebook:a", "notebook:b"],
            target_notebook_id="notebook:target",
        )

    # One write — the dedup collapsed the two same-typed surface forms.
    assert mock_upsert.call_count == 1
    assert report.entities_merged == 1
    # The Entity model the service handed the repo should carry the
    # unioned type_tags with primary_type = highest-confidence label.
    call = mock_upsert.call_args
    entity = call.args[0]
    assert entity.canonical_name == "OECD"
    # Higher-confidence pass (0.9) wins primary_type and the surface text.
    assert entity.primary_type == "Organization"
    assert set(entity.type_tags) == {"Organization", "IGO", "Institution"}


@pytest.mark.asyncio
async def test_same_name_different_type_stays_distinct():
    """REGRESSION (K.1 rev2/rev3): minister vs ministry → 2 distinct entities.

    ``Minister van BZK`` (person) and ``Ministerie van BZK`` (organization)
    must stay two entities. rev2 kept them apart via the ``(name, type)``
    bucket key (both normalized to ``bzk``). rev3 goes further and stops
    stripping the person-role leader, so the two now normalize to DIFFERENT
    strings (``minister van bzk`` vs ``bzk``) — they are trivially distinct at
    BOTH the name level and the (name, type) level, and their relations route
    to different endpoints. Two separate entities are written from the SAME
    notebook and no conflict is raised.
    """
    entities = [
        _entity_row("entity:minister", "Minister van BZK", "person", 0.9),
        _entity_row("entity:ministerie", "Ministerie van BZK", "organization", 0.8),
    ]
    fake = _FakeExecuteQuery(
        sources_by_notebook={
            "notebook:src": ["source:s1"],
            "notebook:target": [],
        },
        entities_by_source={"source:s1": entities},
        relations_by_source={"source:s1": []},
    )
    svc, mock_upsert = _make_service_with_fake_db(fake)

    with patch(
        "app_main.services.notebook_merge_service.execute_query",
        new=fake,
    ):
        report = await svc.merge_notebooks(
            source_notebook_ids=["notebook:src"],
            target_notebook_id="notebook:target",
            type_match_required=True,
        )

    # Two distinct entities written — NOT collapsed into one.
    assert report.entities_merged == 2
    assert mock_upsert.call_count == 2
    # No conflict: distinct types never share a bucket, so the cross-type
    # collision can't arise by construction.
    assert report.conflicts == []
    written = {
        (c.args[0].canonical_name, c.args[0].entity_type)
        for c in mock_upsert.call_args_list
    }
    assert written == {
        ("Minister van BZK", "person"),
        ("Ministerie van BZK", "organization"),
    }
    # Each entity keeps ONLY its own type tag — no cross-type leakage.
    tags_by_type = {
        c.args[0].entity_type: set(c.args[0].type_tags)
        for c in mock_upsert.call_args_list
    }
    assert tags_by_type == {
        "person": {"person"},
        "organization": {"organization"},
    }


@pytest.mark.asyncio
async def test_relation_endpoints_do_not_cross_types():
    """REGRESSION (K.1 rev3): a relation touching a minister vs ministry routes
    to the correct, distinct endpoint.

    Because ``Minister van BZK`` (person) and ``Ministerie van BZK`` (org) now
    normalize to DIFFERENT strings, the relation-endpoint rewrite
    (``name_to_canon``) resolves each name to its own bucket — the RELATE is
    issued against the right canonical text and never collapses the person and
    the org onto one endpoint.
    """
    entities = [
        _entity_row("entity:minister", "Minister van BZK", "person", 0.9),
        _entity_row("entity:ministerie", "Ministerie van BZK", "organization", 0.8),
        _entity_row("entity:wet", "Klimaatwet", "other", 0.7),
    ]
    # Two relations: the minister signs the law; the ministry drafts it. They
    # must keep distinct source endpoints.
    relations = [
        _relation_row("entity:minister", "entity:wet", "SIGNS"),
        _relation_row("entity:ministerie", "entity:wet", "DRAFTS"),
    ]
    fake = _FakeExecuteQuery(
        sources_by_notebook={
            "notebook:src": ["source:s1"],
            "notebook:target": [],
        },
        entities_by_source={"source:s1": entities},
        relations_by_source={"source:s1": relations},
    )
    svc, _ = _make_service_with_fake_db(fake)

    with patch(
        "app_main.services.notebook_merge_service.execute_query",
        new=fake,
    ):
        report = await svc.merge_notebooks(
            source_notebook_ids=["notebook:src"],
            target_notebook_id="notebook:target",
        )

    assert report.relations_created == 2
    # Inspect the RELATE calls: each relation's source endpoint text must be
    # the original (now-distinct) canonical surface form, not a shared 'BZK'.
    relate_sources = {
        (params or {}).get("src_name")
        for sql, params in fake.calls
        if "RELATE" in sql.strip()
    }
    assert "Minister van BZK" in relate_sources
    assert "Ministerie van BZK" in relate_sources


@pytest.mark.asyncio
async def test_relation_endpoints_route_to_type_correct_winner_on_shared_name():
    """K.7a: when a person-X and an org-X share a NORMALIZED name (the
    aggressive-normalizer world K.7b unlocks), each incident relation routes
    to its TYPE-CORRECT merged bucket — not the highest-confidence-of-any-type
    one.

    Simulated by two entities whose surface forms normalize to the same key
    (``"BZK"`` and ``"bzk."`` both normalize to the BZK org canonical) but with
    DISTINCT surface texts and DIFFERENT types, so the chosen ``src_name`` on
    the RELATE reveals which typed bucket each relation resolved to.

    Falsifiability: under the OLD name-only ``name_to_canon`` both relations
    would route to the single highest-confidence bucket (the person, conf 0.95)
    — so the org relation's ``src_name`` would be the PERSON's text. The
    (name, type) key makes each route to its own type's winner.
    """
    # Person bucket: higher confidence (0.95), surface "De BZK".
    # Org bucket: lower confidence (0.80), surface "BZK".
    # "De BZK" and "BZK" normalize to the SAME key (article strip → org alias),
    # so they collide on name but stay distinct by (name, type). Distinct
    # surface texts let the RELATE's src_name reveal which bucket each routed to.
    person = _entity_row("entity:person", "De BZK", "person", 0.95)
    org = _entity_row("entity:org", "BZK", "organization", 0.80)
    wet = _entity_row("entity:wet", "Klimaatwet", "other", 0.7)
    entities = [person, org, wet]
    # Sanity: the two endpoints really do collide on the normalized name.
    assert normalize_entity_name("De BZK") == normalize_entity_name("BZK")

    # One relation from the person, one from the org — same target.
    relations = [
        _relation_row("entity:person", "entity:wet", "LEADS"),
        _relation_row("entity:org", "entity:wet", "FUNDS"),
    ]
    fake = _FakeExecuteQuery(
        sources_by_notebook={
            "notebook:src": ["source:s1"],
            "notebook:target": [],
        },
        entities_by_source={"source:s1": entities},
        relations_by_source={"source:s1": relations},
    )
    svc, _ = _make_service_with_fake_db(fake)

    with patch(
        "app_main.services.notebook_merge_service.execute_query",
        new=fake,
    ):
        await svc.merge_notebooks(
            source_notebook_ids=["notebook:src"],
            target_notebook_id="notebook:target",
        )

    # Map each RELATE's relation_type → the src_name it resolved to.
    relate_by_type = {
        (params or {}).get("rel_type"): (params or {}).get("src_name")
        for sql, params in fake.calls
        if "RELATE" in sql.strip()
    }
    # The person relation routes to the PERSON bucket's text; the org relation
    # to the ORG bucket's text. Under the old name-only key both would be the
    # person's text (0.95 > 0.80).
    assert relate_by_type["LEADS"] == "De BZK"
    assert relate_by_type["FUNDS"] == "BZK"


@pytest.mark.asyncio
async def test_idempotent_re_run():
    """Same merge twice → second run reports zero changes.

    REGRESSION TEST FOR B1 — attempt 1 of this phase relied on
    ``updated_at`` mismatch to drive the idempotency counter; production
    ``upsert_entity`` always bumps ``updated_at = time::now()``, so the
    counter would have inflated to 1 (every contributed entity) on
    re-run. The strict-advance mock (above) faithfully reproduces that
    production behaviour, so this test PASSES only if the service
    compares semantic content rather than timestamps.
    """
    entities = [
        _entity_row(
            "entity:e1",
            "OECD",
            "Organization",
            0.9,
            source_documents=["source:s1"],
        )
    ]
    relations = [_relation_row("entity:e1", "entity:e1", "RELATED")]
    fake = _FakeExecuteQuery(
        sources_by_notebook={
            "notebook:src": ["source:s1"],
            "notebook:target": ["source:t1"],
        },
        entities_by_source={"source:s1": entities, "source:t1": []},
        relations_by_source={"source:s1": relations, "source:t1": []},
    )
    svc, mock_upsert = _make_service_with_fake_db(fake)

    with patch(
        "app_main.services.notebook_merge_service.execute_query",
        new=fake,
    ):
        first = await svc.merge_notebooks(
            source_notebook_ids=["notebook:src"],
            target_notebook_id="notebook:target",
        )
        second = await svc.merge_notebooks(
            source_notebook_ids=["notebook:src"],
            target_notebook_id="notebook:target",
        )

    # First run plants the row + relation.
    assert first.entities_merged == 1
    assert first.relations_created == 1
    # Second run: the upsert STILL runs (so updated_at advances), but
    # the semantic-content check sees the would-be merged Entity equals
    # the row that was just persisted ∴ counter stays at zero.
    assert second.entities_merged == 0
    assert second.relations_created == 0
    # Sanity: both runs DID call upsert (proves we're not just caching
    # the first run's report).
    assert mock_upsert.call_count == 2
    # Sanity: the strict-advance mock did advance the timestamp on the
    # second call (proves the production ``updated_at = time::now()``
    # contract is faithfully simulated).
    second_call_ts = fake.target_after_write[("OECD", "Organization")][
        "updated_at"
    ]
    assert second_call_ts == "2026-06-12T00:00:02Z"


@pytest.mark.asyncio
async def test_cross_type_homographs_stay_distinct_when_required():
    """K.1 rev2: same name, different type across notebooks → 2 distinct entities.

    Formerly ``test_type_collision_records_conflict_when_required``, which
    asserted a CONFLICT was raised for ``Smith`` (Organization) vs
    ``Smith`` (Person) under a name-only bucket key. The (name, type) key
    keeps them in separate buckets, so no collision arises by construction
    — both entities are written, no conflict recorded. This is the correct
    behaviour: a person and an org with the same surface name ARE
    different entities.
    """
    src_a_entities = [_entity_row("entity:a-smith", "Smith", "Organization", 0.9)]
    src_b_entities = [_entity_row("entity:b-smith", "Smith", "Person", 0.85)]
    fake = _FakeExecuteQuery(
        sources_by_notebook={
            "notebook:a": ["source:a"],
            "notebook:b": ["source:b"],
            "notebook:target": [],
        },
        entities_by_source={
            "source:a": src_a_entities,
            "source:b": src_b_entities,
        },
        relations_by_source={"source:a": [], "source:b": []},
    )
    svc, mock_upsert = _make_service_with_fake_db(fake)

    with patch(
        "app_main.services.notebook_merge_service.execute_query",
        new=fake,
    ):
        report = await svc.merge_notebooks(
            source_notebook_ids=["notebook:a", "notebook:b"],
            target_notebook_id="notebook:target",
            type_match_required=True,
        )

    # No conflict — distinct types never collide on the (name, type) key.
    assert report.conflicts == []
    # Two distinct entities written, one per type.
    assert report.entities_merged == 2
    assert mock_upsert.call_count == 2
    written = {
        (c.args[0].canonical_name, c.args[0].entity_type)
        for c in mock_upsert.call_args_list
    }
    assert written == {("Smith", "Organization"), ("Smith", "Person")}


@pytest.mark.asyncio
async def test_cross_type_homographs_stay_distinct_when_not_required():
    """Same scenario with type_match_required=False → still two distinct entities.

    The disjoint-type "union into one entity" outcome is no longer
    reachable: the (name, type) key separates the buckets before any
    union could occur, regardless of ``type_match_required``. Both flag
    values yield two distinct entities — which is the correct, non-corrupt
    result.
    """
    src_a_entities = [_entity_row("entity:a-smith", "Smith", "Organization", 0.9)]
    src_b_entities = [_entity_row("entity:b-smith", "Smith", "Person", 0.85)]
    fake = _FakeExecuteQuery(
        sources_by_notebook={
            "notebook:a": ["source:a"],
            "notebook:b": ["source:b"],
            "notebook:target": [],
        },
        entities_by_source={
            "source:a": src_a_entities,
            "source:b": src_b_entities,
        },
        relations_by_source={"source:a": [], "source:b": []},
    )
    svc, mock_upsert = _make_service_with_fake_db(fake)

    with patch(
        "app_main.services.notebook_merge_service.execute_query",
        new=fake,
    ):
        report = await svc.merge_notebooks(
            source_notebook_ids=["notebook:a", "notebook:b"],
            target_notebook_id="notebook:target",
            type_match_required=False,
        )

    assert report.conflicts == []
    # Two writes — the entities stay distinct, never unioned into one.
    assert mock_upsert.call_count == 2
    written = {
        (c.args[0].canonical_name, c.args[0].entity_type)
        for c in mock_upsert.call_args_list
    }
    assert written == {("Smith", "Organization"), ("Smith", "Person")}
    for c in mock_upsert.call_args_list:
        entity = c.args[0]
        # Each entity keeps only its own type tag — no cross-type union.
        assert set(entity.type_tags) == {entity.entity_type}


@pytest.mark.asyncio
async def test_dry_run_returns_counts_without_writing():
    """dry_run=True → counts populated, upsert never called."""
    entities = [
        _entity_row("entity:e1", "Acme", "Organization", 0.9),
        _entity_row("entity:e2", "Beta Corp", "Organization", 0.8),
    ]
    relations = [_relation_row("entity:e1", "entity:e2", "PARTNER_OF")]
    fake = _FakeExecuteQuery(
        sources_by_notebook={
            "notebook:src": ["source:s1"],
            "notebook:target": [],
        },
        entities_by_source={"source:s1": entities},
        relations_by_source={"source:s1": relations},
    )
    svc, mock_upsert = _make_service_with_fake_db(fake)

    with patch(
        "app_main.services.notebook_merge_service.execute_query",
        new=fake,
    ):
        report = await svc.merge_notebooks(
            source_notebook_ids=["notebook:src"],
            target_notebook_id="notebook:target",
            dry_run=True,
        )

    assert report.dry_run is True
    assert report.entities_merged == 2
    assert report.relations_created == 1
    # Critical: no actual writes during a dry run.
    assert mock_upsert.call_count == 0
