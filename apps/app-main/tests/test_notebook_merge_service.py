"""Tests for the B.6 ``NotebookMergeService``.

Six fixtures cover the acceptance criteria:

1. ``test_single_source_merges_into_target`` — N=1 source with 5 entities
   + 3 relations writes them all into the target.
2. ``test_multi_source_merges_with_dedup`` — same canonical name across
   two notebooks merges with union ``type_tags`` and max confidence.
3. ``test_idempotent_re_run`` — a second run is a no-op (counters zero).
4. ``test_type_collision_records_conflict_when_required`` — disjoint
   ``type_tags`` with ``type_match_required=True`` records a conflict
   and skips the merge.
5. ``test_type_collision_unions_when_not_required`` — same scenario,
   ``type_match_required=False`` unions the tags and merges anyway.
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

        # 4. resolve endpoint canonical_name
        if "FROM entity" in s and "id INSIDE" in s:
            wanted_ids = (params or {}).get("ids", [])
            rows: List[Dict[str, Any]] = []
            seen = set()
            for src_id, ents in self.entities_by_source.items():
                for e in ents:
                    eid = e.get("id")
                    if eid in wanted_ids and eid not in seen:
                        rows.append({"id": eid, "canonical_name": e.get("canonical_name", "")})
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
            # Existence probe for the idempotency counter.
            rn = (params or {}).get("src_name", "")
            tn = (params or {}).get("tgt_name", "")
            rt = (params or {}).get("rel_type", "")
            if (rn, tn, rt) in self.relation_exists_keys:
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
    """Return (service, mock_upsert_entity) with the fake query layer wired in."""
    mock_repo = MagicMock()

    async def _upsert(entity):
        # Record + simulate the post-write row state so the idempotency
        # snapshot can detect the write. We bump ``updated_at`` and
        # store the after-row keyed by (canonical_name, entity_type).
        fake.upsert_calls.append((entity.canonical_name, entity.entity_type))
        before = fake.target_existing.get((entity.canonical_name, entity.entity_type))
        if before is not None:
            # Update path — refresh updated_at to signal a write happened.
            fake.target_after_write[(entity.canonical_name, entity.entity_type)] = {
                **before,
                "updated_at": "2026-06-12T00:00:00Z",
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
    """Same canonical name with two different labels → one entity, two type_tags."""
    src_a_entities = [_entity_row("entity:a-oecd", "OECD", "Organization", 0.9)]
    src_b_entities = [_entity_row("entity:b-oecd", "OECD", "Org", 0.7)]
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
            # Disjoint type_tags by default would conflict; allow the
            # union here since the AC asks for the union outcome.
            type_match_required=False,
        )

    # One write — the dedup collapsed the two surface forms.
    assert mock_upsert.call_count == 1
    assert report.entities_merged == 1
    # The Entity model the service handed the repo should carry the
    # unioned type_tags with primary_type = highest-confidence label.
    call = mock_upsert.call_args
    entity = call.args[0]
    assert entity.canonical_name == "OECD"
    # Higher-confidence pass was "Organization" → that label wins
    # primary_type and the surface text.
    assert entity.primary_type == "Organization"
    assert set(entity.type_tags) == {"Organization", "Org"}


@pytest.mark.asyncio
async def test_idempotent_re_run():
    """Same merge twice → second run reports zero changes."""
    entities = [_entity_row("entity:e1", "OECD", "Organization", 0.9)]
    fake = _FakeExecuteQuery(
        sources_by_notebook={
            "notebook:src": ["source:s1"],
            "notebook:target": ["source:t1"],
        },
        entities_by_source={"source:s1": entities, "source:t1": []},
        relations_by_source={"source:s1": [], "source:t1": []},
    )
    svc, _ = _make_service_with_fake_db(fake)

    with patch(
        "app_main.services.notebook_merge_service.execute_query",
        new=fake,
    ):
        first = await svc.merge_notebooks(
            source_notebook_ids=["notebook:src"],
            target_notebook_id="notebook:target",
        )
        # Simulate post-write state on the target — the second probe
        # round will find the merged entity already there.
        fake.target_existing[("OECD", "Organization")] = {
            "id": "entity:target-oecd",
            "canonical_name": "OECD",
            "entity_type": "Organization",
            "confidence": 0.9,
            "type_tags": ["Organization"],
            "updated_at": "2026-06-12T00:00:00Z",
        }
        # Clear the after-write cache so the next run sees the existing
        # row unchanged unless an actual write happens.
        fake.target_after_write.clear()

        second = await svc.merge_notebooks(
            source_notebook_ids=["notebook:src"],
            target_notebook_id="notebook:target",
        )

    assert first.entities_merged == 1
    # Second run: the upsert still runs but produces the same updated_at
    # (the fake records the bumped timestamp + on second pass the
    # before/after match), so the counter stays at 0.
    assert second.entities_merged == 0
    assert second.relations_created == 0


@pytest.mark.asyncio
async def test_type_collision_records_conflict_when_required():
    """Disjoint type_tags + type_match_required=True → conflict, no merge."""
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

    assert len(report.conflicts) == 1
    conflict = report.conflicts[0]
    assert conflict.normalized_name == "smith"
    assert set(conflict.conflicting_type_tags) == {"Organization", "Person"}
    assert set(conflict.source_notebook_ids) == {"notebook:a", "notebook:b"}
    # No merge took place — upsert was never called for this entity.
    assert mock_upsert.call_count == 0
    assert report.entities_merged == 0


@pytest.mark.asyncio
async def test_type_collision_unions_when_not_required():
    """Same disjoint scenario with type_match_required=False → merged anyway."""
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
    # One write — both contributors collapsed into a single merged entity.
    assert mock_upsert.call_count == 1
    entity = mock_upsert.call_args.args[0]
    assert set(entity.type_tags) == {"Organization", "Person"}
    # Higher-confidence pass (0.9) supplies primary_type.
    assert entity.primary_type == "Organization"


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
