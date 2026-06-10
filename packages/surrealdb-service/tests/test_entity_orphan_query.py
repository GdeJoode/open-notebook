"""Unit tests for :meth:`EntityRepository.list_orphans_for_source` (B.5a).

These tests cover the contract of the orphan query without standing up a
live SurrealDB instance. The repo's ``execute_query`` call is patched at
the import site (``surrealdb_service.repositories.entity.execute_query``)
so each test can simulate:

* the source-SELECT result (entities owned by the source), and
* the per-entity edge-probe SELECT (zero or N edges).

Six paths are pinned to lock the failure / continuation semantics
documented in :meth:`EntityRepository.list_orphans_for_source`:

1. empty ``source_id`` short-circuits to ``[]`` with no DB call
2. entity-SELECT raises -> logged + returns ``[]`` (no exception escapes)
3. per-entity edge-probe raises -> entity EXCLUDED, loop continues
4. no entities for source -> returns ``[]``
5. all entities orphan -> returns all entities
6. mixed (some orphan, some not) -> returns only the orphan rows
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

import surrealdb_service.repositories.entity as entity_module
from surrealdb_service.repositories.entity import EntityRepository


class _ExecuteQueryStub:
    """Stand-in for ``execute_query`` that returns scripted responses.

    The repository issues one entity-SELECT followed by one edge-probe
    per entity. The stub distinguishes them by query substring so the
    test arranges for ``entities`` and ``edges_by_eid`` independently.

    Per-entity edge-probes can also be configured to raise — useful for
    the "loop continues after probe failure" contract test.
    """

    def __init__(
        self,
        *,
        entities_result: Any = None,
        entities_raises: Optional[Exception] = None,
        edges_by_eid: Optional[Dict[str, Any]] = None,
        edges_raises_for: Optional[set[str]] = None,
    ) -> None:
        self.entities_result = entities_result
        self.entities_raises = entities_raises
        self.edges_by_eid = edges_by_eid or {}
        self.edges_raises_for = edges_raises_for or set()
        self.calls: List[Dict[str, Any]] = []

    async def __call__(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
        config: Any = None,
    ) -> Any:
        self.calls.append({"query": query, "params": params or {}})
        # Distinguish the two query shapes by substring. The orphan
        # query uses ``FROM entity`` for the source SELECT and
        # ``FROM relation`` for the edge probe.
        if "FROM entity" in query:
            if self.entities_raises is not None:
                raise self.entities_raises
            return self.entities_result
        if "FROM relation" in query:
            eid = (params or {}).get("eid")
            if eid in self.edges_raises_for:
                raise RuntimeError(f"probe failed for {eid}")
            return self.edges_by_eid.get(eid, [])
        raise AssertionError(f"unexpected query: {query!r}")


@pytest.fixture
def patch_execute_query(monkeypatch: pytest.MonkeyPatch):
    """Return a helper that installs ``stub`` as ``execute_query``."""

    def _install(stub: _ExecuteQueryStub) -> _ExecuteQueryStub:
        monkeypatch.setattr(entity_module, "execute_query", stub)
        return stub

    return _install


# ---------------------------------------------------------------------------
# Path 1: empty source_id short-circuits
# ---------------------------------------------------------------------------


async def test_empty_source_id_returns_empty_without_db_call(
    patch_execute_query,
) -> None:
    """Empty ``source_id`` returns ``[]`` and never touches the DB."""
    stub = patch_execute_query(_ExecuteQueryStub(entities_result=[]))
    repo = EntityRepository()

    result = await repo.list_orphans_for_source("")

    assert result == []
    assert stub.calls == [], "execute_query must not be called for empty source_id"


# ---------------------------------------------------------------------------
# Path 2: entity-SELECT raises -> logged + returns []
# ---------------------------------------------------------------------------


async def test_entity_select_raises_returns_empty_without_propagating(
    patch_execute_query,
) -> None:
    """If the source-entities SELECT raises, return ``[]`` (no exception)."""
    boom = RuntimeError("db down")
    stub = patch_execute_query(_ExecuteQueryStub(entities_raises=boom))
    repo = EntityRepository()

    # Critically: must not propagate. The orphan-detection path is
    # best-effort; a failure must degrade to "no orphans" rather than
    # blowing up the surrounding workflow stage.
    result = await repo.list_orphans_for_source("source:abc")

    assert result == []
    # Only the source-SELECT was attempted; no edge probes after the failure.
    assert len(stub.calls) == 1
    assert "FROM entity" in stub.calls[0]["query"]


# ---------------------------------------------------------------------------
# Path 3: per-entity edge-probe raises -> entity excluded, loop continues
# ---------------------------------------------------------------------------


async def test_edge_probe_raises_excludes_entity_and_continues(
    patch_execute_query,
) -> None:
    """A probe failure must exclude that entity, NOT abort the loop."""
    entities = [
        {"id": "entity:a", "canonical_name": "Alice", "entity_type": "PERSON"},
        {"id": "entity:b", "canonical_name": "Bob", "entity_type": "PERSON"},
        {"id": "entity:c", "canonical_name": "Carol", "entity_type": "PERSON"},
    ]
    # B's probe raises. A and C have zero edges -> they ARE orphans.
    stub = patch_execute_query(
        _ExecuteQueryStub(
            entities_result=entities,
            edges_by_eid={"entity:a": [], "entity:c": []},
            edges_raises_for={"entity:b"},
        )
    )
    repo = EntityRepository()

    result = await repo.list_orphans_for_source("source:abc")

    # B is excluded due to probe failure; A and C remain.
    result_ids = {row["id"] for row in result}
    assert result_ids == {"entity:a", "entity:c"}
    # All three probes were attempted (loop did not abort).
    edge_probes = [c for c in stub.calls if "FROM relation" in c["query"]]
    assert len(edge_probes) == 3


# ---------------------------------------------------------------------------
# Path 4: no entities for source -> []
# ---------------------------------------------------------------------------


async def test_source_with_no_entities_returns_empty(
    patch_execute_query,
) -> None:
    """A source with no entities short-circuits before any edge probes."""
    stub = patch_execute_query(_ExecuteQueryStub(entities_result=[]))
    repo = EntityRepository()

    result = await repo.list_orphans_for_source("source:empty")

    assert result == []
    # Only the source-SELECT was attempted; no probes ran.
    assert len(stub.calls) == 1
    assert "FROM entity" in stub.calls[0]["query"]


# ---------------------------------------------------------------------------
# Path 5: all entities orphan -> returns all
# ---------------------------------------------------------------------------


async def test_all_entities_orphan_returns_all(patch_execute_query) -> None:
    """Every entity reports zero edges -> every entity comes back."""
    entities = [
        {"id": "entity:a", "canonical_name": "Alice", "entity_type": "PERSON"},
        {"id": "entity:b", "canonical_name": "Bob", "entity_type": "PERSON"},
    ]
    patch_execute_query(
        _ExecuteQueryStub(
            entities_result=entities,
            edges_by_eid={"entity:a": [], "entity:b": []},
        )
    )
    repo = EntityRepository()

    result = await repo.list_orphans_for_source("source:abc")

    assert len(result) == 2
    assert {row["id"] for row in result} == {"entity:a", "entity:b"}


# ---------------------------------------------------------------------------
# Path 6: mixed -> returns only orphans
# ---------------------------------------------------------------------------


async def test_mixed_results_returns_only_orphans(patch_execute_query) -> None:
    """Only entities with empty edge-probe results are returned."""
    entities = [
        {"id": "entity:a", "canonical_name": "Alice", "entity_type": "PERSON"},
        {"id": "entity:b", "canonical_name": "Bob", "entity_type": "PERSON"},
        {"id": "entity:c", "canonical_name": "Carol", "entity_type": "PERSON"},
        {"id": "entity:d", "canonical_name": "Dan", "entity_type": "PERSON"},
    ]
    # A and C have edges; B and D are orphans.
    patch_execute_query(
        _ExecuteQueryStub(
            entities_result=entities,
            edges_by_eid={
                "entity:a": [{"id": "relation:1"}],
                "entity:b": [],
                "entity:c": [{"id": "relation:2"}],
                "entity:d": [],
            },
        )
    )
    repo = EntityRepository()

    result = await repo.list_orphans_for_source("source:abc")

    assert {row["id"] for row in result} == {"entity:b", "entity:d"}
    # Order is preserved (B before D, as they appear in entities).
    assert [row["id"] for row in result] == ["entity:b", "entity:d"]
