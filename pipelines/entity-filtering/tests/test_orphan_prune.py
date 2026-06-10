"""Tests for the orphan-prune lifecycle module (B.5b).

Coverage scope (mirrors the acceptance criteria in the B.5b plan):

1. :func:`mark_pending_reconnect` flips status, increments attempts,
   stamps timestamps -- and the second mark on the SAME entity does
   NOT overwrite ``first_orphaned_at`` (idempotent stamp).
2. :func:`retry_pending_reconnects`
   * happy path: the connector confirms a relation, status clears.
   * failure path: connector returns nothing, status stays pending,
     attempts increment.
   * connector-raises: the loop continues, the failed orphan stays
     pending.
3. :func:`archive_stale_orphans`
   * max_attempts threshold flips entities at attempts >= 3 to archived.
   * max_age_days threshold flips entities whose first_orphaned_at is
     older than the cutoff.
   * idempotency: running twice does not double-archive (because the
     second run sees zero ``pending_reconnect`` rows for those entities).

All tests use a thin in-memory mock that satisfies
:class:`OrphanPruneRepoProtocol`. The real connector module is mocked
out of the retry path via the ``orphan_connector_run`` injection point.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import pytest

from entity_filtering.resolution.orphan_prune import (
    DEFAULT_MAX_AGE_DAYS,
    DEFAULT_MAX_ATTEMPTS,
    STATUS_ARCHIVED,
    STATUS_PENDING_RECONNECT,
    archive_stale_orphans,
    mark_pending_reconnect,
    retry_pending_reconnects,
)


# ---------------------------------------------------------------------------
# In-memory mock satisfying OrphanPruneRepoProtocol
# ---------------------------------------------------------------------------


class MockRepo:
    """Faithfully models the repo surface used by the prune module.

    Each entity is a plain dict; the mock mutates fields in place so
    tests can assert on the final state without staging a DB.
    """

    def __init__(self, entities: Optional[Dict[str, Dict[str, Any]]] = None):
        self.entities: Dict[str, Dict[str, Any]] = entities or {}
        # Mapping notebook_id -> list of entity ids that "belong" to the
        # notebook (i.e. their first source's notebook). Tests configure
        # this directly.
        self.notebook_index: Dict[str, List[str]] = {}
        # Recording: list of (entity_id, status, kwargs) every time
        # update_orphan_status was called.
        self.update_calls: List[Dict[str, Any]] = []
        # Recording: list of (source_id) every time the orphan-connector
        # mock was invoked (driven from the test fixtures, not the
        # repo).
        self.connector_calls: List[str] = []

    def attach_to_notebook(self, notebook_id: str, entity_ids: List[str]) -> None:
        self.notebook_index[notebook_id] = list(entity_ids)

    async def list_orphans_for_source(
        self, source_id: str
    ) -> List[Dict[str, Any]]:
        # Required by OrphanEntityRepoProtocol but unused in these tests.
        return [
            dict(row)
            for row in self.entities.values()
            if source_id in (row.get("source_documents") or [])
            and not row.get("orphan_status")
        ]

    async def list_orphans_with_status(
        self,
        notebook_id: str,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        eids = self.notebook_index.get(notebook_id, [])
        out: List[Dict[str, Any]] = []
        for eid in eids:
            row = self.entities.get(eid)
            if not row:
                continue
            if status is None:
                if row.get("orphan_status"):
                    out.append(dict(row))
            elif row.get("orphan_status") == status:
                out.append(dict(row))
        return out

    async def update_orphan_status(
        self,
        entity_id: str,
        status: Optional[str],
        *,
        increment_attempts: bool = False,
        set_first_orphaned_at: bool = False,
        set_last_reconnect_attempt_at: bool = False,
    ) -> bool:
        self.update_calls.append(
            {
                "entity_id": entity_id,
                "status": status,
                "increment_attempts": increment_attempts,
                "set_first_orphaned_at": set_first_orphaned_at,
                "set_last_reconnect_attempt_at": set_last_reconnect_attempt_at,
            }
        )
        row = self.entities.get(entity_id)
        if row is None:
            return False
        row["orphan_status"] = status
        if increment_attempts:
            row["reconnect_attempts"] = int(row.get("reconnect_attempts") or 0) + 1
        if set_first_orphaned_at:
            # Idempotent: only stamp when currently NONE.
            if row.get("first_orphaned_at") in (None, ""):
                row["first_orphaned_at"] = datetime.now(timezone.utc)
        if set_last_reconnect_attempt_at:
            row["last_reconnect_attempt_at"] = datetime.now(timezone.utc)
        return True


def _entity(
    id_: str,
    name: str,
    *,
    source_id: str = "source:s1",
    status: Optional[str] = None,
    attempts: int = 0,
    first_orphaned_at: Optional[datetime] = None,
) -> Dict[str, Any]:
    return {
        "id": id_,
        "canonical_name": name,
        "entity_type": "MISC",
        "source_documents": [source_id],
        "orphan_status": status,
        "reconnect_attempts": attempts,
        "first_orphaned_at": first_orphaned_at,
        "last_reconnect_attempt_at": None,
    }


# ---------------------------------------------------------------------------
# Mock relation + connector for the retry path
# ---------------------------------------------------------------------------


class _FakeRelation:
    """Stand-in for ExtractedRelation -- carries just what we match on."""

    def __init__(self, source_entity: str, target_entity: str):
        self.source_entity = source_entity
        self.target_entity = target_entity


def _make_connector_run(
    *,
    by_source: Dict[str, List[_FakeRelation]] | None = None,
    raise_for: Optional[set[str]] = None,
):
    """Build a fake ``orphan_connector.run`` for tests.

    ``by_source`` maps a source_id to the list of relations the fake
    will return when that source is requested. ``raise_for`` is a set
    of source_ids that should make the fake raise (for the failure-
    path test).
    """
    by_source = by_source or {}
    raise_for = raise_for or set()
    captured: List[str] = []

    async def _fake_run(
        source_id: str,
        chunks,
        entity_repo,
        llm_caller,
        *,
        model: str = "default",
        max_proposals_per_orphan: int = 3,
        min_confidence: float = 0.6,
    ):
        captured.append(source_id)
        if source_id in raise_for:
            raise RuntimeError(f"forced failure for {source_id}")
        return list(by_source.get(source_id, []))

    return _fake_run, captured


# ===========================================================================
# 1. mark_pending_reconnect
# ===========================================================================


class TestMarkPendingReconnect:
    async def test_state_transition_and_timestamps(self):
        repo = MockRepo(
            {
                "entity:o1": _entity("entity:o1", "Alice"),
                "entity:o2": _entity("entity:o2", "Bob"),
            }
        )

        n = await mark_pending_reconnect(["entity:o1", "entity:o2"], repo)

        assert n == 2
        for eid in ("entity:o1", "entity:o2"):
            row = repo.entities[eid]
            assert row["orphan_status"] == STATUS_PENDING_RECONNECT
            assert row["reconnect_attempts"] == 1
            assert row["first_orphaned_at"] is not None
            assert row["last_reconnect_attempt_at"] is not None

    async def test_empty_input_returns_zero(self):
        repo = MockRepo({})
        n = await mark_pending_reconnect([], repo)
        assert n == 0
        assert repo.update_calls == []

    async def test_unknown_id_returns_zero_without_crash(self):
        repo = MockRepo({})
        n = await mark_pending_reconnect(["entity:does-not-exist"], repo)
        # The mock returns False for unknown IDs -> success count is 0.
        assert n == 0
        # But the update was still attempted -- that's the contract.
        assert len(repo.update_calls) == 1

    async def test_first_orphaned_at_is_idempotent(self):
        """Re-marking the same entity does NOT overwrite the first timestamp."""
        seed = datetime(2026, 1, 1, tzinfo=timezone.utc)
        repo = MockRepo(
            {
                "entity:o1": _entity(
                    "entity:o1",
                    "Alice",
                    status=STATUS_PENDING_RECONNECT,
                    attempts=1,
                    first_orphaned_at=seed,
                ),
            }
        )

        await mark_pending_reconnect(["entity:o1"], repo)

        row = repo.entities["entity:o1"]
        assert row["first_orphaned_at"] == seed
        # Attempts still tick.
        assert row["reconnect_attempts"] == 2


# ===========================================================================
# 2. retry_pending_reconnects
# ===========================================================================


class TestRetryPendingReconnects:
    async def test_no_pending_short_circuits(self):
        repo = MockRepo()
        repo.attach_to_notebook("notebook:nb", [])
        outcome = await retry_pending_reconnects(
            "notebook:nb",
            repo,
            chunks=[],
            orphan_connector_run=None,
        )
        assert outcome.attempted == 0
        assert outcome.reconnected == 0
        assert outcome.still_pending == 0

    async def test_success_path_clears_status(self):
        """A confirmed relation referencing the orphan clears its status."""
        repo = MockRepo(
            {
                "entity:o1": _entity(
                    "entity:o1",
                    "Alice",
                    source_id="source:s1",
                    status=STATUS_PENDING_RECONNECT,
                    attempts=1,
                    first_orphaned_at=datetime.now(timezone.utc),
                ),
            }
        )
        repo.attach_to_notebook("notebook:nb", ["entity:o1"])

        # Connector returns a relation mentioning "Alice" as source.
        run, captured = _make_connector_run(
            by_source={
                "source:s1": [_FakeRelation("Alice", "Bob")],
            }
        )

        persisted: List[Any] = []

        async def _persist(nb_id: str, rel):
            persisted.append((nb_id, rel))

        outcome = await retry_pending_reconnects(
            "notebook:nb",
            repo,
            chunks=[{"id": "c1", "text": "Alice and Bob met."}],
            orphan_connector_run=run,
            persist_relation=_persist,
        )

        assert outcome.attempted == 1
        assert outcome.reconnected == 1
        assert outcome.still_pending == 0
        # Status cleared.
        assert repo.entities["entity:o1"]["orphan_status"] is None
        # attempts incremented from 1 -> 2.
        assert repo.entities["entity:o1"]["reconnect_attempts"] == 2
        # persist_relation was called.
        assert len(persisted) == 1
        assert persisted[0][0] == "notebook:nb"
        # Connector was called with the source id.
        assert captured == ["source:s1"]

    async def test_failure_path_increments_but_stays_pending(self):
        """Connector returns no matching relation -> status stays pending."""
        repo = MockRepo(
            {
                "entity:o1": _entity(
                    "entity:o1",
                    "Alice",
                    source_id="source:s1",
                    status=STATUS_PENDING_RECONNECT,
                    attempts=1,
                    first_orphaned_at=datetime.now(timezone.utc),
                ),
            }
        )
        repo.attach_to_notebook("notebook:nb", ["entity:o1"])

        # Connector returns nothing for Alice.
        run, _ = _make_connector_run(by_source={"source:s1": []})

        outcome = await retry_pending_reconnects(
            "notebook:nb",
            repo,
            chunks=[],
            orphan_connector_run=run,
        )

        assert outcome.attempted == 1
        assert outcome.reconnected == 0
        assert outcome.still_pending == 1
        # Status unchanged.
        assert (
            repo.entities["entity:o1"]["orphan_status"]
            == STATUS_PENDING_RECONNECT
        )
        # attempts incremented 1 -> 2.
        assert repo.entities["entity:o1"]["reconnect_attempts"] == 2

    async def test_connector_raises_isolates_failure(self):
        """One orphan's connector failure must not abort the other retries."""
        repo = MockRepo(
            {
                "entity:o1": _entity(
                    "entity:o1",
                    "Alice",
                    source_id="source:s1",
                    status=STATUS_PENDING_RECONNECT,
                    attempts=1,
                ),
                "entity:o2": _entity(
                    "entity:o2",
                    "Bob",
                    source_id="source:s2",
                    status=STATUS_PENDING_RECONNECT,
                    attempts=1,
                ),
            }
        )
        repo.attach_to_notebook("notebook:nb", ["entity:o1", "entity:o2"])

        # s1 raises; s2 succeeds.
        run, captured = _make_connector_run(
            by_source={"source:s2": [_FakeRelation("Bob", "Carol")]},
            raise_for={"source:s1"},
        )

        outcome = await retry_pending_reconnects(
            "notebook:nb",
            repo,
            chunks=[],
            orphan_connector_run=run,
        )

        assert outcome.attempted == 2
        assert outcome.reconnected == 1
        assert outcome.still_pending == 1
        # Both connector calls were attempted.
        assert sorted(captured) == ["source:s1", "source:s2"]
        # o1 stayed pending; o2 cleared.
        assert (
            repo.entities["entity:o1"]["orphan_status"]
            == STATUS_PENDING_RECONNECT
        )
        assert repo.entities["entity:o2"]["orphan_status"] is None

    async def test_empty_notebook_id_returns_zero(self):
        repo = MockRepo()
        outcome = await retry_pending_reconnects("", repo, chunks=[])
        assert outcome.attempted == 0
        assert outcome.reconnected == 0

    # -----------------------------------------------------------------
    # M4 (attempt 2): entity_id_filter narrows to a single orphan.
    # -----------------------------------------------------------------

    async def test_entity_id_filter_narrows_to_one_orphan(self):
        """``entity_id_filter`` retries ONLY the targeted orphan.

        Three pending orphans in the notebook; the per-row dashboard
        action should call the connector for the clicked orphan only,
        not fan out across the other two.
        """
        repo = MockRepo(
            {
                "entity:o1": _entity(
                    "entity:o1",
                    "Alice",
                    source_id="source:s1",
                    status=STATUS_PENDING_RECONNECT,
                    attempts=1,
                ),
                "entity:o2": _entity(
                    "entity:o2",
                    "Bob",
                    source_id="source:s2",
                    status=STATUS_PENDING_RECONNECT,
                    attempts=1,
                ),
                "entity:o3": _entity(
                    "entity:o3",
                    "Carol",
                    source_id="source:s3",
                    status=STATUS_PENDING_RECONNECT,
                    attempts=1,
                ),
            }
        )
        repo.attach_to_notebook(
            "notebook:nb", ["entity:o1", "entity:o2", "entity:o3"]
        )

        run, captured = _make_connector_run(by_source={"source:s1": []})

        outcome = await retry_pending_reconnects(
            "notebook:nb",
            repo,
            chunks=[],
            orphan_connector_run=run,
            entity_id_filter="entity:o2",
        )

        # Only entity:o2 was retried -- connector was invoked once with
        # its source, NOT with s1 or s3.
        assert captured == ["source:s2"]
        assert outcome.attempted == 1
        # The OTHER orphans' attempts counter did not advance.
        assert repo.entities["entity:o1"]["reconnect_attempts"] == 1
        assert repo.entities["entity:o3"]["reconnect_attempts"] == 1
        # Targeted orphan ticked.
        assert repo.entities["entity:o2"]["reconnect_attempts"] == 2

    async def test_entity_id_filter_default_retries_all(self):
        """Without the filter, every pending orphan is still retried.

        Locks in backward-compat: the existing post-extraction sweep
        relies on no-filter calls touching every pending orphan.
        """
        repo = MockRepo(
            {
                "entity:o1": _entity(
                    "entity:o1",
                    "Alice",
                    source_id="source:s1",
                    status=STATUS_PENDING_RECONNECT,
                    attempts=1,
                ),
                "entity:o2": _entity(
                    "entity:o2",
                    "Bob",
                    source_id="source:s2",
                    status=STATUS_PENDING_RECONNECT,
                    attempts=1,
                ),
            }
        )
        repo.attach_to_notebook("notebook:nb", ["entity:o1", "entity:o2"])

        run, captured = _make_connector_run(by_source={})

        outcome = await retry_pending_reconnects(
            "notebook:nb",
            repo,
            chunks=[],
            orphan_connector_run=run,
            # entity_id_filter explicitly omitted -- the default.
        )

        assert outcome.attempted == 2
        assert sorted(captured) == ["source:s1", "source:s2"]

    async def test_entity_id_filter_unknown_id_returns_zero(self):
        """Unknown filter id -> zero attempts, no connector calls.

        Defensive case: the dashboard could send a stale id after a
        background sweep already archived the row. The endpoint should
        return an empty outcome rather than fanning out.
        """
        repo = MockRepo(
            {
                "entity:o1": _entity(
                    "entity:o1",
                    "Alice",
                    source_id="source:s1",
                    status=STATUS_PENDING_RECONNECT,
                    attempts=1,
                ),
            }
        )
        repo.attach_to_notebook("notebook:nb", ["entity:o1"])

        run, captured = _make_connector_run(by_source={})

        outcome = await retry_pending_reconnects(
            "notebook:nb",
            repo,
            chunks=[],
            orphan_connector_run=run,
            entity_id_filter="entity:does-not-exist",
        )

        assert outcome.attempted == 0
        assert captured == []  # connector never invoked


# ===========================================================================
# 3. archive_stale_orphans
# ===========================================================================


class TestArchiveStaleOrphans:
    async def test_max_attempts_threshold_archives(self):
        repo = MockRepo(
            {
                "entity:o1": _entity(
                    "entity:o1",
                    "Alice",
                    status=STATUS_PENDING_RECONNECT,
                    attempts=3,
                    first_orphaned_at=datetime.now(timezone.utc),
                ),
                "entity:o2": _entity(
                    "entity:o2",
                    "Bob",
                    status=STATUS_PENDING_RECONNECT,
                    attempts=2,
                    first_orphaned_at=datetime.now(timezone.utc),
                ),
            }
        )
        repo.attach_to_notebook("notebook:nb", ["entity:o1", "entity:o2"])

        archived = await archive_stale_orphans(
            repo, notebook_id="notebook:nb", max_attempts=3, max_age_days=365
        )

        assert archived == 1
        assert repo.entities["entity:o1"]["orphan_status"] == STATUS_ARCHIVED
        # Bob below the threshold -- still pending.
        assert (
            repo.entities["entity:o2"]["orphan_status"]
            == STATUS_PENDING_RECONNECT
        )

    async def test_max_age_days_threshold_archives(self):
        old_ts = datetime.now(timezone.utc) - timedelta(days=120)
        recent_ts = datetime.now(timezone.utc) - timedelta(days=5)
        repo = MockRepo(
            {
                "entity:old": _entity(
                    "entity:old",
                    "Alice",
                    status=STATUS_PENDING_RECONNECT,
                    attempts=1,
                    first_orphaned_at=old_ts,
                ),
                "entity:fresh": _entity(
                    "entity:fresh",
                    "Bob",
                    status=STATUS_PENDING_RECONNECT,
                    attempts=1,
                    first_orphaned_at=recent_ts,
                ),
            }
        )
        repo.attach_to_notebook("notebook:nb", ["entity:old", "entity:fresh"])

        archived = await archive_stale_orphans(
            repo,
            notebook_id="notebook:nb",
            max_attempts=99,
            max_age_days=90,
        )

        assert archived == 1
        assert (
            repo.entities["entity:old"]["orphan_status"] == STATUS_ARCHIVED
        )
        assert (
            repo.entities["entity:fresh"]["orphan_status"]
            == STATUS_PENDING_RECONNECT
        )

    async def test_either_threshold_alone_triggers(self):
        """An entity hitting EITHER threshold gets archived."""
        repo = MockRepo(
            {
                "entity:attempts": _entity(
                    "entity:attempts",
                    "Alice",
                    status=STATUS_PENDING_RECONNECT,
                    attempts=5,
                    first_orphaned_at=datetime.now(timezone.utc),
                ),
                "entity:age": _entity(
                    "entity:age",
                    "Bob",
                    status=STATUS_PENDING_RECONNECT,
                    attempts=1,
                    first_orphaned_at=datetime.now(timezone.utc)
                    - timedelta(days=180),
                ),
            }
        )
        repo.attach_to_notebook("notebook:nb", ["entity:attempts", "entity:age"])

        archived = await archive_stale_orphans(
            repo,
            notebook_id="notebook:nb",
            max_attempts=3,
            max_age_days=90,
        )

        assert archived == 2

    async def test_idempotency_no_double_archive(self):
        """Re-running on the same DB does not re-archive already-archived rows."""
        repo = MockRepo(
            {
                "entity:o1": _entity(
                    "entity:o1",
                    "Alice",
                    status=STATUS_PENDING_RECONNECT,
                    attempts=3,
                    first_orphaned_at=datetime.now(timezone.utc),
                ),
            }
        )
        repo.attach_to_notebook("notebook:nb", ["entity:o1"])

        first = await archive_stale_orphans(
            repo, notebook_id="notebook:nb", max_attempts=3, max_age_days=365
        )
        second = await archive_stale_orphans(
            repo, notebook_id="notebook:nb", max_attempts=3, max_age_days=365
        )
        assert first == 1
        assert second == 0
        assert repo.entities["entity:o1"]["orphan_status"] == STATUS_ARCHIVED

    async def test_defaults_match_constants(self):
        """Ensure the default-arg values track the documented constants."""
        # If someone changes one without the other the plan's acceptance
        # criteria (max_attempts=3, max_age_days=90) silently drift.
        assert DEFAULT_MAX_ATTEMPTS == 3
        assert DEFAULT_MAX_AGE_DAYS == 90

    async def test_empty_notebook_id_returns_zero(self):
        """Empty notebook_id short-circuits with a WARNING (Minor-3 fix).

        Attempt 2: ``notebook_id`` is now a required positional argument.
        Falsy values still short-circuit to 0 with a warning rather than
        raising, so a stale UI request can't crash the endpoint -- but
        callers can no longer omit the parameter entirely (caught at
        type-check time).
        """
        repo = MockRepo()
        archived = await archive_stale_orphans(repo, "")
        assert archived == 0

    async def test_clamps_invalid_thresholds(self):
        """Negative inputs are clamped, not propagated."""
        repo = MockRepo(
            {
                "entity:o1": _entity(
                    "entity:o1",
                    "Alice",
                    status=STATUS_PENDING_RECONNECT,
                    attempts=1,
                    first_orphaned_at=datetime.now(timezone.utc)
                    - timedelta(days=1),
                ),
            }
        )
        repo.attach_to_notebook("notebook:nb", ["entity:o1"])

        # max_attempts=0 clamps to 1 -> entity:o1 (attempts=1) IS archived.
        archived = await archive_stale_orphans(
            repo,
            notebook_id="notebook:nb",
            max_attempts=0,
            max_age_days=-5,
        )
        assert archived == 1
