"""Triage review queue — append + dedup-by-entity, backed by ``triage_queue`` (Q.4).

The after-extraction pipeline surfaces edge cases (isolated core actors, well-
connected reference candidates, every unsure_review-tier entity, match conflicts)
into a non-blocking review queue an operator works through later. The load-bearing
property is **idempotency**: re-running the same extraction batch must NOT grow
the queue. So the queue is keyed on the entity — each entity has at most ONE OPEN
row, and :meth:`TriageQueueRepository.upsert` UPDATES that row (degree / doc_count
/ reason / batch refresh) instead of appending a duplicate.

A row leaves 'open' only via an operator decision (Q.4 router → :meth:`set_decision`);
once decided, the entity is operator-pinned (``manual_override``) so automation
never re-queues it. This is a thin :func:`execute_query` wrapper (no shared
Pydantic model) mirroring :class:`StatusChangeLogRepository`; the table is declared
in ``migrations/60.surrealql``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from loguru import logger
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query

# Decision states. 'open' is the default (awaiting an operator); the operator
# router writes one of the others. Kept here so the service and router agree.
DECISION_OPEN = "open"


@dataclass(frozen=True)
class QueueItem:
    """One review-queue row, projected for the read-model (B5).

    Carries exactly the fields acceptance criterion 4 asks for —
    ``id``/``name``/``type``/``structural_degree``/``doc_count``/``reason`` — plus
    the queue's own ``decision`` / ``batch_id`` for the router.
    """

    id: str
    entity: str
    name: str
    type: str
    structural_degree: int
    doc_count: int
    reason: str
    decision: str
    batch_id: str

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> "QueueItem":
        """Build a :class:`QueueItem` from a stringified DB row."""
        return cls(
            id=str(row.get("id")),
            entity=str(row.get("entity")),
            name=str(row.get("name") or ""),
            type=str(row.get("type") or ""),
            structural_degree=int(row.get("structural_degree") or 0),
            doc_count=int(row.get("doc_count") or 0),
            reason=str(row.get("reason") or ""),
            decision=str(row.get("decision") or DECISION_OPEN),
            batch_id=str(row.get("batch_id") or ""),
        )


class TriageQueueRepository:
    """Append-with-dedup and read access to the ``triage_queue`` table.

    The dedup contract: at most one OPEN row per entity. :meth:`upsert` looks up
    an existing open row for the entity and UPDATEs it; only when none exists does
    it CREATE. That is what keeps a re-run of the same batch from doubling the
    queue (Q.4 AC2) — and it also keeps the row's signals fresh as later batches
    change an entity's degree/doc-count while it still awaits a decision.
    """

    def __init__(self, config: Optional[SurrealDBConfig] = None) -> None:
        """Initialise the repository.

        Args:
            config: Optional SurrealDB configuration; uses the global config when
                omitted (mirrors ``EntityRepository`` / ``StatusChangeLogRepository``).
        """
        self.config = config

    async def upsert(
        self,
        entity_id: str,
        *,
        name: str,
        type: str,
        structural_degree: int,
        doc_count: int,
        reason: str,
        batch_id: str,
    ) -> str:
        """Append a queue row for ``entity_id`` OR refresh its existing open row.

        Idempotent by entity: a second call for an entity that already has an
        OPEN row updates that row (signals + reason + batch) and returns its id;
        it never creates a duplicate. A new entity (or one whose prior row was
        already decided) gets a fresh CREATE.

        Args:
            entity_id: Record ID of the entity to queue.
            name: Canonical name (read-model display).
            type: ``primary_type`` (read-model display).
            structural_degree: Effective structural degree (Q.2 signal).
            doc_count: Distinct-document count (Q.2 signal).
            reason: The queue-flag reason (Q.3 status decision).
            batch_id: The triage batch that raised/refreshed this row.

        Returns:
            The queue row's record id.
        """
        rid = ensure_record_id(entity_id)
        existing = await execute_query(
            "SELECT id FROM triage_queue "
            "WHERE entity = $entity AND decision = $open LIMIT 1;",
            {"entity": rid, "open": DECISION_OPEN},
            self.config,
        )
        params: Dict[str, Any] = {
            "entity": rid,
            "name": name,
            "type": type,
            "degree": int(structural_degree),
            "doc_count": int(doc_count),
            "reason": reason,
            "batch_id": batch_id,
        }
        if existing:
            queue_id = ensure_record_id(str(existing[0]["id"]))
            rows = await execute_query(
                "UPDATE $id SET "
                "name = $name, type = $type, structural_degree = $degree, "
                "doc_count = $doc_count, reason = $reason, batch_id = $batch_id, "
                "updated_at = time::now() RETURN AFTER;",
                {"id": queue_id, **params},
                self.config,
            )
        else:
            rows = await execute_query(
                "CREATE triage_queue SET "
                "entity = $entity, name = $name, type = $type, "
                "structural_degree = $degree, doc_count = $doc_count, "
                "reason = $reason, batch_id = $batch_id;",
                params,
                self.config,
            )
        if not rows:
            raise RuntimeError(
                f"triage_queue upsert returned no row for {entity_id}"
            )
        new_id = str(rows[0]["id"])
        logger.debug(
            "triage_queue {} entity={} reason='{}' [batch {}]",
            new_id,
            entity_id,
            reason,
            batch_id,
        )
        return new_id

    async def list_open(self) -> List[QueueItem]:
        """Return every OPEN queue row, newest first."""
        rows = await execute_query(
            "SELECT * FROM triage_queue WHERE decision = $open "
            "ORDER BY created_at DESC;",
            {"open": DECISION_OPEN},
            self.config,
        )
        return [QueueItem.from_row(r) for r in rows or []]

    async def list_all(self) -> List[QueueItem]:
        """Return every queue row (any decision), newest first."""
        rows = await execute_query(
            "SELECT * FROM triage_queue ORDER BY created_at DESC;",
            None,
            self.config,
        )
        return [QueueItem.from_row(r) for r in rows or []]

    async def get(self, queue_id: str) -> Optional[QueueItem]:
        """Return one queue row by id, or ``None`` if absent."""
        rows = await execute_query(
            "SELECT * FROM triage_queue WHERE id = $id LIMIT 1;",
            {"id": ensure_record_id(queue_id)},
            self.config,
        )
        if not rows:
            return None
        return QueueItem.from_row(rows[0])

    async def set_decision(self, queue_id: str, decision: str) -> Optional[QueueItem]:
        """Record an operator decision on a queue row (moves it off 'open').

        Args:
            queue_id: The queue row's record id.
            decision: The operator's decision string (anything other than
                'open' closes the row so automation no longer refreshes it).

        Returns:
            The updated :class:`QueueItem`, or ``None`` if the row is absent.
        """
        rows = await execute_query(
            "UPDATE $id SET decision = $decision, updated_at = time::now() "
            "RETURN AFTER;",
            {"id": ensure_record_id(queue_id), "decision": decision},
            self.config,
        )
        if not rows:
            return None
        return QueueItem.from_row(rows[0])
