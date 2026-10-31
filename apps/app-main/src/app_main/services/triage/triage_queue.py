"""Triage review queue — append + dedup-by-entity, backed by ``triage_queue`` (Q.4).

The after-extraction pipeline surfaces edge cases (isolated core actors, well-
connected reference candidates, every unsure_review-tier entity, match conflicts)
into a non-blocking review queue an operator works through later. The load-bearing
property is **idempotency**: re-running the same extraction batch must NOT grow
the queue. So the queue is keyed on the entity AND the row's ``kind`` (migration
83) — each (entity, kind) has at most ONE OPEN row, and
:meth:`TriageQueueRepository.upsert` UPDATES the entity's status row (degree /
doc_count / reason / batch refresh) instead of appending a duplicate.

A status row leaves 'open' only via an operator decision (Q.4 router →
:meth:`set_decision`); once decided, the entity is operator-pinned
(``manual_override``) so automation never re-queues it. A ``tooi`` row (D2) is a
different question — which authority record the entity is — answered link or
dismiss, never pinning status; it is re-queued only when the candidate set
changes (:meth:`TriageQueueRepository.upsert_tooi`). This is a thin
:func:`execute_query` wrapper (no shared Pydantic model) mirroring
:class:`StatusChangeLogRepository`; the table is declared in
``migrations/60.surrealql`` and ``83.surrealql``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories.base import (
    RecordNotFoundError,
    RevWijziging,
    wijzig_met_rev,
)

# Decision states. 'open' is the default (awaiting an operator); the operator
# router writes one of the others. Kept here so the service and router agree.
DECISION_OPEN = "open"

# Row kinds (migration 83). One queue, one row per (entity, kind): a STATUS
# decision (Q.4) and a TOOI link decision (D2) about the same entity never share
# a row, so neither overwrites the other's reason or closes the other's row.
KIND_STATUS = "status"
KIND_TOOI = "tooi"
DECISION_LINKED = "linked"
DECISION_DISMISSED = "dismissed"

# Reason separator. Multiple surfacing paths (a status flag + a match conflict)
# can flag the same entity; their reasons are joined with this so one queue row
# shows both.
_REASON_SEP = "; "


def _merge_reason(existing: str, incoming: str) -> str:
    """Append ``incoming`` to ``existing`` unless it is already present.

    Idempotent: re-running a batch re-appends nothing (an already-present clause
    is detected on the ``"; "``-split parts), so the merged reason stays stable.
    An empty existing reason just becomes the incoming one.
    """
    incoming = (incoming or "").strip()
    if not incoming:
        return existing
    if not existing:
        return incoming
    parts = [p.strip() for p in existing.split(_REASON_SEP) if p.strip()]
    if incoming in parts:
        return existing
    parts.append(incoming)
    return _REASON_SEP.join(parts)


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
    kind: str = KIND_STATUS
    #: For a TOOI row: the authority records a curator chooses between.
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    candidate_key: str = ""
    #: F2-06 (migratie 92): de versie van de rij voor optimistische vergrendeling.
    rev: int = 0

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
            kind=str(row.get("kind") or KIND_STATUS),
            candidates=list(row.get("candidates") or []),
            candidate_key=str(row.get("candidate_key") or ""),
            rev=int(row.get("rev") or 0),
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
        merge_reason: bool = False,
    ) -> str:
        """Append a queue row for ``entity_id`` OR refresh its existing open row.

        Idempotent by entity: a second call for an entity that already has an
        OPEN row updates that row (signals + reason + batch) and returns its id;
        it never creates a duplicate. A new entity (or one whose prior row was
        already decided) gets a fresh CREATE.

        **Signal preservation (Q.4 review minor):** a caller that does not carry
        real signals — the match-conflict surfacing passes
        ``structural_degree=0, doc_count=0`` because it only knows an entity is
        in a fuzzy-match pair, not its degree — must NOT clobber the real
        degree/doc-count a prior status-flag row already recorded for the same
        entity. So on UPDATE, ``structural_degree`` / ``doc_count`` are only
        overwritten when the incoming value is non-zero (or no row exists yet);
        a zero incoming value preserves whatever the row already holds. This
        keeps the queue row's displayed signals honest regardless of which
        surfacing path touched the entity last, and stays idempotent (the same
        zero-signal call is a no-op on the signals).

        Args:
            entity_id: Record ID of the entity to queue.
            name: Canonical name (read-model display).
            type: ``primary_type`` (read-model display).
            structural_degree: Effective structural degree (Q.2 signal). A zero
                here is treated as "unknown" on UPDATE and never overwrites a
                prior non-zero value.
            doc_count: Distinct-document count (Q.2 signal). Same zero-as-unknown
                rule as ``structural_degree``.
            reason: The queue-flag reason (Q.3 status decision).
            batch_id: The triage batch that raised/refreshed this row.
            merge_reason: When True and an OPEN row already exists with a
                different reason, the incoming reason is APPENDED (``; ``-joined)
                rather than replacing the existing one — so a match-conflict
                reason and a status-flag reason coexist on one row. Idempotent:
                an already-present reason is not re-appended. Defaults to False
                (replace, the status-flag surfacing's behaviour).

        Returns:
            The queue row's record id.
        """
        rid = ensure_record_id(entity_id)
        existing = await execute_query(
            # Status rows only (migration 83): a TOOI row for the same entity is
            # a different decision and must not be refreshed — or have its
            # reason replaced — by a status flag.
            "SELECT id, structural_degree, doc_count, reason FROM triage_queue "
            "WHERE entity = $entity AND decision = $open AND kind = $kind LIMIT 1;",
            {"entity": rid, "open": DECISION_OPEN, "kind": KIND_STATUS},
            self.config,
        )
        if existing:
            prior = existing[0]
            # Zero-as-unknown: never clobber a recorded non-zero signal with a
            # signal-less (degree=0/doc_count=0) call (the match-conflict path).
            degree = (
                int(structural_degree)
                if int(structural_degree) != 0
                else int(prior.get("structural_degree") or 0)
            )
            doc_cnt = (
                int(doc_count)
                if int(doc_count) != 0
                else int(prior.get("doc_count") or 0)
            )
            merged_reason = _merge_reason(
                str(prior.get("reason") or ""), reason
            ) if merge_reason else reason
            queue_id = ensure_record_id(str(prior["id"]))
            rows = await execute_query(
                "UPDATE $id SET "
                "name = $name, type = $type, structural_degree = $degree, "
                "doc_count = $doc_count, reason = $reason, batch_id = $batch_id, "
                "updated_at = time::now() RETURN AFTER;",
                {
                    "id": queue_id,
                    "name": name,
                    "type": type,
                    "degree": degree,
                    "doc_count": doc_cnt,
                    "reason": merged_reason,
                    "batch_id": batch_id,
                },
                self.config,
            )
        else:
            params: Dict[str, Any] = {
                "entity": rid,
                "name": name,
                "type": type,
                "degree": int(structural_degree),
                "doc_count": int(doc_count),
                "reason": reason,
                "batch_id": batch_id,
            }
            rows = await execute_query(
                "CREATE triage_queue SET "
                "entity = $entity, name = $name, type = $type, "
                "structural_degree = $degree, doc_count = $doc_count, "
                "reason = $reason, batch_id = $batch_id, kind = 'status';",
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

    async def upsert_tooi(
        self,
        entity_id: str,
        *,
        name: str,
        type: str,
        reason: str,
        candidates: List[Dict[str, Any]],
        batch_id: str,
    ) -> Optional[str]:
        """Queue a TOOI link decision for ``entity_id`` (D2, migration 83).

        One open TOOI row per entity, refreshed in place. **A decision sticks:**
        when a TOOI row for this entity was already decided (linked or dismissed)
        for the SAME candidate set, nothing is queued and ``None`` is returned. A
        different set — the authority added or ended an organisation — is a new
        question and is queued. Status rows are never read or written here.
        """
        rid = ensure_record_id(entity_id)
        key = candidate_key(candidates)
        decided = await execute_query(
            "SELECT id FROM triage_queue WHERE entity = $entity AND kind = $kind "
            "AND decision != $open AND candidate_key = $key LIMIT 1;",
            {"entity": rid, "kind": KIND_TOOI, "open": DECISION_OPEN, "key": key},
            self.config,
        )
        if decided:
            return None
        params: Dict[str, Any] = {
            "entity": rid, "name": name, "type": type, "reason": reason,
            "candidates": candidates, "key": key, "batch_id": batch_id,
        }
        existing = await execute_query(
            "SELECT id FROM triage_queue WHERE entity = $entity AND kind = $kind "
            "AND decision = $open LIMIT 1;",
            {"entity": rid, "kind": KIND_TOOI, "open": DECISION_OPEN},
            self.config,
        )
        if existing:
            rows = await execute_query(
                "UPDATE $id SET name = $name, type = $type, reason = $reason, "
                "candidates = $candidates, candidate_key = $key, batch_id = $batch_id, "
                "updated_at = time::now() RETURN AFTER;",
                {**params, "id": ensure_record_id(str(existing[0]["id"]))},
                self.config,
            )
        else:
            rows = await execute_query(
                "CREATE triage_queue SET entity = $entity, name = $name, type = $type, "
                "reason = $reason, candidates = $candidates, candidate_key = $key, "
                "batch_id = $batch_id, kind = 'tooi';",
                params,
                self.config,
            )
        if not rows:
            raise RuntimeError(f"triage_queue upsert_tooi returned no row for {entity_id}")
        return str(rows[0]["id"])

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

    async def set_decision(
        self,
        queue_id: str,
        decision: str,
        *,
        rev: Optional[int],
        door: str,
        entiteit: Optional[Tuple[str, Dict[str, Any]]] = None,
    ) -> Optional[QueueItem]:
        """Record an operator decision on a queue row (moves it off 'open').

        F2-06 (ADR 0003): written with :func:`wijzig_met_rev`. A stale ``rev``
        raises :class:`RevConflictError` (HTTP 409) and nothing changes.

        Args:
            queue_id: The queue row's record id.
            decision: The operator's decision string (anything other than
                'open' closes the row so automation no longer refreshes it).
            rev: The row version the operator read; ``None`` only for a row
                closed as a consequence of another checked write.
            door: The person of the key.
            entiteit: ``(entity_id, data)`` written to the row's entity in the
                same transaction (the pin, or the TOOI link), unchecked
                (``rev=None``) but stamped with ``modified_by``.

        Returns:
            The updated :class:`QueueItem`, or ``None`` if the row (or the
            entity) is absent; then nothing is written.
        """
        nu = datetime.now(timezone.utc)
        wijzigingen = [RevWijziging(queue_id, rev, {"decision": decision, "updated_at": nu})]
        if entiteit is not None:
            entity_id, data = entiteit
            wijzigingen.append(RevWijziging(entity_id, None, {**data, "updated_at": nu}))
        try:
            rows = await wijzig_met_rev(wijzigingen, door=door, config=self.config)
        except RecordNotFoundError:
            return None
        return QueueItem.from_row(rows[0]) if rows[0] else None


def candidate_key(candidates: List[Dict[str, Any]]) -> str:
    """The identity of a TOOI question: its sorted candidate URIs."""
    return "|".join(sorted(str(c.get("uri") or "") for c in candidates))
