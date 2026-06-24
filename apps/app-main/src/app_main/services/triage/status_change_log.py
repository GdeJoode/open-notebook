"""Thin repository for the ``status_change_log`` audit table (Track Q.3).

Every automation-driven status change writes ONE row here (old/new/reason/
batch_id/changed_at), giving an auditable, restart-safe, queryable history of
how each entity's triage status evolved across extraction batches. The table is
declared in ``migrations/59.surrealql`` (a SurrealDB table, operator decision R3
over an append-only file).

This is intentionally a thin wrapper over :func:`execute_query` rather than a
``BaseRepository`` subclass: like ``EntityRepository`` it operates directly on a
table that has no shared Pydantic model, and the rows are append-only provenance
records the rest of the app reads as plain dicts. Idempotency (one row per
ACTUAL change, none on an unchanged re-run) is the caller's responsibility — the
:class:`~app_main.services.triage.status_assignment_service.StatusAssignmentService`
only appends when the status genuinely changed.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query


class StatusChangeLogRepository:
    """Append-and-read access to the ``status_change_log`` table.

    Writes land via :meth:`append`; reads filter by entity (:meth:`list_for_entity`)
    or batch (:meth:`list_for_batch`), both backed by the indexes defined in
    migration 59.
    """

    def __init__(self, config: Optional[SurrealDBConfig] = None) -> None:
        """Initialise the repository.

        Args:
            config: Optional SurrealDB configuration; uses the global config
                when omitted (mirrors ``EntityRepository``).
        """
        self.config = config

    async def append(
        self,
        entity_id: str,
        old: str,
        new: str,
        reason: str,
        batch_id: str,
    ) -> str:
        """Append one status-change row and return its record id.

        Args:
            entity_id: Record ID of the entity whose status changed.
            old: The status before the change.
            new: The status after the change.
            reason: Human-readable derivation reason (the rule that fired).
            batch_id: Identifier of the triage batch that drove the change,
                so a whole run's changes can be queried together.

        Returns:
            The new ``status_change_log`` row id as a string.

        Raises:
            RuntimeError: If the insert returns no row, or the query fails.
        """
        rid = ensure_record_id(entity_id)
        # ``changed_at`` is left to the schema default (``time::now()``) so the
        # timestamp is the DB's, not the client's — consistent across processes.
        rows = await execute_query(
            """
            CREATE status_change_log SET
                entity = $entity,
                old_status = $old,
                new_status = $new,
                reason = $reason,
                batch_id = $batch_id;
            """,
            {
                "entity": rid,
                "old": old,
                "new": new,
                "reason": reason,
                "batch_id": batch_id,
            },
            self.config,
        )
        if not rows:
            raise RuntimeError(
                f"status_change_log append returned no row for {entity_id}"
            )
        new_id = str(rows[0]["id"])
        logger.debug(
            "status_change_log {}: {} '{}' -> '{}' ({}) [batch {}]",
            new_id,
            entity_id,
            old,
            new,
            reason,
            batch_id,
        )
        return new_id

    async def list_for_entity(self, entity_id: str) -> List[Dict[str, Any]]:
        """Return every status-change row for an entity, oldest first.

        Args:
            entity_id: Record ID of the entity.

        Returns:
            The matching rows as dicts (RecordIDs already stringified by
            ``execute_query``), ordered by ``changed_at`` ascending.
        """
        rid = ensure_record_id(entity_id)
        return await execute_query(
            "SELECT * FROM status_change_log WHERE entity = $entity "
            "ORDER BY changed_at ASC;",
            {"entity": rid},
            self.config,
        )

    async def list_for_batch(self, batch_id: str) -> List[Dict[str, Any]]:
        """Return every status-change row written by one batch, oldest first.

        Args:
            batch_id: The batch identifier passed to :meth:`append`.

        Returns:
            The matching rows as dicts, ordered by ``changed_at`` ascending.
        """
        return await execute_query(
            "SELECT * FROM status_change_log WHERE batch_id = $batch_id "
            "ORDER BY changed_at ASC;",
            {"batch_id": batch_id},
            self.config,
        )
