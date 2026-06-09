"""Repository for the ``notebook_event`` append-only stream (Phase B.3b).

Backs the per-notebook event log declared in ``migrations/46.surrealql``.
The Pydantic mirror lives in ``shared.models.notebook_schema.NotebookEvent``.

Design notes
============

* **Append-only writes, read-with-filter reads.** The hot read path is
  "unread events of type X for notebook Y, newest first" — exactly what
  the composite index on ``(notebook, event_type, created_at)`` covers.
  ``read_at IS NONE`` filters on the sparse field post-index.

* **record(...) returns the new id**, mirroring the
  ``Pass1ResultRepository.record`` contract. The id is needed by the
  caller (B.3b schema edit ops) so we have a clean correlation between
  the schema mutation and the event row in tests.

* **mark_read is idempotent** — re-marking an already-read event is a
  no-op (the UPDATE just rewrites the same value). The caller does not
  need to check first.

* **list_unread filters by ``event_type``** when provided. Most consumers
  care about a single event-type (B.3c reads ``extension_suggested``,
  B.3d reads ``schema_changed``). The ``None`` overload returns every
  unread event for the notebook — useful for the debug panel.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models import NotebookEvent
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories.base import BaseRepository


class NotebookEventRepository(BaseRepository[NotebookEvent]):
    """Append-only repository for the per-notebook event stream.

    Writes land via :meth:`record`; reads filter by notebook + optional
    event_type via :meth:`list_unread`; consumption is marked via
    :meth:`mark_read`.
    """

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(NotebookEvent, config)

    async def record(
        self,
        notebook_id: str,
        event_type: str,
        payload: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Append a new event row to the stream.

        Args:
            notebook_id: Record ID of the owning notebook
                (e.g. ``"notebook:abc123"``).
            event_type: Discriminator string (e.g. ``"schema_changed"``).
            payload: Optional context bag. ``None`` writes ``{}``.

        Returns:
            The record id of the newly created row.

        Raises:
            RuntimeError: On unexpected SurrealDB errors.
        """
        notebook_rid = ensure_record_id(notebook_id)
        data: Dict[str, Any] = {
            "notebook": notebook_rid,
            "event_type": event_type,
            "payload": payload or {},
        }

        try:
            created = await execute_query(
                "CREATE notebook_event CONTENT $data RETURN AFTER;",
                {"data": data},
                self.config,
            )
            if not created:
                raise RuntimeError("CREATE notebook_event returned no rows")
            return str(created[0]["id"])
        except Exception as e:
            logger.exception(
                f"Failed to record notebook_event for {notebook_id}: {e}"
            )
            raise

    async def list_unread(
        self,
        notebook_id: str,
        event_type: Optional[str] = None,
    ) -> List[NotebookEvent]:
        """List events for a notebook that have not yet been marked read.

        Args:
            notebook_id: Record ID of the notebook.
            event_type: When set, restricts the result to events of this
                type. ``None`` returns every unread event.

        Returns:
            List of :class:`NotebookEvent` rows ordered by ``created_at``
            descending; empty list on failure.
        """
        try:
            if event_type is None:
                rows = await execute_query(
                    "SELECT * FROM notebook_event "
                    "WHERE notebook = $notebook AND read_at IS NONE "
                    "ORDER BY created_at DESC;",
                    {"notebook": ensure_record_id(notebook_id)},
                    self.config,
                )
            else:
                rows = await execute_query(
                    "SELECT * FROM notebook_event "
                    "WHERE notebook = $notebook AND event_type = $event_type "
                    "AND read_at IS NONE ORDER BY created_at DESC;",
                    {
                        "notebook": ensure_record_id(notebook_id),
                        "event_type": event_type,
                    },
                    self.config,
                )
            return [NotebookEvent(**row) for row in rows]
        except Exception as e:
            logger.error(
                f"Failed to list_unread notebook_event for {notebook_id}: {e}"
            )
            return []

    async def list_by_notebook(
        self,
        notebook_id: str,
        event_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[NotebookEvent]:
        """List all events (read or unread) for a notebook.

        Used primarily by tests to assert exact event counts after an
        operation. Production code should prefer :meth:`list_unread`.

        Args:
            notebook_id: Record ID of the notebook.
            event_type: When set, restricts to this discriminator.
            limit: Maximum number of rows.

        Returns:
            List of :class:`NotebookEvent` rows ordered by ``created_at``
            descending; empty list on failure.
        """
        try:
            if event_type is None:
                rows = await execute_query(
                    "SELECT * FROM notebook_event "
                    "WHERE notebook = $notebook "
                    "ORDER BY created_at DESC LIMIT $limit;",
                    {
                        "notebook": ensure_record_id(notebook_id),
                        "limit": limit,
                    },
                    self.config,
                )
            else:
                rows = await execute_query(
                    "SELECT * FROM notebook_event "
                    "WHERE notebook = $notebook AND event_type = $event_type "
                    "ORDER BY created_at DESC LIMIT $limit;",
                    {
                        "notebook": ensure_record_id(notebook_id),
                        "event_type": event_type,
                        "limit": limit,
                    },
                    self.config,
                )
            return [NotebookEvent(**row) for row in rows]
        except Exception as e:
            logger.error(
                f"Failed to list notebook_event for {notebook_id}: {e}"
            )
            return []

    async def mark_read(self, event_id: str) -> bool:
        """Mark an event as consumed by setting ``read_at`` to now.

        Idempotent: re-marking an already-read event rewrites the same
        timestamp without raising.

        Args:
            event_id: Record id of the notebook_event row.

        Returns:
            ``True`` on success, ``False`` if the row does not exist.
        """
        try:
            result = await execute_query(
                "UPDATE type::thing($id) SET read_at = $now RETURN AFTER;",
                {
                    "id": event_id,
                    "now": datetime.now(timezone.utc),
                },
                self.config,
            )
            return bool(result)
        except Exception as e:
            logger.exception(
                f"Failed to mark_read notebook_event {event_id}: {e}"
            )
            return False
