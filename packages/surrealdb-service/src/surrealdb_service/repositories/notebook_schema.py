"""
Repositories for the per-notebook ontology state and per-source pass-1 history.

Backs Phase B.1b. The schemas live in ``migrations/45.surrealql`` and the
Pydantic mirrors live in ``shared.models.notebook_schema``.

Two repositories live here:

* :class:`NotebookSchemaRepository` — singleton-per-notebook semantics
  enforced by the UNIQUE index on ``notebook_schema.notebook``. The
  ``upsert`` method explicitly *rewrites* the existing row rather than
  attempting a blind CREATE, because the UNIQUE constraint would
  otherwise raise.

* :class:`Pass1ResultRepository` — append-only writes; reads are
  source-scoped or notebook-scoped.

Both repositories use the canonical :func:`execute_query` entry point
and respect the optional :class:`SurrealDBConfig` for test isolation.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models import NotebookSchema, Pass1Result
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories.base import BaseRepository


class NotebookSchemaRepository(BaseRepository[NotebookSchema]):
    """
    Repository for :class:`NotebookSchema` rows.

    The UNIQUE index on ``notebook`` means at most one row exists per
    notebook. Callers should use :meth:`upsert` rather than ``create``
    so the rewrite-on-conflict semantic is consistent.
    """

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(NotebookSchema, config)

    async def get_by_notebook(
        self, notebook_id: str
    ) -> Optional[NotebookSchema]:
        """Fetch the schema row for a notebook, or ``None`` if absent.

        Args:
            notebook_id: Record ID of the notebook (e.g. ``"notebook:abc"``).

        Returns:
            The :class:`NotebookSchema` row or ``None`` when no row exists
            yet for this notebook.
        """
        try:
            result = await execute_query(
                "SELECT * FROM notebook_schema "
                "WHERE notebook = $notebook LIMIT 1;",
                {"notebook": ensure_record_id(notebook_id)},
                self.config,
            )
            if result:
                return NotebookSchema(**result[0])
            return None
        except Exception as e:
            logger.error(
                f"Failed to fetch notebook_schema for {notebook_id}: {e}"
            )
            return None

    async def upsert(self, schema: NotebookSchema) -> str:
        """Create or rewrite the notebook_schema row for a notebook.

        The UNIQUE index on ``notebook`` means a blind CREATE would
        raise on the second call. This method queries by notebook
        first and chooses CREATE vs UPDATE accordingly — the rewrite
        path returns the existing row's id, while the create path
        returns a freshly generated id.

        Args:
            schema: The :class:`NotebookSchema` to persist. ``notebook``
                MUST be set; ``id`` is ignored (the existing row id is
                preferred when present).

        Returns:
            The record id of the persisted row (e.g.
            ``"notebook_schema:abc"``).

        Raises:
            RuntimeError: On unexpected SurrealDB errors.
        """
        notebook_rid = ensure_record_id(schema.notebook)

        # Build a dict of field values, dropping id/created/updated
        # because we want SurrealDB to manage those.
        data: Dict[str, Any] = schema.model_dump(
            exclude={"id", "created", "updated", "notebook"},
            exclude_none=False,
        )
        data["notebook"] = notebook_rid
        data["last_modified_at"] = datetime.now(timezone.utc)

        try:
            existing = await execute_query(
                "SELECT id FROM notebook_schema "
                "WHERE notebook = $notebook LIMIT 1;",
                {"notebook": notebook_rid},
                self.config,
            )

            if existing:
                existing_id = existing[0]["id"]
                # ``execute_query`` returns string-form record ids (parsed
                # by ``parse_record_ids``); UPDATE statements in
                # SurrealQL want a record-id expression in the target
                # position. Convert back via ``type::thing()`` rather
                # than f-string interpolation — both are safe here, but
                # the parametrised form avoids any risk if the id ever
                # contains characters that need quoting.
                result = await execute_query(
                    "UPDATE type::thing($id) MERGE $data RETURN AFTER;",
                    {"id": existing_id, "data": data},
                    self.config,
                )
                if not result:
                    raise RuntimeError(
                        f"UPDATE notebook_schema {existing_id} returned no rows"
                    )
                return str(result[0]["id"])

            result = await execute_query(
                "CREATE notebook_schema CONTENT $data RETURN AFTER;",
                {"data": data},
                self.config,
            )
            if not result:
                raise RuntimeError(
                    "CREATE notebook_schema returned no rows"
                )
            return str(result[0]["id"])
        except Exception as e:
            logger.exception(
                f"Failed to upsert notebook_schema for {schema.notebook}: {e}"
            )
            raise

    async def add_pending_extension(
        self, notebook_id: str, extension: Dict[str, Any]
    ) -> bool:
        """Append a proposed extension to ``pending_extensions``.

        Reads the current row, appends the extension dict to
        ``pending_extensions``, and writes it back. If the notebook has
        no schema row yet (caller skipped the upsert), this returns
        ``False`` and logs a warning — pending extensions are only
        meaningful once the base ontology has been established.

        Args:
            notebook_id: Record ID of the notebook.
            extension: Extension dict to append (shape is FLEXIBLE; B.1c
                writes ``{extension_id, type_name, parent_type, ...}``).

        Returns:
            ``True`` on success, ``False`` when no schema row exists.
        """
        existing = await self.get_by_notebook(notebook_id)
        if existing is None:
            logger.warning(
                f"add_pending_extension: no notebook_schema row for {notebook_id}"
            )
            return False

        existing.pending_extensions.append(extension)
        await self.upsert(existing)
        return True

    async def accept_pending_extension(
        self, notebook_id: str, extension_id: str
    ) -> bool:
        """Move a pending extension into ``accepted_extensions``.

        Looks up the schema, removes the extension whose ``extension_id``
        matches from ``pending_extensions``, and appends it to
        ``accepted_extensions``. Returns ``False`` if no matching pending
        extension exists.

        Args:
            notebook_id: Record ID of the notebook.
            extension_id: The ``extension_id`` key inside one of the
                pending dicts. B.1c assigns these.

        Returns:
            ``True`` if the extension was moved, ``False`` otherwise.
        """
        existing = await self.get_by_notebook(notebook_id)
        if existing is None:
            return False

        matched: Optional[Dict[str, Any]] = None
        remaining: List[Dict[str, Any]] = []
        for ext in existing.pending_extensions:
            if matched is None and ext.get("extension_id") == extension_id:
                matched = ext
            else:
                remaining.append(ext)

        if matched is None:
            return False

        existing.pending_extensions = remaining
        existing.accepted_extensions.append(matched)
        await self.upsert(existing)
        return True

    async def reject_pending_extension(
        self, notebook_id: str, extension_id: str
    ) -> bool:
        """Drop a pending extension without accepting it.

        Args:
            notebook_id: Record ID of the notebook.
            extension_id: The ``extension_id`` key inside one of the
                pending dicts.

        Returns:
            ``True`` if the extension was removed, ``False`` otherwise.
        """
        existing = await self.get_by_notebook(notebook_id)
        if existing is None:
            return False

        remaining = [
            ext
            for ext in existing.pending_extensions
            if ext.get("extension_id") != extension_id
        ]
        if len(remaining) == len(existing.pending_extensions):
            return False

        existing.pending_extensions = remaining
        await self.upsert(existing)
        return True


class Pass1ResultRepository(BaseRepository[Pass1Result]):
    """
    Repository for append-only :class:`Pass1Result` records.
    """

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(Pass1Result, config)

    async def record(self, result: Pass1Result) -> str:
        """Persist a new pass-1 result row.

        Args:
            result: The :class:`Pass1Result` to persist. ``source`` and
                ``notebook`` must be set.

        Returns:
            The record id of the newly created row.

        Raises:
            RuntimeError: On unexpected SurrealDB errors.
        """
        source_rid = ensure_record_id(result.source)
        notebook_rid = ensure_record_id(result.notebook)

        data: Dict[str, Any] = result.model_dump(
            exclude={"id", "created", "updated", "source", "notebook"},
            exclude_none=False,
        )
        data["source"] = source_rid
        data["notebook"] = notebook_rid

        try:
            created = await execute_query(
                "CREATE pass1_results CONTENT $data RETURN AFTER;",
                {"data": data},
                self.config,
            )
            if not created:
                raise RuntimeError("CREATE pass1_results returned no rows")
            return str(created[0]["id"])
        except Exception as e:
            logger.exception(
                f"Failed to record pass1_result for source {result.source}: {e}"
            )
            raise

    async def list_by_source(self, source_id: str) -> List[Pass1Result]:
        """List every pass-1 result for a source, newest first.

        Args:
            source_id: Record ID of the source.

        Returns:
            List of :class:`Pass1Result` rows ordered by ``created_at``
            descending; empty list on failure.
        """
        try:
            rows = await execute_query(
                "SELECT * FROM pass1_results "
                "WHERE source = $source ORDER BY created_at DESC;",
                {"source": ensure_record_id(source_id)},
                self.config,
            )
            return [Pass1Result(**row) for row in rows]
        except Exception as e:
            logger.error(
                f"Failed to list pass1_results for source {source_id}: {e}"
            )
            return []

    async def list_by_notebook(
        self, notebook_id: str, limit: int = 100
    ) -> List[Pass1Result]:
        """List recent pass-1 results across an entire notebook.

        Args:
            notebook_id: Record ID of the notebook.
            limit: Maximum number of rows to return (default 100).

        Returns:
            List of :class:`Pass1Result` rows ordered by ``created_at``
            descending; empty list on failure.
        """
        try:
            rows = await execute_query(
                "SELECT * FROM pass1_results "
                "WHERE notebook = $notebook "
                "ORDER BY created_at DESC LIMIT $limit;",
                {
                    "notebook": ensure_record_id(notebook_id),
                    "limit": limit,
                },
                self.config,
            )
            return [Pass1Result(**row) for row in rows]
        except Exception as e:
            logger.error(
                f"Failed to list pass1_results for notebook {notebook_id}: {e}"
            )
            return []
