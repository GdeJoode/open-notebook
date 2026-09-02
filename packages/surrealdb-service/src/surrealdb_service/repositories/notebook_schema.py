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

    async def ensure_row(self, notebook_id: str, base_ontology: str) -> bool:
        """Create the notebook's schema row when it does not exist yet.

        Track PC.1. Nothing in production ever created this row. The router's
        ``_ensure_schema_row`` builds one IN MEMORY and returns it without
        persisting — its docstring claims "we materialise the row eagerly so the
        toggle persists across restarts", which is not what the code does — and
        the only writers are the three toggle endpoints, so the row appears only
        if a user happens to flip a switch in the Schema tab.

        Measured on the live corpus: 17 `pass1_results` rows carrying 111
        proposals across 79 distinct type names, and **zero** `notebook_schema`
        rows. So the queue was not merely empty; the record that holds it did not
        exist, and every writer — including this module's own
        :meth:`merge_pending_extensions` — correctly returned "no row" and did
        nothing.

        ``base_ontology`` is a starting value, not a verdict: the curator changes
        it from the Schema tab. Callers with no explicit curator choice should
        pass ``shared.models.notebook_schema.DEFAULT_BASE_ONTOLOGY`` — the value
        every read path already falls back to — and specifically NOT an
        extraction request's ``ontology_name``, which is per-request state with a
        different default; a review measured that writing it here changes
        canonical typing across the graph and breaks the schema TTL download.

        **Name collision, deliberate.** The router's ``_ensure_schema_row`` has
        the opposite semantic: it builds the default row and does NOT persist it.
        This method is the persisting one. Renaming that helper is PC.5's job (it
        is called by the toggle endpoints); until then, the two live one grep
        apart and this paragraph is the disambiguation.

        **The check is not the protection.** ``idx_notebook_schema_notebook`` is
        UNIQUE (migration 45), and that index — not this read-then-write — is
        what actually prevents a duplicate: two concurrent first-extractions on
        the same notebook can both read ``None`` and both attempt a create, and
        the loser's constraint violation raises to the caller. That is safe (no
        duplicate row, no clobbering) but not free: the losing run sees no row
        and takes the legacy path for its document, so it produces no Pass-1
        proposals. The next document recovers, since by then the row exists.

        Returns ``True`` when a row was created, ``False`` when one already
        existed (so a caller can log the transition without re-reading). A
        ``False`` under a race means the same thing to the caller as a ``False``
        without one: read the row again if you need it.
        """
        existing = await self.get_by_notebook(notebook_id)
        if existing is not None:
            return False
        await self.upsert(
            NotebookSchema(notebook=notebook_id, base_ontology=base_ontology)
        )
        logger.info(
            "created notebook_schema for {nb} with base_ontology={base!r}",
            nb=notebook_id,
            base=base_ontology,
        )
        return True

    async def merge_pending_extensions(
        self, notebook_id: str, extensions: List[Dict[str, Any]]
    ) -> int:
        """Add proposals to ``pending_extensions``, skipping ones already known.

        Track PC.1. :meth:`add_pending_extension` appends unconditionally and
        rewrites the row per call, which is right for a single proposal and wrong
        for a Pass-1 batch: proposals are per-DOCUMENT while this list is
        per-NOTEBOOK, so the same type proposed by three documents would appear
        three times, and each append would be its own read-modify-write.

        A proposal is skipped when its ``type_name`` already appears in
        ``pending_extensions``, in ``accepted_extensions``, or in
        ``excluded_types``. The first two are duplicates; the third is the
        curator having explicitly soft-deleted that type (B.3b), which is a "no"
        that must survive the next document proposing it again. Matching is
        case-insensitive on the trimmed name, because the LLM's capitalisation is
        not stable across documents.

        **A REJECTED proposal is NOT remembered.** ``reject_extension`` drops the
        row and records nothing — there is no ``rejected_extensions`` field — so a
        type the curator rejected returns the next time a document proposes it.
        A durable "no" needs a new field and a migration, so it is recorded as a
        follow-up rather than half-solved here. Two tests pin it, one per layer,
        because the repository's own `reject_pending_extension` has no production
        caller: `test_notebook_schema_queue.py` covers this module, and
        `test_schema_edit_service.py` covers the path a curator's Reject button
        actually takes.

        The stored ``type_name`` is stripped, and a name that cannot survive the
        accept/reject route is refused outright: those endpoints take the name as
        a PATH SEGMENT (``/schema/extensions/{type_name}/accept``) with no
        ``:path`` converter, so a model-authored ``"Grant/Funding Source"`` would
        queue a row that can be neither accepted nor rejected — permanently stuck
        on the one surface this exists to make actionable.

        Each stored proposal gets a deterministic ``extension_id``
        (``pass1::<lowercased type_name>``) so a replay is idempotent. The
        production endpoints key on ``type_name``, not on this id; the id is the
        frontend's list key and the key the older ``accept_pending_extension`` /
        ``reject_pending_extension`` pair matches on.

        Returns the number of proposals actually added. ``0`` means everything
        was already known — a normal outcome, not a failure.

        **Known race, inherited and now reachable.** This is a read-modify-write
        over the whole array, and so is ``add_pending_extension`` before it — but
        PC.1 is the first PRODUCTION caller, which is what makes the race
        reachable. Two extractions ingesting different sources into one notebook
        concurrently can lose one job's proposals. Not fixed here: the fix is a
        server-side array append or an optimistic version check, which is a
        change to the repository's write contract rather than to this method.
        Recorded so it is a known limit rather than a surprise the first time a
        bulk upload runs.
        """
        existing = await self.get_by_notebook(notebook_id)
        if existing is None:
            logger.warning(
                f"merge_pending_extensions: no notebook_schema row for {notebook_id}"
            )
            return 0

        def _key(value: Any) -> str:
            return str(value or "").strip().lower()

        known = {_key(e.get("type_name")) for e in existing.pending_extensions}
        known |= {_key(e.get("type_name")) for e in existing.accepted_extensions}
        known |= {_key(t) for t in existing.excluded_types}

        added = 0
        for proposal in extensions or []:
            if not isinstance(proposal, dict):
                continue
            name = str(proposal.get("type_name", "") or "").strip()
            key = _key(name)
            if not key or key in known:
                continue
            if not _is_routable_type_name(name):
                logger.warning(
                    "merge_pending_extensions: refusing {name!r} - it cannot be "
                    "used in the accept/reject route and would be unactionable",
                    name=name,
                )
                continue
            known.add(key)
            existing.pending_extensions.append(
                {**proposal, "type_name": name, "extension_id": f"pass1::{key}"}
            )
            added += 1

        if added:
            await self.upsert(existing)
        return added

    async def set_coverage_pct(self, notebook_id: str, value: float) -> bool:
        """Store the notebook's rolling Pass-1 coverage.

        Track PC.1. The field is documented as driving the B.3c soft-nudge and is
        rendered by the Schema tab, but nothing in the extraction path wrote it —
        after eight documents and fourteen Pass-1 measurements it was still 0.0.

        Clamped to 0.0–1.0: the model constrains the field, and a Pass-1 run that
        reports a percentage instead of a fraction should not make the write fail
        after the extraction already succeeded.
        """
        existing = await self.get_by_notebook(notebook_id)
        if existing is None:
            logger.warning(
                f"set_coverage_pct: no notebook_schema row for {notebook_id}"
            )
            return False
        existing.coverage_pct = max(0.0, min(1.0, float(value)))
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


def _is_routable_type_name(name: str) -> bool:
    """Whether a proposed type name can survive the accept/reject endpoints.

    Those routes take the name as a bare path segment, so a ``/`` splits the path
    and the request 404s (measured). A backslash and control characters are
    refused for the same class of reason. Everything else - spaces, accents,
    ordinary punctuation - round-trips fine through ``encodeURIComponent``.
    """
    if not name:
        return False
    if "/" in name or "\\" in name:
        return False
    return not any(ord(ch) < 32 or ord(ch) == 127 for ch in name)


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
