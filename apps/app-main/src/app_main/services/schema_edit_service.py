"""Schema edit operations service (Phase B.3b).

Pure business logic for the six edit ops the Schema-tab UI exposes:

* ``accept_extension`` — move a pending extension into ``accepted_extensions``.
* ``reject_extension`` — drop a pending extension.
* ``rename_type`` — record a rename in ``accepted_extensions`` as a synonym.
  The base ontology YAML is never mutated.
* ``merge_types`` — record a merge of N types into one; future extractions
  consult ``accepted_extensions`` and remap.
* ``split_type`` — record a split of one type into N, with an LLM hint.
* ``delete_type`` — soft-delete by adding the name to ``excluded_types``.
* ``reparent_type`` — record that N types now hang under a different parent
  (Track N.4d.3). This is how an accepted ``BROADER_THAN`` placement is applied:
  as a schema edit, on the curator's decision, never silently.

Design notes
============

* **Each op emits exactly one ``notebook_event{type:"schema_changed"}`` row**
  via :class:`NotebookEventRepository`. The payload carries an ``op``
  discriminator plus the op-specific fields so consumers (B.3c soft-nudge,
  B.3d re-extract prompt, Track G5 webhooks) can decide whether to react.

* **Idempotent.** Re-running the same op on already-converged state is a
  no-op AND emits ZERO new events. This is critical for the "user
  double-clicks Accept" case + for replays during testing. Idempotency
  is enforced by reading current state and short-circuiting before
  writing.

* **The base ontology YAML is read-only.** Rename / merge / split /
  delete record their intent in the per-notebook schema row. Downstream
  consumers (effective-schema projection, TTL export, Pass 2 extraction)
  consult the same row when computing the user-facing view.

* **Service depends on repositories, not on HTTP types.** Inputs are
  primitive strings + lists; the FastAPI router translates request /
  response bodies. This keeps the service unit-testable without
  spinning up a FastAPI client.

* **All ops return the updated ``NotebookSchema``.** The router serialises
  it; tests assert on it directly. Keeps the contract symmetric: one
  call, one resulting view.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models import NotebookSchema
from surrealdb_service.repositories.notebook_event import NotebookEventRepository
from surrealdb_service.repositories.notebook_schema import (
    NotebookSchemaRepository,
)


SCHEMA_CHANGED_EVENT_TYPE = "schema_changed"


class NotebookSchemaNotFoundError(Exception):
    """Raised when an edit op targets a notebook that has no schema row.

    The notebook itself exists (the router layer guards 404 separately)
    but no pass-1 extraction has run, so there's nothing to edit. The
    router translates this to 404 — the schema row is implicit
    state-not-found, distinct from "notebook does not exist".
    """


class UnknownExtensionError(Exception):
    """Raised when accept_extension / reject_extension targets a type_name
    not present in ``pending_extensions``.

    Router translates to 404. Distinguished from
    :class:`NotebookSchemaNotFoundError` so the API surface can give a
    precise reason — important when a user clicks Accept twice and the
    second click 404s because the first already moved the row.
    """


class SchemaEditService:
    """Service orchestrating the six per-notebook schema edit ops.

    Each public method:

    1. Loads the current :class:`NotebookSchema` row.
    2. Computes the new state.
    3. Returns early (no DB write, no event) when the state has not
       changed — idempotent.
    4. Persists via :meth:`NotebookSchemaRepository.upsert`.
    5. Emits one ``schema_changed`` event via
       :meth:`NotebookEventRepository.record`.
    6. Returns the updated schema row.

    The split between schema_repo and event_repo is deliberate: schema
    state is the source-of-truth, events are the change-stream. A future
    Track G5 webhook fan-out reads events; B.3c's soft-nudge banner
    reads events; the Schema tab itself reads schema_repo directly via
    the existing GET endpoints.
    """

    def __init__(
        self,
        schema_repo: NotebookSchemaRepository,
        event_repo: NotebookEventRepository,
    ):
        self._schema_repo = schema_repo
        self._event_repo = event_repo

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    async def _load_or_raise(self, notebook_id: str) -> NotebookSchema:
        schema = await self._schema_repo.get_by_notebook(notebook_id)
        if schema is None:
            raise NotebookSchemaNotFoundError(
                f"No notebook_schema row for {notebook_id}. "
                "Pass-1 extraction has not run yet."
            )
        return schema

    async def _persist(
        self,
        schema: NotebookSchema,
        notebook_id: str,
        event_payload: Dict[str, Any],
    ) -> NotebookSchema:
        """Upsert schema + emit one schema_changed event.

        The event is emitted AFTER the schema upsert succeeds — if the
        upsert raises, callers see the exception and no event lands.
        Conversely, an event failure does not roll back the schema
        write (the event stream is best-effort additive history;
        losing one row is preferable to leaving schema state in a
        half-updated condition). Errors in event emit are logged.
        """
        await self._schema_repo.upsert(schema)
        try:
            await self._event_repo.record(
                notebook_id,
                SCHEMA_CHANGED_EVENT_TYPE,
                event_payload,
            )
        except Exception as e:
            logger.error(
                f"Schema edit op persisted but event emit failed for "
                f"{notebook_id}: {e}"
            )
        # Re-fetch to pick up server-side last_modified_at + ensure the
        # caller sees the canonical post-write state.
        refreshed = await self._schema_repo.get_by_notebook(notebook_id)
        return refreshed if refreshed is not None else schema

    # ------------------------------------------------------------------
    # Accept / reject pending extensions
    # ------------------------------------------------------------------

    async def accept_extension(
        self,
        notebook_id: str,
        type_name: str,
    ) -> NotebookSchema:
        """Move the matching pending extension into accepted_extensions.

        Idempotent: if no pending extension matches ``type_name`` but one
        already exists in ``accepted_extensions`` with the same name, the
        op is a no-op (returns the current row, no event).

        Raises :class:`UnknownExtensionError` when no extension exists
        with this name in either list.
        """
        schema = await self._load_or_raise(notebook_id)

        already_accepted = any(
            ext.get("type_name") == type_name
            for ext in schema.accepted_extensions
        )
        matched: Optional[Dict[str, Any]] = None
        remaining: List[Dict[str, Any]] = []
        for ext in schema.pending_extensions:
            if matched is None and ext.get("type_name") == type_name:
                matched = ext
            else:
                remaining.append(ext)

        if matched is None:
            if already_accepted:
                return schema  # idempotent no-op
            raise UnknownExtensionError(
                f"No pending or accepted extension named {type_name!r}"
            )

        schema.pending_extensions = remaining
        schema.accepted_extensions.append(matched)
        return await self._persist(
            schema,
            notebook_id,
            {
                "op": "accept_extension",
                "type_name": type_name,
                "extension_id": matched.get("extension_id"),
            },
        )

    async def reject_extension(
        self,
        notebook_id: str,
        type_name: str,
    ) -> NotebookSchema:
        """Remove the matching pending extension.

        Idempotent: re-rejecting a name that is no longer in
        ``pending_extensions`` returns the current row with no event.
        We deliberately do NOT raise on "already gone" because the
        natural UX flow has the user clicking Reject after the row has
        disappeared from another tab.
        """
        schema = await self._load_or_raise(notebook_id)
        remaining = [
            ext for ext in schema.pending_extensions
            if ext.get("type_name") != type_name
        ]
        if len(remaining) == len(schema.pending_extensions):
            return schema  # idempotent no-op
        schema.pending_extensions = remaining
        return await self._persist(
            schema,
            notebook_id,
            {"op": "reject_extension", "type_name": type_name},
        )

    # ------------------------------------------------------------------
    # Rename / merge / split / delete
    # ------------------------------------------------------------------

    async def rename_type(
        self,
        notebook_id: str,
        old_name: str,
        new_name: str,
    ) -> NotebookSchema:
        """Record a rename as a synonym in accepted_extensions.

        The base ontology YAML is not touched. A rename entry is appended
        to ``accepted_extensions`` with shape::

            {
                "rename_id": "<deterministic key>",
                "op": "rename",
                "old_name": ...,
                "new_name": ...,
                "type_name": <new_name>,
            }

        ``type_name`` mirrors the new name so the existing TTL exporter
        + schema browser surface the rename as a regular class
        declaration. ``rename_id`` is deterministic
        (``rename:<old_name>->∶<new_name>``) so idempotent replays
        recognise an existing entry instead of appending a duplicate.

        Idempotent: a rename whose ``rename_id`` already lives in
        ``accepted_extensions`` returns the current row, no event.
        """
        if old_name == new_name:
            return await self._load_or_raise(notebook_id)
        schema = await self._load_or_raise(notebook_id)
        rename_id = self._rename_id(old_name, new_name)

        if any(
            ext.get("rename_id") == rename_id
            for ext in schema.accepted_extensions
        ):
            return schema  # idempotent no-op

        schema.accepted_extensions.append(
            {
                "rename_id": rename_id,
                "op": "rename",
                "old_name": old_name,
                "new_name": new_name,
                "type_name": new_name,
            }
        )
        return await self._persist(
            schema,
            notebook_id,
            {
                "op": "rename",
                "old_name": old_name,
                "new_name": new_name,
            },
        )

    async def merge_types(
        self,
        notebook_id: str,
        type_names: List[str],
        merged_name: str,
    ) -> NotebookSchema:
        """Record a merge of N types into one.

        Inputs:

        * ``type_names`` — the names being merged. Must contain at least
          two distinct names. Duplicate entries are deduplicated.
        * ``merged_name`` — the surviving name.

        The merge entry shape is::

            {
                "merge_id": "<deterministic>",
                "op": "merge",
                "source_types": [...],
                "merged_name": ...,
                "type_name": <merged_name>,
            }

        Idempotent: a merge whose ``merge_id`` already lives in
        ``accepted_extensions`` returns the current row, no event.

        Raises ``ValueError`` if ``type_names`` has fewer than two
        distinct entries — a "merge" of one type is a rename and
        should use :meth:`rename_type` instead.
        """
        deduped = sorted(set(type_names))
        if len(deduped) < 2:
            raise ValueError(
                f"merge requires at least two distinct type names; got {type_names!r}"
            )
        schema = await self._load_or_raise(notebook_id)
        merge_id = self._merge_id(deduped, merged_name)

        if any(
            ext.get("merge_id") == merge_id
            for ext in schema.accepted_extensions
        ):
            return schema  # idempotent no-op

        schema.accepted_extensions.append(
            {
                "merge_id": merge_id,
                "op": "merge",
                "source_types": deduped,
                "merged_name": merged_name,
                "type_name": merged_name,
            }
        )
        return await self._persist(
            schema,
            notebook_id,
            {
                "op": "merge",
                "source_types": deduped,
                "merged_name": merged_name,
            },
        )

    async def split_type(
        self,
        notebook_id: str,
        type_name: str,
        into: List[str],
        criterion: str,
    ) -> NotebookSchema:
        """Record a split of one type into N new ones.

        Inputs:

        * ``type_name`` — the source type.
        * ``into`` — list of new names. Must contain at least two distinct.
        * ``criterion`` — natural-language hint passed to Pass 2 so the
          LLM knows how to disambiguate (e.g. "by year published").

        The split entry shape is::

            {
                "split_id": "<deterministic>",
                "op": "split",
                "source_type": <type_name>,
                "into": [...],
                "criterion": ...,
                "type_name": <type_name>,
            }

        ``type_name`` mirrors the source so existing tooling sees the
        entry as a regular accepted extension declaration; downstream
        Pass 2 consults ``into`` + ``criterion`` for routing.

        Idempotent on (source_type, into, criterion).

        Raises ``ValueError`` if ``into`` has fewer than two distinct
        entries.
        """
        deduped_into = sorted(set(into))
        if len(deduped_into) < 2:
            raise ValueError(
                f"split requires at least two distinct target names; got {into!r}"
            )
        schema = await self._load_or_raise(notebook_id)
        split_id = self._split_id(type_name, deduped_into, criterion)

        if any(
            ext.get("split_id") == split_id
            for ext in schema.accepted_extensions
        ):
            return schema  # idempotent no-op

        schema.accepted_extensions.append(
            {
                "split_id": split_id,
                "op": "split",
                "source_type": type_name,
                "into": deduped_into,
                "criterion": criterion,
                "type_name": type_name,
            }
        )
        return await self._persist(
            schema,
            notebook_id,
            {
                "op": "split",
                "source_type": type_name,
                "into": deduped_into,
                "criterion": criterion,
            },
        )

    async def reparent_type(
        self,
        notebook_id: str,
        type_names: List[str],
        new_parent: str,
    ) -> NotebookSchema:
        """Record that ``type_names`` now hang under ``new_parent``.

        This is the write half of Track N.4d.3. ``type_placement`` decides where
        a PROPOSED type may go and which of its siblings are candidates to move
        under it; the judge selects among those candidates; this records the
        curator's acceptance. Nothing here re-derives the placement — a service
        that re-judged at write time could disagree with what the curator saw.

        One entry per moved type::

            {
                "reparent_id": "reparent::Article->Report",
                "op": "reparent",
                "type_name": "Article",
                "new_parent": "Report",
                "parent_type": "Report",
            }

        N entries rather than one entry naming N types, deliberately: each is
        separately visible in the schema browser, separately reversible, and
        renders through the existing extension tooling without it having to learn
        a list shape. ``parent_type`` mirrors ``new_parent`` for exactly that
        reason (the same mirroring ``rename_type`` does with ``type_name``);
        ``new_parent`` is the authoritative field.

        The edit takes effect through
        :func:`ontology_manager.schema_projection.project_accepted_edits`, which
        applies these entries to COPIES of the applied ontologies. That is what
        lets an entity of a re-parented type resolve through its new ancestor via
        ``canonical_bridge`` with no per-entity write — the type moved, not the
        rows.

        Idempotent per (type, parent): a type already recorded as moving under
        ``new_parent`` is skipped. Re-parenting the same type under a DIFFERENT
        parent appends a second entry rather than rewriting the first, and the
        projection applies entries in order, so the latest decision wins while the
        earlier one stays in the audit trail.

        Emits ONE event for the whole call, listing only the types this call
        actually moved.

        Raises ``ValueError`` when ``new_parent`` is blank, when no distinct type
        name is given, or when a type is moved under itself.
        """
        parent = (new_parent or "").strip()
        if not parent:
            raise ValueError("reparent requires a non-empty new_parent")

        wanted: List[str] = []
        seen: set = set()
        for name in type_names or []:
            cleaned = (name or "").strip()
            if not cleaned or cleaned.lower() in seen:
                continue
            if cleaned.lower() == parent.lower():
                raise ValueError(
                    f"cannot re-parent {cleaned!r} under itself"
                )
            seen.add(cleaned.lower())
            wanted.append(cleaned)
        if not wanted:
            raise ValueError(
                f"reparent requires at least one type name; got {type_names!r}"
            )

        schema = await self._load_or_raise(notebook_id)
        # No isinstance guard: `NotebookSchema.accepted_extensions` is typed
        # `List[Dict[str, Any]]` and Pydantic rejects a non-dict row at load, so
        # a guard here could not be reached — and an unreachable fence reads as a
        # demonstrated one.
        existing = {ext.get("reparent_id") for ext in schema.accepted_extensions}

        moved: List[str] = []
        for name in wanted:
            reparent_id = self._reparent_id(name, parent)
            if reparent_id in existing:
                continue  # already recorded — idempotent
            schema.accepted_extensions.append(
                {
                    "reparent_id": reparent_id,
                    "op": "reparent",
                    "type_name": name,
                    "new_parent": parent,
                    "parent_type": parent,
                }
            )
            moved.append(name)

        if not moved:
            return schema  # idempotent no-op

        return await self._persist(
            schema,
            notebook_id,
            {
                "op": "reparent",
                "type_names": moved,
                "new_parent": parent,
            },
        )

    async def delete_type(
        self,
        notebook_id: str,
        type_name: str,
    ) -> NotebookSchema:
        """Soft-delete by appending ``type_name`` to ``excluded_types``.

        Idempotent: re-deleting a type already in ``excluded_types`` is a
        no-op. The base ontology YAML is never mutated; the projection
        layer filters the entity types list against ``excluded_types``
        when building the effective schema.
        """
        schema = await self._load_or_raise(notebook_id)
        if type_name in schema.excluded_types:
            return schema  # idempotent no-op
        schema.excluded_types = [*schema.excluded_types, type_name]
        return await self._persist(
            schema,
            notebook_id,
            {"op": "delete", "type_name": type_name},
        )

    # ------------------------------------------------------------------
    # Deterministic id helpers
    # ------------------------------------------------------------------
    #
    # These keep replays idempotent without needing to compare entry
    # dicts structurally. A plain ``"::"`` delimiter is fine because
    # type names never contain it in practice; for defensive belt-and-
    # braces we strip the delimiter from inputs.

    @staticmethod
    def _rename_id(old_name: str, new_name: str) -> str:
        return f"rename::{old_name.replace('::', '_')}->{new_name.replace('::', '_')}"

    @staticmethod
    def _merge_id(source_types: List[str], merged_name: str) -> str:
        joined = "+".join(t.replace("::", "_") for t in source_types)
        return f"merge::{joined}->{merged_name.replace('::', '_')}"

    @staticmethod
    def _reparent_id(type_name: str, new_parent: str) -> str:
        return (
            f"reparent::{type_name.replace('::', '_')}->"
            f"{new_parent.replace('::', '_')}"
        )

    @staticmethod
    def _split_id(type_name: str, into: List[str], criterion: str) -> str:
        joined = "+".join(t.replace("::", "_") for t in into)
        # Include the criterion so two splits with different criteria are
        # treated as distinct operations.
        # We hash on a stable string rather than carrying the raw criterion
        # because it can be long-form natural language.
        import hashlib

        crit_hash = hashlib.sha256(criterion.encode("utf-8")).hexdigest()[:12]
        return (
            f"split::{type_name.replace('::', '_')}->"
            f"{joined}@{crit_hash}"
        )
