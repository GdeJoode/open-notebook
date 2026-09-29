"""The notebook's Pass-1 proposal queue: accept into the shared ontology, or reject.

Pass 1 proposes entity types a document needed and the ontologies lacked; they
queue per notebook in ``notebook_schema.pending_extensions`` (PC.1). The
ontologies themselves are SHARED by every notebook (owner, 2026-09-28; OE-D2
(a)), so a decision on a proposal is made there:

* ``accept_extension`` adds the type to the shared ontology the curator picks,
  through the ontology editor's gate and store
  (:meth:`OntologyEditService.publish_new_entity_type`): a blocking finding
  refuses it (422, nothing stored), and it is previewed first with
  ``preview_accept``. On success the proposal leaves the queue and is recorded
  in ``accepted_extensions`` with the ontology and version it was published to.
* ``mark_covered`` takes out a proposal the shared ontologies already define
  (by name or alias, as typing matches): recorded as accepted with the
  defining ontology, nothing published, no rejection. Accept refuses such a
  proposal (409 ``defined_in``) instead of publishing a second definition.
* ``reject_extension`` drops it and remembers the "no" (PC.5).

``accepted_extensions`` is the queue's history. Since OE.5 nothing reads it for
typing, detection or the Pass-2 prompt; it keeps a published proposal from
re-queueing and satisfies the ``review_required`` gate. The per-notebook edit
ops (rename, merge, split, re-parent, delete) are gone: those are edits of the
shared ontology now, made in the editor. Rows they stored earlier stay in the
database and are ignored.

Each decision emits one ``notebook_event{type:"schema_changed"}``, which the
re-extract banner (B.3d) reacts to. Re-running a decision that already took
effect is a no-op without an event.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from loguru import logger
from shared.models import NotebookSchema
from surrealdb_service.repositories.notebook_event import NotebookEventRepository
from surrealdb_service.repositories.notebook_schema import (
    NotebookSchemaRepository,
    record_rejection,
)

from app_main.services.ontology_edit_service import (
    OntologyEditError,
    OntologyEditService,
    defined_in,
)
from app_main.services.ontology_preview_service import load_own_ontologies

SCHEMA_CHANGED_EVENT_TYPE = "schema_changed"

# The discriminator on an ``accepted_extensions`` entry that records a re-parent
# rather than a new type (N.4d.3). No longer written; older rows still carry it.
REPARENT_OP = "reparent"

#: The ontology a proposal goes to by default when no ontology defines its
#: parent: the root the other shipped ontologies extend.
FALLBACK_ONTOLOGY = "schema_core"


class NotebookSchemaNotFoundError(Exception):
    """The notebook has no ``notebook_schema`` row, so there is no queue.

    The router translates this to 404, distinct from "notebook does not exist".
    """


class UnknownExtensionError(Exception):
    """No pending or accepted proposal has this name. The router makes it 404."""


def default_ontology(ontologies: Dict[str, Any], parent_type: Optional[str]) -> str:
    """Where a proposal goes unless the curator picks otherwise: the ontology
    that introduces its parent, else :data:`FALLBACK_ONTOLOGY`.

    "Introduces": defines it (by name or alias, case-insensitive) with the
    fewest ``extends`` above it; among those, the one most other ontologies
    build on, then by name. ``Organization``, defined by both roots ``base`` and
    ``schema_core``, goes to ``schema_core``, which seven ontologies extend.
    """
    wanted = (parent_type or "").strip().lower()

    def chain(name: str) -> List[str]:
        out: List[str] = []
        current = ontologies.get(name)
        while current is not None and current.metadata.extends and name not in out:
            out.append(name)
            name = current.metadata.extends
            current = ontologies.get(name)
        return out + ([name] if name not in out else [])

    def builds_on(name: str) -> int:
        return sum(1 for other in ontologies if other != name and name in chain(other))

    defining = [
        name for name, ontology in ontologies.items()
        if wanted and any(
            wanted in {str(key).lower(), str(d.name or "").lower(),
                       *(str(a).lower() for a in (d.aliases or []))}
            for key, d in ontology.entity_types.items()
        )
    ]
    if defining:
        return min(defining, key=lambda n: (len(chain(n)), -builds_on(n), n))
    return FALLBACK_ONTOLOGY


class SchemaEditService:
    """Accept or reject the proposals queued for a notebook."""

    def __init__(
        self,
        schema_repo: NotebookSchemaRepository,
        event_repo: NotebookEventRepository,
        ontology_edit: Optional[OntologyEditService] = None,
    ):
        self._schema_repo = schema_repo
        self._event_repo = event_repo
        self._ontology_edit = ontology_edit or OntologyEditService()

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
        """Upsert the row, then emit one ``schema_changed`` event.

        An event failure is logged and does not undo the write: the event stream
        is best-effort history, the row is the state.
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
        refreshed = await self._schema_repo.get_by_notebook(notebook_id)
        return refreshed if refreshed is not None else schema

    # ------------------------------------------------------------------
    # Accept into the shared ontology
    # ------------------------------------------------------------------

    @staticmethod
    def _pending(
        schema: NotebookSchema, type_name: str
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]]]:
        matched: Optional[Dict[str, Any]] = None
        remaining: List[Dict[str, Any]] = []
        for ext in schema.pending_extensions:
            if matched is None and ext.get("type_name") == type_name:
                matched = ext
            else:
                remaining.append(ext)
        return matched, remaining

    @staticmethod
    def _already_accepted(schema: NotebookSchema, type_name: str) -> bool:
        # A legacy re-parent row carries the moved type's name in `type_name`
        # too; it is not an acceptance of that name.
        return any(
            ext.get("type_name") == type_name and ext.get("op") != REPARENT_OP
            for ext in schema.accepted_extensions
        )

    @staticmethod
    def _fields(proposal: Dict[str, Any], parent_type: Optional[str]) -> Dict[str, Any]:
        """The entity type the proposal becomes. ``parent_type`` is the
        curator's choice when given, else the proposal's."""
        parent = parent_type if parent_type is not None else proposal.get("parent_type")
        fields: Dict[str, Any] = {"parent_type": (parent or "").strip() or None}
        description = proposal.get("description") or proposal.get("rationale")
        if isinstance(description, str) and description.strip():
            fields["description"] = description.strip()
        return fields

    async def _proposal(self, notebook_id: str, type_name: str) -> Dict[str, Any]:
        schema = await self._load_or_raise(notebook_id)
        matched, _ = self._pending(schema, type_name)
        if matched is None:
            raise UnknownExtensionError(f"No pending extension named {type_name!r}")
        return matched

    async def accept_target(self, notebook_id: str, type_name: str) -> Dict[str, Any]:
        """The proposal, and the ontology it goes to by default."""
        proposal = await self._proposal(notebook_id, type_name)
        ontologies = await load_own_ontologies()
        parent = self._fields(proposal, None)["parent_type"]
        return {
            "type_name": type_name,
            "parent_type": parent,
            "default_ontology": default_ontology(ontologies, parent),
            "ontologies": sorted(ontologies),
            # Where the proposed name is already a shared type (review M3):
            # then it is not accepted but marked as covered.
            "defined_in": defined_in(ontologies, type_name),
        }

    async def preview_accept(
        self,
        notebook_id: str,
        type_name: str,
        ontology: str,
        parent_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """What accepting into ``ontology`` would do (the OE.1 preview). Writes
        nothing."""
        proposal = await self._proposal(notebook_id, type_name)
        return await self._ontology_edit.preview_new_entity_type(
            ontology, type_name, self._fields(proposal, parent_type)
        )

    async def accept_extension(
        self,
        notebook_id: str,
        type_name: str,
        ontology: str,
        parent_type: Optional[str] = None,
    ) -> Tuple[NotebookSchema, Optional[Dict[str, Any]]]:
        """Publish the proposal as a type of the shared ``ontology``, then move
        it out of the queue.

        Returns the updated row and the publish result (``None`` when the
        proposal was already accepted: idempotent, nothing published, no event).
        :class:`OntologyEditError` from the publish (422 blocking findings, 409
        open draft or existing type, 404 unknown ontology) propagates, and the
        queue is left as it was.
        """
        schema = await self._load_or_raise(notebook_id)
        matched, _ = self._pending(schema, type_name)
        if matched is None:
            if self._already_accepted(schema, type_name):
                return schema, None
            raise UnknownExtensionError(
                f"No pending or accepted extension named {type_name!r}"
            )

        fields = self._fields(matched, parent_type)
        published = await self._ontology_edit.publish_new_entity_type(
            ontology, type_name, fields,
            note=f"accepted from {notebook_id}'s Pass-1 proposals",
        )
        # Re-read: the publish takes a while, and proposals merged or decided
        # meanwhile must not be overwritten (review m5). Only this proposal
        # moves.
        schema = await self._dequeue(notebook_id, type_name, {
            **matched,
            "parent_type": fields["parent_type"],
            "published_to": ontology,
            "published_version": published["version"],
        })
        updated = await self._persist(
            schema,
            notebook_id,
            {
                "op": "accept_extension",
                "type_name": type_name,
                "extension_id": matched.get("extension_id"),
                "ontology": ontology,
                "version": published["version"],
            },
        )
        return updated, published

    async def _dequeue(
        self, notebook_id: str, type_name: str, record: Dict[str, Any]
    ) -> NotebookSchema:
        """A FRESH read of the row with only ``type_name`` moved from the queue
        into ``accepted_extensions`` as ``record``."""
        schema = await self._load_or_raise(notebook_id)
        schema.pending_extensions = [
            e for e in schema.pending_extensions if e.get("type_name") != type_name
        ]
        schema.accepted_extensions.append(record)
        return schema

    async def mark_covered(
        self, notebook_id: str, type_name: str
    ) -> NotebookSchema:
        """Take a proposal the shared ontologies already define out of the
        queue, as accepted, without publishing anything.

        ``published_to`` is the ontology that introduces the name (the
        :func:`default_ontology` rule over its definitions), ``published_version``
        its active version, ``covered`` marks that nothing was published. Not a
        rejection: nothing goes to ``rejected_extensions``. 409 when no shared
        ontology defines the name (then it is accepted or rejected instead).
        """
        schema = await self._load_or_raise(notebook_id)
        matched, _ = self._pending(schema, type_name)
        if matched is None:
            if self._already_accepted(schema, type_name):
                return schema
            raise UnknownExtensionError(
                f"No pending or accepted extension named {type_name!r}"
            )
        ontologies = await load_own_ontologies()
        existing = defined_in(ontologies, type_name)
        if not existing:
            raise OntologyEditError(409, (
                f"'{type_name}' is not defined in any shared ontology; accept or "
                "reject it"))
        covered_by = default_ontology(
            {e["ontology"]: ontologies[e["ontology"]] for e in existing}, type_name
        )
        schema = await self._dequeue(notebook_id, type_name, {
            **matched,
            "published_to": covered_by,
            "published_version": ontologies[covered_by].metadata.version,
            "covered": True,
        })
        return await self._persist(
            schema,
            notebook_id,
            {"op": "mark_covered", "type_name": type_name, "ontology": covered_by},
        )

    # ------------------------------------------------------------------
    # Reject
    # ------------------------------------------------------------------

    async def reject_extension(
        self,
        notebook_id: str,
        type_name: str,
    ) -> NotebookSchema:
        """Remove the matching pending extension.

        The rejection is remembered in ``rejected_extensions`` (PC.5), which
        ``merge_pending_extensions`` honours, so the type does not return with
        the next document.

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
        # PC.5: the "no" is recorded, so the next document proposing the same
        # type does not put it back in the queue (migration 84).
        record_rejection(schema, type_name)
        return await self._persist(
            schema,
            notebook_id,
            {"op": "reject_extension", "type_name": type_name},
        )
