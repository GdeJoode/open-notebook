"""Track N.4d.3 — project a notebook's accepted schema edits onto its ontologies.

Why this module exists
======================
:func:`canonical_bridge.resolve_ontology_type` reasons over ``Ontology`` objects
and never sees ``notebook_schema``. So a curator's accepted edit — a new type, or
a re-parent decided by N.4d.1/N.4d.2 — is invisible to the persist path unless
something materialises it onto the ontologies the bridge is handed.

Two facts were MEASURED before writing this, because assuming either would have
shipped a re-parent that quietly does nothing:

1. **The registry hands out shared objects.** ``OntologyRegistry.get('deals')``
   returns the identical object on the second call, ``entity_types`` dict
   included. A projection that edited in place would leak one notebook's
   vocabulary into every other notebook served by the same process, so this
   module works on ``model_copy(deep=True)`` and the inputs are never touched.

2. **The bridge prefers ``schema_org_type`` over ``parent_type``.** Across the
   eleven shipped ontologies, 13 of 277 applied type entries declare a
   ``schema_org_type``, and for 10 of them rewriting ``parent_type`` alone leaves
   the canonical UNCHANGED — the bridge terminates at step 1 and never walks the
   new parent. Every one of those is a root a curator plausibly re-parents
   (``Person``, ``Organization``, ``Location``, ``Event``, ``Topic``,
   ``Technology``). So a re-parent CLEARS ``schema_org_type``: the curator's
   placement overrides the declaration it replaces, and leaving the field would
   make the edit a no-op for exactly the types most likely to receive one.

The rule this implements, stated once: **a notebook's applied vocabulary is the
base ontologies plus the notebook's accepted schema edits.** Accepting an
extension therefore also makes that type resolvable by the bridge, which it was
not before — a deliberate behaviour change, recorded here rather than discovered
later.

What is deliberately NOT done
=============================
``rename`` / ``merge`` / ``split`` / ``delete`` entries are ignored. They are
recorded curator ops with their own downstream semantics (the schema browser, the
TTL export, the Pass-2 prompt); reading them as type declarations here would
invent a vocabulary nobody accepted. Only genuine accepted extensions and
``op == "reparent"`` entries are projected.

Every refusal names an OBSERVATION with exactly one cause (the N.4a evidence
discipline): what was looked for, and what was found instead — never the
inference a curator would draw from it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger

from .canonical_bridge import resolve_ontology_type
from .schema import EntityTypeDefinition
from .type_placement import find_type, known_schema_org_base, would_cycle

# Actions an entry can produce.
MATERIALISED = "materialised"
REPARENTED = "reparented"
REFUSED = "refused"

ACTIONS = (MATERIALISED, REPARENTED, REFUSED)

# Reason codes. Each names what was OBSERVED.
EV_DEFINED = "accepted_extension_now_defined"
EV_PARENT_REWRITTEN = "parent_type_rewritten"
EV_NO_SCHEMAS = "no_applied_schemas"
EV_NO_TYPE_NAME = "entry_declares_no_type_name"
EV_NAME_ALREADY_DEFINED = "name_already_defined_in_applied_schemas"
EV_TYPE_NOT_FOUND = "type_not_defined_in_applied_schemas"
EV_PARENT_NOT_FOUND = "new_parent_neither_defined_nor_a_mapped_base"
EV_CYCLE = "new_parent_descends_from_this_type"
EV_CHAIN_ORPHANS = "chain_stopped_reaching_a_mapped_base"

REASON_CODES = (
    EV_DEFINED,
    EV_PARENT_REWRITTEN,
    EV_NO_SCHEMAS,
    EV_NO_TYPE_NAME,
    EV_NAME_ALREADY_DEFINED,
    EV_TYPE_NOT_FOUND,
    EV_PARENT_NOT_FOUND,
    EV_CYCLE,
    EV_CHAIN_ORPHANS,
)

REPARENT_OP = "reparent"

# B.3c's resume sentinel is infrastructure, not ontology content. The contract in
# `routers/schemas.py` requires every new `accepted_extensions` consumer to filter
# it; this module is the sixth such site.
_SENTINEL_KEY = "is_resume_sentinel"

# The ops that are recorded in `accepted_extensions` but are not type
# declarations. Listed positively so a future op is ignored by default rather
# than silently materialised as a type.
_NON_DECLARATION_OPS = ("rename", "merge", "split", "delete")


@dataclass(frozen=True)
class ProjectionOutcome:
    """What happened to one accepted-extensions entry.

    ``type_name`` is the type the entry is about. One re-parent entry moves ONE
    type: a curator accepting a placement over N siblings records N entries, so
    each is separately visible, separately refusable, and separately reversible.
    A single entry carrying a list would also have to be read defensively — a
    string is iterable by character — which is the blocker N.4d.2 shipped a fix
    for one phase earlier.
    """

    type_name: str
    action: str
    reason_code: str
    detail: str = ""


@dataclass
class Projection:
    """Deep-copied ontologies with the notebook's accepted edits applied."""

    schemas: List[Any] = field(default_factory=list)
    outcomes: List[ProjectionOutcome] = field(default_factory=list)

    @property
    def applied(self) -> List[ProjectionOutcome]:
        return [o for o in self.outcomes if o.action != REFUSED]

    @property
    def refused(self) -> List[ProjectionOutcome]:
        return [o for o in self.outcomes if o.action == REFUSED]


def _norm(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def _is_sentinel(entry: Any) -> bool:
    return isinstance(entry, dict) and entry.get(_SENTINEL_KEY) is True


def _entry_type_name(entry: Dict[str, Any]) -> str:
    name = entry.get("type_name")
    return name.strip() if isinstance(name, str) else ""


def _target_ontology(entry: Dict[str, Any], schemas: Sequence[Any]) -> Any:
    """Which applied ontology a materialised type is attached to.

    ``schema_name`` when it names one of the applied schemas, else the first —
    mirroring the broadcast rule `_run_multi_schema` already uses for the Pass-2
    extension map, so a type does not land in one place for the prompt and
    another for the bridge. The choice only decides WHERE the definition hangs;
    the bridge searches every applied schema, so resolution is unaffected.
    """
    wanted = entry.get("schema_name")
    if isinstance(wanted, str) and wanted.strip():
        target = _norm(wanted)
        for ontology in schemas:
            metadata = getattr(ontology, "metadata", None)
            if metadata is not None and _norm(getattr(metadata, "name", "")) == target:
                return ontology
    return schemas[0]


def _materialise(
    entry: Dict[str, Any], schemas: List[Any]
) -> ProjectionOutcome:
    """Define an accepted extension type on the applied schemas."""
    name = _entry_type_name(entry)
    if not name:
        return ProjectionOutcome("", REFUSED, EV_NO_TYPE_NAME)

    if find_type(name, schemas) is not None:
        # A shipped definition is never overwritten by an accepted extension:
        # the YAML is the authored vocabulary, and an extension row that
        # collides with it is a naming clash for a curator to resolve, not a
        # licence to redefine the type underneath every other notebook.
        return ProjectionOutcome(name, REFUSED, EV_NAME_ALREADY_DEFINED)

    parent = entry.get("parent_type")
    definition = EntityTypeDefinition(
        name=name,
        description=entry.get("description")
        if isinstance(entry.get("description"), str)
        else None,
        parent_type=parent.strip() if isinstance(parent, str) and parent.strip() else None,
    )
    _target_ontology(entry, schemas).entity_types[name] = definition
    return ProjectionOutcome(name, MATERIALISED, EV_DEFINED, f"parent={definition.parent_type}")


def _reparent_one(
    type_name: str, new_parent: str, schemas: List[Any]
) -> ProjectionOutcome:
    """Rewrite one type's parent, refusing anything that cannot be shown safe."""
    definition = find_type(type_name, schemas)
    if definition is None:
        return ProjectionOutcome(type_name, REFUSED, EV_TYPE_NOT_FOUND)

    if find_type(new_parent, schemas) is None and not known_schema_org_base(new_parent):
        return ProjectionOutcome(type_name, REFUSED, EV_PARENT_NOT_FOUND, f"parent={new_parent}")

    if would_cycle(type_name, new_parent, schemas):
        return ProjectionOutcome(type_name, REFUSED, EV_CYCLE, f"parent={new_parent}")

    before = resolve_ontology_type(definition.name or type_name, schemas)

    previous_parent = definition.parent_type
    previous_base = definition.schema_org_type
    definition.parent_type = new_parent
    # See the module docstring: leaving `schema_org_type` in place makes the
    # re-parent a no-op for 10 of the 13 types that declare one, because the
    # bridge terminates on it before ever walking the parent.
    definition.schema_org_type = None

    after = resolve_ontology_type(definition.name or type_name, schemas)
    if before is not None and after is None:
        # The type resolved to a canonical before this edit and does not after:
        # its chain no longer reaches a mapped schema.org base. Applying it would
        # silently drop every entity of this type onto the alias fallback, so the
        # edit is rolled back and the observation is reported instead.
        definition.parent_type = previous_parent
        definition.schema_org_type = previous_base
        return ProjectionOutcome(type_name, REFUSED, EV_CHAIN_ORPHANS, f"parent={new_parent}")

    return ProjectionOutcome(
        type_name,
        REPARENTED,
        EV_PARENT_REWRITTEN,
        f"{previous_parent or previous_base or '(none)'} -> {new_parent}",
    )


def project_accepted_edits(
    schemas: Optional[Sequence[Any]],
    accepted_extensions: Optional[Sequence[Any]],
) -> Projection:
    """Apply a notebook's accepted schema edits to copies of ``schemas``.

    Materialisation runs before re-parenting, so a type may be moved under a
    parent the same curator accepted a moment earlier; the reverse order would
    refuse that edit for a parent that does exist. Re-parents are applied in
    recorded order, so a type moved twice ends under the parent named LAST — the
    curator's most recent decision.

    The inputs are never mutated. Callers pass the registry's cached ontologies
    and keep them intact for the next notebook.
    """
    projected: List[Any] = [
        s.model_copy(deep=True) if hasattr(s, "model_copy") else s
        for s in (schemas or [])
    ]
    outcomes: List[ProjectionOutcome] = []

    entries = [e for e in (accepted_extensions or []) if isinstance(e, dict)]
    entries = [e for e in entries if not _is_sentinel(e)]

    if not projected:
        # Nothing to project onto. Reported rather than silently empty: a
        # notebook with accepted edits and no applied schema is a wiring
        # problem, and "no edits applied" would read as "no edits exist".
        if entries:
            outcomes.append(ProjectionOutcome("", REFUSED, EV_NO_SCHEMAS))
        return Projection(schemas=projected, outcomes=outcomes)

    reparents = [e for e in entries if e.get("op") == REPARENT_OP]
    declarations = [
        e
        for e in entries
        if e.get("op") not in _NON_DECLARATION_OPS and e.get("op") != REPARENT_OP
    ]

    for entry in declarations:
        outcomes.append(_materialise(entry, projected))

    for entry in reparents:
        name = _entry_type_name(entry)
        new_parent = entry.get("new_parent")
        new_parent = new_parent.strip() if isinstance(new_parent, str) else ""
        if not name:
            outcomes.append(ProjectionOutcome("", REFUSED, EV_NO_TYPE_NAME))
            continue
        if not new_parent:
            outcomes.append(
                ProjectionOutcome(name, REFUSED, EV_PARENT_NOT_FOUND, "parent=(none)")
            )
            continue
        outcomes.append(_reparent_one(name, new_parent, projected))

    refused = [o for o in outcomes if o.action == REFUSED]
    if refused:
        logger.info(
            "schema projection: {applied} applied, {refused} refused ({codes})",
            applied=len(outcomes) - len(refused),
            refused=len(refused),
            codes=", ".join(sorted({o.reason_code for o in refused})),
        )
    return Projection(schemas=projected, outcomes=outcomes)
