"""Where does a PROPOSED type belong in the existing hierarchy? (Track N.4d.1)

A new entity TYPE enters this system at three moments, and all three currently
fill the parent slot with an unvalidated guess:

1. ``pass1_schema_validation`` proposes an extension — the extraction LLM guesses
   ``parent_type``;
2. ``OntologyEvolutionAgent.create_proposal_from_gap`` sets
   ``definition["parent_type"] = gap.entity_type_guess`` — the gap's raw guess;
3. ``SchemaEditService.accept_extension`` carries whichever guess arrived into
   ``accepted_extensions``, unchecked.

This module fills that gap. It answers, deterministically and with evidence,
whether a proposed type is actually NEW, whether its declared parent exists, and
which existing types could plausibly move UNDER it.

Why here and not per entity (D-N4-12)
=====================================
Three earlier attempts placed concepts by comparing ENTITIES, and all three failed
identically, because **subsumption relates TYPES while the entity table stores
MENTIONS** — no writer in this codebase creates an entity row denoting a type. At
this boundary both sides of the question are types, so the question is well-posed
for the first time.

The bounded candidate set that makes BROADER_THAN tractable
===========================================================
A proposed type ``P`` with declared parent ``G`` can only become the parent of an
existing type ``T`` by being inserted BETWEEN ``T`` and ``T``'s current parent.
That is structurally valid only when ``T`` also hangs from ``G`` — i.e. ``T`` is a
SIBLING of ``P``. Anything whose chain already passes through something narrower
than ``P`` cannot be re-parented under it.

So the candidate set is derived from DECLARATIONS, never guessed from names or
vectors: :func:`sibling_types` enumerates it, and N.4d.2's judge then selects
within it. On the shipped ``deals`` ontology a type proposed under ``Deal`` yields
exactly ``{RegioDeal, Woondeal, CityDeal}`` — three definitions to weigh, not a
graph-sized search.

Scope: **verdicts and enumeration only.** No LLM (N.4d.2), no schema write
(N.4d.3), no gap recording (N.4d.4). Everything here is a pure function of
``(proposal, applied ontologies)``.

Evidence discipline (D-N4-7)
============================
Inherited verbatim from the entity-side work, because it is what made that work
trustworthy: every verdict names **what was observed**, never the inference it
would license. "No applied ontology defines this parent" is a fact about the
lookup; "this type is top-level" is a conclusion that may be false. Each
``EV_*`` code below has exactly one cause.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from loguru import logger

# -- verdicts ---------------------------------------------------------------

#: The proposal is not a new type: an applied ontology already defines this name
#: or declares it as an alias. It needs merging or rejecting, not placing.
DUPLICATE = "DUPLICATE"
#: The declared parent resolves to a known type, and the placement is structurally
#: valid. Note this VALIDATES a declaration; it does not infer one.
PLACED = "PLACED"
#: A parent was declared but no applied ontology defines it.
PARENT_UNKNOWN = "PARENT_UNKNOWN"
#: No parent was declared at all.
UNPARENTED = "UNPARENTED"
#: The declared parent's own chain runs through this type — accepting it would
#: make the hierarchy cyclic.
CYCLIC = "CYCLIC"

VERDICTS = (DUPLICATE, PLACED, PARENT_UNKNOWN, UNPARENTED, CYCLIC)

# -- evidence codes: each names an OBSERVATION with exactly one cause -------

EV_NO_SCHEMAS = "no_applied_schemas"
EV_NO_NAME = "proposal_has_no_name"
EV_NAME_TAKEN = "name_already_defined"
EV_ALIAS_TAKEN = "name_declared_as_alias"
EV_PARENT_RESOLVED = "declared_parent_resolves"
EV_PARENT_UNKNOWN = "declared_parent_not_defined"
EV_NO_PARENT_DECLARED = "no_parent_declared"
EV_CYCLE = "declared_parent_descends_from_this_type"

REASON_CODES = (
    EV_NO_SCHEMAS,
    EV_NO_NAME,
    EV_NAME_TAKEN,
    EV_ALIAS_TAKEN,
    EV_PARENT_RESOLVED,
    EV_PARENT_UNKNOWN,
    EV_NO_PARENT_DECLARED,
    EV_CYCLE,
)

#: Bound on any chain walk, so a hand-authored ontology loop cannot hang a caller.
#: ``canonical_bridge`` uses the same bound for the same reason.
_MAX_DEPTH = 16


@dataclass(frozen=True)
class TypePlacement:
    """Where a proposed type sits, and what could move under it.

    ``reason_code`` is the machine-checkable half of the evidence and ``evidence``
    its human-readable expansion; both are always populated, because a curator has
    to be able to audit and reverse the decision this feeds.

    ``descendant_candidates`` is the BOUNDED set N.4d.2's judge selects within —
    the proposal's siblings, which are the only existing types that could be
    re-parented under it. It is empty for every verdict except ``PLACED``: without
    a resolved parent there is no sibling set to speak of.
    """

    verdict: str
    reason_code: str
    evidence: str
    parent: Optional[str] = None
    duplicate_of: Optional[str] = None
    descendant_candidates: Tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Lookup over the applied ontologies
# ---------------------------------------------------------------------------


def _norm(text: Optional[str]) -> str:
    return (text or "").strip().lower()


def _iter_types(schemas: Optional[Sequence[Any]]) -> Iterable[Tuple[str, Any]]:
    """``(name, definition)`` over every applied ontology's entity types.

    The registry resolves inheritance before handing an ``Ontology`` over, so each
    one already carries its merged parent types — a single pass sees the whole
    layered vocabulary. Malformed members are skipped rather than raising: this
    runs on curator input.
    """
    for ontology in schemas or ():
        types = getattr(ontology, "entity_types", None)
        if not isinstance(types, dict):
            continue
        for key, definition in types.items():
            name = getattr(definition, "name", None) or key
            if name:
                yield str(name), definition


def known_schema_org_base(name: str) -> bool:
    """True when ``name`` is a schema.org base ``canonical_bridge`` maps.

    A parent can be valid in two ways, and conflating them misreports half the
    shipped vocabulary. ``deals.yaml`` roots ``Deal``/``Akkoord``/
    ``BeleidsProgramma`` at ``GovernmentService``, which no ontology DEFINES —
    the bridge terminates its walk on the base name itself. Treating that as an
    unknown parent would call four of eight declarations broken when they are
    exactly how the vocabulary is meant to be authored.

    Reads the bridge's own map so the two cannot drift apart; a bridge that ever
    stops exporting it degrades to "not a base" rather than raising.
    """
    target = _norm(name)
    if not target:
        return False
    try:
        from ontology_manager.canonical_bridge import _CANONICAL_BY_SCHEMA_ORG
    except Exception as exc:  # noqa: BLE001 - degrade, never raise on curator input
        logger.debug("type_placement: schema.org base map unavailable ({e})", e=exc)
        return False
    return any(_norm(base) == target for base in _CANONICAL_BY_SCHEMA_ORG)


def find_type(name: str, schemas: Optional[Sequence[Any]]) -> Optional[Any]:
    """The definition whose name matches ``name``, case-insensitively.

    Aliases are deliberately NOT matched here — :func:`alias_owner` reports those
    separately, because "the name is taken" and "the name is somebody's alias" are
    different observations and a curator needs to tell them apart.
    """
    target = _norm(name)
    if not target:
        return None
    for type_name, definition in _iter_types(schemas):
        if _norm(type_name) == target:
            return definition
    return None


def alias_owner(name: str, schemas: Optional[Sequence[Any]]) -> Optional[str]:
    """The type that declares ``name`` as one of its aliases, if any."""
    target = _norm(name)
    if not target:
        return None
    for type_name, definition in _iter_types(schemas):
        for alias in getattr(definition, "aliases", None) or ():
            if _norm(alias) == target:
                return type_name
    return None


def ancestors_of(type_name: str, schemas: Optional[Sequence[Any]]) -> List[str]:
    """The declared ``parent_type`` chain above ``type_name``, nearest first.

    Terminates on an unknown parent (returning what it walked) rather than
    raising: a chain that leaves the applied set is a real, reportable state, not
    an error. Cycle- and depth-guarded.
    """
    out: List[str] = []
    seen = {_norm(type_name)}
    current = find_type(type_name, schemas)
    for _ in range(_MAX_DEPTH):
        parent = getattr(current, "parent_type", None) if current is not None else None
        if not parent or _norm(parent) in seen:
            break
        out.append(str(parent))
        seen.add(_norm(parent))
        current = find_type(parent, schemas)
    return out


def sibling_types(
    parent: str, schemas: Optional[Sequence[Any]], *, exclude: str = ""
) -> Tuple[str, ...]:
    """Existing types that declare ``parent`` as their direct parent.

    This is the whole bounded candidate set for BROADER_THAN: a proposed type can
    only be inserted between these and their shared parent. Order follows the
    applied ontologies so the result is stable for a given input.
    """
    target = _norm(parent)
    skip = _norm(exclude)
    if not target:
        return ()
    out: List[str] = []
    for type_name, definition in _iter_types(schemas):
        if _norm(type_name) == skip:
            continue
        if _norm(getattr(definition, "parent_type", None)) == target:
            if type_name not in out:
                out.append(type_name)
    return tuple(out)


def would_cycle(
    child: str, new_parent: str, schemas: Optional[Sequence[Any]]
) -> bool:
    """True when making ``new_parent`` the parent of ``child`` closes a loop.

    That happens when ``new_parent`` already descends from ``child``. Unreachable
    for a genuinely new type — nothing can descend from what does not exist yet —
    but N.4d.3 re-parents EXISTING types, where it is reachable, so the check
    lives with the other structural rules rather than being invented later.
    """
    if _norm(child) == _norm(new_parent):
        return True
    return _norm(child) in {_norm(a) for a in ancestors_of(new_parent, schemas)}


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------


def place_proposed_type(
    name: str,
    declared_parent: Optional[str],
    schemas: Optional[Sequence[Any]],
) -> TypePlacement:
    """Validate a proposed type against the applied ontologies.

    This VALIDATES the declaration the proposer made; it does not invent one. A
    resolved parent means "the proposal's own claim checks out", which is a
    different and much stronger thing than an inferred subsumption — and it is
    why this boundary works where the entity-level attempts did not.
    """
    if not (name or "").strip():
        return TypePlacement(
            verdict=UNPARENTED,
            reason_code=EV_NO_NAME,
            evidence="the proposal carries no type name, so nothing could be looked up",
        )
    if not schemas:
        return TypePlacement(
            verdict=UNPARENTED,
            reason_code=EV_NO_SCHEMAS,
            evidence=(
                f"no ontologies were applied, so {name!r} was never compared "
                "against anything — this says nothing about whether it is new"
            ),
        )

    existing = find_type(name, schemas)
    if existing is not None:
        owner = getattr(existing, "name", None) or name
        return TypePlacement(
            verdict=DUPLICATE,
            reason_code=EV_NAME_TAKEN,
            evidence=f"an applied ontology already defines the type {owner!r}",
            duplicate_of=str(owner),
        )
    owner = alias_owner(name, schemas)
    if owner is not None:
        return TypePlacement(
            verdict=DUPLICATE,
            reason_code=EV_ALIAS_TAKEN,
            evidence=(
                f"{name!r} is already declared as an alias of the existing type "
                f"{owner!r}"
            ),
            duplicate_of=owner,
        )

    if not (declared_parent or "").strip():
        return TypePlacement(
            verdict=UNPARENTED,
            reason_code=EV_NO_PARENT_DECLARED,
            evidence=(
                f"the proposal for {name!r} declares no parent type; where it "
                "belongs is undecided, not top-level"
            ),
        )
    parent_def = find_type(declared_parent, schemas)
    is_base = parent_def is None and known_schema_org_base(declared_parent)
    if parent_def is None and not is_base:
        return TypePlacement(
            verdict=PARENT_UNKNOWN,
            reason_code=EV_PARENT_UNKNOWN,
            evidence=(
                f"the proposal declares parent {declared_parent!r}, which no "
                "applied ontology defines and which is not a schema.org base the "
                "canonical bridge maps — the placement could not be checked"
            ),
        )
    parent_name = str(
        getattr(parent_def, "name", None) if parent_def is not None else declared_parent
    )
    if would_cycle(name, parent_name, schemas):
        return TypePlacement(
            verdict=CYCLIC,
            reason_code=EV_CYCLE,
            evidence=(
                f"{parent_name!r} already descends from {name!r}, so declaring it "
                "as the parent would make the hierarchy cyclic"
            ),
            parent=parent_name,
        )

    siblings = sibling_types(parent_name, schemas, exclude=name)
    kind = (
        "a schema.org base the canonical bridge maps"
        if is_base
        else "a type defined in an applied ontology"
    )
    logger.debug(
        "type_placement: {n!r} placed under {p!r} with {c} descendant candidate(s)",
        n=name, p=parent_name, c=len(siblings),
    )
    return TypePlacement(
        verdict=PLACED,
        reason_code=EV_PARENT_RESOLVED,
        evidence=(
            f"the declared parent {parent_name!r} resolves as {kind}; "
            f"{len(siblings)} existing type(s) declare it as their parent and are "
            f"therefore the only candidates that could move under {name!r}"
        ),
        parent=parent_name,
        descendant_candidates=siblings,
    )


__all__ = [
    "TypePlacement",
    "DUPLICATE",
    "PLACED",
    "PARENT_UNKNOWN",
    "UNPARENTED",
    "CYCLIC",
    "VERDICTS",
    "EV_NO_SCHEMAS",
    "EV_NO_NAME",
    "EV_NAME_TAKEN",
    "EV_ALIAS_TAKEN",
    "EV_PARENT_RESOLVED",
    "EV_PARENT_UNKNOWN",
    "EV_NO_PARENT_DECLARED",
    "EV_CYCLE",
    "REASON_CODES",
    "known_schema_org_base",
    "find_type",
    "alias_owner",
    "ancestors_of",
    "sibling_types",
    "would_cycle",
    "place_proposed_type",
]
