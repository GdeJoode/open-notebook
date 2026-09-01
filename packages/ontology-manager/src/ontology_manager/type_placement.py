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
SIBLING of ``P``. A grandchild is excluded not because of any breadth relation
between it and ``P``, which nothing establishes, but because re-parenting it would
move it away from ITS own parent, which is a different and unsound edit.

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


def _strip_schema_prefix(value: str) -> str:
    """``"schema:Person"`` → ``"Person"``. Mirrors ``canonical_bridge``."""
    v = (value or "").strip()
    return v.split(":", 1)[1].strip() if ":" in v else v


def known_schema_org_base(name: str) -> bool:
    """True when ``name`` is a schema.org base ``canonical_bridge`` maps.

    A parent is valid in two ways, and conflating them misreports much of the
    shipped vocabulary: ``deals.yaml`` roots types at ``GovernmentService``, which
    no ontology DEFINES — the bridge terminates its walk on the base name itself.

    The lookup mirrors the bridge EXACTLY: strip a ``schema:`` prefix (``base.yaml``
    writes ``schema:Person``), then an exact-case map lookup, because the bridge's
    own ``_CANONICAL_BY_SCHEMA_ORG.get(base)`` is case-sensitive. Being lenient
    here would be worse than being strict: the value of this module is that it
    predicts what the bridge will do, so accepting a spelling the bridge rejects
    would make the placement a lie about the outcome. ``test_agrees_with_the_bridge``
    pins that agreement so the two cannot drift.
    """
    base = _strip_schema_prefix(name)
    if not base:
        return False
    try:
        from ontology_manager.canonical_bridge import _CANONICAL_BY_SCHEMA_ORG
    except Exception as exc:  # noqa: BLE001 - degrade, never raise on curator input
        logger.debug("type_placement: schema.org base map unavailable ({e})", e=exc)
        return False
    return base in _CANONICAL_BY_SCHEMA_ORG


def roots_at(definition: Any) -> Optional[str]:
    """What a type hangs from: its ``schema_org_type`` base, else ``parent_type``.

    Ordered the way ``canonical_bridge.resolve_ontology_type`` orders it — an
    explicit ``schema_org_type`` wins over the parent walk. Reading only
    ``parent_type`` would have missed the entire default vocabulary: ``general``
    and ``base`` declare **zero** ``parent_type`` and root all their types by
    ``schema_org_type``, so a sibling enumeration blind to it returns nothing on
    the default ontology while claiming to have found the only candidates.
    """
    explicit = getattr(definition, "schema_org_type", None)
    if explicit:
        return _strip_schema_prefix(str(explicit))
    parent = getattr(definition, "parent_type", None)
    return str(parent) if parent else None


def find_type(name: str, schemas: Optional[Sequence[Any]]) -> Optional[Any]:
    """The definition whose NAME matches ``name``, case-insensitively.

    Aliases are deliberately not matched here: in the NAME slot "this name is
    taken" and "this name is somebody's alias" are different observations a
    curator acts on differently, and :func:`alias_owner` reports the second. In
    the PARENT slot that distinction has no force, so :func:`resolve_parent` does
    match aliases — mirroring ``canonical_bridge._find_definition``.
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


def resolve_parent(
    name: str, schemas: Optional[Sequence[Any]]
) -> Tuple[Optional[Any], Optional[str]]:
    """``(definition, resolved_name)`` for a declared parent, as the bridge sees it.

    Matches a type NAME or one of its ALIASES, case-insensitively — exactly
    ``canonical_bridge._find_definition``. Aliases matter here because the parent
    slot is filled from free text: ``evolution.create_proposal_from_gap`` writes
    ``definition["parent_type"] = gap.entity_type_guess``, an LLM's words, and
    ``general.Topic`` really does ship ``aliases: [Subject, Theme, Category]``.
    Refusing "Theme" as unknown while the bridge resolves it to ``Topic`` would be
    a false negative about the bridge's own behaviour.
    """
    definition = find_type(name, schemas)
    if definition is not None:
        return definition, str(getattr(definition, "name", None) or name)
    owner = alias_owner(name, schemas)
    if owner is not None:
        return find_type(owner, schemas), owner
    return None, None


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
        parent = roots_at(current) if current is not None else None
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

    "Declare" here means :func:`roots_at` — an explicit ``schema_org_type`` base
    or, failing that, ``parent_type`` — matching how the bridge resolves a type.
    This is the whole bounded candidate set for BROADER_THAN: a proposed type can
    only be inserted between these and their shared parent. Order follows the
    applied ontologies so the result is stable for a given input.
    """
    target = _norm(parent)
    skip = _norm(exclude)
    if not target:
        return ()
    out: List[str] = []
    seen: set = set()
    for type_name, definition in _iter_types(schemas):
        if _norm(type_name) == skip:
            continue
        if _norm(roots_at(definition)) == target and _norm(type_name) not in seen:
            seen.add(_norm(type_name))
            out.append(type_name)
    return tuple(out)


def would_cycle(
    child: str, new_parent: str, schemas: Optional[Sequence[Any]]
) -> bool:
    """True when making ``new_parent`` the parent of ``child`` closes a loop.

    That happens when ``new_parent`` already descends from ``child``.

    It is NOT unreachable for a proposal, which an earlier draft of this docstring
    claimed, reasoning that nothing can descend from what does not exist yet. That
    assumes a binary exists/does-not-exist, and this module's own model has a third
    state: a name REFERENCED as a parent but DEFINED nowhere. N.4d.3's re-parent of
    an existing type reaches this check too.
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
    if known_schema_org_base(name):
        # Symmetry: this module accepts a schema.org base as an EXISTING type in
        # the parent slot, so it must not report the same string as new in the
        # name slot. GovernmentService is the live case — no ontology defines it,
        # yet the bridge maps it to the `programme` canonical.
        return TypePlacement(
            verdict=DUPLICATE,
            reason_code=EV_NAME_TAKEN,
            evidence=(
                f"{name!r} is a schema.org base the canonical bridge already maps, "
                "so it is an existing type rather than a new one — even though no "
                "applied ontology defines it"
            ),
            duplicate_of=_strip_schema_prefix(name),
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
    parent_def, resolved = resolve_parent(declared_parent, schemas)
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
    parent_name = resolved or _strip_schema_prefix(str(declared_parent))
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
    "roots_at",
    "resolve_parent",
    "find_type",
    "alias_owner",
    "ancestors_of",
    "sibling_types",
    "would_cycle",
    "place_proposed_type",
]
