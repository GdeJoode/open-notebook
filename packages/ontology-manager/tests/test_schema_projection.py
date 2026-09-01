"""Track N.4d.3 — projecting a notebook's accepted edits onto its ontologies.

Run against the REAL shipped ontologies wherever the question is "does this hold
on our data", in the shape production assembles them (`detect_applicable_schemas`
uses `top_k=3`, so an applied set holds THREE ontologies). N.4d.2 was rejected
for asserting a cross-ontology property against a single-ontology load, where it
held by the name-keyed `entity_types` dict rather than by the mechanism.

Hand-built ontologies appear only where a structure has to be FORCED — an
orphaning chain, a cycle — which the shipped vocabulary does not contain.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from ontology_manager.canonical_bridge import resolve_ontology_type
from ontology_manager.schema import (
    EntityTypeDefinition,
    Ontology,
    OntologyMetadata,
)
from ontology_manager.schema_projection import (
    EV_CHAIN_ORPHANS,
    EV_CYCLE,
    EV_DEFINED,
    EV_NAME_ALREADY_DEFINED,
    EV_NO_SCHEMAS,
    EV_NO_TYPE_NAME,
    EV_PARENT_NOT_FOUND,
    EV_PARENT_REWRITTEN,
    EV_TYPE_NOT_FOUND,
    MATERIALISED,
    REFUSED,
    REPARENTED,
    project_accepted_edits,
)

# Every ontology that ships with the package, so a sweep covers the vocabulary a
# notebook can actually be given.
ALL = (
    "base",
    "deals",
    "general",
    "government",
    "instruments",
    "policy",
    "policy_themes",
    "regiodeal",
    "scholarly",
    "schema_core",
    "social_profiles",
)


def _load(*names):
    """Load through `registry.get`, which is what production calls.

    `get` resolves `metadata.extends`; `_load_from_file` does not, and the two
    disagree by 45 types on `deals` alone. Its DB probe is stubbed so the unit
    suite neither opens a socket per ontology nor describes DB rows instead of
    the shipped vocabulary.
    """
    from ontology_manager.registry import OntologyRegistry

    loop = asyncio.new_event_loop()
    try:
        with patch.object(
            OntologyRegistry, "_load_from_db", AsyncMock(return_value=None)
        ):
            registry = OntologyRegistry()
            loaded = [
                loop.run_until_complete(registry.get(name)) for name in names
            ]
    finally:
        loop.close()
    if any(o is None for o in loaded):  # pragma: no cover - files ship with the package
        pytest.skip(f"ontologies not available: {names}")
    return loaded


@pytest.fixture(scope="module")
def applied():
    """A production-shaped applied set: three ontologies, inheritance resolved."""
    return _load("general", "deals", "government")


def _ont(name="t", **types):
    return Ontology(
        metadata=OntologyMetadata(name=name, version="1"),
        entity_types=dict(types),
    )


def _reparent(new_parent, *type_names):
    """One entry per moved type — the shape `SchemaEditService.reparent_type` records."""
    return [
        {
            "reparent_id": f"reparent::{name}->{new_parent}",
            "op": "reparent",
            "type_name": name,
            "new_parent": new_parent,
            "parent_type": new_parent,
        }
        for name in type_names
    ]


def _extension(type_name, parent_type=None, **extra):
    entry = {"extension_id": f"ext:{type_name}", "type_name": type_name}
    if parent_type is not None:
        entry["parent_type"] = parent_type
    entry.update(extra)
    return entry


# ---------------------------------------------------------------------------
# The registry's objects are shared; the projection must not touch them
# ---------------------------------------------------------------------------


def test_the_inputs_are_left_untouched(applied):
    """Measured before this module was written: `registry.get` returns the SAME
    object on a second call, `entity_types` dict included. So an in-place edit
    would give every other notebook in the process one notebook's vocabulary.
    """
    before = {
        id(ontology): {
            name: (d.parent_type, d.schema_org_type)
            for name, d in ontology.entity_types.items()
        }
        for ontology in applied
    }

    result = project_accepted_edits(applied, _reparent("Technology", "Person"))
    assert [o.action for o in result.outcomes] == [REPARENTED]

    after = {
        id(ontology): {
            name: (d.parent_type, d.schema_org_type)
            for name, d in ontology.entity_types.items()
        }
        for ontology in applied
    }
    assert after == before
    assert all(
        projected is not original
        for projected, original in zip(result.schemas, applied)
    )


def test_a_second_load_still_sees_the_shipped_declaration(applied):
    """The stronger form of the same property, stated the way the leak would be
    noticed in production: the NEXT notebook asks the registry for `general` and
    must get `Person` as its author wrote it.
    """
    project_accepted_edits(applied, _reparent("Technology", "Person"))
    fresh, = _load("general")
    assert fresh.entity_types["Person"].schema_org_type == "schema:Person"
    assert fresh.entity_types["Person"].parent_type is None


# ---------------------------------------------------------------------------
# Why a re-parent clears schema_org_type
# ---------------------------------------------------------------------------


def test_a_reparent_moves_the_canonical_of_a_type_rooted_by_its_base(applied):
    """`Person` roots at `schema:Person`, and the bridge PREFERS that field over
    `parent_type` — so rewriting the parent alone leaves the canonical exactly
    where it was. This is the assertion that dies if the clear is removed.
    """
    assert resolve_ontology_type("Person", applied).canonical == "person"

    result = project_accepted_edits(applied, _reparent("Technology", "Person"))

    assert resolve_ontology_type("Person", result.schemas).canonical == "technology"


def test_every_base_rooted_type_actually_follows_its_new_parent(applied):
    """The property over the whole applied set rather than one pinned case.

    Across the eleven shipped ontologies 13 of 277 type entries declare a
    `schema_org_type`, and for 10 of them a `parent_type` rewrite alone is a
    no-op. Each one is a root a curator plausibly re-parents, so the sweep asserts
    on all of them present here — and asserts the set is non-empty, because a
    vocabulary change that removed the last such type would otherwise leave this
    test green and meaningless.
    """
    rooted = [
        name
        for ontology in applied
        for name, d in ontology.entity_types.items()
        if d.schema_org_type
    ]
    assert rooted, "no type in the applied set roots at a schema.org base"

    moved = 0
    for name in rooted:
        before = resolve_ontology_type(name, applied)
        result = project_accepted_edits(applied, _reparent("Location", name))
        outcome, = result.outcomes
        if outcome.action == REFUSED:
            # Only ever because the move would close a loop or orphan the chain;
            # never because the edit silently failed to take.
            assert outcome.reason_code in (EV_CYCLE, EV_CHAIN_ORPHANS)
            continue
        after = resolve_ontology_type(name, result.schemas)
        assert after is not None
        assert after.canonical == resolve_ontology_type("Location", applied).canonical
        if before is not None and name != "Location":
            assert after.canonical != before.canonical
            moved += 1
    # Without a floor this test passes when EVERY move is refused — a guard that
    # only ever measures the direction that is not at risk (the N.4d.2 lesson).
    assert moved >= 3, f"only {moved} of {len(rooted)} base-rooted types moved"


# ---------------------------------------------------------------------------
# Refusals name an observation
# ---------------------------------------------------------------------------


def test_a_type_that_is_not_defined_is_refused(applied):
    result = project_accepted_edits(applied, _reparent("Person", "NotATypeAnywhere"))
    outcome, = result.outcomes
    assert (outcome.action, outcome.reason_code) == (REFUSED, EV_TYPE_NOT_FOUND)


def test_a_parent_that_is_neither_defined_nor_a_base_is_refused(applied):
    result = project_accepted_edits(applied, _reparent("NoSuchParent", "Person"))
    outcome, = result.outcomes
    assert (outcome.action, outcome.reason_code) == (REFUSED, EV_PARENT_NOT_FOUND)
    assert "NoSuchParent" in outcome.detail


def test_a_parent_that_is_a_mapped_base_is_accepted_without_being_defined(applied):
    """`GovernmentService` is declared only as a PARENT in the shipped YAML — no
    definition exists in this applied set — but the bridge maps it, so it is a
    legitimate destination. (`Deal` reads like the same case and is not: `deals`
    defines it. Measured rather than assumed, after the first draft asserted the
    wrong one.)
    """
    from ontology_manager.type_placement import find_type

    assert find_type("GovernmentService", applied) is None
    result = project_accepted_edits(applied, _reparent("GovernmentService", "Person"))
    outcome, = result.outcomes
    assert (outcome.action, outcome.reason_code) == (REPARENTED, EV_PARENT_REWRITTEN)
    assert resolve_ontology_type("Person", result.schemas).canonical == "programme"


def test_a_move_under_ones_own_descendant_is_refused(applied):
    """Real chains, not a forced one: pick a type whose parent is defined and try
    to hang the parent under it.
    """
    pairs = [
        (name, d.parent_type)
        for ontology in applied
        for name, d in ontology.entity_types.items()
        if d.parent_type
    ]
    child, parent = next(
        (c, p)
        for c, p in pairs
        if any(p in o.entity_types for o in applied)
    )
    result = project_accepted_edits(applied, _reparent(child, parent))
    outcome, = result.outcomes
    assert (outcome.action, outcome.reason_code) == (REFUSED, EV_CYCLE)


def test_a_move_that_orphans_the_chain_is_refused_and_rolled_back():
    """Forced structure: `Loose` reaches no mapped base, so moving a resolvable
    type under it would drop every entity of that type onto the alias fallback.
    """
    ontology = _ont(
        Person=EntityTypeDefinition(name="Person", schema_org_type="schema:Person"),
        Loose=EntityTypeDefinition(name="Loose"),
    )
    assert resolve_ontology_type("Person", [ontology]).canonical == "person"

    result = project_accepted_edits([ontology], _reparent("Loose", "Person"))
    outcome, = result.outcomes
    assert (outcome.action, outcome.reason_code) == (REFUSED, EV_CHAIN_ORPHANS)
    # Rolled back on the COPY too — a refusal leaves no half-applied edit behind.
    assert resolve_ontology_type("Person", result.schemas).canonical == "person"


def test_a_reparent_with_no_parent_named_is_refused(applied):
    result = project_accepted_edits(
        applied, [{"op": "reparent", "type_name": "Person"}]
    )
    outcome, = result.outcomes
    assert (outcome.action, outcome.reason_code) == (REFUSED, EV_PARENT_NOT_FOUND)


def test_edits_without_any_applied_schema_are_reported_not_dropped():
    result = project_accepted_edits([], _reparent("Thing", "Person"))
    assert [(o.action, o.reason_code) for o in result.outcomes] == [
        (REFUSED, EV_NO_SCHEMAS)
    ]


def test_no_schemas_and_no_edits_is_silent():
    assert project_accepted_edits([], []).outcomes == []


# ---------------------------------------------------------------------------
# Malformed rows
# ---------------------------------------------------------------------------


def test_a_reparent_without_a_type_name_is_refused(applied):
    result = project_accepted_edits(
        applied, [{"op": "reparent", "new_parent": "Technology"}]
    )
    outcome, = result.outcomes
    assert (outcome.action, outcome.reason_code) == (REFUSED, EV_NO_TYPE_NAME)


def test_a_non_string_type_name_is_refused(applied):
    result = project_accepted_edits(
        applied, [{"op": "reparent", "new_parent": "Technology", "type_name": ["Person"]}]
    )
    outcome, = result.outcomes
    assert (outcome.action, outcome.reason_code) == (REFUSED, EV_NO_TYPE_NAME)
    assert resolve_ontology_type("Person", result.schemas).canonical == "person"


def test_a_non_string_parent_is_refused(applied):
    result = project_accepted_edits(
        applied,
        [{"op": "reparent", "type_name": "Person", "new_parent": {"Technology": True}}],
    )
    outcome, = result.outcomes
    assert (outcome.action, outcome.reason_code) == (REFUSED, EV_PARENT_NOT_FOUND)


def test_the_last_recorded_move_wins(applied):
    """A type re-parented twice ends under the parent named LAST — the curator's
    most recent decision, not the first one they ever made.
    """
    result = project_accepted_edits(
        applied, _reparent("Technology", "Person") + _reparent("Deal", "Person")
    )
    assert [o.action for o in result.outcomes] == [REPARENTED, REPARENTED]
    assert resolve_ontology_type("Person", result.schemas).canonical == "programme"


# ---------------------------------------------------------------------------
# Accepting an extension makes it resolvable
# ---------------------------------------------------------------------------


def test_an_accepted_extension_becomes_resolvable(applied):
    """Before this phase an accepted extension was rendered into the Pass-2
    prompt and the TTL but was invisible to the bridge, so entities carrying its
    label fell to the alias fallback. Materialising it is the behaviour change
    this module documents.
    """
    assert resolve_ontology_type("Regiodeal Tranche", applied) is None

    result = project_accepted_edits(
        applied, [_extension("Regiodeal Tranche", parent_type="Deal")]
    )
    outcome, = result.outcomes
    assert (outcome.action, outcome.reason_code) == (MATERIALISED, EV_DEFINED)

    resolved = resolve_ontology_type("Regiodeal Tranche", result.schemas)
    assert resolved.canonical == "programme"
    assert resolved.ontology_type == "Regiodeal Tranche"


def test_a_shipped_definition_is_never_overwritten(applied):
    result = project_accepted_edits(
        applied, [_extension("Person", parent_type="Technology")]
    )
    outcome, = result.outcomes
    assert (outcome.action, outcome.reason_code) == (REFUSED, EV_NAME_ALREADY_DEFINED)
    assert resolve_ontology_type("Person", result.schemas).canonical == "person"


def test_an_entry_without_a_type_name_is_refused(applied):
    result = project_accepted_edits(applied, [{"extension_id": "ext:1"}])
    outcome, = result.outcomes
    assert (outcome.action, outcome.reason_code) == (REFUSED, EV_NO_TYPE_NAME)


def test_a_type_can_be_moved_under_a_parent_accepted_in_the_same_pass(applied):
    """Materialisation runs first for exactly this case; the reverse order refuses
    the move for a parent that does exist.
    """
    result = project_accepted_edits(
        applied,
        [
            _extension("Regional Programme", parent_type="Deal"),
            *_reparent("Regional Programme", "Person"),
        ],
    )
    assert [o.action for o in result.outcomes] == [MATERIALISED, REPARENTED]
    assert resolve_ontology_type("Person", result.schemas).canonical == "programme"


def test_the_extension_lands_on_the_schema_it_names(applied):
    result = project_accepted_edits(
        applied, [_extension("Tranche", parent_type="Deal", schema_name="deals")]
    )
    holder = next(
        o for o in result.schemas if o.metadata.name == "deals"
    )
    assert "Tranche" in holder.entity_types


def test_an_unnamed_schema_lands_on_the_first_applied_ontology(applied):
    result = project_accepted_edits(applied, [_extension("Tranche", parent_type="Deal")])
    assert "Tranche" in result.schemas[0].entity_types


# ---------------------------------------------------------------------------
# Entries this module must NOT read as type declarations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entry",
    [
        {"op": "rename", "old_name": "A", "new_name": "B", "type_name": "B"},
        {"op": "merge", "source_types": ["A", "B"], "type_name": "AB"},
        {"op": "split", "source_type": "A", "into": ["B", "C"], "type_name": "A"},
        {"op": "delete", "type_name": "A"},
    ],
)
def test_a_recorded_op_is_not_a_type_declaration(applied, entry):
    """Reading these as declarations would invent a vocabulary nobody accepted —
    a merge would define `AB`, a rename would define `B`.
    """
    assert project_accepted_edits(applied, [entry]).outcomes == []


def test_the_resume_sentinel_is_filtered(applied):
    """The sixth sentinel filter site; `routers/schemas.py` requires every new
    `accepted_extensions` consumer to add one.
    """
    sentinel = {
        "type_name": "_resumed_without_extensions",
        "is_resume_sentinel": True,
    }
    assert project_accepted_edits(applied, [sentinel]).outcomes == []


def test_a_non_dict_entry_is_ignored(applied):
    assert project_accepted_edits(applied, ["Person", None, 7]).outcomes == []


# ---------------------------------------------------------------------------
# Re-running changes nothing (the plan's second acceptance criterion)
# ---------------------------------------------------------------------------


def test_projecting_twice_gives_the_same_vocabulary(applied):
    edits = [
        _extension("Regional Programme", parent_type="Deal"),
        *_reparent("Regional Programme", "Person", "Organization"),
    ]
    first = project_accepted_edits(applied, edits)
    second = project_accepted_edits(applied, edits)

    def _shape(projection):
        return [
            {
                name: (d.parent_type, d.schema_org_type)
                for name, d in ontology.entity_types.items()
            }
            for ontology in projection.schemas
        ]

    assert _shape(first) == _shape(second)
    assert [(o.type_name, o.action) for o in first.outcomes] == [
        (o.type_name, o.action) for o in second.outcomes
    ]


def test_projecting_the_projection_is_a_fixed_point(applied):
    """Re-running against ALREADY-projected schemas: the materialisation is now a
    name collision and the re-parent is already in place, so nothing moves twice.
    """
    edits = _reparent("Technology", "Person")
    once = project_accepted_edits(applied, edits)
    twice = project_accepted_edits(once.schemas, edits)
    assert resolve_ontology_type("Person", twice.schemas).canonical == "technology"


# ---------------------------------------------------------------------------
# Whole-vocabulary safety
# ---------------------------------------------------------------------------


def test_no_reparent_of_any_shipped_type_can_silently_lose_its_canonical():
    """The safety property, swept over every shipped ontology in production-shaped
    applied sets of three: after projecting a re-parent, a type that resolved to a
    canonical before still resolves to one — or the edit was REFUSED and named
    why. No type may resolve before and quietly stop resolving after.
    """
    sets = [ALL[i : i + 3] for i in range(0, len(ALL), 3)]
    checked = applied_count = refused = 0
    for names in sets:
        schemas = _load(*names)
        for ontology in schemas:
            for type_name in list(ontology.entity_types):
                before = resolve_ontology_type(type_name, schemas)
                if before is None:
                    continue
                result = project_accepted_edits(schemas, _reparent("Person", type_name))
                outcome, = result.outcomes
                checked += 1
                if outcome.action == REFUSED:
                    refused += 1
                    continue
                applied_count += 1
                assert resolve_ontology_type(type_name, result.schemas) is not None, (
                    f"{type_name} resolved before the re-parent and not after"
                )
    assert checked > 200, f"the sweep only reached {checked} types"
    # Same floor, same reason: a projection that refused everything would satisfy
    # the safety property above and deliver nothing.
    assert applied_count > checked * 0.8, (
        f"{applied_count} applied vs {refused} refused of {checked}"
    )
