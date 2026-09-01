"""Track N.4d.1 — placing a PROPOSED type in the existing hierarchy.

Run against the REAL shipped ontologies wherever the question is "does this hold
on our data", and against small hand-built ones only where a structure has to be
forced (cycles, aliases). That split is deliberate: N.4a shipped a regression test
that monkeypatched the very function it claimed to verify, and it would have
passed against any implementation. A stub here would hide exactly the thing these
tests exist to check — that the declared chains in `deals.yaml` really do bound
the candidate set.
"""

from __future__ import annotations

import pytest
from ontology_manager.schema import (
    EntityTypeDefinition,
    Ontology,
    OntologyMetadata,
)
from ontology_manager.type_placement import (
    CYCLIC,
    DUPLICATE,
    EV_ALIAS_TAKEN,
    EV_CYCLE,
    EV_NAME_IS_BASE,
    EV_NAME_TAKEN,
    EV_NO_NAME,
    EV_NO_PARENT_DECLARED,
    EV_NO_SCHEMAS,
    EV_PARENT_RESOLVED,
    EV_PARENT_UNKNOWN,
    PARENT_UNKNOWN,
    PLACED,
    UNPARENTED,
    alias_owner,
    ancestors_of,
    find_type,
    known_schema_org_base,
    place_proposed_type,
    resolve_parent,
    roots_at,
    sibling_types,
    would_cycle,
)


def _load(name):
    """Load an ontology through `registry.get`, which is what production calls.

    The first draft used the private `_load_from_file`, and the review measured
    the gap: raw `deals` has 8 types, the applied vocabulary has 53, because
    `metadata.extends: instruments` is resolved only by `get`. An assertion about
    "the real shipped ontology" made against the raw file is an assertion about
    something production never sees — the N.4a M2 lesson, one level down.
    """
    import asyncio
    from unittest.mock import AsyncMock, patch

    from ontology_manager.registry import OntologyRegistry

    # `get` is the production path and resolves `extends`; its DB probe is
    # stubbed out so the unit suite neither opens a socket per ontology (which
    # BLOCKS rather than fails fast where the port is filtered) nor silently
    # describes DB rows instead of the shipped vocabulary.
    loop = asyncio.new_event_loop()
    try:
        with patch.object(OntologyRegistry, "_load_from_db", AsyncMock(return_value=None)):
            ontology = loop.run_until_complete(OntologyRegistry().get(name))
    finally:
        loop.close()
    if ontology is None:  # pragma: no cover - the files ship with the package
        pytest.skip(f"{name} ontology not available")
    return [ontology]


@pytest.fixture(scope="module")
def deals():
    """The real `deals` vocabulary as applied: 53 types, inheritance resolved."""
    return _load("deals")


@pytest.fixture(scope="module")
def general():
    """`general` is DEFAULT_ONTOLOGY and roots every type by `schema_org_type`."""
    return _load("general")


def _ont(**types):
    return Ontology(
        metadata=OntologyMetadata(name="t", version="1"),
        entity_types={k: v for k, v in types.items()},
    )


# ---------------------------------------------------------------------------
# The bounded candidate set, on real declarations
# ---------------------------------------------------------------------------


def test_siblings_bound_the_candidates(deals):
    # The applied vocabulary declares exactly these three under Deal. A type
    # proposed under Deal can only be inserted between those and Deal.
    assert set(sibling_types("Deal", deals)) == {"RegioDeal", "Woondeal", "CityDeal"}


def test_candidates_are_the_shared_parents_children(deals):
    placement = place_proposed_type("Stadsdeal", "Deal", deals)
    assert placement.verdict == PLACED
    assert placement.reason_code == EV_PARENT_RESOLVED
    assert placement.parent == "Deal"
    assert set(placement.descendant_candidates) == {"RegioDeal", "Woondeal", "CityDeal"}


def test_proposal_excludes_itself(deals):
    # Re-proposing an existing sibling name would be a DUPLICATE, but the
    # exclusion is asserted directly so the guard cannot rot behind that check.
    assert "Woondeal" not in sibling_types("Deal", deals, exclude="Woondeal")


def test_ancestors_follow_declared_chain(deals):
    assert ancestors_of("RegioDeal", deals) == ["Deal", "GovernmentService"]
    assert ancestors_of("Deal", deals) == ["GovernmentService"]


def test_find_type_is_case_insensitive(deals):
    assert find_type("regiodeal", deals) is not None
    assert find_type("Onbekend Type", deals) is None


# ---------------------------------------------------------------------------
# A proposal that is not new
# ---------------------------------------------------------------------------


def test_existing_name_is_a_duplicate(deals):
    placement = place_proposed_type("RegioDeal", "Deal", deals)
    assert placement.verdict == DUPLICATE
    assert placement.reason_code == EV_NAME_TAKEN
    assert placement.duplicate_of == "RegioDeal"
    # a duplicate has no sibling set: it is not being placed at all
    assert placement.descendant_candidates == ()


def test_existing_alias_is_a_duplicate():
    schemas = [_ont(
        Deal=EntityTypeDefinition(name="Deal", parent_type="GovernmentService"),
        Akkoord=EntityTypeDefinition(name="Akkoord", parent_type="GovernmentService",
                                     aliases=["Convenant"]),
    )]
    assert alias_owner("convenant", schemas) == "Akkoord"
    placement = place_proposed_type("Convenant", "Deal", schemas)
    assert placement.verdict == DUPLICATE
    assert placement.reason_code == EV_ALIAS_TAKEN
    assert placement.duplicate_of == "Akkoord"


def test_name_and_alias_have_distinct_codes():
    # Different observations, so different codes: a curator merges a duplicate
    # name but may want to keep an alias distinct.
    schemas = [_ont(
        Deal=EntityTypeDefinition(name="Deal", aliases=["Overeenkomst"]),
    )]
    assert place_proposed_type("Deal", None, schemas).reason_code == EV_NAME_TAKEN
    assert place_proposed_type("Overeenkomst", None, schemas).reason_code == EV_ALIAS_TAKEN


# ---------------------------------------------------------------------------
# Falsifiable evidence (D-N4-7): each code has exactly one cause
# ---------------------------------------------------------------------------


def test_unknown_parent_is_unchecked(deals):
    placement = place_proposed_type("Stadsdeal", "Verzonnen", deals)
    assert placement.verdict == PARENT_UNKNOWN
    assert placement.reason_code == EV_PARENT_UNKNOWN
    assert "could not be checked" in placement.evidence
    assert placement.descendant_candidates == ()


def test_no_declared_parent_is_undecided(deals):
    placement = place_proposed_type("Stadsdeal", None, deals)
    assert placement.verdict == UNPARENTED
    assert placement.reason_code == EV_NO_PARENT_DECLARED
    assert "undecided, not top-level" in placement.evidence


def test_no_ontologies_claims_nothing():
    placement = place_proposed_type("Stadsdeal", "Deal", [])
    assert placement.reason_code == EV_NO_SCHEMAS
    assert "never compared against anything" in placement.evidence
    assert "says nothing about whether it is new" in placement.evidence


def test_nameless_proposal_says_so(deals):
    placement = place_proposed_type("   ", "Deal", deals)
    assert placement.reason_code == EV_NO_NAME


def test_every_verdict_carries_evidence(deals):
    for name, parent in [("Stadsdeal", "Deal"), ("RegioDeal", "Deal"),
                         ("Stadsdeal", "Verzonnen"), ("Stadsdeal", None),
                         ("   ", "Deal")]:
        placement = place_proposed_type(name, parent, deals)
        assert placement.reason_code, (name, parent)
        assert placement.evidence, (name, parent)


# ---------------------------------------------------------------------------
# Structural guards
# ---------------------------------------------------------------------------


def test_cycle_is_refused():
    # Reachable for N.4d.3's re-parent of an EXISTING type, not for a new one.
    schemas = [_ont(
        A=EntityTypeDefinition(name="A", parent_type="B"),
        B=EntityTypeDefinition(name="B"),
    )]
    assert would_cycle("A", "A", schemas) is True
    assert would_cycle("B", "A", schemas) is True   # A already descends from B
    assert would_cycle("A", "B", schemas) is False


def test_duplicate_is_reported_before_cycle():
    """Precedence, not an oversight: proposing "Deal" under "Deal" is caught as a
    DUPLICATE rather than CYCLIC. "This type already exists" is the more specific
    and more actionable observation — a curator merges or rejects it, and the
    cycle is moot because the proposal never becomes a new type."""
    schemas = [_ont(Deal=EntityTypeDefinition(name="Deal"))]
    assert place_proposed_type("Deal", "Deal", schemas).verdict == DUPLICATE


def test_authored_loop_cannot_hang():
    schemas = [_ont(
        A=EntityTypeDefinition(name="A", parent_type="B"),
        B=EntityTypeDefinition(name="B", parent_type="A"),
    )]
    assert ancestors_of("A", schemas) == ["B"]


def test_chain_may_leave_applied_set(deals):
    # GovernmentService is declared as a parent in deals.yaml but defined
    # elsewhere; the walk stops there instead of erroring.
    assert ancestors_of("Deal", deals) == ["GovernmentService"]
    assert find_type("GovernmentService", deals) is None


def test_malformed_members_are_skipped():
    class _Broken:
        entity_types = "not a dict"

    assert find_type("Deal", [_Broken()]) is None
    assert sibling_types("Deal", [_Broken()]) == ()


# ---------------------------------------------------------------------------
# The two ways a parent can be valid
# ---------------------------------------------------------------------------


def test_schema_org_base_is_valid_parent(deals):
    """`deals.yaml` roots half its types at bases no ontology DEFINES.

    `canonical_bridge` terminates its walk on the base name itself, so treating
    such a declaration as unknown would call four of eight shipped declarations
    broken when they are exactly how the vocabulary is meant to be authored.
    Found by running against the real ontology rather than a fixture.
    """
    assert find_type("GovernmentService", deals) is None  # not defined...
    assert known_schema_org_base("GovernmentService") is True  # ...but valid

    placement = place_proposed_type("Streekakkoord", "GovernmentService", deals)
    assert placement.verdict == PLACED
    assert "schema.org base" in placement.evidence
    # The APPLIED vocabulary, not the raw file: `deals` extends `instruments`,
    # which contributes five more children of GovernmentService.
    assert set(placement.descendant_candidates) == {
        "Deal", "Akkoord", "BeleidsProgramma",
        "Subsidieregeling", "Fonds", "Investering", "Convenant", "Bestuursakkoord",
    }
    # NOT the grandchildren: re-parenting those would move them away from their
    # own parent, which is a different and unsound edit.
    assert "RegioDeal" not in placement.descendant_candidates


def test_evidence_names_the_parent_kind(deals):
    defined = place_proposed_type("Stadsdeal", "Deal", deals)
    base = place_proposed_type("Streekakkoord", "GovernmentService", deals)
    assert "defined in an applied ontology" in defined.evidence
    assert "schema.org base" in base.evidence


def test_invented_parent_is_neither(deals):
    placement = place_proposed_type("Stadsdeal", "Verzonnen", deals)
    assert placement.verdict == PARENT_UNKNOWN
    assert known_schema_org_base("Verzonnen") is False
    assert "not a schema.org base" in placement.evidence


# ---------------------------------------------------------------------------
# Agreement with canonical_bridge — pinned, not promised
# ---------------------------------------------------------------------------


def test_agrees_with_the_bridge(general, deals):
    """This module's value is that it predicts what the bridge will do.

    The first draft only *claimed* the two could not drift, and drifted three
    ways: it normalised case where the bridge's map lookup is case-sensitive, it
    did not strip the `schema:` prefix that `base.yaml` writes, and it ignored
    aliases in the parent slot although the bridge resolves them. Asserted here so
    a future divergence is a test failure rather than a docstring that is quietly
    wrong.
    """
    from ontology_manager.canonical_bridge import resolve_ontology_type

    # a prefixed base: base.yaml writes `schema:Person`
    assert known_schema_org_base("schema:Person") is True
    # case-sensitivity mirrors the bridge's own dict lookup
    assert known_schema_org_base("GovernmentService") is True
    assert known_schema_org_base("governmentservice") is False
    # an alias the bridge resolves must resolve here too
    assert resolve_ontology_type("Theme", general) is not None
    assert resolve_parent("Theme", general)[1] == "Topic"


def test_schema_org_rooted_types_are_enumerated(general):
    """`general` is DEFAULT_ONTOLOGY and declares ZERO `parent_type`.

    Every one of its types roots via `schema_org_type`, so an enumerator reading
    only `parent_type` returns nothing there — while the evidence claims it found
    the only candidates. That was the blocker; `roots_at` closes it.
    """
    assert all(d.parent_type is None for d in general[0].entity_types.values())
    assert sibling_types("DefinedTerm", general) == ("Topic",)

    placement = place_proposed_type("Vakgebied", "DefinedTerm", general)
    assert placement.verdict == PLACED
    assert placement.descendant_candidates == ("Topic",)


# ---------------------------------------------------------------------------
# Precedence and the structural guards
# ---------------------------------------------------------------------------


def test_a_referenced_but_undefined_name_can_cycle(deals):
    """`would_cycle` is reachable from a proposal, contrary to the first draft.

    `GovernmentService` is referenced as a parent but defined nowhere, so it is
    neither plainly existing nor plainly new — the third state the binary
    reasoning missed. It is caught earlier as a DUPLICATE (it is a mapped base),
    so the cycle check is reached through `would_cycle` directly.
    """
    assert would_cycle("GovernmentService", "Deal", deals) is True
    assert place_proposed_type("GovernmentService", "Deal", deals).verdict == DUPLICATE


def test_cyclic_verdict_is_reachable():
    """Asserts the VERDICT, not just the helper.

    The first version of this test asserted PLACED and `would_cycle(...) is True`,
    so deleting the whole CYCLIC branch left it green — a test named for a verdict
    it never produced, which is the same defect corrected one commit earlier in
    this file. "Ghost" is referenced as A's parent and defined nowhere, and is not
    a mapped base, so it survives the duplicate checks and reaches the cycle test.
    """
    schemas = [_ont(A=EntityTypeDefinition(name="A", parent_type="Ghost"))]
    placement = place_proposed_type("Ghost", "A", schemas)
    assert placement.verdict == CYCLIC
    assert placement.reason_code == EV_CYCLE
    assert placement.parent == "A"


def test_name_check_precedes_alias_check():
    """Both observations are true for the same string; the more direct one wins.

    Unasserted in the first draft, so swapping the two checks changed nothing —
    the evidence would then report the alias owner while a type of that exact
    name existed.
    """
    schemas = [_ont(
        Deal=EntityTypeDefinition(name="Deal"),
        Akkoord=EntityTypeDefinition(name="Akkoord", aliases=["Deal"]),
    )]
    placement = place_proposed_type("Deal", None, schemas)
    assert placement.reason_code == EV_NAME_TAKEN
    assert placement.duplicate_of == "Deal"


def test_a_mapped_base_is_not_reportable_as_new(deals):
    """Symmetry: the module accepts a base as an existing PARENT, so it must not
    call the same string a new TYPE in the name slot."""
    placement = place_proposed_type("GovernmentService", "Deal", deals)
    assert placement.verdict == DUPLICATE
    # Its OWN code: "already defined" and "is a mapped base" are different
    # observations, and gap recording will gate on these.
    assert placement.reason_code == EV_NAME_IS_BASE
    assert placement.reason_code != EV_NAME_TAKEN
    # ...and no merge target, because nothing defines it
    assert placement.duplicate_of is None
    assert "no definition to merge into" in placement.evidence


def test_sibling_dedup_is_case_insensitive():
    # Spelled DEAL, not `deal`: with the lowercase spelling a case-SENSITIVE
    # dedup coincidentally returns one entry too, so the test passed against the
    # implementation it claims to exclude.
    schemas = [
        _ont(Deal=EntityTypeDefinition(name="Deal", parent_type="X")),
        _ont(DEAL=EntityTypeDefinition(name="DEAL", parent_type="X")),
    ]
    assert len(sibling_types("X", schemas)) == 1


# ---------------------------------------------------------------------------
# The safety invariant, over every shipped ontology
# ---------------------------------------------------------------------------


ALL_ONTOLOGIES = [
    "base", "deals", "general", "government", "instruments", "policy",
    "policy_themes", "regiodeal", "schema_core", "scholarly", "social_profiles",
]


def test_no_candidate_can_close_a_cycle():
    """The property the candidate set must have, measured on the real vocabulary.

    A proposal P is new and therefore has no descendants, so the ONLY way that
    accepting a candidate C as P's child can create a cycle is C == P's parent.
    Excluding that is necessary and sufficient — so this asserts it for every
    parent in every shipped ontology rather than for one hand-picked case.

    It exists because the previous fix was measured on the DEFECT but not on the
    FIX: stripping the `schema:` prefix made `Person` (which declares
    `schema_org_type: schema:Person`) root at itself, so the declared parent was
    offered as a candidate to become the proposal's child. Four of `general`'s
    eight types were affected.
    """
    checked = 0
    for name in ALL_ONTOLOGIES:
        schemas = _load(name)
        parents = {roots_at(d) for d in schemas[0].entity_types.values() if roots_at(d)}
        for parent in parents:
            for candidate in sibling_types(parent, schemas):
                assert candidate.lower() != parent.lower(), (name, parent, candidate)
                checked += 1
    assert checked > 200, f"expected a substantial sample, enumerated {checked}"


def test_a_self_rooting_type_is_not_its_own_sibling(general):
    # `general` roots Person/Organization/Event/Product at themselves via
    # schema_org_type, which is what made this reachable.
    self_rooting = [
        n for n, d in general[0].entity_types.items() if (roots_at(d) or "") == n
    ]
    assert self_rooting, "fixture no longer exercises the self-rooting case"
    for name in self_rooting:
        assert name not in sibling_types(name, general)
        assert name not in place_proposed_type(
            "Nieuw" + name, name, general
        ).descendant_candidates
