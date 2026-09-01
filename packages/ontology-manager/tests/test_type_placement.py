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
    sibling_types,
    would_cycle,
)


@pytest.fixture(scope="module")
def deals():
    """The real shipped `deals` ontology, loaded the way production loads it."""
    from ontology_manager.registry import OntologyRegistry

    ontology = OntologyRegistry()._load_from_file("deals")
    if ontology is None:  # pragma: no cover - the file ships with the package
        pytest.skip("deals ontology not available")
    return [ontology]


def _ont(**types):
    return Ontology(
        metadata=OntologyMetadata(name="t", version="1"),
        entity_types={k: v for k, v in types.items()},
    )


# ---------------------------------------------------------------------------
# The bounded candidate set, on real declarations
# ---------------------------------------------------------------------------


def test_siblings_bound_the_candidate_set_on_the_real_ontology(deals):
    # deals.yaml declares RegioDeal, Woondeal and CityDeal under Deal. A type
    # proposed under Deal can only be inserted between those and Deal.
    assert set(sibling_types("Deal", deals)) == {"RegioDeal", "Woondeal", "CityDeal"}


def test_a_proposal_under_deal_gets_exactly_those_candidates(deals):
    placement = place_proposed_type("Stadsdeal", "Deal", deals)
    assert placement.verdict == PLACED
    assert placement.reason_code == EV_PARENT_RESOLVED
    assert placement.parent == "Deal"
    assert set(placement.descendant_candidates) == {"RegioDeal", "Woondeal", "CityDeal"}


def test_the_proposal_never_lists_itself_as_its_own_candidate(deals):
    # Re-proposing an existing sibling name would be a DUPLICATE, but the
    # exclusion is asserted directly so the guard cannot rot behind that check.
    assert "Woondeal" not in sibling_types("Deal", deals, exclude="Woondeal")


def test_ancestors_follow_the_real_declared_chain(deals):
    assert ancestors_of("RegioDeal", deals) == ["Deal", "GovernmentService"]
    assert ancestors_of("Deal", deals) == ["GovernmentService"]


def test_find_type_is_case_insensitive_on_real_types(deals):
    assert find_type("regiodeal", deals) is not None
    assert find_type("Onbekend Type", deals) is None


# ---------------------------------------------------------------------------
# A proposal that is not new
# ---------------------------------------------------------------------------


def test_an_existing_name_is_a_duplicate_not_a_placement(deals):
    placement = place_proposed_type("RegioDeal", "Deal", deals)
    assert placement.verdict == DUPLICATE
    assert placement.reason_code == EV_NAME_TAKEN
    assert placement.duplicate_of == "RegioDeal"
    # a duplicate has no sibling set: it is not being placed at all
    assert placement.descendant_candidates == ()


def test_an_existing_alias_is_a_duplicate_and_says_whose():
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


def test_name_taken_and_alias_taken_are_distinguishable():
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


def test_an_unknown_parent_is_reported_as_unchecked_not_as_top_level(deals):
    placement = place_proposed_type("Stadsdeal", "Verzonnen", deals)
    assert placement.verdict == PARENT_UNKNOWN
    assert placement.reason_code == EV_PARENT_UNKNOWN
    assert "could not be checked" in placement.evidence
    assert placement.descendant_candidates == ()


def test_no_declared_parent_is_undecided_not_top_level(deals):
    placement = place_proposed_type("Stadsdeal", None, deals)
    assert placement.verdict == UNPARENTED
    assert placement.reason_code == EV_NO_PARENT_DECLARED
    assert "undecided, not top-level" in placement.evidence


def test_without_applied_ontologies_nothing_is_claimed():
    placement = place_proposed_type("Stadsdeal", "Deal", [])
    assert placement.reason_code == EV_NO_SCHEMAS
    assert "never compared against anything" in placement.evidence
    assert "says nothing about whether it is new" in placement.evidence


def test_a_nameless_proposal_says_so(deals):
    placement = place_proposed_type("   ", "Deal", deals)
    assert placement.reason_code == EV_NO_NAME


def test_every_verdict_carries_both_halves_of_its_evidence(deals):
    for name, parent in [("Stadsdeal", "Deal"), ("RegioDeal", "Deal"),
                         ("Stadsdeal", "Verzonnen"), ("Stadsdeal", None),
                         ("   ", "Deal")]:
        placement = place_proposed_type(name, parent, deals)
        assert placement.reason_code, (name, parent)
        assert placement.evidence, (name, parent)


# ---------------------------------------------------------------------------
# Structural guards
# ---------------------------------------------------------------------------


def test_a_cycle_is_refused():
    # Reachable for N.4d.3's re-parent of an EXISTING type, not for a new one.
    schemas = [_ont(
        A=EntityTypeDefinition(name="A", parent_type="B"),
        B=EntityTypeDefinition(name="B"),
    )]
    assert would_cycle("A", "A", schemas) is True
    assert would_cycle("B", "A", schemas) is True   # A already descends from B
    assert would_cycle("A", "B", schemas) is False


def test_a_self_declared_parent_is_cyclic():
    schemas = [_ont(Deal=EntityTypeDefinition(name="Deal"))]
    placement = place_proposed_type("Deal", "Deal", schemas)
    # caught as a duplicate first — the earlier, more specific observation
    assert placement.verdict == DUPLICATE


def test_a_hand_authored_loop_cannot_hang_the_walk():
    schemas = [_ont(
        A=EntityTypeDefinition(name="A", parent_type="B"),
        B=EntityTypeDefinition(name="B", parent_type="A"),
    )]
    assert ancestors_of("A", schemas) == ["B"]


def test_a_chain_leaving_the_applied_set_is_reported_not_raised(deals):
    # GovernmentService is declared as a parent in deals.yaml but defined
    # elsewhere; the walk stops there instead of erroring.
    assert ancestors_of("Deal", deals) == ["GovernmentService"]
    assert find_type("GovernmentService", deals) is None


def test_malformed_members_are_skipped_not_raised():
    class _Broken:
        entity_types = "not a dict"

    assert find_type("Deal", [_Broken()]) is None
    assert sibling_types("Deal", [_Broken()]) == ()


# ---------------------------------------------------------------------------
# The two ways a parent can be valid
# ---------------------------------------------------------------------------


def test_a_schema_org_base_is_a_valid_parent(deals):
    """`deals.yaml` roots half its types at bases no ontology DEFINES.

    `canonical_bridge` terminates its walk on the base name itself, so treating
    such a declaration as unknown would call four of eight shipped declarations
    broken when they are exactly how the vocabulary is meant to be authored.
    Found by running against the real ontology rather than a fixture.
    """
    assert find_type("GovernmentService", deals) is None  # not defined...
    assert known_schema_org_base("GovernmentService") is True  # ...but valid

    placement = place_proposed_type("Bestuursakkoord", "GovernmentService", deals)
    assert placement.verdict == PLACED
    assert "schema.org base" in placement.evidence
    assert set(placement.descendant_candidates) == {
        "Deal", "Akkoord", "BeleidsProgramma",
    }
    # NOT the grandchildren: re-parenting those would move them away from their
    # own parent, which is a different and unsound edit.
    assert "RegioDeal" not in placement.descendant_candidates


def test_the_evidence_distinguishes_the_two_kinds_of_parent(deals):
    defined = place_proposed_type("Stadsdeal", "Deal", deals)
    base = place_proposed_type("Bestuursakkoord", "GovernmentService", deals)
    assert "defined in an applied ontology" in defined.evidence
    assert "schema.org base" in base.evidence


def test_an_invented_parent_is_neither(deals):
    placement = place_proposed_type("Stadsdeal", "Verzonnen", deals)
    assert placement.verdict == PARENT_UNKNOWN
    assert known_schema_org_base("Verzonnen") is False
    assert "not a schema.org base" in placement.evidence
