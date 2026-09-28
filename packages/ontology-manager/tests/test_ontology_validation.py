"""OE.1: the structural validation rules — each fed the broken state it refuses.

The candidate is the shipped set with one edit applied, exactly as the editor
will submit it; the active set is the shipped set. The typing rules (what a
label actually types as) are tested beside that typing, in app-main
(``test_ontology_validation_service.py``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ontology_manager.schema import (
    EntityTypeDefinition,
    Ontology,
    PropertyDefinition,
    RelationshipTypeDefinition,
)
from ontology_manager.validation import (
    BLOCK,
    WARN,
    blocking,
    resolve_set,
    structural_findings,
    validate,
)

ONTOLOGY_DIR = Path(__file__).resolve().parents[1] / "ontologies"


def _shipped():
    return {p.stem: Ontology.from_yaml(p) for p in sorted(ONTOLOGY_DIR.glob("*.yaml"))}


def _edited(edit):
    candidate = {k: v.model_copy(deep=True) for k, v in _shipped().items()}
    edit(candidate)
    return candidate


def _keys(findings, severity=BLOCK):
    return {(f.rule, f.ontology, f.subject) for f in findings if f.severity == severity}


def test_the_shipped_set_holds_exactly_the_known_findings():
    """Pinned, so a new pre-existing problem is noticed rather than absorbed as
    'already there' by the regression semantics."""
    assert {(f.rule, f.ontology, f.subject) for f in structural_findings(_shipped())} == {
        ("unmapped_schema_org_type", "base", "Concept"),        # schema:Intangible, deliberate
        ("unmapped_schema_org_type", "general", "Product"),     # schema:Product, deliberate
        ("relationship_endpoint_unknown", None, "DISCUSSES:source"),
        ("relationship_endpoint_unknown", None, "MENTIONS:source"),
        ("relationship_endpoint_unknown", None, "CONTRIBUTES_TO_REPO:SoftwareSourceCode"),
    }


def test_an_unchanged_candidate_has_nothing_to_block():
    assert blocking(validate(_shipped(), _shipped())) == []


def _add(ontology, name, **fields):
    def edit(c):
        c[ontology].entity_types[name] = EntityTypeDefinition(name=name, **fields)
    return edit


@pytest.mark.parametrize("edit, expected", [
    # A base-named type pointed at a DIFFERENT mapped base (review M3).
    (lambda c: setattr(c["schema_core"].entity_types["Grant"], "schema_org_type", "schema:Event"),
     ("base_named_type_points_elsewhere", "schema_core", "Grant")),
    # A typo in schema_org_type (review M4).
    (_add("general", "Stichting", schema_org_type="schema:Persn"),
     ("unmapped_schema_org_type", "general", "Stichting")),
    # A parent_type loop.
    (lambda c: (c["deals"].entity_types.__setitem__(
        "A", EntityTypeDefinition(name="A", parent_type="B")),
        c["deals"].entity_types.__setitem__(
        "B", EntityTypeDefinition(name="B", parent_type="A"))),
     ("parent_cycle", None, "a,b")),
    # One name, two types.
    (_add("deals", "Fondsje", parent_type="Grant", aliases=["Project"]),
     ("alias_collision", None, "project")),
    # A literal declared as a type.
    (_add("general", "Date", schema_org_type="schema:Event"),
     ("literal_declared_as_type", "general", "Date")),
    # An ontology extending one that does not exist.
    (lambda c: setattr(c["deals"].metadata, "extends", "instrumentz"),
     ("unknown_parent_ontology", "deals", "instrumentz")),
    # An inheritance loop.
    (lambda c: setattr(c["schema_core"].metadata, "extends", "regiodeal"),
     ("inheritance_cycle", None, "deals,government,instruments,regiodeal,schema_core")),
    # A relationship naming no type.
    (lambda c: c["deals"].relationship_types.__setitem__(
        "FUNDS", RelationshipTypeDefinition(name="FUNDS", domain=["Grant"], range=["Projct"])),
     ("relationship_endpoint_unknown", None, "FUNDS:Projct")),
])
def test_each_rule_blocks_the_state_it_refuses(edit, expected):
    findings = validate(_edited(edit), _shipped())
    assert expected in _keys(findings), [f.detail for f in findings]


def test_a_base_named_type_pointing_at_the_same_canonical_is_fine():
    """`general.Technology` is `schema:TechArticle`: both map to `technology`."""
    assert not [f for f in structural_findings(_shipped())
                if f.rule == "base_named_type_points_elsewhere"]


def test_a_pre_existing_finding_warns_and_does_not_block():
    broken = _edited(_add("general", "Stichting", schema_org_type="schema:Persn"))
    findings = validate(broken, active=broken)
    assert blocking(findings) == []
    assert ("unmapped_schema_org_type", "general", "Stichting") in _keys(findings, WARN)


def test_a_finding_that_gets_worse_blocks():
    """Review m2: keyed on (rule, ontology, subject) alone, a collision that
    grew from two types to three stayed a warning."""
    two = _edited(_add("deals", "Fondsje", parent_type="Grant", aliases=["Project"]))
    three = _edited(lambda c: (
        _add("deals", "Fondsje", parent_type="Grant", aliases=["Project"])(c),
        _add("deals", "Fondsbeheer", parent_type="Grant", aliases=["Project"])(c)))
    assert ("alias_collision", None, "project") in _keys(validate(three, active=two))


def test_an_inherited_problem_is_reported_once_where_it_is_defined():
    """Review m3: a new ontology extending one with a known warning re-reported
    it under its own name and blocked."""
    broken = _edited(_add("policy", "Stichting", schema_org_type="schema:Persn"))
    with_child = {**{k: v.model_copy(deep=True) for k, v in broken.items()},
                  "policy_nl": Ontology(metadata={"name": "policy_nl", "version": "1",
                                                  "extends": "policy"})}
    assert blocking(validate(with_child, active=broken)) == []


def test_includes_without_extends_are_not_merged():
    """Review m1: the registry resolves inheritance only when `extends` is set."""
    own = {"a": Ontology(metadata={"name": "a", "version": "1"},
                         entity_types=[{"name": "OnlyInA", "parent_type": "Thing"}]),
           "b": Ontology(metadata={"name": "b", "version": "1", "includes": ["a"]})}
    merged, _ = resolve_set(own)
    assert "OnlyInA" not in merged["b"].entity_types


@pytest.mark.parametrize("edit, subject", [
    (_add("deals", "X", parent_type="Grant", parent_typ="oops"), "X.parent_typ"),
    (lambda c: c["deals"].entity_types["Project"].properties.append(
        PropertyDefinition(name="budget", data_type="string", requird=True)), "Project.budget.requird"),
    (lambda c: c["deals"].metadata.__pydantic_extra__.update(namespace={}), "deals.namespace"),
])
def test_an_unrecognised_key_only_warns_at_every_level(edit, subject):
    """Review m4: only entity-type extras were checked."""
    findings = validate(_edited(edit), _shipped())
    assert subject in {f.subject for f in findings if f.rule == "unknown_key" and f.severity == WARN}
    assert blocking(findings) == []
