"""OE.1: the validation gate — each rule fed the broken state it refuses.

The candidate is the shipped set with one edit applied, exactly as the editor
will submit it; the active set is the shipped set. A finding the edit
introduces must BLOCK; the shipped set itself must be clean.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ontology_manager.schema import EntityTypeDefinition, Ontology
from ontology_manager.validation import BLOCK, WARN, blocking, validate

ONTOLOGY_DIR = Path(__file__).resolve().parents[1] / "ontologies"


def _shipped():
    return {p.stem: Ontology.from_yaml(p) for p in sorted(ONTOLOGY_DIR.glob("*.yaml"))}


def _edited(edit):
    candidate = {k: v.model_copy(deep=True) for k, v in _shipped().items()}
    edit(candidate)
    return candidate


def _rules(findings, severity=BLOCK):
    return {(f.rule, f.ontology, f.subject) for f in findings if f.severity == severity}


def test_the_shipped_set_is_clean():
    assert validate(_shipped()) == []


def test_an_unchanged_candidate_has_nothing_to_block():
    assert blocking(validate(_shipped(), _shipped())) == []


def _add(ontology, name, **fields):
    def edit(c):
        c[ontology].entity_types[name] = EntityTypeDefinition(name=name, **fields)
    return edit


@pytest.mark.parametrize("edit, expected", [
    # The PC.4 bug class: a base-named type with no schema_org_type walks to Thing.
    (lambda c: setattr(c["policy"].entity_types["Legislation"], "schema_org_type", None),
     ("base_named_type_misses_its_base", "policy", "Legislation")),
    # A second ontology types an existing label differently.
    (_add("scholarly", "Gemeente", parent_type="Organization"),
     ("two_canonical_answers", None, "gemeente")),
    # A parent that exists nowhere and is no mapped base: a typo.
    (_add("deals", "Streekfonds", parent_type="GovermentService"),
     ("orphan", "deals", "Streekfonds")),
    # A parent_type loop.
    (lambda c: (c["deals"].entity_types.__setitem__(
        "A", EntityTypeDefinition(name="A", parent_type="B")),
        c["deals"].entity_types.__setitem__(
        "B", EntityTypeDefinition(name="B", parent_type="A"))),
     ("parent_cycle", "deals", "a")),
    # One name, two types.
    (_add("deals", "Fondsje", parent_type="Grant", aliases=["Project"]),
     ("alias_collision", "deals", "project")),
    # A literal declared as a type.
    (_add("general", "Date", schema_org_type="schema:Event"),
     ("literal_declared_as_type", "general", "Date")),
    # An ontology extending one that does not exist.
    (lambda c: setattr(c["deals"].metadata, "extends", "instrumentz"),
     ("unknown_parent_ontology", "deals", "instrumentz")),
    # An inheritance loop.
    (lambda c: setattr(c["schema_core"].metadata, "extends", "regiodeal"),
     ("inheritance_cycle", None, "deals,government,instruments,regiodeal,schema_core")),
])
def test_each_rule_blocks_the_state_it_refuses(edit, expected):
    findings = validate(_edited(edit), _shipped())
    assert expected in _rules(findings), [f.detail for f in findings]


def test_a_pre_existing_finding_warns_and_does_not_block():
    """An imperfect vocabulary can still be edited; only a regression blocks."""
    broken = _edited(lambda c: setattr(
        c["policy"].entity_types["Legislation"], "schema_org_type", None))
    findings = validate(broken, active=broken)
    assert blocking(findings) == []
    assert ("base_named_type_misses_its_base", "policy", "Legislation") in _rules(findings, WARN)


def test_a_change_to_a_parent_is_seen_in_its_children():
    """`regiodeal` extends `schema_core` three levels down; breaking `Grant` in
    `schema_core` is found in every ontology that inherits it."""
    findings = validate(_edited(lambda c: setattr(
        c["schema_core"].entity_types["Grant"], "schema_org_type", None)), _shipped())
    hit = {f.ontology for f in blocking(findings) if f.subject == "Grant"}
    assert {"schema_core", "regiodeal", "deals"} <= hit


def test_an_inheritance_loop_is_one_finding():
    findings = validate(_edited(lambda c: setattr(c["schema_core"].metadata, "extends", "regiodeal")))
    assert [f.subject for f in findings if f.rule == "inheritance_cycle"] == [
        "deals,government,instruments,regiodeal,schema_core"]


def test_an_unrecognised_key_only_warns():
    candidate = _edited(lambda c: c["deals"].entity_types.__setitem__(
        "X", EntityTypeDefinition(name="X", parent_type="Grant", parent_typ="oops")))
    findings = validate(candidate, _shipped())
    assert ("unknown_key", "deals", "X.parent_typ") in _rules(findings, WARN)
    assert blocking(findings) == []
