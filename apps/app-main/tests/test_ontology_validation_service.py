"""OE.1: the typing rules of the ontology editor's gate.

Modelled on what a label actually types as — ``_resolve_entity_type`` over
``typing_vocabulary``, the path persistence takes since PC.4. The first version
modelled its own idea of typing; it passed the PC.4 bug class (review B1) and
blocked 553 of 935 legitimate re-parents (M1). Both are pinned here, the
re-parents over the whole space.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from app_main.services.entity_persistence_service import (
    _resolve_entity_type,
    typing_vocabulary,
)
from app_main.services.ontology_validation_service import (
    all_findings,
    validate_candidate,
)
from ontology_manager.schema import EntityTypeDefinition, Ontology
from ontology_manager.validation import BLOCK, classify, labels_of, resolve_set

ONTOLOGY_DIR = (Path(__file__).resolve().parents[3]
                / "packages" / "ontology-manager" / "ontologies")


def _shipped():
    return {p.stem: Ontology.from_yaml(p) for p in sorted(ONTOLOGY_DIR.glob("*.yaml"))}


def _edited(edit):
    candidate = {k: v.model_copy(deep=True) for k, v in _shipped().items()}
    edit(candidate)
    return candidate


def _blocks(findings):
    return {(f.rule, f.ontology, f.subject) for f in findings if f.severity == BLOCK}


def _answers(own, label):
    """The type of `label` with each ontology leading, and with none — the
    production typing, computed independently of the service."""
    merged = list(resolve_set(own)[0].values())
    orders = [typing_vocabulary(None, merged)] + [typing_vocabulary([o], merged) for o in merged]
    return {_resolve_entity_type(label, v).entity_type for v in orders}


def test_the_shipped_set_has_no_typing_findings():
    assert [f for f in all_findings(_shipped())
            if f.rule in {"selection_dependent_type", "base_named_type_misses_its_base",
                          "orphan", "subtype_disagrees_with_parent"}] == []


def test_an_unchanged_candidate_has_nothing_to_block():
    assert _blocks(validate_candidate(_shipped(), _shipped())) == set()


@pytest.mark.parametrize("edit, expected", [
    # Review B1: a label redeclared under a declared-but-unmapped base types
    # differently depending on which ontology leads.
    (lambda c: c["general"].entity_types.__setitem__(
        "Wijk", EntityTypeDefinition(name="Wijk", schema_org_type="schema:Product")),
     ("selection_dependent_type", None, "wijk")),
    # Review M2: the dict key is a label too — the bridge matches it.
    (lambda c: c["scholarly"].entity_types.__setitem__(
        "Gemeente", EntityTypeDefinition(name="Xyzzy", parent_type="Event")),
     ("selection_dependent_type", None, "gemeente")),
    # The PC.4 bug class: a base-named type without its base.
    (lambda c: setattr(c["policy"].entity_types["Legislation"], "schema_org_type", None),
     ("base_named_type_misses_its_base", "policy", "Legislation")),
    # A parent that is neither defined nor a mapped base: a typo.
    (lambda c: c["deals"].entity_types.__setitem__(
        "Streekfonds", EntityTypeDefinition(name="Streekfonds", parent_type="GovermentService")),
     ("orphan", "deals", "Streekfonds")),
    # Review M5: `grant` (not `Grant`) misses the bridge's case-sensitive base
    # lookup and walks past `Grant`'s base to `Thing`.
    (lambda c: c["deals"].entity_types.__setitem__(
        "Streekfonds", EntityTypeDefinition(name="Streekfonds", parent_type="grant")),
     ("subtype_disagrees_with_parent", "deals", "Streekfonds")),
])
def test_each_rule_blocks_the_state_it_refuses(edit, expected):
    findings = validate_candidate(_edited(edit), _shipped())
    assert expected in _blocks(findings), [f.detail for f in findings]


def test_the_selection_dependent_finding_is_real():
    """Vacuity guard: the finding's claim, checked against production typing."""
    candidate = _edited(lambda c: c["general"].entity_types.__setitem__(
        "Wijk", EntityTypeDefinition(name="Wijk", schema_org_type="schema:Product")))
    assert len(_answers(candidate, "Wijk")) > 1
    assert len(_answers(_shipped(), "Wijk")) == 1


def test_a_re_parent_blocks_only_when_production_typing_proves_it_across_the_space():
    """Review M1, derived: a new type under EVERY declared type name, in EVERY
    ontology (the first version blocked 553 of these). A block is allowed only
    when production typing, computed independently, confirms it: the new label
    types differently per selection, or differently from its parent (a child of
    `Product` falls to `concept` while `Product` itself types `product` — a real
    surprise for a curator). Never as `orphan`."""
    shipped = _shipped()
    known = all_findings(shipped)
    parents = sorted({label for o in resolve_set(shipped)[0].values()
                      for label, owner in labels_of(o) if label == owner})
    checked = wrongly_blocked = 0
    for ontology in sorted(shipped):
        for parent in parents:
            candidate = {k: v.model_copy(deep=True) for k, v in shipped.items()}
            candidate[ontology].entity_types["Nieuwtype"] = EntityTypeDefinition(
                name="Nieuwtype", parent_type=parent)
            blocks = [f for f in classify(all_findings(candidate), known) if f.severity == BLOCK]
            checked += 1
            for f in blocks:
                if f.rule == "selection_dependent_type":
                    confirmed = len(_answers(candidate, f.subject)) > 1
                elif f.rule == "subtype_disagrees_with_parent":
                    confirmed = _answers(candidate, "Nieuwtype") != _answers(candidate, parent)
                else:
                    confirmed = False
                wrongly_blocked += not confirmed
    assert checked > 900
    assert wrongly_blocked == 0


def test_a_selection_dependency_that_gets_worse_blocks():
    """Weight = number of different answers; two to three blocks."""
    two = _edited(lambda c: c["general"].entity_types.__setitem__(
        "Wijk", EntityTypeDefinition(name="Wijk", schema_org_type="schema:Product")))
    three = _edited(lambda c: (
        c["general"].entity_types.__setitem__(
            "Wijk", EntityTypeDefinition(name="Wijk", schema_org_type="schema:Product")),
        c["scholarly"].entity_types.__setitem__(
            "Wijk", EntityTypeDefinition(name="Wijk", parent_type="Event"))))
    assert len(_answers(three, "Wijk")) == 3
    assert ("selection_dependent_type", None, "wijk") in _blocks(validate_candidate(three, two))
