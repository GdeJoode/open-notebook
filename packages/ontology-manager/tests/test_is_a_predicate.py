"""Track N.5b — ``is_a`` is a declared predicate, not one that survives by accident.

The review finding (I3) was not that Hearst mining is bad. It was that N.2 shipped
a producer whose output lived only because nobody looked: ``is_a`` appeared in no
ontology, ``OntologyValidator`` downgrades an unknown predicate to a WARNING
outside strict mode, and the filtering stage that would have applied the
validation is off by default. Setting ``strict_mode`` deleted every mined
hierarchy edge, silently.

Both halves of the decision are pinned here:

* the predicate is declared in every shipped ontology, so ``strict_mode`` cannot
  change the outcome unnoticed — which is the phase's stated acceptance criterion;
* the miner ships explicitly OFF, so nothing flows until somebody decides it
  should. Measured while it defaulted on: 220 raw pairs over 3823 chunks and zero
  edges in the graph.
"""

from __future__ import annotations

import asyncio

import pytest
from ontology_manager.registry import OntologyRegistry
from ontology_manager.validator import OntologyValidator, ValidationSeverity


def _ontology_names():
    """Every shipped ontology, resolved once at collection time.

    Parametrising over the discovered set rather than a hard-coded list is
    deliberate: a new ROOT ontology added later (one with no ``extends``) would
    silently reintroduce exactly the gap this phase closes, and a fixed list
    would not notice.
    """
    names = sorted(asyncio.run(OntologyRegistry().list_ontologies()))
    assert names, "no ontologies discovered — the sweep below would be vacuous"
    return names


ONTOLOGY_NAMES = _ontology_names()


def _relationship():
    return {"predicate": "is_a", "subject": "dorpshuizen", "object": "ontmoetingspunten"}


@pytest.mark.parametrize("ontology_name", ONTOLOGY_NAMES)
async def test_is_a_is_declared_in_every_shipped_ontology(ontology_name):
    """Declared directly or inherited through ``extends`` — the registry resolves
    inheritance on load, so three declarations (schema_core, base, policy_themes)
    cover all eleven. The sweep is over every ontology because a new root added
    later would otherwise reintroduce the gap unnoticed.
    """
    ontology = await OntologyRegistry().get(ontology_name)
    assert ontology is not None, f"{ontology_name} failed to load"
    assert ontology.get_relationship_type("is_a") is not None


@pytest.mark.parametrize("ontology_name", ONTOLOGY_NAMES)
async def test_strict_mode_does_not_change_the_verdict_on_is_a(ontology_name):
    """The acceptance criterion, stated as the property rather than as a config.

    Before N.5b this failed for every ontology: lenient gave a WARNING and strict
    gave an ERROR plus an early return, so flipping one flag removed the edges
    with nothing in the record to say why.
    """
    ontology = await OntologyRegistry().get(ontology_name)
    lenient = OntologyValidator(ontology, strict=False).validate_relationship(
        _relationship()
    )
    strict = OntologyValidator(ontology, strict=True).validate_relationship(
        _relationship()
    )

    for report in (lenient, strict):
        assert not [
            i
            for i in report.issues
            if i.severity == ValidationSeverity.ERROR
            and i.path == "relationship.predicate"
        ], f"{ontology_name}: is_a reported as an unknown predicate"

    assert lenient.is_valid == strict.is_valid


async def test_an_undeclared_predicate_still_differs_between_the_two_modes():
    """Vacuity guard. The test above asserts that two reports AGREE; without this
    it would pass just as well if the validator had stopped distinguishing the
    modes at all, or if `validate_relationship` had become a no-op.
    """
    ontology = await OntologyRegistry().get("schema_core")
    made_up = {"predicate": "flurbs_at", "subject": "a", "object": "b"}

    lenient = OntologyValidator(ontology, strict=False).validate_relationship(made_up)
    strict = OntologyValidator(ontology, strict=True).validate_relationship(made_up)

    severities = {
        i.severity for i in lenient.issues if i.path == "relationship.predicate"
    }
    assert ValidationSeverity.WARNING in severities
    assert ValidationSeverity.ERROR in {
        i.severity for i in strict.issues if i.path == "relationship.predicate"
    }
    assert lenient.is_valid != strict.is_valid


def test_the_hearst_miner_ships_off():
    """The other half of the decision, and the half a config drift would undo.

    Measured while this defaulted on: 220 raw pairs across 3823 chunks of the
    project's own corpus, 138 distinct, and ZERO `is_a` edges in the graph. The
    precision gate requires both endpoints to be entities the LLM extracted for
    the same chunk; under the looser "exists anywhere in this notebook" reading
    only 15 distinct pairs survive, including `banken is_a voedselketen` and
    `PD is_a Control variables`.
    """
    import os
    from unittest.mock import patch

    from ontology_extraction.pass2_typed_extraction import _hearst_isa_enabled

    with patch.dict(os.environ, {}, clear=False):
        os.environ.pop("EXTRACTION_HEARST_ISA", None)
        assert _hearst_isa_enabled() is False

    # ...and it is still a flag, not a deletion: N.2's machinery stays available.
    with patch.dict(os.environ, {"EXTRACTION_HEARST_ISA": "true"}):
        assert _hearst_isa_enabled() is True
