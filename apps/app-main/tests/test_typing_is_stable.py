"""PC.4: a label's type does not depend on which ontologies a document selected.

Detection selects up to 3 ontologies per document (``top_k=3``, ranked), because
a model cannot always hold more in its prompt. Typing ran against that selection
only, so a label from an unselected ontology fell to ``concept``: measured, 69 of
the 91 labels the shipped ontologies declare got a different type under a
different selection, and ``Person`` / ``Organization`` / ``Event`` got two
answers even inside the ontologies (``schema_core`` carried no
``schema_org_type``).

The space is derived, not sampled: every label any shipped ontology declares
(names and aliases), under every ordered selection detection can produce — every
sequence of 1 to 3 distinct shipped ontologies.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace

import pytest
from app_main.services.entity_extraction_service import EntityExtractionService
from app_main.services.entity_persistence_service import (
    _resolve_entity_type,
    load_typing_ontologies,
    typing_vocabulary,
)
from ontology_manager.canonical_bridge import resolve_ontology_type

pytestmark = pytest.mark.asyncio


async def _shipped():
    """The shipped YAML ontologies (conftest keeps the registry off any database)."""
    return {o.metadata.name: o for o in await load_typing_ontologies()}


def _svc():
    return EntityExtractionService.__new__(EntityExtractionService)


def _typing(onts, selection, edits=None):
    """What the extraction service types against for this selection."""
    notebook_schema = SimpleNamespace(accepted_extensions=edits) if edits else None
    return _svc()._typing_schemas_for(
        [(onts[n], 0.9) for n in selection], list(onts.values()), notebook_schema, "notebook:t"
    )


def _selections(names):
    return [p for k in (1, 2, 3) for p in itertools.permutations(names, k)]


def _labels(onts):
    labels = set()
    for ontology in onts.values():
        for key, definition in ontology.entity_types.items():
            labels.add(definition.name or key)
            labels.update(definition.aliases or [])
    return sorted(labels, key=str.lower)


async def test_no_label_has_two_canonical_answers_across_the_ontologies():
    onts = await _shipped()
    assert len(onts) >= 11, sorted(onts)
    split = {}
    for label in _labels(onts):
        answers = {
            name: r.canonical
            for name, o in onts.items()
            if (r := resolve_ontology_type(label, [o])) is not None
        }
        if len(set(answers.values())) > 1:
            split[label] = answers
    assert split == {}


async def test_a_type_named_after_a_mapped_base_reaches_that_base():
    """Consistent is not correct: `policy.Legislation` typed as `creative_work`
    in EVERY ontology, so the sweep above saw no conflict. A type whose own name
    is a schema.org base the bridge maps must resolve to that base's canonical,
    in every ontology that declares it — its `schema_org:` URL is documentation
    the loader does not read, so `schema_org_type` must say it."""
    from ontology_manager.canonical_bridge import _CANONICAL_BY_SCHEMA_ORG

    onts = await _shipped()
    wrong = {}
    for name, ontology in onts.items():
        for key, definition in ontology.entity_types.items():
            label = definition.name or key
            expected = _CANONICAL_BY_SCHEMA_ORG.get(label)
            # A type that declares its own base is the author's choice
            # (`base.Concept` is `schema:Intangible`, so `concept`); the bug
            # class is a type that declares none.
            if expected is None or definition.schema_org_type:
                continue
            got = resolve_ontology_type(label, [ontology])
            if got is None or got.canonical != expected:
                wrong[(name, label)] = got.canonical if got else None
    assert wrong == {}


async def test_the_type_is_the_same_under_every_selection():
    onts = await _shipped()
    names = sorted(onts)
    labels = _labels(onts)
    reference = {
        label: _resolve_entity_type(label, [onts[n] for n in names]).entity_type
        for label in labels
    }
    selections = _selections(names)
    differs = {}
    for selection in selections:
        schemas = typing_vocabulary([onts[n] for n in selection], list(onts.values()))
        for label in labels:
            got = _resolve_entity_type(label, schemas).entity_type
            if got != reference[label]:
                differs.setdefault(label, set()).add((selection, got))
    assert differs == {}


@pytest.mark.parametrize("label, expected", [
    ("Person", "person"),
    ("Organization", "organization"),
    ("Event", "event"),
    ("Place", "location"),
    ("CreativeWork", "creative_work"),
    ("DefinedTerm", "topic"),
    ("Grant", "grant"),
    ("Report", "policy_document"),
    ("Dataset", "dataset"),
])
async def test_the_schema_core_types_reach_their_base(label, expected):
    """Resolved against schema_core ALONE: the fix is in the ontology, not the order."""
    onts = await _shipped()
    assert _resolve_entity_type(label, [onts["schema_core"]]).entity_type == expected


async def test_a_label_outside_the_selection_is_still_typed():
    """The measured case: a Regio Deal document whose selection missed `regiodeal`."""
    onts = await _shipped()
    selection = [onts["policy_themes"], onts["base"], onts["general"]]
    assert resolve_ontology_type("RegioDeal", selection) is None
    schemas = typing_vocabulary(selection, list(onts.values()))
    assert _resolve_entity_type("RegioDeal", schemas).entity_type == "programme"


@pytest.mark.parametrize("edit, label", [
    # A re-parent of a type only some selections contain.
    ({"reparent_id": "r1", "op": "reparent", "type_name": "Gemeente",
      "new_parent": "GovernmentOrganization", "parent_type": "GovernmentOrganization"},
     "Gemeente"),
    # An extension whose name an ontology already declares as an alias.
    ({"extension_id": "ext:Theme", "type_name": "Theme", "parent_type": "Organization"},
     "Theme"),
])
async def test_a_curator_edit_takes_the_same_way_under_every_selection(edit, label):
    """Review M2: edits projected over the selection alone made the type depend
    on it again — the re-parent was refused when the selection lacked
    `Gemeente`, the extension materialised when it lacked the alias."""
    onts = await _shipped()
    answers = {
        _resolve_entity_type(label, _typing(onts, sel, [edit])).entity_type
        for sel in _selections(sorted(onts))
    }
    assert len(answers) == 1, answers


async def test_the_re_parent_is_applied_not_refused():
    """Vacuity guard for the test above: one answer could be "always refused"."""
    onts = await _shipped()
    edit = {"reparent_id": "r1", "op": "reparent", "type_name": "Gemeente",
            "new_parent": "GovernmentOrganization", "parent_type": "GovernmentOrganization"}
    assert _resolve_entity_type("Gemeente", _typing(onts, ["policy_themes"], [edit])).entity_type         == "government_organization"
    assert _resolve_entity_type("Gemeente", _typing(onts, ["policy_themes"])).entity_type         == "administrative_area"


async def test_the_selection_leads():
    onts = await _shipped()
    schemas = typing_vocabulary([onts["regiodeal"], onts["base"]], list(onts.values()))
    assert [o.metadata.name for o in schemas[:2]] == ["regiodeal", "base"]
    assert len({o.metadata.name for o in schemas}) == len(schemas) == len(onts)


async def test_a_project_is_a_programme():
    """PC.4b (owner): `deals.Project` hung under CreativeWork, contradicting
    L.3's alias `project -> programme`; the ontology now agrees with the alias."""
    onts = await _shipped()
    for label in ("Project", "project"):
        assert _resolve_entity_type(label, list(onts.values())).entity_type == "programme"
        assert _resolve_entity_type(label, None).entity_type == "programme"
