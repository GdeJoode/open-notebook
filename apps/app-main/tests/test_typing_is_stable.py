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

OE.5 extends it from "every selection" to "every notebook": the ontologies are
shared by every notebook (owner, 2026-09-28), so nothing a notebook stores may
change a type. The notebooks are derived from what a ``notebook_schema`` row can
hold that the extraction path reads — its declared base (none, each shipped
ontology, one that does not exist) crossed with an empty layer and a layer
holding one entry of every kind the retired per-notebook layer ever wrote — and
each is run through the service's real ``_run_multi_schema`` under every
selection.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from app_main.services.entity_extraction_service import EntityExtractionService
from app_main.services.entity_persistence_service import (
    _resolve_entity_type,
    load_typing_ontologies,
    typing_vocabulary,
)
from ontology_manager.canonical_bridge import resolve_ontology_type
from shared.models.notebook_schema import NotebookSchema

pytestmark = pytest.mark.asyncio


async def _shipped():
    """The shipped YAML ontologies (conftest keeps the registry off any database)."""
    return {o.metadata.name: o for o in await load_typing_ontologies()}


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


# One entry of every kind the retired per-notebook layer ever stored. Each
# names something that, had it taken effect, would have changed a type or the
# prompt: a re-parent the projection applied (`Gemeente` -> administrative_area
# becomes government_organization), an extension on a name an ontology declares
# as an alias (`Theme`), a new type, a rename, a merge, a split, the resume
# sentinel, a soft-delete. `schema_name` on an extension used to force that
# ontology into the selection.
_LAYER_NAMES = ("ZzNotebookOnlyType", "ZzRenamed", "ZzMerged", "ZzSplitA", "ZzSplitB")
_EVERY_KIND = [
    {"reparent_id": "r1", "op": "reparent", "type_name": "Gemeente",
     "new_parent": "GovernmentOrganization", "parent_type": "GovernmentOrganization"},
    {"extension_id": "ext:Theme", "type_name": "Theme", "parent_type": "Organization"},
    {"extension_id": "ext:new", "type_name": "ZzNotebookOnlyType",
     "parent_type": "Person", "schema_name": "scholarly", "rationale": "notebook only"},
    {"rename_id": "rn", "op": "rename", "old_name": "Person", "new_name": "ZzRenamed",
     "type_name": "ZzRenamed"},
    {"merge_id": "mg", "op": "merge", "source_types": ["Person", "Organization"],
     "merged_name": "ZzMerged", "type_name": "ZzMerged"},
    {"split_id": "sp", "op": "split", "source_type": "Event",
     "into": ["ZzSplitA", "ZzSplitB"], "criterion": "by year", "type_name": "Event"},
    {"type_name": "_resumed_without_extensions", "is_resume_sentinel": True},
]


def _notebooks(names):
    """Every distinct stored layer the extraction path reads, plus no row."""
    rows = [None]
    for base in ["", *names, "does_not_exist"]:
        rows.append(NotebookSchema(notebook="notebook:t", base_ontology=base))
        rows.append(NotebookSchema(
            notebook="notebook:t", base_ontology=base,
            accepted_extensions=[dict(e) for e in _EVERY_KIND],
            excluded_types=["Person", "Gemeente"],
        ))
    return rows


class _Harness:
    """The service's real `_run_multi_schema`, with its I/O faked.

    Detection is replaced by a fixed selection (the space under test); the
    notebook row, the source and the Pass-1 history come from fakes; the
    workflow records what it was handed. Everything between — the notebook's
    base forcing, the typing vocabulary, what reaches Pass 2 — is production
    code.
    """

    def __init__(self, onts):
        self.onts = onts
        self.schema_repo = MagicMock()
        self.schema_repo.merge_pending_extensions = AsyncMock(return_value=0)
        self.schema_repo.set_coverage_pct = AsyncMock(return_value=True)
        self.schema_repo.ensure_row = AsyncMock(return_value=False)
        pass1 = MagicMock()
        pass1.list_by_notebook = AsyncMock(return_value=[])
        source_repo = MagicMock()
        source_repo.get = AsyncMock(return_value=SimpleNamespace(metadata={}))
        events = MagicMock()
        events.list_unread = AsyncMock(return_value=[])
        events.record = AsyncMock(return_value="notebook_event:t")
        self.svc = EntityExtractionService(
            source_repo=source_repo, notebook_schema_repo=self.schema_repo,
            pass1_repo=pass1, notebook_event_repo=events,
        )
        self.svc._make_routed_caller = AsyncMock(return_value=None)
        self.manager = MagicMock()
        self.manager.list_ontologies = AsyncMock(return_value=sorted(onts))
        self.manager.get_ontology = AsyncMock(side_effect=lambda n: onts.get(n))
        self.detect = AsyncMock()
        self.workflow = MagicMock()
        self.workflow.extract = AsyncMock(return_value=SimpleNamespace(metadata={}))

    def patches(self):
        return (
            patch("ontology_manager.get_ontology_manager", return_value=self.manager),
            patch("app_main.services.entity_extraction_service.detect_applicable_schemas",
                  new=self.detect),
        )

    async def run(self, notebook, selection):
        self.schema_repo.get_by_notebook = AsyncMock(return_value=notebook)
        self.detect.return_value = [(self.onts[n], 0.9) for n in selection]
        await self.svc._run_multi_schema(
            self.workflow, "source:t", "notebook:t", [{"text": "t"}], None
        )
        return self.workflow.extract.await_args.kwargs


async def test_the_type_is_the_same_in_every_notebook():
    """The AC of OE.5. Every declared label, and every name the old layer
    stored, types the same in every notebook under every selection — and the
    typing vocabulary is the registry's own objects, nothing projected."""
    onts = await _shipped()
    names = sorted(onts)
    labels = sorted({*_labels(onts), *_LAYER_NAMES}, key=str.lower)
    everything = [onts[n] for n in names]
    reference = {label: _resolve_entity_type(label, everything).entity_type
                 for label in labels}
    registry_objects = {id(o) for o in everything}

    harness = _Harness(onts)
    vocabularies = {}
    longest = 0
    notebooks = _notebooks(names)
    first, second = harness.patches()
    with first, second:
        for notebook in notebooks:
            for selection in _selections(names):
                kwargs = await harness.run(notebook, selection)
                assert "accepted_extensions_by_schema" not in kwargs
                typing = harness.svc._typing_schemas
                assert {id(o) for o in typing} <= registry_objects
                assert {id(o) for o, _c in kwargs["applicable_schemas"]} <= registry_objects
                vocabularies.setdefault(tuple(id(o) for o in typing), typing)
                longest = max(longest, len(kwargs["applicable_schemas"]))

    differs = {}
    for typing in vocabularies.values():
        for label in labels:
            got = _resolve_entity_type(label, typing).entity_type
            if got != reference[label]:
                differs.setdefault(label, set()).add(
                    (tuple(o.metadata.name for o in typing[:5]), got))
    assert differs == {}
    # The space reached what it claims: the forced bases lengthened selections
    # past detection's three, and the layer rows were read.
    assert longest > 3
    assert len(notebooks) == 2 * (len(names) + 2) + 1


async def test_the_old_layer_would_have_changed_a_type():
    """Vacuity guard for the test above: under the retired projection the same
    layer DID change types, so "no label differs" is a statement about OE.5 and
    not about a layer that never did anything."""
    onts = await _shipped()
    policy = [onts["policy_themes"]]
    assert _resolve_entity_type("Gemeente", typing_vocabulary(policy, list(onts.values()))
                                ).entity_type == "administrative_area"
    # What the N.4d.3 projection returned for the re-parent in `_EVERY_KIND`.
    moved = [o.model_copy(deep=True) for o in typing_vocabulary(policy, list(onts.values()))]
    for o in moved:
        if "Gemeente" in o.entity_types:
            o.entity_types["Gemeente"].parent_type = "GovernmentOrganization"
    assert _resolve_entity_type("Gemeente", moved).entity_type == "government_organization"


async def test_nothing_of_the_old_layer_reaches_the_pass_2_prompt():
    """The prompt as the model receives it, through the real workflow and
    orchestrator: a notebook carrying every kind of old entry, each selection
    of one ontology. The layer's names and its section never appear; the
    ontology's own types do."""
    from ontology_extraction.config import ExtractionConfig
    from ontology_extraction.prompts.pass2 import PASS2_SYSTEM_PROMPT
    from ontology_extraction.workflow import ExtractionWorkflow

    onts = await _shipped()
    harness = _Harness(onts)
    prompts = {}

    def llm(system_prompt, user_prompt, model):
        if system_prompt == PASS2_SYSTEM_PROMPT:
            prompts.setdefault(current, []).append(user_prompt)
            return '{"entities": [], "relations": []}'
        return ('{"detected_schema": "x", "confidence_in_choice": 0.9, '
                '"coverage_pct": 0.9, "uncovered_concepts": [], '
                '"proposed_extensions": [], "alternative_schemas": []}')

    harness.svc._make_routed_caller = AsyncMock(return_value=llm)
    notebook = _notebooks(sorted(onts))[-1]
    assert notebook.accepted_extensions, "the layer under test is the full one"
    first, second = harness.patches()
    with first, second:
        for current in sorted(onts):
            harness.schema_repo.get_by_notebook = AsyncMock(return_value=notebook)
            harness.detect.return_value = [(onts[current], 0.9)]
            await harness.svc._run_multi_schema(
                ExtractionWorkflow(ExtractionConfig()), "source:t", "notebook:t",
                [{"text": "De gemeente en de provincie sluiten een Regio Deal."}], None,
            )

    assert sorted(prompts) == sorted(onts)
    for name, sent in prompts.items():
        for prompt in sent:
            assert "Accepted Extension" not in prompt
            assert "_resumed_without_extensions" not in prompt
            for layer_name in _LAYER_NAMES:
                assert layer_name not in prompt, (name, layer_name)
            assert "notebook only" not in prompt
    assert any("Gemeente" in p for p in prompts["regiodeal"])


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


async def test_every_literal_stays_out_of_the_type_layer_under_every_ontology():
    """PC.4b: a literal lands on `other` through the WHOLE typing path, not just
    the residual map — an ontology that later declared `Date` as a type would
    take it back into the type layer, and this fails then."""
    from shared.utils.entity_type_aliases import NON_TYPE_LABELS

    onts = await _shipped()
    for label in sorted(NON_TYPE_LABELS):
        resolved = _resolve_entity_type(label, list(onts.values()))
        assert (resolved.entity_type, resolved.is_noise) == ("other", True), label
