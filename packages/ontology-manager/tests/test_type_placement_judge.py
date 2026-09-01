"""Track N.4d.2 — the judge that selects among a proposal's siblings.

Pure: this module builds a prompt and parses a reply, so every test here runs
without a model. The fences are what matter, and each one exists because an
entity-side judge failed that exact way earlier in this track — so they are
asserted as properties rather than as single cases.
"""

from __future__ import annotations

import json

import pytest
from ontology_manager.type_placement import place_proposed_type
from ontology_manager.type_placement_judge import (
    JUDGE_SYSTEM_PROMPT,
    JudgeSelection,
    build_judge_prompt,
    candidates_from_ontologies,
    parse_judge_response,
)


def _cands(*names):
    return tuple((str(i), n, f"{n} description") for i, n in enumerate(names))


def _reply(*ids):
    return json.dumps({"move_under_proposal": list(ids)})


# ---------------------------------------------------------------------------
# The core AC: the judge selects within the set, and can never widen it
# ---------------------------------------------------------------------------


def test_selects_within_the_offered_set():
    candidates = _cands("RegioDeal", "Woondeal", "CityDeal")
    selection = parse_judge_response(_reply("0", "2"), candidates)
    assert selection.selected == ("0", "2")
    assert selection.considered == ("0", "1", "2")
    assert selection.widened is False


@pytest.mark.parametrize("reply", [
    _reply("99"),                      # an id that was never offered
    _reply("0", "99"),                 # one real, one invented
    _reply("Woondeal"),                # a NAME rather than an id
    json.dumps({"move_under_proposal": ["0"], "also_move": ["99"]}),
    json.dumps({"move_under_proposal": [{"id": "99"}]}),
])
def test_the_judge_can_never_widen_the_set(reply):
    """The property that makes delegating this decision safe.

    Asserted over a range of shapes rather than one, because the failure it
    guards against — an entity-side judge inventing a link target — arrived as a
    plausible-looking reply, not a malformed one.
    """
    candidates = _cands("RegioDeal", "Woondeal")
    selection = parse_judge_response(reply, candidates)
    assert selection.widened is False
    assert set(selection.selected).issubset({"0", "1"})


def test_an_unoffered_id_is_reported_not_silently_dropped():
    selection = parse_judge_response(_reply("0", "99"), _cands("A", "B"))
    assert selection.selected == ("0",)
    assert "ignored 1 entr(y/ies) that were not offered" in selection.evidence


# ---------------------------------------------------------------------------
# Silence is not a weak yes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reply,why", [
    ("", "empty reply"),
    ("no json here", "prose only"),
    ("{not valid json", "malformed"),
    (json.dumps({"move_under_proposal": []}), "explicit empty list"),
    (json.dumps({"something_else": ["0"]}), "wrong key"),
])
def test_nothing_is_moved_without_an_explicit_choice(reply, why):
    selection = parse_judge_response(reply, _cands("A", "B"))
    assert selection.selected == (), why
    assert selection.considered == ("0", "1")


def test_an_unmentioned_candidate_is_left_where_it_is():
    selection = parse_judge_response(_reply("0"), _cands("A", "B", "C"))
    assert selection.selected == ("0",)
    assert "1" not in selection.selected and "2" not in selection.selected


# ---------------------------------------------------------------------------
# Distinguishable states, and other fences
# ---------------------------------------------------------------------------


def test_asked_nothing_differs_from_chose_nothing():
    """An empty selection over five candidates is a decision; over zero it is not.

    The entity-side work was rejected twice for collapsing exactly this kind of
    pair, so the two carry different evidence.
    """
    nothing_asked = parse_judge_response(_reply(), ())
    chose_nothing = parse_judge_response(_reply(), _cands("A", "B"))
    assert nothing_asked.selected == chose_nothing.selected == ()
    assert "nothing was asked" in nothing_asked.evidence
    assert "nothing was asked" not in chose_nothing.evidence
    assert nothing_asked.considered == ()
    assert chose_nothing.considered == ("0", "1")


def test_a_repeated_id_counts_once():
    assert parse_judge_response(_reply("0", "0"), _cands("A")).selected == ("0",)


def test_ids_not_names_are_the_key():
    """Two ontologies may define the same type name; the id keeps them apart.

    Descendant of the entity-side defect where a batch keyed by surface form let
    one ruling satisfy two items and link one to the other's target.
    """
    candidates = (("0", "Deal", "from ontology A"), ("1", "Deal", "from ontology B"))
    selection = parse_judge_response(_reply("1"), candidates)
    assert selection.selected == ("1",)


def test_selection_is_immutable():
    selection = JudgeSelection(selected=("0",), considered=("0",))
    with pytest.raises(Exception):
        selection.selected = ("1",)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


def test_prompt_carries_the_ids_and_the_parent():
    prompt = build_judge_prompt(
        "Stadsdeal", "een deal met een stad", "Deal", _cands("RegioDeal", "Woondeal")
    )
    assert "id=0: RegioDeal" in prompt and "id=1: Woondeal" in prompt
    assert '"Stadsdeal"' in prompt and '"Deal"' in prompt
    assert "een deal met een stad" in prompt


def test_prompt_says_an_empty_answer_is_valid():
    # The judge must not feel obliged to move something; over-moving is the
    # damaging direction, since it changes everyone's vocabulary.
    prompt = build_judge_prompt("P", "d", "G", _cands("A"))
    assert "empty list is a valid" in prompt
    assert "leave it where it is" in JUDGE_SYSTEM_PROMPT


def test_prompt_handles_a_description_free_type():
    prompt = build_judge_prompt("P", "", "G", (("0", "A", ""),))
    assert "(none given)" in prompt and "(no description)" in prompt


# ---------------------------------------------------------------------------
# End to end with the real vocabulary
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def deals():
    import asyncio
    from unittest.mock import AsyncMock, patch

    from ontology_manager.registry import OntologyRegistry

    loop = asyncio.new_event_loop()
    try:
        with patch.object(OntologyRegistry, "_load_from_db", AsyncMock(return_value=None)):
            ontology = loop.run_until_complete(OntologyRegistry().get("deals"))
    finally:
        loop.close()
    if ontology is None:  # pragma: no cover - ships with the package
        pytest.skip("deals ontology not available")
    return [ontology]


def test_the_candidate_set_comes_from_the_deterministic_placement(deals):
    """The judge is only ever offered what `type_placement` bounded.

    This is the seam that keeps the delegation safe: the model chooses among a
    handful of definitions that already share a parent, never a graph.
    """
    placement = place_proposed_type("Stadsdeal", "Deal", deals)
    candidates = candidates_from_ontologies(placement.descendant_candidates, deals)
    assert {c[1] for c in candidates} == {"RegioDeal", "Woondeal", "CityDeal"}
    assert [c[0] for c in candidates] == ["0", "1", "2"]

    # and a reply can only ever select within it
    selection = parse_judge_response(_reply("0", "99"), candidates)
    assert selection.widened is False
    assert selection.selected == ("0",)


def test_candidates_carry_real_descriptions(deals):
    candidates = candidates_from_ontologies(("RegioDeal",), deals)
    assert candidates[0][1] == "RegioDeal"
    assert candidates[0][2], "the real ontology defines a description"


def test_an_unknown_type_name_still_yields_a_candidate(deals):
    # `type_placement` only ever passes names it found, but the helper must not
    # raise if a caller passes something else.
    candidates = candidates_from_ontologies(("Verzonnen",), deals)
    assert candidates == (("0", "Verzonnen", ""),)


# ---------------------------------------------------------------------------
# The fences, swept over the whole shipped vocabulary
# ---------------------------------------------------------------------------


def _all_ontologies():
    """Every shipped ontology, derived rather than hardcoded.

    A hardcoded list silently loses coverage when an ontology is added, and
    raises confusingly when one is renamed.
    """
    from ontology_manager.registry import OntologyRegistry

    directory = OntologyRegistry()._find_ontology_dir()
    return sorted(f.stem for f in directory.glob("*.yaml"))


def _load(*names):
    """Load an APPLIED SET — one or more ontologies, as production assembles it.

    `detect_applicable_schemas(..., top_k=3)` means a real applied set normally
    holds three ontologies. That matters here rather than being a detail: within
    ONE `Ontology`, `entity_types` is a name-keyed dict, so a single-ontology load
    makes same-name collisions impossible *by the dict* and any assertion about
    them unfalsifiable. Measured on the combined set the de-duplication fires 42
    times; on `deals` alone, never once.
    """
    import asyncio
    from unittest.mock import AsyncMock, patch

    from ontology_manager.registry import OntologyRegistry

    loop = asyncio.new_event_loop()
    try:
        with patch.object(OntologyRegistry, "_load_from_db", AsyncMock(return_value=None)):
            registry = OntologyRegistry()
            return [
                loop.run_until_complete(registry.get(name)) for name in names
            ]
    finally:
        loop.close()


def test_no_reply_can_widen_any_real_candidate_set():
    """Swept rather than sampled, and in BOTH directions.

    The first version fed only "every valid id plus invented ones", which can
    detect widening — a direction that was never actually at risk. The blocker
    review then mutated the parser to `chosen = list(offered)` (move everything,
    regardless of the reply) and this test still passed, while the real defect
    lived exactly there: a reply of `"10"` iterated as characters and moved two
    types the model never named.

    So the sweep now also asserts the OVER-MOVE direction: every reply shape that
    should select nothing must select nothing, on every parent of every shipped
    ontology.
    """
    from ontology_manager import type_placement as tp

    # Shapes that must yield NOTHING. Each is a real model failure mode, and the
    # first two are the blocker: a string iterates by character, a dict by key.
    refuse = [
        json.dumps({"move_under_proposal": "10"}),
        json.dumps({"move_under_proposal": {"0": False, "1": True}}),
        json.dumps({"move_under_proposal": 2}),
        json.dumps({"move_under_proposal": True}),
        json.dumps({"move_under_proposal": None}),
        json.dumps({"move_under_proposal": [["0"]]}),
        json.dumps({"move_under_proposal": [{"id": "0"}]}),
        '[{"move_under_proposal": ["0"]}]',
        json.dumps({"something_else": ["0"]}),
        "no json at all",
        "",
    ]

    # Applied SETS of three, the shape `detect_applicable_schemas(top_k=3)`
    # produces, plus the whole vocabulary as one set. A single-ontology sweep
    # never exercises cross-ontology de-duplication at all.
    every = _all_ontologies()
    applied_sets = [tuple(every[i : i + 3]) for i in range(0, len(every), 3)]
    applied_sets.append(tuple(every))

    names_seen = 0
    parents = 0
    candidate_count = 0
    for combination in applied_sets:
        schemas = _load(*combination)
        names_seen += len(combination)
        roots = {
            tp.roots_at(d)
            for ontology in schemas
            for d in ontology.entity_types.values()
            if tp.roots_at(d)
        }
        for parent in roots:
            names = tp.sibling_types(parent, schemas)
            if not names:
                continue
            parents += 1
            candidates = candidates_from_ontologies(names, schemas)
            candidate_count += len(candidates)
            ids = [c[0] for c in candidates]

            assert len(set(ids)) == len(ids), (combination, parent)
            assert tuple(c[1] for c in candidates) == tuple(names), (combination, parent)

            # (a) cannot widen
            selection = parse_judge_response(
                json.dumps({"move_under_proposal": ids + ["999", "Verzonnen"]}),
                candidates,
            )
            assert selection.widened is False, (combination, parent)
            assert set(selection.selected) == set(ids), (combination, parent)

            # (b) cannot over-move — the direction the blocker lived in
            for reply in refuse:
                got = parse_judge_response(reply, candidates)
                assert got.selected == (), (combination, parent, reply[:40])

    assert names_seen == 2 * len(every) >= 22, names_seen
    assert parents > 60, f"expected a substantial sweep, covered {parents} parents"
    assert candidate_count > 250, f"enumerated only {candidate_count} candidates"


def test_same_named_types_are_separated_by_their_parents():
    """The fact that makes the id key harmless — over a MULTI-ontology set.

    The first version of this test loaded one ontology at a time, where
    `entity_types` is a name-keyed dict and the property therefore holds by the
    dict rather than by the mechanism: deleting `sibling_types`' de-duplication
    left every test green. It now runs over the whole applied vocabulary, where
    the de-duplication genuinely fires — 250 duplicate NAMES collapsed across the
    swept applied sets — so removing it fails here. (A looser reading of 260 also
    counts the 10 self-rooting types excluded by `sibling_types`' own
    `type_name == target` guard, which is a different mechanism; 250 is the number
    this test is about.)

    What it establishes: no candidate set ever contains two types with the same
    normalised name. Where two ontologies do define one — `Person`,
    `Organization` and `Event` in `base` vs `general` — they root at DIFFERENT
    parents and so never share a set.
    """
    from ontology_manager import type_placement as tp

    schemas = _load(*_all_ontologies())
    checked = 0
    for parent in {tp.roots_at(d) for o in schemas for d in o.entity_types.values()
                   if tp.roots_at(d)}:
        names = tp.sibling_types(parent, schemas)
        lowered = [n.lower() for n in names]
        assert len(set(lowered)) == len(lowered), (parent, names)
        checked += len(names)
    assert checked > 50, f"only {checked} candidates seen; the sweep lost coverage"


def test_a_same_named_pair_really_exists_to_be_deduplicated():
    """Guards the test above from becoming vacuous if the vocabulary changes.

    If no two ontologies ever defined the same type name again, the property
    would hold trivially and stop testing anything.
    """
    schemas = _load(*_all_ontologies())
    seen: dict = {}
    for ontology in schemas:
        for name in ontology.entity_types:
            seen.setdefault(name.lower(), set()).add(ontology.metadata.name)
    shared = {n: o for n, o in seen.items() if len(o) > 1}
    assert shared, "no type name is defined by two ontologies; dedup is untested"


def test_duplicate_caller_ids_are_reported_not_resolved_silently():
    # Unreachable via `candidates_from_ontologies`, but a caller could build it,
    # and the reported NAME would otherwise be arbitrary.
    candidates = (("0", "A", ""), ("0", "B", ""))
    selection = parse_judge_response(_reply("0"), candidates)
    assert "duplicate candidate ids" in selection.evidence


def test_non_string_caller_ids_still_match():
    # A caller passing ints must not silently get a total refusal that reads
    # identical to "the judge chose nothing".
    candidates = ((0, "A", ""), (1, "B", ""))
    assert parse_judge_response(_reply("1"), candidates).selected == ("1",)


@pytest.mark.parametrize("spelling", [
    '[{"move_under_proposal": ["0"]}]',
    '```json\n[{"move_under_proposal": ["0"]}]\n```',
    'Result: [{"move_under_proposal": ["0"]}]',
])
def test_a_top_level_array_is_refused_however_it_is_spelled(spelling):
    """One shape, three spellings — a fence or preamble must not smuggle it past.

    The first version tested `startswith("[")`, so only the bare form was
    refused while the fenced and prose-wrapped forms were descended into.
    """
    assert parse_judge_response(spelling, _cands("A", "B")).selected == ()
