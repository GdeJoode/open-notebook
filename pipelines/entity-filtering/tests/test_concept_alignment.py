"""Track N.4a — concept-alignment verdicts (no seeding, no workflow stage).

Deterministic tiers are pure; the LLM-judge is mocked at the injected
``llm_caller`` seam. These tests encode the v2 design decisions and guard the
regressions the attempt-1 review found:

* D-N4-1 — lexical containment is an ALIAS CANDIDATE, never ``is_a``.
* D-N4-2 — subsumption only from the ontology's declared ancestor chain.
* D-N4-3 — candidates are fetched by CANONICAL type, never the rich label.
* D-N4-7 — evidence is falsifiable: "nothing was queried" is never reported as
  "nothing comparable exists".
* method/judged accounting stays mutually consistent.
"""

from __future__ import annotations

import json

from entity_filtering.resolution import concept_alignment as ca
from entity_filtering.resolution.concept_alignment import (
    EV_NO_CANDIDATES,
    EV_NO_REPO,
    EV_NO_TYPE,
    EV_NO_VECTORS,
    EV_NONE_CLOSE,
    METHOD_EMBEDDING,
    METHOD_JUDGE,
    METHOD_NONE,
    METHOD_TYPE_CHAIN,
    NARROWER_THAN,
    NOVEL,
    RELATED_TO,
    ConceptAligner,
    build_judge_prompt,
    lexical_alias_candidates,
    nearest_by_embedding,
    parse_judge_response,
    type_chain_subsumption,
)


def _row(name, ident="entity:1", embedding=None):
    """A find_by_type row: id/name/embedding/weight — NO type column."""
    row = {"id": ident, "name": name, "weight": 0.0}
    if embedding is not None:
        row["embedding"] = embedding
    return row


def _entity(text, label="", embedding=None, is_new=True, properties=...):
    if properties is not ...:
        return {"text": text, "label": label, "properties": properties}
    props = {"is_new": is_new}
    if embedding is not None:
        props["embedding"] = embedding
    return {"text": text, "label": label, "properties": props}


class _Repo:
    def __init__(self, by_type):
        self._by_type = by_type
        self.calls = []

    async def find_by_type(self, entity_type, limit=100):
        self.calls.append(entity_type)
        return list(self._by_type.get(entity_type, []))


# --- D-N4-1: lexical containment is an alias candidate, never is_a ----------


def test_containment_yields_alias_candidate_not_subsumption():
    got = lexical_alias_candidates(
        "Tweede Kamer der Staten-Generaal", [_row("Tweede Kamer", "entity:tk")]
    )
    assert len(got) == 1
    assert got[0].candidate_name == "Tweede Kamer"
    assert got[0].candidate_id == "entity:tk"
    assert "alias" in got[0].evidence
    assert "NOT a subtype" in got[0].evidence


def test_alias_candidates_are_direction_agnostic():
    # short novel name contained in a longer existing one also pairs
    got = lexical_alias_candidates("Regio Deal", [_row("Regio Deal Midden-Limburg")])
    assert len(got) == 1
    assert "contains" in got[0].evidence


def test_sibling_names_do_not_pair():
    assert lexical_alias_candidates("Gemeente Den Haag", [_row("Gemeente Den Bosch")]) == []


def test_single_shared_word_does_not_pair():
    assert lexical_alias_candidates("Gemeente Groningen", [_row("Gemeente")]) == []


def test_matching_is_token_bounded_not_substring():
    assert lexical_alias_candidates("Regio Dealer Groep", [_row("Regio Deal")]) == []


def test_identical_names_do_not_pair():
    assert lexical_alias_candidates("Regio Deal", [_row("regio  deal")]) == []


async def test_alias_candidates_surface_in_report_without_changing_verdict():
    # The alias pair is reported, but the verdict still comes from the tiers.
    repo = _Repo({"programme": [_row("Tweede Kamer", "entity:tk", embedding=[0.0, 1.0])]})
    aligner = ConceptAligner(repo, schemas=["s"])
    ents, report = await _align_with_types(
        aligner,
        [_entity("Tweede Kamer der Staten-Generaal", "Orgaan", embedding=[1.0, 0.0])],
        canonical="programme",
        ancestors=[],
    )
    assert len(report["alias_candidates"]) == 1
    assert report["alias_candidates"][0]["candidate_name"] == "Tweede Kamer"
    # containment did NOT become a subsumption verdict
    assert ents[0]["properties"]["concept_alignment"] != NARROWER_THAN


# --- D-N4-2: subsumption only from the ontology chain ----------------------


def test_type_chain_fires_only_when_ancestor_is_materialised():
    got = type_chain_subsumption(["Deal"], [_row("Deal", "entity:deal")])
    assert got.verdict == NARROWER_THAN
    assert got.method == METHOD_TYPE_CHAIN
    assert got.target_id == "entity:deal"


def test_type_chain_returns_none_when_ancestor_is_not_a_node():
    # Deliberate: restating the entity's own declared type is an empty claim.
    assert type_chain_subsumption(["Deal"], [_row("Iets anders")]) is None


def test_type_chain_without_ancestors_is_skipped():
    assert type_chain_subsumption([], [_row("Deal")]) is None
    assert type_chain_subsumption(["Deal"], []) is None


def test_type_chain_prefers_the_nearest_ancestor():
    got = type_chain_subsumption(
        ["Deal", "Thing"], [_row("Thing", "entity:thing"), _row("Deal", "entity:deal")]
    )
    assert got.target_id == "entity:deal"


# --- D-N4-3: candidates are fetched by CANONICAL type ----------------------


async def _align_with_types(aligner, entities, *, canonical, ancestors, monkeypatch=None):
    """Run align() with resolve_types stubbed (ontology-manager is optional)."""
    import entity_filtering.resolution.concept_alignment as mod

    original = mod.resolve_types
    mod.resolve_types = lambda label, schemas: (canonical, ancestors)
    try:
        return await aligner.align(entities)
    finally:
        mod.resolve_types = original


async def test_fetch_uses_canonical_type_not_the_rich_label():
    repo = _Repo({"programme": [_row("Regio Deal Zuid", "entity:z", embedding=[1.0, 0.0])]})
    aligner = ConceptAligner(repo, schemas=["s"])
    await _align_with_types(
        aligner,
        [_entity("Regio Deal Noord", "RegioDeal", embedding=[1.0, 0.0])],
        canonical="programme",
        ancestors=["Deal"],
    )
    # queried the canonical enum, never the rich Track-L label
    assert repo.calls == ["programme"]
    assert "RegioDeal" not in repo.calls


async def test_candidate_fetch_is_cached_per_batch():
    repo = _Repo({"programme": [_row("X", "entity:x", embedding=[0.0, 1.0])]})
    aligner = ConceptAligner(repo, schemas=["s"])
    ents = [
        _entity("A", "RegioDeal", embedding=[1.0, 0.0]),
        _entity("B", "RegioDeal", embedding=[1.0, 0.0]),
        _entity("C", "RegioDeal", embedding=[1.0, 0.0]),
    ]
    await _align_with_types(aligner, ents, canonical="programme", ancestors=[])
    assert repo.calls == ["programme"]  # one fetch for three entities


# --- D-N4-7: falsifiable evidence -------------------------------------------


async def test_no_repo_reports_no_repo_not_nothing_comparable():
    aligner = ConceptAligner(None, schemas=["s"])
    ents, report = await _align_with_types(
        aligner, [_entity("X", "RegioDeal")], canonical="programme", ancestors=[]
    )
    props = ents[0]["properties"]
    assert props["concept_alignment"] == NOVEL
    assert props["alignment_reason_code"] == EV_NO_REPO
    assert "never queried" in props["alignment_evidence"]
    assert report["reason_counts"][EV_NO_REPO] == 1


async def test_unresolvable_label_reports_no_type():
    repo = _Repo({})
    aligner = ConceptAligner(repo, schemas=["s"])
    ents, _ = await _align_with_types(
        aligner, [_entity("X", "Onbekend")], canonical=None, ancestors=[]
    )
    assert ents[0]["properties"]["alignment_reason_code"] == EV_NO_TYPE


async def test_empty_fetch_reports_no_candidates_not_none_close():
    repo = _Repo({})  # query runs, returns nothing
    aligner = ConceptAligner(repo, schemas=["s"])
    ents, _ = await _align_with_types(
        aligner, [_entity("X", "RegioDeal")], canonical="programme", ancestors=[]
    )
    props = ents[0]["properties"]
    assert props["alignment_reason_code"] == EV_NO_CANDIDATES
    assert "holds no concepts of canonical type" in props["alignment_evidence"]


async def test_candidates_without_vectors_report_no_vectors():
    repo = _Repo({"programme": [_row("Y", "entity:y")]})  # no embedding
    aligner = ConceptAligner(repo, schemas=["s"])
    ents, _ = await _align_with_types(
        aligner, [_entity("X", "RegioDeal", embedding=[1.0, 0.0])],
        canonical="programme", ancestors=[],
    )
    assert ents[0]["properties"]["alignment_reason_code"] == EV_NO_VECTORS


async def test_far_neighbour_reports_none_close_with_the_distance():
    repo = _Repo({"programme": [_row("Y", "entity:y", embedding=[0.0, 1.0])]})
    aligner = ConceptAligner(repo, schemas=["s"])
    ents, _ = await _align_with_types(
        aligner, [_entity("X", "RegioDeal", embedding=[1.0, 0.0])],
        canonical="programme", ancestors=[],
    )
    props = ents[0]["properties"]
    assert props["alignment_reason_code"] == EV_NONE_CLOSE
    assert "cosine" in props["alignment_evidence"]


# --- embedding band ---------------------------------------------------------


def test_nearest_by_embedding_picks_highest():
    near, score = nearest_by_embedding(
        [1.0, 0.0],
        [_row("far", "entity:f", embedding=[0.0, 1.0]),
         _row("close", "entity:c", embedding=[1.0, 0.05])],
    )
    assert near["id"] == "entity:c"
    assert score > 0.9


def test_nearest_by_embedding_without_vectors():
    assert nearest_by_embedding(None, [_row("x")]) == (None, 0.0)
    assert nearest_by_embedding([1.0, 0.0], [_row("x")]) == (None, 0.0)


def test_orthogonal_neighbour_still_counts_as_compared():
    # cosine 0.0 is a REAL comparison. Seeding best_score at 0.0 would return
    # (None, 0.0) here and the caller would falsely report "no comparable
    # vectors" (D-N4-7).
    near, score = nearest_by_embedding([1.0, 0.0], [_row("orth", "entity:o",
                                                         embedding=[0.0, 1.0])])
    assert near is not None
    assert near["id"] == "entity:o"
    assert score == 0.0


def test_opposed_neighbour_still_counts_as_compared():
    near, score = nearest_by_embedding([1.0, 0.0], [_row("opp", "entity:p",
                                                         embedding=[-1.0, 0.0])])
    assert near is not None
    assert score < 0.0


def test_nearest_by_embedding_tolerates_null_properties():
    row = {"id": "entity:n", "name": "n", "properties": None}
    assert nearest_by_embedding([1.0, 0.0], [row]) == (None, 0.0)


async def test_high_cosine_is_related_by_embedding_alone():
    repo = _Repo({"programme": [_row("Twin", "entity:t", embedding=[1.0, 0.0])]})
    aligner = ConceptAligner(repo, schemas=["s"])
    ents, report = await _align_with_types(
        aligner, [_entity("X", "RegioDeal", embedding=[1.0, 0.0])],
        canonical="programme", ancestors=[],
    )
    props = ents[0]["properties"]
    assert props["concept_alignment"] == RELATED_TO
    assert props["alignment_method"] == METHOD_EMBEDDING
    assert report["judged_count"] == 0  # no judge needed


# --- judge: prompt / parse fencing (pure) -----------------------------------


def test_build_judge_prompt_lists_neighbours():
    p = build_judge_prompt([("Brede Welvaart", ["Leefbaarheid"])])
    assert '"Brede Welvaart"' in p and '"Leefbaarheid"' in p
    assert RELATED_TO in p and NOVEL in p


def test_parse_judge_accepts_related_and_novel():
    items = [("A", ["N1"]), ("B", ["N2"])]
    raw = json.dumps({"alignments": [
        {"text": "A", "verdict": "RELATED_TO", "target": "N1"},
        {"text": "B", "verdict": "NOVEL", "target": None},
    ]})
    assert parse_judge_response(raw, items) == {"A": (RELATED_TO, "N1"), "B": (NOVEL, None)}


def test_parse_judge_rejects_subsumption_verdicts():
    items = [("A", ["N1"])]
    raw = json.dumps({"alignments": [{"text": "A", "verdict": "NARROWER_THAN",
                                      "target": "N1"}]})
    assert parse_judge_response(raw, items) == {}


def test_parse_judge_downgrades_fabricated_target():
    items = [("A", ["N1"])]
    raw = json.dumps({"alignments": [{"text": "A", "verdict": "RELATED_TO",
                                      "target": "Invented"}]})
    assert parse_judge_response(raw, items) == {"A": (NOVEL, None)}


def test_parse_judge_rejects_a_borrowed_target_from_another_item():
    items = [("A", ["N1"]), ("B", ["N2"])]
    raw = json.dumps({"alignments": [{"text": "A", "verdict": "RELATED_TO",
                                      "target": "N2"}]})
    assert parse_judge_response(raw, items) == {"A": (NOVEL, None)}


def test_parse_judge_garbage_returns_empty():
    items = [("A", ["N1"])]
    assert parse_judge_response("not json", items) == {}
    assert parse_judge_response("", items) == {}


# --- judge: orchestration + accounting consistency --------------------------


def _band_repo():
    # cosine ≈ 0.86 → inside [0.75, 0.90) → ambiguous band
    return _Repo({"programme": [_row("Leefbaarheid", "entity:leef",
                                     embedding=[1.0, 0.6])]})


async def test_judge_related_links_but_does_not_merge():
    def caller(system, user, model):
        return json.dumps({"alignments": [
            {"text": "Brede Welvaart", "verdict": "RELATED_TO",
             "target": "Leefbaarheid"}]})

    aligner = ConceptAligner(_band_repo(), schemas=["s"], llm_caller=caller)
    ents, report = await _align_with_types(
        aligner, [_entity("Brede Welvaart", "Thema", embedding=[1.0, 0.0])],
        canonical="programme", ancestors=[],
    )
    props = ents[0]["properties"]
    assert props["concept_alignment"] == RELATED_TO
    assert props["alignment_target_name"] == "Leefbaarheid"
    assert ents[0]["text"] == "Brede Welvaart"  # linked, not merged
    assert report["judged_count"] == 1
    assert report["method_counts"][METHOD_JUDGE] == 1


async def test_silent_judge_item_is_not_stamped_as_judged():
    # The judge rules on A only; B must NOT claim a judge verdict.
    def caller(system, user, model):
        return json.dumps({"alignments": [
            {"text": "A", "verdict": "NOVEL", "target": None}]})

    aligner = ConceptAligner(_band_repo(), schemas=["s"], llm_caller=caller)
    ents, report = await _align_with_types(
        aligner,
        [_entity("A", "Thema", embedding=[1.0, 0.0]),
         _entity("B", "Thema", embedding=[1.0, 0.0])],
        canonical="programme", ancestors=[],
    )
    by_text = {e["text"]: e["properties"] for e in ents}
    assert by_text["A"]["alignment_method"] == METHOD_JUDGE
    assert by_text["B"]["alignment_method"] == METHOD_NONE
    assert "no judge verdict was obtained" in by_text["B"]["alignment_evidence"]
    # accounting stays consistent: judged_count counts only explicit rulings
    assert report["judged_count"] == 1
    assert report["method_counts"][METHOD_JUDGE] == 1


async def test_judge_disabled_leaves_band_novel_without_calling():
    calls = {"n": 0}

    def caller(system, user, model):
        calls["n"] += 1
        return "{}"

    aligner = ConceptAligner(_band_repo(), schemas=["s"], llm_caller=caller,
                             judge_enabled=False)
    ents, report = await _align_with_types(
        aligner, [_entity("X", "Thema", embedding=[1.0, 0.0])],
        canonical="programme", ancestors=[],
    )
    assert ents[0]["properties"]["concept_alignment"] == NOVEL
    assert ents[0]["properties"]["alignment_method"] == METHOD_NONE
    assert calls["n"] == 0
    assert report["judged_count"] == 0


async def test_judge_failure_falls_back_to_novel():
    def caller(system, user, model):
        raise RuntimeError("transport down")

    aligner = ConceptAligner(_band_repo(), schemas=["s"], llm_caller=caller)
    ents, report = await _align_with_types(
        aligner, [_entity("X", "Thema", embedding=[1.0, 0.0])],
        canonical="programme", ancestors=[],
    )
    assert ents[0]["properties"]["concept_alignment"] == NOVEL
    assert report["judged_count"] == 0


# --- non-destructiveness / robustness ---------------------------------------


async def test_already_matched_entities_are_untouched():
    repo = _Repo({"programme": [_row("Y", "entity:y", embedding=[1.0, 0.0])]})
    aligner = ConceptAligner(repo, schemas=["s"])
    ents, report = await _align_with_types(
        aligner, [_entity("X", "RegioDeal", is_new=False)],
        canonical="programme", ancestors=[],
    )
    assert "concept_alignment" not in ents[0]["properties"]
    assert report["aligned_count"] == 0


async def test_null_properties_entity_does_not_crash():
    repo = _Repo({})
    aligner = ConceptAligner(repo, schemas=["s"])
    ents, report = await _align_with_types(
        aligner,
        [_entity("X", "RegioDeal", properties=None),
         _entity("Y", "RegioDeal")],
        canonical="programme", ancestors=[],
    )
    # the null-properties entity is simply not is_new → skipped, no exception
    assert report["aligned_count"] == 1


async def test_no_relations_are_emitted_in_n4a():
    # N.4a is verdicts-only: align() returns exactly (entities, report).
    repo = _Repo({"programme": [_row("Deal", "entity:deal", embedding=[1.0, 0.0])]})
    aligner = ConceptAligner(repo, schemas=["s"])
    result = await _align_with_types(
        aligner, [_entity("X", "RegioDeal", embedding=[1.0, 0.0])],
        canonical="programme", ancestors=["Deal"],
    )
    assert len(result) == 2


async def test_verdict_and_method_counts_sum_to_aligned_count():
    repo = _Repo({"programme": [_row("Deal", "entity:deal", embedding=[0.0, 1.0])]})
    aligner = ConceptAligner(repo, schemas=["s"])
    ents, report = await _align_with_types(
        aligner,
        [_entity("A", "RegioDeal", embedding=[1.0, 0.0]),
         _entity("B", "RegioDeal", embedding=[1.0, 0.0])],
        canonical="programme", ancestors=["Deal"],
    )
    assert sum(report["verdict_counts"].values()) == report["aligned_count"] == 2
    assert sum(report["method_counts"].values()) == report["aligned_count"]


def test_alignment_dataclass_is_frozen():
    a = ca.Alignment(verdict=NOVEL, method=METHOD_NONE, confidence=0.5, evidence="x")
    try:
        a.verdict = NARROWER_THAN  # type: ignore[misc]
    except Exception:
        return
    raise AssertionError("Alignment must be immutable")
