"""Track N.4a — concept-alignment verdicts (no seeding, no workflow stage).

These tests encode the v2 design decisions and guard the regressions found in the
attempt-1 review AND the attempt-2 review. The recurring defect class is
EVIDENCE OVERCLAIM: reporting an observation as the inference it would license.
Most of the file exists to pin each observation to exactly one reason code.

* D-N4-1/9 — lexical containment is an ALIAS CANDIDATE, never ``is_a``.
* D-N4-12 — this module produces NO subsumption. Three attempts failed at
  instance level because subsumption relates TYPES while this table stores
  MENTIONS; it now lives at the type boundary (N.4d). Only RELATED_TO/NOVEL here.
* D-N4-3  — candidates fetched by CANONICAL type, asserted against the REAL
  ``canonical_bridge``, not a stub of itself.
* D-N4-7  — every negative path names what was observed, never what it implies.
"""

from __future__ import annotations

import json

import pytest
from entity_filtering.resolution import concept_alignment as ca
from entity_filtering.resolution.concept_alignment import (
    EV_EMPTY_TEXT,
    EV_ERROR,
    EV_FETCH_FAILED,
    EV_INCOMPARABLE_VECTORS,
    EV_NO_CANDIDATE_VECTORS,
    EV_NO_QUERY_VECTOR,
    EV_NO_REPO,
    EV_NO_ROWS,
    EV_NO_TYPE,
    EV_NONE_CLOSE,
    METHOD_EMBEDDING,
    METHOD_JUDGE,
    METHOD_NONE,
    NOVEL,
    RELATED_TO,
    ConceptAligner,
    build_judge_prompt,
    lexical_alias_candidates,
    parse_judge_response,
    probe_neighbours,
    resolve_canonical_type,
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
    def __init__(self, by_type, raises=False):
        self._by_type = by_type
        self._raises = raises
        self.calls = []

    async def find_by_type(self, entity_type, limit=100):
        self.calls.append(entity_type)
        if self._raises:
            raise RuntimeError("db down")
        return list(self._by_type.get(entity_type, []))[:limit]


async def _align(aligner, entities, *, canonical, monkeypatch):
    """Run align() with resolve_canonical_type stubbed. The REAL contract of that
    function is asserted separately in the canonical_bridge tests below — this
    stub only isolates the orchestrator."""
    monkeypatch.setattr(ca, "resolve_canonical_type", lambda label, schemas: canonical)
    return await aligner.align(entities)


# ===========================================================================
# D-N4-3 — resolve_canonical_type against the REAL canonical_bridge (review M2)
# ===========================================================================


def _ontology():
    """A minimal real Ontology whose chain terminates on a mapped schema.org base."""
    schema = pytest.importorskip("ontology_manager.schema")
    return schema.Ontology(
        metadata=schema.OntologyMetadata(name="test", version="1"),
        entity_types={
            "RegioDeal": schema.EntityTypeDefinition(
                name="RegioDeal", parent_type="Deal"
            ),
            "Gemeente": schema.EntityTypeDefinition(
                name="Gemeente", parent_type="AdministrativeArea"
            ),
            "Alias": schema.EntityTypeDefinition(
                name="Alias", parent_type="Deal", aliases=["Bijnaam"]
            ),
        },
    )


def test_resolve_returns_the_canonical_enum_not_the_rich_label():
    # The regression that made attempt 1 a silent no-op: the fetch type MUST be
    # the canonical entity_type column value, never the rich Track-L label.
    canonical = resolve_canonical_type("RegioDeal", [_ontology()])
    assert canonical == "programme"          # the DB column value
    assert canonical != "RegioDeal"          # not the rich label


def test_resolve_maps_a_second_type():
    assert resolve_canonical_type("Gemeente", [_ontology()]) == "administrative_area"


def test_resolve_degrades_without_schemas_or_label():
    assert resolve_canonical_type("RegioDeal", None) is None
    assert resolve_canonical_type("", [_ontology()]) is None
    assert resolve_canonical_type("Onbekend", [_ontology()]) is None


async def test_fetch_uses_the_canonical_type_end_to_end(monkeypatch):
    # No stub: the real bridge drives the real fetch.
    repo = _Repo({"programme": [_row("Iets", "entity:i", embedding=[0.0, 1.0])]})
    aligner = ConceptAligner(repo, schemas=[_ontology()])
    await aligner.align([_entity("Regio Deal Noord", "RegioDeal", embedding=[1.0, 0.0])])
    assert repo.calls == ["programme"]
    assert "RegioDeal" not in repo.calls


# ===========================================================================
# D-N4-7 — every negative path names an OBSERVATION, never an inference
# ===========================================================================


async def test_no_repo_says_nothing_was_queried(monkeypatch):
    aligner = ConceptAligner(None, schemas=["s"])
    ents, report = await _align(aligner, [_entity("X", "L")],
                                canonical="programme", monkeypatch=monkeypatch)
    props = ents[0]["properties"]
    assert props["alignment_reason_code"] == EV_NO_REPO
    assert "never queried" in props["alignment_evidence"]
    assert report["reason_counts"][EV_NO_REPO] == 1


async def test_empty_surface_form_has_its_own_code(monkeypatch):
    aligner = ConceptAligner(_Repo({}), schemas=["s"])
    ents, _ = await _align(aligner, [_entity("   ", "L")],
                           canonical="programme", monkeypatch=monkeypatch)
    assert ents[0]["properties"]["alignment_reason_code"] == EV_EMPTY_TEXT


async def test_unresolvable_label_says_no_query_could_be_formed(monkeypatch):
    aligner = ConceptAligner(_Repo({}), schemas=["s"])
    ents, _ = await _align(aligner, [_entity("X", "Onbekend")],
                           canonical=None, monkeypatch=monkeypatch)
    props = ents[0]["properties"]
    assert props["alignment_reason_code"] == EV_NO_TYPE
    assert "no query could be formed" in props["alignment_evidence"]


async def test_fetch_failure_is_not_reported_as_an_empty_graph(monkeypatch):
    # Review B1: a raised fetch must NOT become "the graph holds no concepts".
    aligner = ConceptAligner(_Repo({}, raises=True), schemas=["s"])
    ents, report = await _align(aligner, [_entity("X", "L")],
                                canonical="programme", monkeypatch=monkeypatch)
    props = ents[0]["properties"]
    assert props["alignment_reason_code"] == EV_FETCH_FAILED
    assert "raised" in props["alignment_evidence"]
    assert "nothing was established" in props["alignment_evidence"]
    assert EV_NO_ROWS not in report["reason_counts"]


async def test_fetch_failure_does_not_poison_the_rest_of_the_batch(monkeypatch):
    # B1 aggravation (a): one blip must not mislabel every entity of that type as
    # "no rows". They all get the honest FETCH_FAILED code instead.
    aligner = ConceptAligner(_Repo({}, raises=True), schemas=["s"])
    ents, report = await _align(
        aligner, [_entity("A", "L"), _entity("B", "L"), _entity("C", "L")],
        canonical="programme", monkeypatch=monkeypatch,
    )
    assert report["reason_counts"] == {EV_FETCH_FAILED: 3}


async def test_empty_result_is_stated_as_no_rows_with_the_caveat(monkeypatch):
    # The repository reports a FAILED query as an empty result, so "no rows"
    # cannot claim the graph is empty — the evidence must say so.
    aligner = ConceptAligner(_Repo({}), schemas=["s"])
    ents, _ = await _align(aligner, [_entity("X", "L")],
                           canonical="programme", monkeypatch=monkeypatch)
    props = ents[0]["properties"]
    assert props["alignment_reason_code"] == EV_NO_ROWS
    assert "returned no rows" in props["alignment_evidence"]
    assert "does not by itself prove" in props["alignment_evidence"]


async def test_missing_entity_vector_is_not_blamed_on_the_graph(monkeypatch):
    # Review B2: the NEW entity has no embedding while candidates DO have one.
    repo = _Repo({"programme": [_row("A", "entity:a", embedding=[1.0, 0.0]),
                                _row("B", "entity:b", embedding=[0.0, 1.0])]})
    aligner = ConceptAligner(repo, schemas=["s"])
    ents, _ = await _align(aligner, [_entity("X", "L")],  # no embedding
                           canonical="programme", monkeypatch=monkeypatch)
    props = ents[0]["properties"]
    assert props["alignment_reason_code"] == EV_NO_QUERY_VECTOR
    assert "this entity carries no embedding" in props["alignment_evidence"]
    assert "says nothing about the graph" in props["alignment_evidence"]


async def test_candidates_without_vectors_report_that_specifically(monkeypatch):
    repo = _Repo({"programme": [_row("A", "entity:a")]})  # no embedding on the row
    aligner = ConceptAligner(repo, schemas=["s"])
    ents, _ = await _align(aligner, [_entity("X", "L", embedding=[1.0, 0.0])],
                           canonical="programme", monkeypatch=monkeypatch)
    assert ents[0]["properties"]["alignment_reason_code"] == EV_NO_CANDIDATE_VECTORS


async def test_dimension_mismatch_is_incomparable_not_distant(monkeypatch):
    # Review B3: a 1024-vs-768 pair is NOT "compared and not close".
    repo = _Repo({"programme": [_row("A", "entity:a", embedding=[1.0, 0.0, 0.0])]})
    aligner = ConceptAligner(repo, schemas=["s"])
    ents, _ = await _align(aligner, [_entity("X", "L", embedding=[1.0, 0.0])],
                           canonical="programme", monkeypatch=monkeypatch)
    props = ents[0]["properties"]
    assert props["alignment_reason_code"] == EV_INCOMPARABLE_VECTORS
    assert props["alignment_reason_code"] != EV_NONE_CLOSE
    assert "no comparison was performed" in props["alignment_evidence"]


async def test_zero_norm_vector_is_incomparable(monkeypatch):
    repo = _Repo({"programme": [_row("A", "entity:a", embedding=[0.0, 0.0])]})
    aligner = ConceptAligner(repo, schemas=["s"])
    ents, _ = await _align(aligner, [_entity("X", "L", embedding=[1.0, 0.0])],
                           canonical="programme", monkeypatch=monkeypatch)
    assert ents[0]["properties"]["alignment_reason_code"] == EV_INCOMPARABLE_VECTORS


async def test_internal_error_has_its_own_code(monkeypatch):
    # Review B4: a crash must not be stamped "no rows"/"none close" — N.4c filters
    # gap-recording on this code.
    repo = _Repo({"programme": [_row("A", "entity:a", embedding=[1.0, 0.0])]})
    aligner = ConceptAligner(repo, schemas=["s"])
    monkeypatch.setattr(ca, "resolve_canonical_type",
                        lambda label, schemas: (_ for _ in ()).throw(RuntimeError("boom")))
    ents, report = await aligner.align([_entity("X", "L", embedding=[1.0, 0.0])])
    props = ents[0]["properties"]
    assert props["alignment_reason_code"] == EV_ERROR
    assert report["reason_counts"] == {EV_ERROR: 1}


async def test_genuine_distance_is_the_only_none_close_path(monkeypatch):
    repo = _Repo({"programme": [_row("A", "entity:a", embedding=[0.0, 1.0])]})
    aligner = ConceptAligner(repo, schemas=["s"])
    ents, _ = await _align(aligner, [_entity("X", "L", embedding=[1.0, 0.0])],
                           canonical="programme", monkeypatch=monkeypatch)
    props = ents[0]["properties"]
    assert props["alignment_reason_code"] == EV_NONE_CLOSE
    assert "compared concepts" in props["alignment_evidence"]
    assert props["alignment_similarity"] == 0.0  # orthogonal, genuinely compared


# --- M4: NOVEL must not imply the whole graph was seen ---------------------


async def test_capped_sample_is_disclosed_in_the_evidence(monkeypatch):
    rows = [_row(f"E{i}", f"entity:{i}", embedding=[0.0, 1.0]) for i in range(5)]
    aligner = ConceptAligner(_Repo({"programme": rows}), schemas=["s"], max_candidates=5)
    ents, report = await _align(aligner, [_entity("X", "L", embedding=[1.0, 0.0])],
                                canonical="programme", monkeypatch=monkeypatch)
    ev = ents[0]["properties"]["alignment_evidence"]
    assert "arbitrary sample" in ev and "LIMIT-capped" in ev
    assert report["capped_type_fetches"] == ["programme"]
    assert report["candidate_cap"] == 5


async def test_uncapped_fetch_makes_no_sampling_claim(monkeypatch):
    rows = [_row("E0", "entity:0", embedding=[0.0, 1.0])]
    aligner = ConceptAligner(_Repo({"programme": rows}), schemas=["s"], max_candidates=5)
    ents, report = await _align(aligner, [_entity("X", "L", embedding=[1.0, 0.0])],
                                canonical="programme", monkeypatch=monkeypatch)
    assert "arbitrary sample" not in ents[0]["properties"]["alignment_evidence"]
    assert report["capped_type_fetches"] == []


# ===========================================================================
# probe_neighbours — the cause-separating primitive
# ===========================================================================


def test_probe_separates_the_three_causes():
    no_query = probe_neighbours(None, [_row("a", embedding=[1.0, 0.0])])
    assert no_query.reason_code() == EV_NO_QUERY_VECTOR

    no_cand = probe_neighbours([1.0, 0.0], [_row("a")])
    assert no_cand.reason_code() == EV_NO_CANDIDATE_VECTORS

    incomparable = probe_neighbours([1.0, 0.0], [_row("a", embedding=[1.0, 0.0, 0.0])])
    assert incomparable.reason_code() == EV_INCOMPARABLE_VECTORS
    assert incomparable.skipped_incomparable == 1


def test_probe_counts_a_real_comparison():
    p = probe_neighbours([1.0, 0.0], [_row("a", "entity:a", embedding=[1.0, 0.0])])
    assert p.reason_code() is None
    assert p.compared == 1
    assert p.nearest["id"] == "entity:a"
    assert p.score == pytest.approx(1.0)


def test_orthogonal_and_opposed_neighbours_still_count_as_compared():
    # Any numeric sentinel inside the cosine range drops one of these.
    orth = probe_neighbours([1.0, 0.0], [_row("o", "entity:o", embedding=[0.0, 1.0])])
    assert orth.nearest is not None and orth.score == 0.0 and orth.compared == 1

    opp = probe_neighbours([1.0, 0.0], [_row("p", "entity:p", embedding=[-1.0, 0.0])])
    assert opp.nearest is not None and opp.score < 0.0 and opp.compared == 1


def test_probe_picks_the_highest_and_tolerates_null_properties():
    p = probe_neighbours(
        [1.0, 0.0],
        [{"id": "entity:n", "name": "n", "properties": None},
         _row("far", "entity:f", embedding=[0.0, 1.0]),
         _row("close", "entity:c", embedding=[1.0, 0.05])],
    )
    assert p.nearest["id"] == "entity:c"
    assert p.skipped_no_vector == 1


# ===========================================================================
# D-N4-1/9 — lexical containment is an alias candidate, never a verdict
# ===========================================================================


def test_containment_yields_alias_candidate_not_subsumption():
    got = lexical_alias_candidates(
        "Tweede Kamer der Staten-Generaal", [_row("Tweede Kamer", "entity:tk")]
    )
    assert len(got) == 1
    assert got[0].candidate_id == "entity:tk"
    assert "alias" in got[0].evidence and "NOT a subtype" in got[0].evidence


def test_alias_candidates_are_direction_agnostic():
    got = lexical_alias_candidates("Regio Deal", [_row("Regio Deal Midden-Limburg")])
    assert len(got) == 1


def test_sibling_and_single_word_and_substring_do_not_pair():
    assert lexical_alias_candidates("Gemeente Den Haag", [_row("Gemeente Den Bosch")]) == []
    assert lexical_alias_candidates("Gemeente Groningen", [_row("Gemeente")]) == []
    assert lexical_alias_candidates("Regio Dealer Groep", [_row("Regio Deal")]) == []
    assert lexical_alias_candidates("Regio Deal", [_row("regio  deal")]) == []


async def test_alias_candidates_are_reported_without_becoming_a_verdict(monkeypatch):
    repo = _Repo({"programme": [_row("Tweede Kamer", "entity:tk", embedding=[0.0, 1.0])]})
    aligner = ConceptAligner(repo, schemas=["s"])
    ents, report = await _align(
        aligner,
        [_entity("Tweede Kamer der Staten-Generaal", "L", embedding=[1.0, 0.0])],
        canonical="programme", monkeypatch=monkeypatch,
    )
    assert report["alias_candidates"][0]["candidate_name"] == "Tweede Kamer"
    # containment is a review candidate, never a verdict
    assert ents[0]["properties"]["concept_alignment"] in (RELATED_TO, NOVEL)


# ===========================================================================
# D-N4-12 — this module produces NO subsumption
# ===========================================================================


def test_only_related_and_novel_are_producible():
    """Three attempts failed at instance level before subsumption moved to the
    type boundary. The taxonomy here is deliberately two-valued now, so a
    reintroduction has to change this test rather than slip past it."""
    assert set(ca.VERDICTS) == {RELATED_TO, NOVEL}
    assert not hasattr(ca, "type_chain_subsumption")
    assert not hasattr(ca, "build_is_a_seeds")


async def test_align_returns_only_entities_and_a_report(monkeypatch):
    # Module-level counterpart to the workflow's no-relations assertion: align()
    # has no relation channel at all, so nothing can be emitted by accident.
    aligner = ConceptAligner(_Repo({}), schemas=["s"])
    result = await _align(aligner, [_entity("X", "L")],
                          canonical="programme", monkeypatch=monkeypatch)
    assert len(result) == 2
    entities, report = result
    assert isinstance(entities, list) and isinstance(report, dict)
    assert not any("relation" in k for k in report)


# ===========================================================================
# Judge — fencing and accounting
# ===========================================================================


def _items():
    return [("0", "A", ["N1"]), ("1", "B", ["N2"])]


def test_build_judge_prompt_carries_ids_and_neighbours():
    p = build_judge_prompt(_items())
    assert "id=0" in p and '"A"' in p and '"N1"' in p
    assert RELATED_TO in p and NOVEL in p


def test_parse_judge_accepts_related_and_novel():
    raw = json.dumps({"alignments": [
        {"id": "0", "verdict": "RELATED_TO", "target": "N1"},
        {"id": "1", "verdict": "NOVEL", "target": None},
    ]})
    assert parse_judge_response(raw, _items()) == {"0": (RELATED_TO, "N1"),
                                                   "1": (NOVEL, None)}


def test_parse_judge_rejects_subsumption_and_unknown_ids():
    assert parse_judge_response(json.dumps({"alignments": [
        {"id": "0", "verdict": "NARROWER_THAN", "target": "N1"}]}), _items()) == {}
    assert parse_judge_response(json.dumps({"alignments": [
        {"id": "99", "verdict": "RELATED_TO", "target": "N1"}]}), _items()) == {}


def test_parse_judge_downgrades_invented_and_borrowed_targets():
    invented = json.dumps({"alignments": [
        {"id": "0", "verdict": "RELATED_TO", "target": "Invented"}]})
    assert parse_judge_response(invented, _items()) == {"0": (NOVEL, None)}
    # N2 belongs to item 1, not item 0 — borrowing is not a link (review M1)
    borrowed = json.dumps({"alignments": [
        {"id": "0", "verdict": "RELATED_TO", "target": "N2"}]})
    assert parse_judge_response(borrowed, _items()) == {"0": (NOVEL, None)}


def test_parse_judge_garbage_returns_empty():
    assert parse_judge_response("not json", _items()) == {}
    assert parse_judge_response("", _items()) == {}


def _band_repo():
    # cosine ≈ 0.86 → inside [0.75, 0.90) → the ambiguous band
    return _Repo({"programme": [_row("Leefbaarheid", "entity:leef",
                                     embedding=[1.0, 0.6])]})


async def test_judge_related_links_but_does_not_merge(monkeypatch):
    def caller(system, user, model):
        return json.dumps({"alignments": [
            {"id": "0", "verdict": "RELATED_TO", "target": "Leefbaarheid"}]})

    aligner = ConceptAligner(_band_repo(), schemas=["s"], llm_caller=caller)
    ents, report = await _align(aligner,
                                [_entity("Brede Welvaart", "L", embedding=[1.0, 0.0])],
                                canonical="programme", monkeypatch=monkeypatch)
    props = ents[0]["properties"]
    assert props["concept_alignment"] == RELATED_TO
    assert props["alignment_target_name"] == "Leefbaarheid"
    assert props["alignment_target_id"] == "entity:leef"
    assert ents[0]["text"] == "Brede Welvaart"  # linked, not merged
    assert report["judged_count"] == 1 == report["method_counts"][METHOD_JUDGE]


async def test_duplicate_surface_forms_do_not_share_one_ruling(monkeypatch):
    """Review M1: two novel entities with the SAME name must be judged separately.

    Keying the batch on text let one ruling satisfy both items, skewed
    judged_count vs method_counts, and let item A be linked to item B's neighbour
    — with target_id and target_name then pointing at different nodes.
    """
    repo = _Repo({"programme": [_row("Alpha", "entity:alpha", embedding=[1.0, 0.6])],
                  "location": [_row("Beta", "entity:beta", embedding=[1.0, 0.6])]})

    def caller(system, user, model):
        # rules on item 0 only
        return json.dumps({"alignments": [
            {"id": "0", "verdict": "RELATED_TO", "target": "Alpha"}]})

    aligner = ConceptAligner(repo, schemas=["s"], llm_caller=caller)
    calls = {"n": 0}

    def fake_resolve(label, schemas):
        calls["n"] += 1
        return "programme" if label == "Gemeente" else "location"

    monkeypatch.setattr(ca, "resolve_canonical_type", fake_resolve)
    ents, report = await aligner.align([
        _entity("Den Haag", "Gemeente", embedding=[1.0, 0.0]),
        _entity("Den Haag", "Locatie", embedding=[1.0, 0.0]),
    ])

    first, second = ents[0]["properties"], ents[1]["properties"]
    assert first["concept_alignment"] == RELATED_TO
    assert first["alignment_target_name"] == "Alpha"
    assert first["alignment_target_id"] == "entity:alpha"  # same node, not Beta's
    # the unruled twin is NOT stamped as judged
    assert second["concept_alignment"] == NOVEL
    assert second["alignment_method"] == METHOD_NONE
    # accounting stays consistent
    assert report["judged_count"] == 1 == report["method_counts"][METHOD_JUDGE]


async def test_silent_judge_item_is_not_stamped_as_judged(monkeypatch):
    def caller(system, user, model):
        return json.dumps({"alignments": [
            {"id": "0", "verdict": "NOVEL", "target": None}]})

    aligner = ConceptAligner(_band_repo(), schemas=["s"], llm_caller=caller)
    ents, report = await _align(
        aligner,
        [_entity("A", "L", embedding=[1.0, 0.0]), _entity("B", "L", embedding=[1.0, 0.0])],
        canonical="programme", monkeypatch=monkeypatch,
    )
    assert ents[0]["properties"]["alignment_method"] == METHOD_JUDGE
    assert ents[1]["properties"]["alignment_method"] == METHOD_NONE
    assert "no judge verdict was obtained" in ents[1]["properties"]["alignment_evidence"]
    assert report["judged_count"] == 1 == report["method_counts"][METHOD_JUDGE]


async def test_judge_disabled_and_judge_failure_both_fall_back_to_novel(monkeypatch):
    calls = {"n": 0}

    def caller(system, user, model):
        calls["n"] += 1
        return "{}"

    off = ConceptAligner(_band_repo(), schemas=["s"], llm_caller=caller, judge_enabled=False)
    ents, report = await _align(off, [_entity("X", "L", embedding=[1.0, 0.0])],
                                canonical="programme", monkeypatch=monkeypatch)
    assert ents[0]["properties"]["concept_alignment"] == NOVEL
    assert calls["n"] == 0 and report["judged_count"] == 0

    def boom(system, user, model):
        raise RuntimeError("transport down")

    failed = ConceptAligner(_band_repo(), schemas=["s"], llm_caller=boom)
    ents2, report2 = await _align(failed, [_entity("X", "L", embedding=[1.0, 0.0])],
                                  canonical="programme", monkeypatch=monkeypatch)
    assert ents2[0]["properties"]["concept_alignment"] == NOVEL
    assert report2["judged_count"] == 0


async def test_async_judge_caller_is_awaited(monkeypatch):
    async def caller(system, user, model):
        return json.dumps({"alignments": [
            {"id": "0", "verdict": "RELATED_TO", "target": "Leefbaarheid"}]})

    aligner = ConceptAligner(_band_repo(), schemas=["s"], llm_caller=caller)
    ents, report = await _align(aligner, [_entity("X", "L", embedding=[1.0, 0.0])],
                                canonical="programme", monkeypatch=monkeypatch)
    assert ents[0]["properties"]["concept_alignment"] == RELATED_TO
    assert report["judged_count"] == 1


# ===========================================================================
# Confidence units, non-destructiveness, accounting
# ===========================================================================


async def test_cosine_is_not_written_as_confidence(monkeypatch):
    # A raw similarity must not outrank an ontological confidence (review minor 5).
    repo = _Repo({"programme": [_row("Twin", "entity:t", embedding=[1.0, 0.0])]})
    aligner = ConceptAligner(repo, schemas=["s"])
    ents, _ = await _align(aligner, [_entity("X", "L", embedding=[1.0, 0.0])],
                           canonical="programme", monkeypatch=monkeypatch)
    props = ents[0]["properties"]
    assert props["concept_alignment"] == RELATED_TO
    assert props["alignment_method"] == METHOD_EMBEDDING
    assert props["alignment_similarity"] == pytest.approx(1.0)
    assert props["alignment_confidence"] == ca._CONF_EMBEDDING


async def test_canonical_type_is_carried_for_n4b(monkeypatch):
    repo = _Repo({"programme": [_row("Twin", "entity:t", embedding=[1.0, 0.0])]})
    aligner = ConceptAligner(repo, schemas=["s"])
    ents, _ = await _align(aligner, [_entity("X", "L", embedding=[1.0, 0.0])],
                           canonical="programme", monkeypatch=monkeypatch)
    assert ents[0]["properties"]["alignment_canonical_type"] == "programme"


async def test_already_matched_entities_are_untouched(monkeypatch):
    repo = _Repo({"programme": [_row("Y", "entity:y", embedding=[1.0, 0.0])]})
    aligner = ConceptAligner(repo, schemas=["s"])
    ents, report = await _align(aligner, [_entity("X", "L", is_new=False)],
                                canonical="programme", monkeypatch=monkeypatch)
    assert "concept_alignment" not in ents[0]["properties"]
    assert report["aligned_count"] == 0


async def test_null_properties_entity_is_skipped_not_crashed(monkeypatch):
    aligner = ConceptAligner(_Repo({}), schemas=["s"])
    ents, report = await _align(
        aligner, [_entity("X", "L", properties=None), _entity("Y", "L")],
        canonical="programme", monkeypatch=monkeypatch,
    )
    assert report["aligned_count"] == 1  # the null-properties row is not is_new


async def test_counts_are_mutually_consistent(monkeypatch):
    repo = _Repo({"programme": [_row("Deal", "entity:deal", embedding=[0.0, 1.0])]})
    aligner = ConceptAligner(repo, schemas=["s"])
    _, report = await _align(
        aligner,
        [_entity("A", "L", embedding=[1.0, 0.0]), _entity("B", "L", embedding=[1.0, 0.0])],
        canonical="programme", monkeypatch=monkeypatch,
    )
    assert sum(report["verdict_counts"].values()) == report["aligned_count"] == 2
    assert sum(report["method_counts"].values()) == report["aligned_count"]
    assert sum(report["reason_counts"].values()) <= report["aligned_count"]
    assert report["method_counts"][METHOD_JUDGE] <= report["judged_count"]


def test_alignment_is_immutable():
    a = ca.Alignment(verdict=NOVEL, method=METHOD_NONE, confidence=0.5, evidence="x")
    with pytest.raises(Exception):
        a.verdict = RELATED_TO  # type: ignore[misc]


# ===========================================================================
# Carried items C2 / C4 (review M4) — the disclosures added in N.4b
# ===========================================================================


async def test_cap_is_disclosed_when_no_candidate_had_a_vector(monkeypatch):
    rows = [_row(f"E{i}", f"entity:{i}") for i in range(3)]
    aligner = ConceptAligner(_Repo({"programme": rows}), schemas=["s"],
                             max_candidates=3)
    ents, _ = await _align(aligner, [_entity("X", "L", embedding=[1.0, 0.0])],
                           canonical="programme", monkeypatch=monkeypatch)
    props = ents[0]["properties"]
    assert props["alignment_reason_code"] == EV_NO_CANDIDATE_VECTORS
    assert "arbitrary sample" in props["alignment_evidence"]


async def test_cap_is_not_disclosed_when_the_entity_itself_lacks_a_vector(monkeypatch):
    # A fact about the INPUT — the sample size is irrelevant to it.
    rows = [_row(f"E{i}", f"entity:{i}", embedding=[1.0, 0.0]) for i in range(3)]
    aligner = ConceptAligner(_Repo({"programme": rows}), schemas=["s"],
                             max_candidates=3)
    ents, _ = await _align(aligner, [_entity("X", "L")],
                           canonical="programme", monkeypatch=monkeypatch)
    props = ents[0]["properties"]
    assert props["alignment_reason_code"] == EV_NO_QUERY_VECTOR
    assert "arbitrary sample" not in props["alignment_evidence"]


async def test_cap_is_disclosed_on_the_judge_path(monkeypatch):
    rows = [_row("Leefbaarheid", "entity:leef", embedding=[1.0, 0.6])]
    aligner = ConceptAligner(_Repo({"programme": rows}), schemas=["s"],
                             max_candidates=1, judge_enabled=False)
    ents, _ = await _align(aligner, [_entity("X", "L", embedding=[1.0, 0.0])],
                           canonical="programme", monkeypatch=monkeypatch)
    assert "arbitrary sample" in ents[0]["properties"]["alignment_evidence"]


async def test_no_repo_verdict_still_carries_the_canonical_type(monkeypatch):
    # C4: the type resolved fine; only the repository was absent.
    aligner = ConceptAligner(None, schemas=["s"])
    ents, _ = await _align(aligner, [_entity("X", "L")],
                           canonical="programme", monkeypatch=monkeypatch)
    props = ents[0]["properties"]
    assert props["alignment_reason_code"] == EV_NO_REPO
    assert props["alignment_canonical_type"] == "programme"
