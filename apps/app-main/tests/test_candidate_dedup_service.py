"""Unit tests for the K.5 CandidateDedupService.

Covers the over-merge guards that make this a ×2.0 phase:

* a typo pair the deterministic normalizer misses is caught (fuzzy) — AC1;
* the review band is honoured and NEVER auto-applied (no merge write) — AC2;
* over the extended must-NOT corpus, auto-merge proposals contain ZERO must-NOT
  pairs at the tuned threshold — AC3 / AC6;
* a force-split overlay vetoes a pair even above the auto-threshold — AC4;
* a force-merge overlay is honoured, scoped per-notebook — AC5;
* type-awareness: a person and an org with the same name never propose.

DB-free: a fake repo + fake overlay service stand in. The testcontainer
roundtrip for the overlay table lives in ``test_alias_overlay_roundtrip.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
from app_main.services.entity_resolution.candidate_dedup_service import (
    AUTO_MERGE,
    REVIEW,
    CandidateDedupService,
)
from app_main.services.entity_resolution.overlay_service import OverlayRule
from entity_filtering.config import EmbeddingDedupConfig, FuzzyDedupConfig

_FIXTURES = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "entity_resolution"
)


# --- Fakes --------------------------------------------------------------------


class FakeEntityRepo:
    """In-memory entity store exposing the embedding loader the service uses."""

    def __init__(self, rows: List[Dict[str, Any]]) -> None:
        self._rows = [dict(r) for r in rows]

    async def list_active_entities_with_embeddings(
        self, source_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        out = [dict(r) for r in self._rows]
        if source_ids is not None:
            out = [
                r
                for r in out
                if set(r.get("source_documents") or []) & set(source_ids)
            ]
        return out


class FakeOverlayService:
    """Stand-in overlay that returns canned split pairs + merge rules."""

    def __init__(
        self,
        split_pairs: Optional[set] = None,
        merge_rules: Optional[List[OverlayRule]] = None,
    ) -> None:
        self._split = split_pairs or set()
        self._merge = merge_rules or []

    async def split_pairs(self, notebook_id: Optional[str] = None) -> set:
        return set(self._split)

    async def merge_rules(
        self, notebook_id: Optional[str] = None
    ) -> List[OverlayRule]:
        return list(self._merge)


def _entity(
    eid: str,
    name: str,
    etype: str = "organization",
    confidence: float = 1.0,
    embedding: Optional[list] = None,
    sources: Optional[list] = None,
) -> Dict[str, Any]:
    return {
        "id": eid,
        "canonical_name": name,
        "entity_type": etype,
        "confidence": confidence,
        "embedding": embedding or [],
        "source_documents": sources or [],
    }


def _service(
    rows: List[Dict[str, Any]],
    *,
    split_pairs: Optional[set] = None,
    merge_rules: Optional[List[OverlayRule]] = None,
    fuzzy: Optional[FuzzyDedupConfig] = None,
    embedding: Optional[EmbeddingDedupConfig] = None,
) -> CandidateDedupService:
    return CandidateDedupService(
        entity_repo=FakeEntityRepo(rows),
        overlay_service=FakeOverlayService(split_pairs, merge_rules),
        fuzzy_config=fuzzy or FuzzyDedupConfig(enabled=True),
        embedding_config=embedding or EmbeddingDedupConfig(enabled=True),
    )


def _all_pairs(report) -> set:
    pairs = set()
    for c in [*report.auto_merge, *report.review]:
        pairs.add(frozenset({c.name_a, c.name_b}))
    return pairs


# --- AC1: typo caught by fuzzy ------------------------------------------------


@pytest.mark.asyncio
async def test_typo_pair_proposed() -> None:
    """A typo pair the deterministic normalizer misses is proposed (AC1)."""
    rows = [
        _entity("entity:a", "Koninkrijksrelaties"),
        _entity("entity:b", "Koninkrijksreiaties"),  # OCR-mangled 'l'->'i'
    ]
    report = await _service(rows).propose_candidates()
    proposed = _all_pairs(report)
    assert frozenset({"Koninkrijksrelaties", "Koninkrijksreiaties"}) in proposed
    # ~0.947 → auto-merge band.
    assert report.auto_merge_count == 1
    assert report.auto_merge[0].method == "fuzzy"


# --- AC2: review band not auto-applied ----------------------------------------


@pytest.mark.asyncio
async def test_review_band_returned_not_applied() -> None:
    """A pair in the review band is returned as REVIEW and never written (AC2).

    The candidate has no winner/loser side-effect: ``propose_candidates`` is
    read-only (the fake repo exposes no write path), so producing a review
    candidate cannot have merged anything.
    """
    # Construct a pair scoring inside [0.86, 0.93): "Samenwerkingsverband" vs a
    # one-char variant lands in the review band.
    a = "Regiodeal Oost"
    b = "Regiodeal Oast"  # single-substitution → ~0.92 (review band)
    rows = [_entity("entity:a", a), _entity("entity:b", b)]
    report = await _service(rows).propose_candidates()
    assert report.auto_merge_count == 0
    assert report.review_count == 1
    cand = report.review[0]
    assert cand.band == REVIEW
    assert 0.86 <= cand.score < 0.93


# --- AC3 / AC6: zero must-NOT auto-merges -------------------------------------


def _load_must_not() -> List[Dict[str, Any]]:
    path = _FIXTURES / "must_not_merge.jsonl"
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            pairs.append(json.loads(line))
    return pairs


@pytest.mark.asyncio
async def test_must_not_corpus_zero_auto_merges() -> None:
    """Over the must-NOT corpus, ZERO pairs land in auto-merge (AC3/AC6).

    Each must-NOT pair is seeded as two same-typed entities; running the
    proposer must never auto-merge any of them at the tuned threshold. (Some
    may legitimately appear in the review band for a human; the gate is that
    none auto-merge — and in practice none even reach review here.)
    """
    must_not = _load_must_not()
    false_auto = 0
    for pair in must_not:
        a = pair["a"]
        b = pair["b"]
        name_a = a["name"] if isinstance(a, dict) else a
        name_b = b["name"] if isinstance(b, dict) else b
        type_a = a.get("type", "other") if isinstance(a, dict) else "other"
        type_b = b.get("type", "other") if isinstance(b, dict) else "other"
        rows = [
            _entity("entity:a", name_a, etype=type_a),
            _entity("entity:b", name_b, etype=type_b),
        ]
        report = await _service(rows).propose_candidates()
        for cand in report.auto_merge:
            if frozenset({cand.name_a, cand.name_b}) == frozenset(
                {name_a, name_b}
            ):
                false_auto += 1
    assert false_auto == 0, f"{false_auto} must-NOT pairs auto-merged"


# --- K.5 rev2: discriminator guard --------------------------------------------


_DISCRIMINATOR_PAIRS = [
    ("Gemeente Bergen NH", "Gemeente Bergen NB"),  # province code NH/NB
    ("Wet open overheid 2021", "Wet open overheid 2022"),  # annual version
    ("Kamerstuk 35000-A", "Kamerstuk 35000-B"),  # trailing -A/-B suffix
    ("Tranche I Regio Deal", "Tranche II Regio Deal"),  # short ordinal token
    # Width-varying Roman-numeral ordinals (K.5 rev3): iii(3)/viii(4) and
    # vii(3)/viii(4) are NOT both ≤3 chars nor equal-length, so they bypass the
    # short-token / equal-length branches and need the Roman-numeral rule.
    ("Tranche III Regio Deal", "Tranche VIII Regio Deal"),
    ("Tranche VII Regio Deal", "Tranche VIII Regio Deal"),
]


@pytest.mark.parametrize("name_a,name_b", _DISCRIMINATOR_PAIRS)
@pytest.mark.asyncio
async def test_discriminator_pair_never_auto_merges(
    name_a: str, name_b: str
) -> None:
    """A short trailing/embedded discriminator → REVIEW, never AUTO (K.5 rev2).

    Each pair scores in the fuzzy AUTO band (≈0.94) over a long shared prefix,
    yet the differing discriminator (NH/NB, a year, -A/-B, an ordinal) marks two
    genuinely distinct entities. The guard demotes them to REVIEW so nothing is
    silently collapsed.
    """
    rows = [
        _entity("entity:a", name_a, etype="other"),
        _entity("entity:b", name_b, etype="other"),
    ]
    report = await _service(rows).propose_candidates()
    assert report.auto_merge_count == 0, f"{name_a} / {name_b} auto-merged"
    # The pair is preserved for a human, not dropped.
    assert frozenset({name_a, name_b}) in _all_pairs(report)
    assert report.review_count == 1
    assert report.review[0].band == REVIEW


@pytest.mark.asyncio
async def test_typo_still_auto_merges_through_guard() -> None:
    """A 1-char typo INSIDE a word is unaffected by the discriminator guard.

    The guard only fires on a differing discriminator *token* or digit run; a
    substitution inside a single word preserves token structure, so the typo
    pair still auto-merges (AC1).
    """
    rows = [
        _entity("entity:a", "Koninkrijksrelaties"),
        _entity("entity:b", "Koninkrijksreiaties"),
    ]
    report = await _service(rows).propose_candidates()
    assert report.auto_merge_count == 1
    assert report.auto_merge[0].method == "fuzzy"


def test_roman_numeral_validity_guard_rejects_real_words() -> None:
    """The Roman-numeral detector accepts numerals, rejects Roman-letter words.

    A naive ``^[ivxlcdm]+$`` test would mis-classify real words built from
    Roman letters (``did``, ``dim``, ``mid``, ``civic``). Anchoring to the
    canonical subtractive grammar rejects those, so the rev3 ordinal demotion
    fires ONLY on genuine numeral discriminators.
    """
    is_roman = CandidateDedupService._is_roman_numeral
    for good in ("i", "ii", "iii", "iv", "v", "vii", "viii", "xiii", "xx"):
        assert is_roman(good), good
    for bad in ("did", "dim", "mid", "civic", "lid", "mill", ""):
        assert not is_roman(bad), bad


@pytest.mark.asyncio
async def test_roman_letter_real_word_token_still_auto_merges() -> None:
    """A single differing token that is a real word (not a numeral) still AUTOs.

    ``... beleid`` vs ``... heleid`` (a 1-char OCR typo of a non-numeral word)
    must NOT be demoted by the rev3 Roman rule: the differing tokens are not
    valid Roman numerals, so the guard leaves the typo to auto-merge (AC1).
    """
    rows = [
        _entity("entity:a", "Programma Stedelijk Beleid"),
        _entity("entity:b", "Programma Stedelijk Beieid"),  # OCR l->i typo
    ]
    report = await _service(rows).propose_candidates()
    assert report.auto_merge_count == 1
    assert report.auto_merge[0].method == "fuzzy"


# --- type-awareness -----------------------------------------------------------


@pytest.mark.asyncio
async def test_cross_type_homograph_reaches_review_but_never_auto() -> None:
    """Same name, different type: a REVIEW candidate, never an auto-merge.

    PC.2 deliberately relaxed this guard, and this test records the new contract
    rather than the old one, so the change cannot happen again by accident.

    **What the guard was for, and still is**: type-blind auto-merging would fuse a
    person and an organisation that share a name. That is preserved — the band for
    :data:`FOLD_EQUAL_CROSS_TYPE` has an auto floor of 2.0, which no score can
    reach, so cross-type pairs are structurally unable to auto-merge.

    **What it was over-reaching on**: suppressing the candidate ENTIRELY assumed
    ``entity_type`` is reliable. Measured on the live graph, it is not — 4 of 539
    distinct names exist twice with byte-identical spelling and two different
    types (`Regio Deal` as both `programme` and `topic`; `Nij Begun Academie` as
    both). Those are one entity that unstable typing split, and under the old rule
    no curator could ever see them. That evidence belongs to PC.4.

    So: the human decides, and the machine still cannot decide this one alone.
    """
    rows = [
        _entity("entity:p", "Mark Rutte", etype="person"),
        _entity("entity:o", "Mark Rutte", etype="organization"),
    ]
    report = await _service(rows).propose_candidates()
    assert report.auto_merge_count == 0
    assert report.review_count == 1
    assert report.review[0].method == "fold_equal_cross_type"


@pytest.mark.asyncio
async def test_cross_type_never_auto_merges_even_at_maximum_score() -> None:
    """The cross-type auto floor is unreachable, not merely high.

    A guard expressed as a threshold is only a guard if no input can cross it.
    Drives the band directly with the largest score the pipeline can produce (1.0)
    and asserts the band is still ``review``.
    """
    from app_main.services.entity_resolution.candidate_dedup_service import (
        FOLD_EQUAL_CROSS_TYPE,
    )

    svc = _service([])
    score, method, band = svc._strongest_band({FOLD_EQUAL_CROSS_TYPE: 1.0})
    assert (score, method, band) == (1.0, FOLD_EQUAL_CROSS_TYPE, "review")


# --- AC4: force-split hard veto -----------------------------------------------


@pytest.mark.asyncio
async def test_force_split_vetoes_above_auto_threshold() -> None:
    """A force-split overlay prevents a proposal even at similarity 1.0 (AC4).

    Two near-identical names that would auto-merge are vetoed by a split rule.
    """
    a = "Koninkrijksrelaties"
    b = "Koninkrijksreiaties"
    split = {frozenset({a.lower(), b.lower()})}
    # The service normalizes names; both lowercase to themselves here.
    from shared.utils.name_normalizer import normalize_entity_name

    split = {frozenset({normalize_entity_name(a), normalize_entity_name(b)})}
    rows = [_entity("entity:a", a), _entity("entity:b", b)]
    report = await _service(rows, split_pairs=split).propose_candidates()
    assert report.auto_merge_count == 0
    assert report.review_count == 0


# --- AC5: force-merge hard include --------------------------------------------


@pytest.mark.asyncio
async def test_force_merge_injects_auto_candidate() -> None:
    """A force-merge overlay injects an auto-merge even for dissimilar names."""
    rows = [
        _entity("entity:a", "Stichting Alfa", etype="organization"),
        _entity("entity:b", "Beta Foundation", etype="organization"),
    ]
    rule = OverlayRule(
        id="alias_overlay:1",
        kind="merge",
        scope="global",
        name_a="Stichting Alfa",
        name_b="Beta Foundation",
        entity_type="organization",
    )
    report = await _service(rows, merge_rules=[rule]).propose_candidates()
    assert report.auto_merge_count == 1
    cand = report.auto_merge[0]
    assert cand.method == "overlay"
    assert cand.band == AUTO_MERGE


@pytest.mark.asyncio
async def test_force_split_beats_contradicting_force_merge() -> None:
    """A split rule on the same pair wins over a merge rule (no over-merge)."""
    from shared.utils.name_normalizer import normalize_entity_name

    rows = [
        _entity("entity:a", "Stichting Alfa", etype="organization"),
        _entity("entity:b", "Beta Foundation", etype="organization"),
    ]
    merge_rule = OverlayRule(
        id="alias_overlay:1",
        kind="merge",
        scope="global",
        name_a="Stichting Alfa",
        name_b="Beta Foundation",
        entity_type="organization",
    )
    split = {
        frozenset(
            {
                normalize_entity_name("Stichting Alfa"),
                normalize_entity_name("Beta Foundation"),
            }
        )
    }
    report = await _service(
        rows, split_pairs=split, merge_rules=[merge_rule]
    ).propose_candidates()
    assert report.auto_merge_count == 0


# --- embedding fallback -------------------------------------------------------


@pytest.mark.asyncio
async def test_embedding_pair_proposed_when_vectors_present() -> None:
    """Two dissimilar names with near-identical embeddings are caught (embedding)."""
    rows = [
        _entity(
            "entity:a", "Stichting Alfa", embedding=[1.0, 0.0, 0.0]
        ),
        _entity(
            "entity:b", "Beta Foundation", embedding=[0.999, 0.0447, 0.0]
        ),
    ]
    report = await _service(rows).propose_candidates()
    # Cosine ≈ 0.999 → auto band, via embedding (fuzzy would reject these names).
    assert report.auto_merge_count == 1
    assert report.auto_merge[0].method == "embedding"


@pytest.mark.asyncio
async def test_no_embedding_falls_back_to_fuzzy_only() -> None:
    """With empty embeddings, only the fuzzy path runs (no crash, no embedding)."""
    rows = [
        _entity("entity:a", "Koninkrijksrelaties", embedding=[]),
        _entity("entity:b", "Koninkrijksreiaties", embedding=[]),
    ]
    report = await _service(rows).propose_candidates()
    assert report.auto_merge_count == 1
    assert report.auto_merge[0].method == "fuzzy"


# --- P.1: embedding band gated on vectors -------------------------------------


@pytest.mark.asyncio
async def test_p1_embedding_band_fires_only_when_vectors_present() -> None:
    """P.1 regression: the SAME pair the fuzzy resolver rejects on names is
    proposed by the embedding band ONCE vectors exist — and NOT before.

    This is the whole point of P.1: pre-fix every entity had ``embedding=[]``, so
    the embedding band never fired and dissimilar-but-equivalent names (different
    surface forms, near-identical meaning) were never proposed. After the forward
    fix + backfill populate vectors, the band fires and dedup finds candidates it
    could not see before.
    """
    # Names too dissimilar for fuzzy to propose (no shared stem).
    name_a, name_b = "Stichting Alfa", "Beta Foundation"

    # 1) Empty embeddings (the broken corpus): fuzzy rejects -> zero proposals.
    empty = [
        _entity("entity:a", name_a, embedding=[]),
        _entity("entity:b", name_b, embedding=[]),
    ]
    report_empty = await _service(empty).propose_candidates()
    assert report_empty.auto_merge_count == 0
    assert report_empty.review_count == 0

    # 2) Populated embeddings (post-backfill): the embedding band fires.
    embedded = [
        _entity("entity:a", name_a, embedding=[1.0, 0.0, 0.0]),
        _entity("entity:b", name_b, embedding=[0.999, 0.0447, 0.0]),
    ]
    report_embedded = await _service(embedded).propose_candidates()
    assert report_embedded.auto_merge_count == 1
    assert report_embedded.auto_merge[0].method == "embedding"


# --- to_merge_cluster reuse ---------------------------------------------------


@pytest.mark.asyncio
async def test_candidate_maps_to_merge_cluster() -> None:
    """An auto-merge candidate projects onto a K.3 MergeCluster for apply."""
    rows = [
        _entity("entity:a", "Koninkrijksrelaties", confidence=0.9),
        _entity("entity:b", "Koninkrijksreiaties", confidence=0.5),
    ]
    report = await _service(rows).propose_candidates()
    cluster = report.auto_merge[0].to_merge_cluster()
    # Winner is the higher-confidence entity; loser is the other.
    assert cluster.winner_id == "entity:a"
    assert cluster.loser_ids == ["entity:b"]
    assert cluster.entity_type == "organization"
    # new_canonical reports the WINNER's name.
    assert cluster.new_canonical == "Koninkrijksrelaties"


@pytest.mark.asyncio
async def test_merge_cluster_new_canonical_follows_winner_when_b_wins() -> None:
    """When id_b wins on confidence, new_canonical is name_b, not name_a (MAJOR 2).

    K.3's apply repoints relations onto ``winner_id``; reporting the loser's
    name would mislabel the surviving entity. Here entity:b has the higher
    confidence, so it must win and ``new_canonical`` must be its name.
    """
    rows = [
        _entity("entity:a", "Koninkrijksreiaties", confidence=0.4),
        _entity("entity:b", "Koninkrijksrelaties", confidence=0.95),
    ]
    report = await _service(rows).propose_candidates()
    assert report.auto_merge_count == 1
    cluster = report.auto_merge[0].to_merge_cluster()
    assert cluster.winner_id == "entity:b"
    assert cluster.loser_ids == ["entity:a"]
    # The winner is id_b → new_canonical must be name_b, NOT the hardcoded name_a.
    assert cluster.new_canonical == "Koninkrijksrelaties"


# --- PC.2: one identity ------------------------------------------------------


@pytest.mark.asyncio
async def test_case_only_duplicate_reaches_the_curator() -> None:
    """Two names differing only in case are an auto-merge proposal.

    Measured on the live graph before this generator existed: of 15 case-only
    duplicate groups among active entities, 9 reached the curator by the fuzzy or
    embedding tier and 6 did not — a name short enough or a vector far enough
    apart and the pair was invisible. Equality under the comparison fold is not a
    similarity score, so it does not depend on either tier's threshold.
    """
    rows = [
        _entity("entity:a", "REGIO DEAL"),
        _entity("entity:b", "Regio Deal"),
    ]
    report = await _service(rows).propose_candidates()
    assert report.auto_merge_count == 1
    assert report.auto_merge[0].method == "fold_equal"


@pytest.mark.asyncio
async def test_fold_equal_survives_a_low_embedding_score() -> None:
    """Fold-equality is not outranked by a weak score from another tier.

    The regression this pins: `_record` originally kept one (score, method) pair
    per candidate and overwrote on a higher raw score, so a 0.945 embedding could
    replace a 0.94 fuzzy and DEMOTE the pair from auto to review, because the two
    tiers have different thresholds. Scores from different scales are not
    comparable; the band is.
    """
    svc = _service([])
    best: dict = {}
    svc._record(best, "entity:a", "entity:b", 0.94, "fuzzy")
    svc._record(best, "entity:a", "entity:b", 0.945, "embedding")
    score, method, band = svc._strongest_band(best[("entity:a", "entity:b")])
    assert band == "auto_merge"
    assert (score, method) == (0.94, "fuzzy")


@pytest.mark.asyncio
async def test_head_affix_short_form_reaches_review() -> None:
    """`Gemeente Leudal` / `Leudal`: a review proposal, never an auto-merge.

    Levenshtein puts this pair at 0.53 — no threshold the dedup config would
    accept can reach it, and one that could would also merge `Regio Deal
    Groningen` with `Regio Deal Drenthe`.
    """
    rows = [
        _entity("entity:a", "Gemeente Leudal"),
        _entity("entity:b", "Leudal"),
    ]
    report = await _service(rows).propose_candidates()
    assert report.auto_merge_count == 0
    assert report.review_count == 1
    assert report.review[0].method == "containment"


@pytest.mark.asyncio
async def test_tail_qualified_siblings_are_not_proposed() -> None:
    """`Regio Deal Groningen` / `Regio Deal Drenthe` / `Regio Deal`: no candidate.

    The pair the dedup config names as the reason it will not lower the fuzzy
    threshold. Containment must not smuggle it back in: the extra tokens sit at
    the TAIL, so they discriminate rather than qualify.
    """
    rows = [
        _entity("entity:a", "Regio Deal"),
        _entity("entity:b", "Regio Deal Groningen"),
        _entity("entity:c", "Regio Deal Drenthe"),
    ]
    report = await _service(rows).propose_candidates()
    assert report.auto_merge_count == 0
    assert report.review_count == 0


@pytest.mark.asyncio
async def test_place_name_in_tail_position_is_not_a_short_form() -> None:
    """An organisation named after a place is not that place.

    The failure mode of head-anchored containment with a free length guard: on the
    live graph `Het Hogeland` paired with seven organisations operating there. The
    removed head run is a proper name, not a curated qualifier, so the curated rule
    rejects it.
    """
    rows = [
        _entity("entity:a", "Het Hogeland"),
        _entity("entity:b", "Mensenwerk Het Hogeland"),
        _entity("entity:c", "Ondernemersplatform Het Hogeland"),
    ]
    report = await _service(rows).propose_candidates()
    assert report.auto_merge_count == 0
    assert report.review_count == 0


@pytest.mark.asyncio
async def test_force_split_still_vetoes_the_new_generators() -> None:
    """A force-split overlay beats fold-equality and containment too.

    Without this, adding generators would quietly route around the one control a
    curator has for saying *these are not the same*. Asserted per generator, because
    a veto applied in only one code path is not a veto.
    """
    from shared.utils.name_normalizer import normalize_entity_name

    fold_rows = [_entity("entity:a", "REGIO DEAL"), _entity("entity:b", "Regio Deal")]
    split = {frozenset({normalize_entity_name("REGIO DEAL"), normalize_entity_name("Regio Deal")})}
    report = await _service(fold_rows, split_pairs=split).propose_candidates()
    assert report.auto_merge_count == 0 and report.review_count == 0

    cont_rows = [_entity("entity:c", "Gemeente Leudal"), _entity("entity:d", "Leudal")]
    split = {frozenset({normalize_entity_name("Gemeente Leudal"), normalize_entity_name("Leudal")})}
    report = await _service(cont_rows, split_pairs=split).propose_candidates()
    assert report.auto_merge_count == 0 and report.review_count == 0


@pytest.mark.asyncio
async def test_curated_head_run_does_not_pair_unrelated_names() -> None:
    """A curated head run is not enough — the remainder must actually match.

    Found by mutation: deleting the suffix check left every test green, because
    `Regio Deal` / `Regio Deal Groningen` is rejected by the curated list anyway
    (`regio` is not an affix). The check earns its place on a different shape —
    a curated head over a DIFFERENT tail — where dropping it pairs two unrelated
    municipalities.
    """
    from shared.utils.org_affixes import head_affix

    assert head_affix(["gemeente", "amsterdam"], ["utrecht"]) is None
    assert head_affix(["gemeente", "amsterdam"], ["amsterdam"]) == ("gemeente",)

    rows = [
        _entity("entity:a", "Gemeente Amsterdam"),
        _entity("entity:b", "Utrecht"),
    ]
    report = await _service(rows).propose_candidates()
    assert report.auto_merge_count == 0
    assert report.review_count == 0


@pytest.mark.asyncio
async def test_the_auto_merge_band_never_carries_a_cross_type_pair() -> None:
    """Asserted on the LIST, not on the band function.

    `okf_import_service._run_dedup` iterates `report.auto_merge` and applies every
    entry with `apply_merge` — no human, no confirmation. `fold_equal` is a new
    AUTO producer at score 1.0, so PC.2 widens what that unattended path merges:
    a case-only duplicate of the same type now merges during an OKF import.

    That is intended — same type and equal under the fold is what a duplicate IS,
    and the fold strips no punctuation and folds no diacritics. What must never
    reach it is a cross-type pair, where the graph holds two rows precisely
    because something is unresolved. `_strongest_band` is tested separately; this
    tests the collection that path actually reads.
    """
    rows = [
        _entity("entity:p", "Mark Rutte", etype="person"),
        _entity("entity:o", "Mark Rutte", etype="organization"),
        _entity("entity:c", "REGIO DEAL", etype="programme"),
        _entity("entity:d", "Regio Deal", etype="programme"),
    ]
    report = await _service(rows).propose_candidates()
    assert [c.method for c in report.auto_merge] == ["fold_equal"]
    assert all(c.entity_type_b == "" for c in report.auto_merge), (
        "a candidate carrying two types reached the unattended apply path"
    )
