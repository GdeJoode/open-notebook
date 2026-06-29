"""Unit tests for the contradiction judge (Track Z.2) — LLM fully MOCKED.

The judge's job is precision: only a *confident* contradicts/reinforces verdict
becomes a ``source_verdict`` edge; everything else (neutral, low-confidence,
malformed) writes NOTHING. These tests prove the SERVICE drives that correctly
against AsyncMock collaborators — no DB, no live model. The Z.1 ``relate_verdict``
idempotency / injection-safety is proven against a real container in
``packages/surrealdb-service/tests/test_source_verdict_relate.py``; here we prove
the judge calls it (and only it) for the right pairs.

Coverage:
  * candidate generation: dedup, self-exclusion, top-k bound, order;
  * verdict parse: each class (contradicts / reinforces / neutral) + malformed →
    neutral, never a crash;
  * the precision gate: confident contradicts → edge; confident reinforces →
    edge; neutral → NO edge; low-confidence contradicts → NO edge; malformed →
    NO edge;
  * idempotency-driving: every persisted verdict goes through ``relate_verdict``
    (itself clear-before-relate), so re-running re-issues identical calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from app_main.services.contradiction_judge_service import (
    DEFAULT_MIN_CONFIDENCE,
    MAX_K,
    ContradictionJudgeService,
    JudgeVerdict,
    build_candidate_pairs,
    parse_verdict,
)

# ``asyncio_mode = "auto"`` (pyproject) runs the async tests; the pure
# candidate/parse tests below are plain sync — no module-level asyncio mark.


# --- candidate generation (pure) -------------------------------------------


def test_candidate_pairs_basic_pairs_with_source_first():
    pairs = build_candidate_pairs("source:a", ["source:b", "source:c"], k=5)
    assert pairs == [("source:a", "source:b"), ("source:a", "source:c")]


def test_candidate_pairs_excludes_self():
    pairs = build_candidate_pairs(
        "source:a", ["source:a", "source:b"], k=5
    )
    assert pairs == [("source:a", "source:b")]
    assert all(b != "source:a" for _, b in pairs)


def test_candidate_pairs_dedups_repeated_related():
    pairs = build_candidate_pairs(
        "source:a", ["source:b", "source:b", "source:c", "source:b"], k=5
    )
    assert pairs == [("source:a", "source:b"), ("source:a", "source:c")]


def test_candidate_pairs_bounded_by_topk_in_rank_order():
    related = [f"source:{i}" for i in range(10)]
    pairs = build_candidate_pairs("source:x", related, k=3)
    assert len(pairs) == 3
    # ranking order preserved
    assert pairs == [
        ("source:x", "source:0"),
        ("source:x", "source:1"),
        ("source:x", "source:2"),
    ]


def test_candidate_pairs_k_zero_or_negative_is_empty():
    assert build_candidate_pairs("source:a", ["source:b"], k=0) == []
    assert build_candidate_pairs("source:a", ["source:b"], k=-1) == []


def test_candidate_pairs_skips_empty_ids():
    pairs = build_candidate_pairs("source:a", ["", "source:b", None], k=5)  # type: ignore[list-item]
    assert pairs == [("source:a", "source:b")]


# --- verdict parse: each class + malformed ---------------------------------


def test_parse_contradicts():
    v = parse_verdict(
        '{"verdict": "contradicts", "confidence": 0.91, "reasoning": "A says X, B says not-X"}'
    )
    assert v.verdict == "contradicts"
    assert v.confidence == pytest.approx(0.91)
    assert "not-X" in v.reasoning


def test_parse_reinforces():
    v = parse_verdict(
        '{"verdict": "reinforces", "confidence": 0.8, "reasoning": "same conclusion"}'
    )
    assert v.verdict == "reinforces"
    assert v.confidence == pytest.approx(0.8)


def test_parse_neutral():
    v = parse_verdict(
        '{"verdict": "neutral", "confidence": 0.2, "reasoning": "different aspects"}'
    )
    assert v.verdict == "neutral"


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "not json at all",
        "{not: valid, json}",
        '{"confidence": 0.9}',  # missing verdict -> neutral
        '{"verdict": "definitely-contradicts", "confidence": 0.99}',  # unknown label
        "[1, 2, 3]",  # JSON array, not object
        None,
    ],
)
def test_parse_malformed_degrades_to_neutral(raw):
    v = parse_verdict(raw)
    assert v.verdict == "neutral"
    # A malformed/unknown response must never be persistable.
    assert not v.is_persistable(DEFAULT_MIN_CONFIDENCE)


def test_parse_strips_fences_and_prose():
    raw = (
        "Here is my judgment:\n```json\n"
        '{"verdict": "contradicts", "confidence": 0.95, "reasoning": "clear conflict"}'
        "\n```\nThat is final."
    )
    v = parse_verdict(raw)
    assert v.verdict == "contradicts"
    assert v.confidence == pytest.approx(0.95)


def test_parse_clamps_confidence_out_of_range():
    assert parse_verdict('{"verdict": "neutral", "confidence": 5}').confidence == 1.0
    assert parse_verdict('{"verdict": "neutral", "confidence": -3}').confidence == 0.0


def test_parse_valid_verdict_with_bad_confidence_is_not_persistable():
    # A valid label but a non-numeric confidence: the label is preserved, the
    # confidence degrades to 0.0, so the precision gate (not the parser) blocks
    # the edge. Proves a parse hiccup can only ever SUPPRESS, never fabricate.
    v = parse_verdict('{"verdict": "contradicts", "confidence": "high"}')
    assert v.verdict == "contradicts"
    assert v.confidence == 0.0
    assert not v.is_persistable(DEFAULT_MIN_CONFIDENCE)


def test_is_persistable_gate_logic():
    assert JudgeVerdict("contradicts", 0.9, "").is_persistable(0.7)
    assert JudgeVerdict("reinforces", 0.7, "").is_persistable(0.7)  # boundary >=
    assert not JudgeVerdict("contradicts", 0.69, "").is_persistable(0.7)
    assert not JudgeVerdict("neutral", 1.0, "").is_persistable(0.7)  # never neutral


# --- judge_pair precision gate (each verdict class -> edge / no edge) -------


def _src(sid: str):
    s = MagicMock()
    s.id = sid
    s.title = f"Title {sid}"
    s.full_text = f"Full text of {sid}"
    return s


def _service(llm_response: str, *, judge_model: str = "test-judge"):
    """Build a judge with AsyncMock collaborators + a fixed LLM response."""
    source_service = MagicMock()
    source_service.get = AsyncMock(side_effect=lambda sid: _src(sid))

    related_service = MagicMock()
    source_repo = MagicMock()
    source_repo.relate_verdict = AsyncMock(return_value=True)

    llm_caller = AsyncMock(return_value=llm_response)

    svc = ContradictionJudgeService(
        source_service=source_service,
        related_service=related_service,
        source_repo=source_repo,
        llm_caller=llm_caller,
        judge_model=judge_model,
    )
    return svc, source_repo, llm_caller


async def test_confident_contradicts_writes_edge():
    svc, repo, _ = _service(
        '{"verdict": "contradicts", "confidence": 0.9, "reasoning": "conflict"}'
    )
    verdict, written = await svc.judge_pair("source:a", "source:b")
    assert verdict.verdict == "contradicts"
    assert written is True
    repo.relate_verdict.assert_awaited_once()
    _, kwargs = repo.relate_verdict.call_args
    assert kwargs["verdict"] == "contradicts"
    assert kwargs["confidence"] == pytest.approx(0.9)
    assert kwargs["judge_model"] == "test-judge"


async def test_confident_reinforces_writes_edge():
    svc, repo, _ = _service(
        '{"verdict": "reinforces", "confidence": 0.85, "reasoning": "corroborates"}'
    )
    verdict, written = await svc.judge_pair("source:a", "source:b")
    assert verdict.verdict == "reinforces"
    assert written is True
    repo.relate_verdict.assert_awaited_once()


async def test_neutral_writes_no_edge():
    svc, repo, _ = _service(
        '{"verdict": "neutral", "confidence": 0.99, "reasoning": "same topic only"}'
    )
    verdict, written = await svc.judge_pair("source:a", "source:b")
    assert verdict.verdict == "neutral"
    assert written is False
    repo.relate_verdict.assert_not_awaited()


async def test_low_confidence_contradicts_writes_no_edge():
    svc, repo, _ = _service(
        '{"verdict": "contradicts", "confidence": 0.4, "reasoning": "maybe"}'
    )
    verdict, written = await svc.judge_pair("source:a", "source:b")
    assert verdict.verdict == "contradicts"
    assert written is False  # below DEFAULT_MIN_CONFIDENCE
    repo.relate_verdict.assert_not_awaited()


async def test_malformed_response_writes_no_edge_no_crash():
    svc, repo, _ = _service("the documents seem related but I cannot output JSON")
    verdict, written = await svc.judge_pair("source:a", "source:b")
    assert verdict.verdict == "neutral"
    assert written is False
    repo.relate_verdict.assert_not_awaited()


async def test_custom_min_confidence_overrides_default():
    # A 0.6-confidence contradicts is below the 0.7 default but persists at 0.5.
    svc, repo, _ = _service(
        '{"verdict": "contradicts", "confidence": 0.6, "reasoning": "conflict"}'
    )
    _, written_default = await svc.judge_pair("source:a", "source:b")
    assert written_default is False

    _, written_low = await svc.judge_pair(
        "source:a", "source:b", min_confidence=0.5
    )
    assert written_low is True


async def test_missing_source_skips_pair():
    svc, repo, _ = _service(
        '{"verdict": "contradicts", "confidence": 0.99, "reasoning": "x"}'
    )
    svc.source_service.get = AsyncMock(return_value=None)
    verdict, written = await svc.judge_pair("source:a", "source:b")
    assert verdict.verdict == "neutral"
    assert written is False
    repo.relate_verdict.assert_not_awaited()


async def test_llm_failure_is_neutral_no_edge():
    svc, repo, llm = _service("ignored")
    llm.side_effect = RuntimeError("model down")
    verdict, written = await svc.judge_pair("source:a", "source:b")
    assert verdict.verdict == "neutral"
    assert written is False
    repo.relate_verdict.assert_not_awaited()


async def test_relate_verdict_decline_reports_no_edge():
    svc, repo, _ = _service(
        '{"verdict": "contradicts", "confidence": 0.95, "reasoning": "x"}'
    )
    repo.relate_verdict = AsyncMock(return_value=False)  # repo refused
    verdict, written = await svc.judge_pair("source:a", "source:b")
    assert verdict.verdict == "contradicts"
    assert written is False


# --- judge_source: candidates from substrate, tally, top-k -----------------


def _source_level_service(related_rows, llm_responses):
    """Judge wired with a hybrid related service + per-pair LLM responses."""
    source_service = MagicMock()
    source_service.get = AsyncMock(side_effect=lambda sid: _src(sid))

    related_service = MagicMock()
    related_service.find_related_hybrid = AsyncMock(return_value=related_rows)

    source_repo = MagicMock()
    source_repo.relate_verdict = AsyncMock(return_value=True)

    # Each judge_pair call consumes the next canned LLM response.
    llm_caller = AsyncMock(side_effect=list(llm_responses))

    svc = ContradictionJudgeService(
        source_service=source_service,
        related_service=related_service,
        source_repo=source_repo,
        llm_caller=llm_caller,
        judge_model="test-judge",
    )
    return svc, source_repo, related_service


async def test_judge_source_uses_related_substrate_and_tallies():
    related = [{"id": "source:b"}, {"id": "source:c"}, {"id": "source:d"}]
    responses = [
        '{"verdict": "contradicts", "confidence": 0.9, "reasoning": "x"}',
        '{"verdict": "reinforces", "confidence": 0.8, "reasoning": "y"}',
        '{"verdict": "neutral", "confidence": 0.5, "reasoning": "z"}',
    ]
    svc, repo, related_service = _source_level_service(related, responses)

    result = await svc.judge_source("source:a", k=5)

    related_service.find_related_hybrid.assert_awaited_once_with("source:a", 5)
    assert result.status == "judged"
    assert result.judged == 3
    assert result.contradicts == 1
    assert result.reinforces == 1
    assert result.neutral == 1
    assert result.edges_written == 2  # only the confident contradicts/reinforces
    assert repo.relate_verdict.await_count == 2
    # The pair details carry per-pair outcomes.
    assert {p["verdict"] for p in result.verdict_pairs} == {
        "contradicts",
        "reinforces",
        "neutral",
    }


async def test_judge_source_below_threshold_counted_not_written():
    related = [{"id": "source:b"}]
    responses = ['{"verdict": "contradicts", "confidence": 0.3, "reasoning": "weak"}']
    svc, repo, _ = _source_level_service(related, responses)

    result = await svc.judge_source("source:a")
    assert result.contradicts == 1
    assert result.below_threshold == 1
    assert result.edges_written == 0
    repo.relate_verdict.assert_not_awaited()


async def test_judge_source_top_k_bounds_pairs_judged():
    related = [{"id": f"source:{i}"} for i in range(10)]
    responses = [
        '{"verdict": "neutral", "confidence": 0.1, "reasoning": "n"}'
    ] * 10
    svc, _, related_service = _source_level_service(related, responses)

    result = await svc.judge_source("source:a", k=3)
    # k passed to the substrate AND caps the pairs judged.
    related_service.find_related_hybrid.assert_awaited_once_with("source:a", 3)
    assert result.judged == 3
    assert result.candidates_considered == 3


async def test_judge_source_excludes_self_from_candidates():
    # The substrate shouldn't return self, but if it ever did, the candidate
    # generator drops it — defense-in-depth.
    related = [{"id": "source:a"}, {"id": "source:b"}]
    responses = ['{"verdict": "neutral", "confidence": 0.1, "reasoning": "n"}']
    svc, _, _ = _source_level_service(related, responses)

    result = await svc.judge_source("source:a", k=5)
    assert result.judged == 1
    assert result.verdict_pairs[0]["to"] == "source:b"


async def test_judge_source_not_found():
    svc, repo, _ = _source_level_service([], [])
    svc.source_service.get = AsyncMock(return_value=None)
    result = await svc.judge_source("source:missing")
    assert result.status == "not_found"
    assert result.judged == 0
    repo.relate_verdict.assert_not_awaited()


async def test_judge_source_no_related_is_clean_empty_run():
    svc, repo, related_service = _source_level_service([], [])
    result = await svc.judge_source("source:a")
    assert result.status == "judged"
    assert result.judged == 0
    assert result.edges_written == 0
    repo.relate_verdict.assert_not_awaited()


async def test_judge_source_falls_back_to_find_related_when_no_hybrid():
    source_service = MagicMock()
    source_service.get = AsyncMock(side_effect=lambda sid: _src(sid))

    # A related service WITHOUT find_related_hybrid (dense-only substrate).
    related_service = MagicMock(spec=["find_related"])
    related_service.find_related = AsyncMock(return_value=[{"id": "source:b"}])

    source_repo = MagicMock()
    source_repo.relate_verdict = AsyncMock(return_value=True)

    svc = ContradictionJudgeService(
        source_service=source_service,
        related_service=related_service,
        source_repo=source_repo,
        llm_caller=AsyncMock(
            return_value='{"verdict": "contradicts", "confidence": 0.9, "reasoning": "x"}'
        ),
        judge_model="test-judge",
    )
    result = await svc.judge_source("source:a", k=4)
    related_service.find_related.assert_awaited_once_with("source:a", 4)
    assert result.edges_written == 1


async def test_judge_source_idempotent_drives_relate_verdict_identically():
    """Re-running a source's judge re-issues the SAME relate_verdict calls.

    Idempotency at the edge level lives in Z.1's clear-before-relate (proven in
    the container suite). Here we prove the SERVICE is deterministic: identical
    inputs → identical relate_verdict invocations, so a re-run can only converge
    on the same single edge per pair, never accrue a second one.
    """
    related = [{"id": "source:b"}]
    confident = '{"verdict": "contradicts", "confidence": 0.9, "reasoning": "x"}'

    def make():
        return _source_level_service(related, [confident])

    svc1, repo1, _ = make()
    await svc1.judge_source("source:a")
    first_call = repo1.relate_verdict.call_args

    svc2, repo2, _ = make()
    await svc2.judge_source("source:a")
    second_call = repo2.relate_verdict.call_args

    assert first_call == second_call
    assert repo1.relate_verdict.await_count == 1
    assert repo2.relate_verdict.await_count == 1


# --- param clamping --------------------------------------------------------


async def test_k_clamped_to_max():
    svc, _, related_service = _source_level_service([], [])
    await svc.judge_source("source:a", k=10_000)
    related_service.find_related_hybrid.assert_awaited_once_with("source:a", MAX_K)


async def test_min_confidence_clamped_into_unit_range():
    svc, repo, _ = _service(
        '{"verdict": "contradicts", "confidence": 0.5, "reasoning": "x"}'
    )
    # min_confidence above 1.0 clamps to 1.0 → a 0.5 verdict can't pass.
    _, written = await svc.judge_pair("source:a", "source:b", min_confidence=5.0)
    assert written is False
