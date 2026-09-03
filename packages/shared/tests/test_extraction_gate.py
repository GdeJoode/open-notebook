"""Track N.5d — the extraction regression gate's decision logic.

The measurement needs Ollama, a database and minutes per document. The rule that
turns a measurement into a verdict is arithmetic, and it is what these pin, so
the gate's logic is exercised on every run of the suite rather than only during
an opt-in live run. A gate whose logic executes once a fortnight is a gate nobody
trusts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from shared.regression import (
    GateOutcome,
    compare_against_baseline,
    summarise_run,
)

BASELINE_PATH = (
    Path(__file__).resolve().parents[3] / "tests/regression/n_extraction_baseline.json"
)


def _doc(entities, relations=0, counters=None):
    result = {"entity_count": entities, "relation_count": relations}
    if counters is not None:
        result["counters"] = counters
    return {"result": result}


class TestSummariseRun:
    def test_counts_are_totalled_across_documents(self):
        s = summarise_run([_doc(17), _doc(22), _doc(0)])
        assert s["documents"] == 3
        assert s["total_entities"] == 39
        assert s["documents_with_entities"] == 2

    def test_an_unmeasured_rate_is_none_not_zero(self):
        """The distinction the whole missing-baseline design rests on.

        Before N.5a, `_merge_results` discarded the per-pass counters, so a
        merged run reported nothing. Recording that as 0.0 would claim the run
        culled nothing — which is exactly the false statement N.5a fixed, moved
        one layer up into the gate.
        """
        s = summarise_run([_doc(5), _doc(3)])
        assert s["over_generation_rate"] is None
        assert s["abstain_rate"] is None

    def test_a_measured_zero_is_zero(self):
        """The vacuity guard for the test above: a run that genuinely culled
        nothing must report 0.0, or `None` would be indistinguishable from a
        clean run and the gate would skip every real one.
        """
        s = summarise_run(
            [
                _doc(
                    5,
                    counters={
                        "entities_extracted": 5,
                        "entities_kept": 5,
                        "abstained_chunks": 0,
                        "chunk_count": 10,
                    },
                )
            ]
        )
        assert s["over_generation_rate"] == 0.0
        assert s["abstain_rate"] == 0.0

    def test_rates_are_computed_over_the_totals_not_per_document(self):
        s = summarise_run(
            [
                _doc(3, counters={"entities_extracted": 9, "entities_kept": 3,
                                  "abstained_chunks": 2, "chunk_count": 10}),
                _doc(2, counters={"entities_extracted": 5, "entities_kept": 2,
                                  "abstained_chunks": 7, "chunk_count": 10}),
            ]
        )
        assert s["over_generation_rate"] == pytest.approx(9 / 14)
        assert s["abstain_rate"] == pytest.approx(9 / 20)


class TestTheGateFailsOnTheThingsItExistsFor:
    def test_a_recall_drop_fails(self):
        """The binding half. A change that halves the LLM calls by extracting
        nothing is a regression, and a cost-only gate waves it through.
        """
        base = summarise_run([_doc(60), _doc(64)])
        now = summarise_run([_doc(30), _doc(30)])
        result = compare_against_baseline(base, now)
        assert not result.passed
        assert [d.name for d in result.failures] == ["total_entities"]

    def test_a_document_going_silent_fails_even_when_the_total_holds(self):
        """Totals can be carried by one verbose document while another stops
        producing entirely — which is the `Bennett_test.pdf` shape.
        """
        base = summarise_run([_doc(60), _doc(64)])
        now = summarise_run([_doc(124), _doc(0)])
        result = compare_against_baseline(base, now)
        assert not result.passed
        assert "documents_with_entities" in [d.name for d in result.failures]

    def test_rising_over_generation_fails(self):
        counters = {"entities_extracted": 10, "entities_kept": 9,
                    "abstained_chunks": 1, "chunk_count": 10}
        base = summarise_run([_doc(9, counters=counters)])
        worse = summarise_run(
            [_doc(2, counters={"entities_extracted": 10, "entities_kept": 2,
                               "abstained_chunks": 1, "chunk_count": 10})]
        )
        result = compare_against_baseline(base, worse, recall_tolerance=1.0)
        assert not result.passed
        assert "over_generation_rate" in [d.name for d in result.failures]

    def test_an_unchanged_run_passes(self):
        counters = {"entities_extracted": 10, "entities_kept": 9,
                    "abstained_chunks": 1, "chunk_count": 10}
        s = summarise_run([_doc(9, counters=counters)])
        result = compare_against_baseline(s, s)
        assert result.passed
        assert not result.skipped

    def test_an_improvement_passes(self):
        base = summarise_run(
            [_doc(9, counters={"entities_extracted": 20, "entities_kept": 9,
                               "abstained_chunks": 5, "chunk_count": 10})]
        )
        better = summarise_run(
            [_doc(12, counters={"entities_extracted": 14, "entities_kept": 12,
                                "abstained_chunks": 2, "chunk_count": 10})]
        )
        assert compare_against_baseline(base, better).passed


class TestTheGateCannotPassVacuously:
    """The failure mode this design is built against: a gate that reports as a
    guard while having verified nothing.
    """

    def test_a_missing_baseline_dimension_skips_rather_than_passes(self):
        base = summarise_run([_doc(10)])  # no counters -> rates are None
        now = summarise_run(
            [_doc(10, counters={"entities_extracted": 100, "entities_kept": 10,
                                "abstained_chunks": 9, "chunk_count": 10})]
        )
        result = compare_against_baseline(base, now)

        skipped = {d.name for d in result.skipped}
        assert skipped == {"over_generation_rate", "abstain_rate"}
        # The run above has 90% over-generation. It is not called a pass on that
        # dimension — it is not called anything, and the report says so.
        for d in result.dimensions:
            if d.name in skipped:
                assert d.outcome is GateOutcome.SKIPPED
        assert "skip" in result.report()

    def test_a_baseline_with_nothing_comparable_is_inconclusive_not_green(self):
        empty = {"total_entities": None, "documents_with_entities": None}
        result = compare_against_baseline(empty, summarise_run([_doc(10)]))
        assert result.inconclusive
        assert not result.passed
        assert "INCONCLUSIVE" in result.report()

    def test_a_non_numeric_baseline_value_does_not_pass_and_does_not_raise(self):
        """A baseline is a JSON file somebody may hand-edit. A string where a
        number belongs must degrade to "not measured", never to "fine".
        """
        base = {"total_entities": "lots", "documents_with_entities": 6}
        result = compare_against_baseline(base, summarise_run([_doc(1)]))
        names = {d.name: d.outcome for d in result.dimensions}
        assert names["total_entities"] is GateOutcome.SKIPPED
        assert names["documents_with_entities"] is GateOutcome.FAILED


class TestAPartialCounterSetIsNotAMeasurement:
    """The blocker the SEAM introduced, found by a review one round after the
    seam was built — the third instance of this track's own thesis inside it.

    `ExtractionWorkflow`'s legacy single-schema branch emits `chunk_count` and
    none of the abstention counters: it drives a pluggable extractor rather than
    `run_pass2`, so it genuinely cannot count abstention. The first version of
    `summarise_run` set one `saw_counters` flag from whole-dict truthiness and
    then guarded `abstain_rate` on `chunk_count > 0` — its DENOMINATOR — so a
    legacy run reported `abstain_rate: 0.0` for something never counted.
    `over_generation_rate` escaped only because its guard happened to test its own
    numerator.

    Two production routes reach that path (no `notebook_id` or
    `multi_schema_enabled=false`, and the fallback when no schema clears the
    applicability floor), and the checked-in baseline says its corpus mostly took
    it. So this was reachable through the step the docs recommend next.
    """

    LEGACY = {"chunk_count": 10}
    MULTI = {
        "chunk_count": 20,
        "entities_extracted": 14,
        "entities_kept": 5,
        "abstained_chunks": 9,
    }

    def test_a_denominator_without_its_numerator_measures_nothing(self):
        s = summarise_run([_doc(3, counters=self.LEGACY)])
        assert s["abstain_rate"] is None
        assert s["over_generation_rate"] is None

    def test_the_shortfall_is_counted_so_a_skip_can_say_why(self):
        s = summarise_run([_doc(3, counters=self.LEGACY)])
        assert s["documents_missing_counters"] == {
            "over_generation_rate": 1,
            "abstain_rate": 1,
        }

    def test_a_mixed_corpus_does_not_understate_the_rate(self):
        """Half legacy, half multi-schema. Summing legacy `chunk_count` into the
        denominator while only multi documents can contribute to the numerator
        would understate abstention by exactly the legacy share — a wrong number
        rather than a missing one.
        """
        mixed = summarise_run(
            [_doc(3, counters=self.LEGACY), _doc(5, counters=self.MULTI)]
        )
        assert mixed["abstain_rate"] is None
        assert mixed["documents_missing_counters"]["abstain_rate"] == 1

    def test_each_rate_is_guarded_separately_not_by_the_other_s_luck(self):
        """A mutation found this gap: dropping the shortfall check from
        `over_generation_rate` alone left every test green.

        The reason is that the legacy fixture carries only `chunk_count`, so
        `entities_extracted` sums to 0 and the `extracted > 0` guard catches it by
        accident — which is exactly the accident the review described, where
        over-generation "escapes only by luck". A document carrying the ABSTENTION
        inputs and not the extraction ones separates the two guards, and nothing
        in the suite had one.
        """
        abstention_only = {"abstained_chunks": 4, "chunk_count": 10}
        s = summarise_run([_doc(3, counters=abstention_only)])

        assert s["abstain_rate"] == pytest.approx(0.4)  # its own inputs are there
        assert s["over_generation_rate"] is None  # and these are not
        assert s["documents_missing_counters"] == {"over_generation_rate": 1}

        # And the case that actually separates the guards: one document DOES
        # supply the extraction inputs, so the `extracted > 0` guard is satisfied
        # and only the shortfall check can still refuse. The first version of this
        # test used the fixture above alone, where `extracted` sums to 0 and the
        # accidental guard catches it — so the mutation survived.
        mixed = summarise_run(
            [
                _doc(3, counters=abstention_only),
                _doc(4, counters={"entities_extracted": 10, "entities_kept": 4,
                                  "abstained_chunks": 1, "chunk_count": 10}),
            ]
        )
        assert mixed["over_generation_rate"] is None
        assert mixed["documents_missing_counters"] == {"over_generation_rate": 1}

    def test_the_mirror_case_extraction_without_abstention(self):
        extraction_only = {"entities_extracted": 10, "entities_kept": 4}
        s = summarise_run([_doc(4, counters=extraction_only)])

        assert s["over_generation_rate"] == pytest.approx(0.6)
        assert s["abstain_rate"] is None
        assert s["documents_missing_counters"] == {"abstain_rate": 1}

    def test_a_complete_set_still_measures(self):
        """The vacuity guard: strictness must not make every rate `None`, or the
        gate's cost half would be permanently skipped for a different reason.
        """
        s = summarise_run([_doc(5, counters=self.MULTI)])
        assert s["over_generation_rate"] == pytest.approx(9 / 14)
        assert s["abstain_rate"] == pytest.approx(9 / 20)
        assert s["documents_missing_counters"] == {}

    def test_the_gate_refuses_to_compare_a_legacy_run_to_a_measured_baseline(self):
        """And the current-absent rule now actually fires on this path. Before
        the fix the phantom 0.0 sailed past it as a 0.45 -> 0.0 improvement.
        """
        baseline = summarise_run([_doc(5, counters=self.MULTI)])
        current = summarise_run([_doc(5, counters=self.LEGACY)])
        result = compare_against_baseline(baseline, current)

        assert not result.passed
        assert {d.name for d in result.failures} == {
            "over_generation_rate",
            "abstain_rate",
        }
        assert "carried no inputs for abstain_rate" in result.report()

    def test_the_failure_message_names_both_causes_not_one(self):
        """A review pointed out the message asserted a cause. An operator who
        simply ran with `multi_schema_enabled=false` gets the same sentence as
        one whose merge stopped carrying counters, and they open different files.
        """
        baseline = summarise_run([_doc(5, counters=self.MULTI)])
        current = summarise_run([_doc(5, counters=self.LEGACY)])
        detail = next(
            d.detail
            for d in compare_against_baseline(baseline, current).dimensions
            if d.name == "abstain_rate"
        )
        assert "measured nothing" in detail
        assert "legacy single-schema path" in detail
        assert "incomparable" in detail


class TestTheMeasurementDisappearing:
    """A review found the mirror image of the missing-baseline rule.

    Both comparators skipped whenever EITHER side was absent, with a message
    hard-coded to "no baseline value". So a run whose counters stopped arriving —
    which is N.5a's own defect recurring — reported green and printed a reason
    that was false. The rule the phase wrote covers "this metric did not exist
    when the baseline was taken"; it did not cover "this metric existed and this
    run lost it", which is the case that catches a regression in this track's own
    code.
    """

    @staticmethod
    def _with_rates():
        return summarise_run(
            [_doc(9, counters={"entities_extracted": 20, "entities_kept": 9,
                               "abstained_chunks": 1, "chunk_count": 10})]
        )

    def test_a_current_run_that_lost_its_counters_fails(self):
        baseline = self._with_rates()
        current = summarise_run([_doc(9)])  # same entities, no counters
        result = compare_against_baseline(baseline, current)

        assert not result.passed
        failed = {d.name for d in result.failures}
        assert failed == {"over_generation_rate", "abstain_rate"}
        assert "measured nothing" in result.report()

    def test_the_skip_message_no_longer_blames_the_baseline_wrongly(self):
        """The reason string is part of the guard: an operator reads it and
        decides whether to act. Saying "no baseline value" about a run that lost
        its own measurement sends them to the wrong file.
        """
        baseline = self._with_rates()
        current = summarise_run([_doc(9)])
        result = compare_against_baseline(baseline, current)
        detail = next(
            d.detail for d in result.dimensions if d.name == "over_generation_rate"
        )
        assert "no baseline value" not in detail
        assert "this run measured nothing" in detail

    def test_a_metric_new_since_the_baseline_still_skips(self):
        """The vacuity guard for the two above: the legitimate direction must
        keep skipping, or every added metric would fail its first run.
        """
        baseline = summarise_run([_doc(9)])
        current = self._with_rates()
        result = compare_against_baseline(baseline, current)
        assert {d.name for d in result.skipped} == {
            "over_generation_rate",
            "abstain_rate",
        }
        assert result.passed

    def test_never_measured_on_either_side_is_neither_pass_nor_fail(self):
        base = summarise_run([_doc(9)])
        result = compare_against_baseline(base, summarise_run([_doc(9)]))
        skipped = {d.name for d in result.skipped}
        assert skipped == {"over_generation_rate", "abstain_rate"}
        assert "never measured on either side" in result.report()


class TestAZeroBaselineHoldsNothingUp:
    """A review reached this by recording a baseline from a corpus where every
    document yielded nothing — Ollama down, a plausible accident. The floor is
    then `0 * (1 - tolerance)` = 0.0, every run clears it, and the gate is
    permanently green while reporting `inconclusive=False`. Zeros rather than
    nulls, but the same failure class the design targets.
    """

    def test_a_zero_floor_skips_rather_than_passing_anything(self):
        baseline = summarise_run([_doc(0), _doc(0)])
        assert baseline["total_entities"] == 0

        result = compare_against_baseline(baseline, summarise_run([_doc(0)]))
        assert {d.name for d in result.skipped} >= {
            "total_entities",
            "documents_with_entities",
        }
        assert result.inconclusive
        assert not result.passed

    def test_a_normal_baseline_is_unaffected(self):
        """Vacuity guard: skipping on zero must not creep into ordinary
        comparisons, or the recall floor would stop working entirely.
        """
        baseline = summarise_run([_doc(60), _doc(64)])
        # Same document count on both sides, so this isolates the recall floor
        # from the per-document liveness dimension.
        assert compare_against_baseline(
            baseline, summarise_run([_doc(70), _doc(70)])
        ).passed
        assert not compare_against_baseline(
            baseline, summarise_run([_doc(15), _doc(15)])
        ).passed


class TestTheCheckedInBaseline:
    def test_the_baseline_file_is_loadable_and_says_what_it_measured(self):
        baseline = json.loads(BASELINE_PATH.read_text())
        assert baseline["total_entities"] == 124
        assert baseline["documents"] == 7
        provenance = baseline["_provenance"]
        assert provenance["harness"].endswith("n_pipeline_review_run.py")
        assert len(provenance["documents"]) == 7

    def test_the_baseline_declares_its_two_unmeasured_dimensions(self):
        """Recorded as null rather than 0.0, and the provenance note says why:
        the counters that would have produced them were being discarded by the
        merge when this corpus was measured.

        An earlier version of this docstring said "anyone re-measuring after N.5a
        gets real numbers", and a review showed that was false at the time: N.5a
        put the counters into `ExtractionResult.metadata` but `run_extraction`
        never returned them, so the gate read a key nothing wrote. That seam is
        closed now (`_observability_counters`), which is what makes the sentence
        true — but it took a second fix, not the first.
        """
        baseline = json.loads(BASELINE_PATH.read_text())
        assert baseline["over_generation_rate"] is None
        assert baseline["abstain_rate"] is None
        assert "N.5a" in baseline["_provenance"]["note"]

    def test_the_gate_run_against_its_own_baseline_passes_and_skips_two(self):
        """The self-check: today's baseline compared with itself must pass on
        recall and skip both cost dimensions. If this ever fails, either the
        baseline or the gate changed shape without the other.
        """
        baseline = json.loads(BASELINE_PATH.read_text())
        result = compare_against_baseline(baseline, baseline)
        assert result.passed
        assert {d.name for d in result.skipped} == {
            "over_generation_rate",
            "abstain_rate",
        }
