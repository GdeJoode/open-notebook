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
        merge when this corpus was measured. Anyone re-measuring after N.5a gets
        real numbers; until then the gate must skip, not pass.
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
