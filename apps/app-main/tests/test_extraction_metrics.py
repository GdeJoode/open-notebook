"""Track N.3 — over-generation & abstention metrics (pure derivation)."""

from __future__ import annotations

import pytest
from app_main.services.extraction_chunking.extraction_metrics import (
    ExtractionMetrics,
    abstain_rate,
    measure_extraction,
    over_generation_rate,
)


def test_over_generation_rate_basic():
    assert over_generation_rate(10, 7) == pytest.approx(0.3)
    assert over_generation_rate(0, 0) == 0.0          # nothing extracted → 0
    assert over_generation_rate(5, 5) == 0.0          # nothing removed → 0
    assert over_generation_rate(4, 0) == 1.0          # all removed → 1


def test_over_generation_rate_clamps_bad_input():
    # survivors > extracted signals a caller bug; clamp rather than go negative
    assert over_generation_rate(3, 9) == 0.0
    assert over_generation_rate(3, -2) == 1.0


def test_abstain_rate_basic():
    assert abstain_rate(2, 10) == 0.2
    assert abstain_rate(0, 0) == 0.0
    assert abstain_rate(5, 5) == 1.0
    assert abstain_rate(9, 5) == 1.0  # clamped


def test_measure_extraction_from_metadata():
    meta = {
        "entities_extracted": 20,
        "entities_kept": 15,
        "abstained_chunks": 3,
        "chunk_count": 12,
    }
    m = measure_extraction(meta)
    assert isinstance(m, ExtractionMetrics)
    assert m.extracted == 20
    assert m.survivors == 15
    assert m.rejected == 5
    assert m.over_generation_rate == 0.25
    assert m.abstained_chunks == 3
    assert m.total_chunks == 12
    assert m.abstain_rate == 0.25


def test_measure_extraction_falls_back_to_total_entities():
    # a run from before the N.3 counts existed: no entities_extracted/kept →
    # all-zero baseline (no over-generation), not a crash.
    meta = {"total_entities": 8, "chunk_count": 4}
    m = measure_extraction(meta)
    assert m.extracted == 0
    assert m.survivors == 8  # falls back to total_entities
    assert m.over_generation_rate == 0.0
    assert m.abstain_rate == 0.0


def test_measure_extraction_empty_metadata():
    m = measure_extraction({})
    assert m.extracted == 0
    assert m.survivors == 0
    assert m.over_generation_rate == 0.0
    assert m.abstain_rate == 0.0


class TestMetricsSurviveTheMultiSchemaMerge:
    """Track N.5a — the metrics module measured against the merge that feeds it.

    `measure_extraction` reads `entities_extracted`, `entities_kept`,
    `chunk_count` and `abstained_chunks` out of a `run_pass2` result. On the
    multi-schema path — the path production takes — `_merge_results` rebuilt the
    metadata from four keys and none of those four survived, so every merged run
    measured as a run that over-generated nothing and abstained on nothing.

    These live here rather than in the pipeline package because the defect is
    only visible where the producer and the consumer meet: the merge's key list
    looks complete on its own, and the metric's defaults look prudent on their
    own. Only together do they turn a missing measurement into a false one.
    """

    @staticmethod
    def _passes():
        from shared.models.extraction import ExtractedEntity, ExtractionResult

        def one(name, kept, extracted, abstained, removed):
            return (
                name,
                ExtractionResult(
                    entities=[
                        ExtractedEntity(text=f"{name}-{i}", label="L")
                        for i in range(kept)
                    ],
                    metadata={
                        "chunk_count": 10,
                        "entities_extracted": extracted,
                        "entities_kept": kept,
                        "not_a_concept_removed": removed,
                        "not_a_concept_judged": removed,
                        "abstained_chunks": abstained,
                    },
                ),
            )

        return [one("deals", 3, 9, 2, 6), one("policy", 2, 5, 7, 3)]

    def test_a_merged_run_measures_what_its_passes_did(self):
        from ontology_extraction.multi_schema_orchestrator import _merge_results

        merged = _merge_results(self._passes())
        m = measure_extraction(merged.metadata)

        assert m.extracted == 14
        assert m.survivors == 5
        assert m.rejected == 9
        assert m.over_generation_rate == pytest.approx(9 / 14)
        assert m.abstained_chunks == 9
        assert m.total_chunks == 20
        assert m.abstain_rate == pytest.approx(9 / 20)

    def test_a_merged_run_does_not_measure_as_a_clean_one(self):
        """The defect's signature, asserted as the negative it actually was.

        Before N.5a this same fixture measured `over_generation_rate` 0.00 and
        `abstain_rate` 0.00 — not "unknown", but "nothing was culled and nothing
        abstained", which is a claim rather than a gap.
        """
        from ontology_extraction.multi_schema_orchestrator import _merge_results

        m = measure_extraction(_merge_results(self._passes()).metadata)
        assert m.over_generation_rate > 0.0
        assert m.abstain_rate > 0.0

    def test_an_all_zero_measurement_is_still_reachable(self):
        """A vacuity guard for the test above: a run that genuinely culled
        nothing must still measure as 0.0, or the assertion would pass for any
        merge at all.
        """
        from ontology_extraction.multi_schema_orchestrator import _merge_results
        from shared.models.extraction import ExtractedEntity, ExtractionResult

        clean = [
            (
                name,
                ExtractionResult(
                    entities=[ExtractedEntity(text=f"{name}", label="L")],
                    metadata={
                        "chunk_count": 4,
                        "entities_extracted": 1,
                        "entities_kept": 1,
                        "abstained_chunks": 0,
                    },
                ),
            )
            for name in ("deals", "policy")
        ]
        m = measure_extraction(_merge_results(clean).metadata)
        assert m.over_generation_rate == 0.0
        assert m.abstain_rate == 0.0
