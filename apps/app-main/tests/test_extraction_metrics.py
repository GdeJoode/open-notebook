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
