"""Unit tests for V.2 region segmentation (pure, offline heuristics).

Cover the three segmentation strategies (numbered / blank-separated /
author-year start-detection), bounded over-segmentation (a wrapped multi-line
entry stays ONE entry), the empty-region case, and the optional LLM-classifier
seam (which the default path never invokes).
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Sequence

from shared.references.segmenter import segment_region

_FIXTURES = Path(__file__).parent / "fixtures" / "references"


def _fixture(name: str) -> str:
    return (_FIXTURES / name).read_text(encoding="utf-8")


# -- numbered lists ---------------------------------------------------------


def test_segments_numbered_bibliography_into_n_entries():
    entries = segment_region(_fixture("references_numbered.txt"))
    assert len(entries) == 3
    # The list marker is stripped from each entry's head.
    assert entries[0].startswith("D. Acemoglu")
    assert not entries[0].startswith("[1]")


def test_numbered_wrapped_lines_stay_one_entry():
    """A reference wrapped across physical lines is NOT over-segmented."""
    entries = segment_region(_fixture("references_numbered.txt"))
    # Entry 2 spans two source lines; both halves must be in the same entry.
    assert "Cambridge" in entries[1] and "Rodríguez-Pose" in entries[1]
    # Entry 3 spans three source lines incl. a DOI continuation.
    assert "10.1111/j.1467-9787.2011.00756.x" in entries[2]
    assert "Barca" in entries[2]


def test_numbered_various_marker_forms():
    text = "1. First entry here.\n2) Second entry here.\n(3) Third entry here."
    entries = segment_region(text)
    assert len(entries) == 3
    assert entries[0] == "First entry here."
    assert entries[2] == "Third entry here."


# -- author-year (blank-line separated) -------------------------------------


def test_segments_author_year_bibliography_blank_separated():
    entries = segment_region(_fixture("references_apa.txt"))
    assert len(entries) == 4
    assert entries[0].startswith("Acemoglu, D., & Robinson")
    # A DOI-bearing entry keeps its DOI in the same entry.
    assert "10.1257/aer.91.5.1369" in entries[1]


def test_dutch_section_with_kamerstuk_segments():
    entries = segment_region(_fixture("references_dutch.txt"))
    assert len(entries) == 3
    assert any("Kamerstukken II 2023/24, 36410" in e for e in entries)


# -- author-year (no blank separators — start detection) --------------------


def test_author_year_without_blank_lines_uses_start_detection():
    text = (
        "Acemoglu, D., & Robinson, J. A. (2012). Why Nations Fail. Crown.\n"
        "Rodríguez-Pose, A. (2018). The revenge of the places. CJRES, 11(1)."
    )
    entries = segment_region(text)
    assert len(entries) == 2
    assert entries[0].startswith("Acemoglu")
    assert entries[1].startswith("Rodríguez-Pose")


def test_wrapped_continuation_without_blank_lines_stays_one_entry():
    """A lowercase/venue continuation line must not start a new entry."""
    text = (
        "Acemoglu, D., & Robinson, J. A. (2012). Why Nations Fail: The Origins\n"
        "of Power, Prosperity, and Poverty. Crown Business.\n"
        "Barca, F. (2012). The case for regional development. JRS."
    )
    entries = segment_region(text)
    assert len(entries) == 2
    assert "of Power, Prosperity" in entries[0]
    assert entries[1].startswith("Barca")


# -- empty / degenerate -----------------------------------------------------


def test_empty_region_returns_empty_list():
    assert segment_region("") == []
    assert segment_region("   \n\n  \t ") == []


# -- LLM-classifier seam (default path never calls it) ----------------------


def test_ambiguity_resolver_seam_is_optional_and_overrides():
    calls: List[Sequence[str]] = []

    def resolver(region_text: str, heuristic: Sequence[str]) -> Sequence[str]:
        calls.append(heuristic)
        return ["one merged entry"]

    out = segment_region(
        _fixture("references_apa.txt"), ambiguity_resolver=resolver
    )
    assert out == ["one merged entry"]
    # It was handed the heuristic segmentation to refine.
    assert len(calls) == 1 and len(calls[0]) == 4


def test_default_path_does_not_invoke_a_resolver():
    """Sanity: with no resolver, segmentation is purely heuristic (deterministic)."""
    a = segment_region(_fixture("references_apa.txt"))
    b = segment_region(_fixture("references_apa.txt"))
    assert a == b and len(a) == 4
