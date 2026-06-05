"""Smoke tests for the threshold-tuning CLI.

The CLI itself wraps Docling (an async, network/heavy-disk operation),
so we don't drive it end-to-end here. Instead we cover the small
pure-Python helpers used by the report shape: ``format_header``,
``format_row``, ``dominant_signal``. If any of those drift the
``threshold-tuning.md`` table format breaks silently — which is the
exact thing this test pins.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The script lives outside ``src/`` so it's not auto-installed; add
# scripts/ to sys.path the same way the script does for itself.
_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import score_pdf_corpus as cli  # noqa: E402

from app_main.services.parsing.confidence import (  # noqa: E402
    DEFAULT_THRESHOLD,
    DoclingConfidenceScore,
)


def _sample_score(
    overall: float = 0.91,
    signals: dict[str, float] | None = None,
    decision: str = "fallback",
) -> DoclingConfidenceScore:
    return DoclingConfidenceScore(
        overall=overall,
        signals=signals
        or {
            "ocr_confidence": 1.0,
            "text_density": 0.85,
            "heading_rate": 0.50,
            "table_success": 1.0,
            "image_text_ratio": 1.0,
            "unknown_element_ratio": 0.95,
        },
        decision=decision,  # type: ignore[arg-type]
        threshold=DEFAULT_THRESHOLD,
    )


def test_format_header_lists_all_six_signals_in_order() -> None:
    header = cli.format_header()
    # Header line + separator
    lines = header.split("\n")
    assert len(lines) == 2
    head_cells = [c.strip() for c in lines[0].strip("|").split("|")]
    # fixture + overall + 6 signals + decision + dominant_signal = 10
    assert len(head_cells) == 10
    assert head_cells[0] == "fixture"
    assert head_cells[1] == "overall"
    # Per the SIGNAL_ORDER list — short labels.
    assert head_cells[2:8] == [
        "ocr", "density", "heading", "table", "image", "unknown",
    ]
    assert head_cells[8].startswith("decision @")
    assert head_cells[9] == "dominant_signal"


def test_format_row_emits_correct_column_count() -> None:
    score = _sample_score()
    row = cli.format_row("synthetic_clean_text.pdf", score)
    # Same column count as the header.
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert len(cells) == 10
    assert cells[0] == "synthetic_clean_text.pdf"
    assert cells[1] == "0.910"
    # Signal cells are 2-decimal floats, decision is "fallback",
    # dominant_signal is a non-empty string.
    for sig_cell in cells[2:8]:
        # 2-decimal float
        float(sig_cell)
        assert len(sig_cell.split(".")[1]) == 2
    assert cells[8] == "fallback"
    assert cells[9]  # non-empty


def test_dominant_signal_picks_lowest_weighted_contribution() -> None:
    # heading_rate has weight 0.15 and score 0.1 → 0.015
    # ocr_confidence has weight 0.30 and score 0.05 → 0.015
    # image_text_ratio has weight 0.10 and score 0.0 → 0.0 (winner)
    signals = {
        "ocr_confidence": 0.05,
        "text_density": 1.0,
        "heading_rate": 0.10,
        "table_success": 1.0,
        "image_text_ratio": 0.0,
        "unknown_element_ratio": 1.0,
    }
    score = _sample_score(signals=signals)
    assert cli.dominant_signal(score) == "image_text_ratio"


def test_dominant_signal_returns_sentinel_for_empty_signals() -> None:
    score = DoclingConfidenceScore(overall=1.0, signals={}, decision="accept")
    assert cli.dominant_signal(score) == "n/a"


def test_signal_order_matches_weights_keys() -> None:
    """If a new signal is added without updating SIGNAL_ORDER the
    Markdown table would silently drop or duplicate a column. Pin
    the contract here."""
    from app_main.services.parsing.confidence import SIGNAL_WEIGHTS

    assert set(cli.SIGNAL_ORDER) == set(SIGNAL_WEIGHTS.keys())


def test_main_exits_nonzero_when_target_missing(tmp_path: Path) -> None:
    missing = tmp_path / "no-such.pdf"
    assert cli.main([str(missing)]) == 1


def test_iter_pdf_paths_sorted(tmp_path: Path) -> None:
    """Directory enumeration is sorted for reproducible output."""
    for name in ["c.pdf", "a.pdf", "b.pdf"]:
        (tmp_path / name).write_bytes(b"%PDF-1.4\n%%EOF\n")
    # Drop a non-PDF — must be ignored.
    (tmp_path / "ignore.txt").write_text("not a pdf")

    paths = list(cli._iter_pdf_paths(tmp_path))
    assert [p.name for p in paths] == ["a.pdf", "b.pdf", "c.pdf"]


def test_iter_pdf_paths_single_file(tmp_path: Path) -> None:
    """Passing a single PDF should yield just that one path."""
    pdf = tmp_path / "single.pdf"
    pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    paths = list(cli._iter_pdf_paths(pdf))
    assert paths == [pdf]
