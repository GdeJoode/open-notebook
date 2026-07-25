"""Docling extraction confidence scoring (Phase A.1c).

Pure function ``score_docling_extraction`` that turns an ``IngestionResult``
into a 6-signal weighted score. The score gates the auto-fallback to
MinerU in ``auto_fallback.py``: when overall confidence drops below the
configured threshold, the orchestrator re-runs the document through
MinerU and trusts that result (Q-A-2 — no comparative scoring in V1).

Signals and weights are tuned to favour clean, native PDFs (high text
density, OCR not needed, headings visible, tables parsed) and penalise
scanned/image-only documents where Docling's text layer is thin. The
weights sum to exactly 1.0 (asserted in tests).

Design notes:

* Pure function — no I/O, no logging side effects in the hot path. Keeps
  per-document cost negligible (acceptance criterion #9: < 50 ms for a
  50-page document).
* Tolerant of partial / minimal ``IngestionResult`` shapes. Tests fake
  these out with lightweight dataclasses, and audio/transcription
  results (``document is None``) collapse to neutral signals rather
  than crashing.
* ``ElementType`` membership lookups use the enum's ``.value`` strings
  rather than importing the enum so consumers (notebooks, scripts) can
  pass plain string element_types without paying for the enum import.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Optional


# Signal weights — MUST sum to 1.0. Asserted in tests.
SIGNAL_WEIGHTS: Dict[str, float] = {
    "ocr_confidence": 0.30,
    "text_density": 0.20,
    "heading_rate": 0.15,
    "table_success": 0.15,
    "image_text_ratio": 0.10,
    "unknown_element_ratio": 0.10,
}

# Default threshold below which auto-mode falls back to MinerU. Users
# can override via ``ContentSettings.docling_min_confidence`` per
# Phase A.1c. The 0.95 value comes from the roadmap; Phase A.3
# tuning may revise it.
DEFAULT_THRESHOLD: float = 0.95


@dataclass(frozen=True)
class DoclingConfidenceScore:
    """Result of scoring a Docling ``IngestionResult``.

    ``overall`` is the weighted aggregate in [0.0, 1.0]. ``signals``
    is the per-signal breakdown so the UI can surface a tooltip
    explaining why a document scored low (acceptance criterion #8).
    ``decision`` is the orchestrator-visible verdict; ``threshold`` is
    echoed back so callers can persist the cutoff that produced the
    decision (useful when the user later raises the bar via
    ``processing_overrides.docling_min_confidence``).
    """

    overall: float
    signals: Dict[str, float] = field(default_factory=dict)
    decision: Literal["accept", "fallback"] = "accept"
    threshold: float = DEFAULT_THRESHOLD


@dataclass(frozen=True)
class PageConfidence:
    """Per-page confidence — the H0.1 granularity the doc-level score aggregates.

    Same 6 signals as :class:`DoclingConfidenceScore`, but scoped to ONE page's
    elements/tables/images/text (``page_count`` = 1). ``page_number`` is docling's
    1-indexed page number (or the 1-based position when unset). This is the input
    the H0.2 hybrid merge routes on — a page whose ``decision == "fallback"`` is a
    candidate for MinerU re-parse while its ``accept`` neighbours keep docling.
    """

    page_number: int
    overall: float
    signals: Dict[str, float] = field(default_factory=dict)
    decision: Literal["accept", "fallback"] = "accept"
    threshold: float = DEFAULT_THRESHOLD


def score_docling_extraction(
    result: Any,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> DoclingConfidenceScore:
    """Return a :class:`DoclingConfidenceScore` for a Docling result.

    ``result`` is typically an ``ingestion.models.IngestionResult`` but
    only the attribute *shape* matters — tests pass lightweight fakes.
    The function tolerates a missing or ``None`` document (audio /
    failed extractions) by returning a neutral 1.0 overall, on the
    grounds that "nothing to score" is not a fallback trigger.
    """
    document = getattr(result, "document", None)
    if document is None:
        # Audio/video transcriptions or failed extractions don't get
        # scored — fallback is decided by ``success`` upstream, not
        # confidence. Return a neutral score so auto-mode short-circuits
        # in the accept branch.
        signals = {name: 1.0 for name in SIGNAL_WEIGHTS}
        return DoclingConfidenceScore(
            overall=1.0,
            signals=signals,
            decision="accept" if 1.0 >= threshold else "fallback",
            threshold=threshold,
        )

    elements = list(_iter_elements(document))
    tables = list(getattr(document, "all_tables", []) or [])
    images = list(getattr(document, "all_images", []) or [])

    page_count = max(1, int(getattr(document, "page_count", 0) or 0) or _infer_page_count(document))

    signals = _signals(
        elements=elements,
        tables=tables,
        images=images,
        text=getattr(document, "full_text", "") or "",
        page_count=page_count,
    )

    overall = _weighted_sum(signals)
    decision: Literal["accept", "fallback"] = (
        "accept" if overall >= threshold else "fallback"
    )

    return DoclingConfidenceScore(
        overall=overall,
        signals=signals,
        decision=decision,
        threshold=threshold,
    )


def score_docling_pages(
    result: Any,
    *,
    threshold: float = DEFAULT_THRESHOLD,
) -> list[PageConfidence]:
    """Score each page of a Docling result INDEPENDENTLY (Track H0.1, Optie A).

    Returns one :class:`PageConfidence` per page (in page order), each scored with
    the same 6 weighted signals as :func:`score_docling_extraction` but over only
    that page's elements/tables/images/text. A page with no elements still scores
    (its signals default to the neutral 1.0s, same as the doc-level empty case).

    Additive to the doc-level score, which is intentionally left byte-identical so
    the A.3-tuned threshold and the existing accept/fallback behaviour are
    unchanged. A ``None`` document (audio / failed extraction) → ``[]``: there are
    no pages to route per-segment.
    """
    document = getattr(result, "document", None)
    if document is None:
        return []
    pages = getattr(document, "pages", None) or []
    out: list[PageConfidence] = []
    for idx, page in enumerate(pages):
        p_elements = list(_iter_page_elements(page))
        p_tables = list(getattr(page, "tables", []) or [])
        p_images = list(getattr(page, "images", []) or [])
        signals = _signals(
            elements=p_elements,
            tables=p_tables,
            images=p_images,
            text=_page_text(page),
            page_count=1,
        )
        overall = _weighted_sum(signals)
        page_number = int(getattr(page, "page_number", 0) or 0) or (idx + 1)
        out.append(
            PageConfidence(
                page_number=page_number,
                overall=overall,
                signals=signals,
                decision="accept" if overall >= threshold else "fallback",
                threshold=threshold,
            )
        )
    return out


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Element-type values that count as "headings" for the structural signal.
# Matches the strings used by ``ingestion.models.document.ElementType``.
_HEADING_VALUES: frozenset[str] = frozenset({"title", "heading"})

# Element-type values considered "text" for the image/text ratio. Anything
# not in this set (tables/images/formulas) is excluded from the denominator
# so a chart-heavy doc isn't penalised for thin captions.
_TEXT_VALUES: frozenset[str] = frozenset(
    {"text", "paragraph", "title", "heading", "list_item", "caption", "code", "footnote"}
)


def _iter_elements(document: Any) -> Iterable[Any]:
    """Yield every ``ExtractedElement`` across all pages.

    Tolerates pages that store elements/tables/images on separate
    attributes (``elements``, ``tables``, ``images``) — matches the
    ``PageContent`` shape in ``ingestion.models.document``.
    """
    pages = getattr(document, "pages", None) or []
    for page in pages:
        for elem in getattr(page, "elements", []) or []:
            yield elem
        for tbl in getattr(page, "tables", []) or []:
            yield tbl
        for img in getattr(page, "images", []) or []:
            yield img


def _infer_page_count(document: Any) -> int:
    """Fall back to ``len(document.pages)`` when ``page_count`` is unset."""
    pages = getattr(document, "pages", None) or []
    return len(pages) or 1


def _element_type_value(elem: Any) -> str:
    """Return the string value of ``elem.element_type``.

    Element types can be an ``ElementType`` enum (with ``.value``) or
    a plain string in tests/fixtures. We accept both.
    """
    et = getattr(elem, "element_type", None)
    if et is None:
        return ""
    value = getattr(et, "value", None)
    if isinstance(value, str):
        return value
    return str(et).lower()


def _ocr_confidence(elements: list[Any]) -> float:
    """Mean confidence across elements with ``source == "ocr"``.

    When no OCR ran (most native PDFs) we return 1.0 so the signal
    doesn't drag the score down for documents that simply didn't need
    OCR. This matches the table in ``docs/tracks/A-mineru/plan.md``.
    """
    ocr_confs = [
        float(getattr(elem, "confidence", 1.0) or 0.0)
        for elem in elements
        if getattr(elem, "source", "native") == "ocr"
    ]
    if not ocr_confs:
        return 1.0
    return _clamp01(sum(ocr_confs) / len(ocr_confs))


def _signals(
    *,
    elements: list[Any],
    tables: list[Any],
    images: list[Any],
    text: str,
    page_count: int,
) -> Dict[str, float]:
    """Compute the 6 weighted signals over a scope (whole doc OR one page).

    Shared by :func:`score_docling_extraction` (doc scope: all elements +
    ``document.full_text``) and :func:`score_docling_pages` (page scope: one
    page's elements + its joined text, ``page_count=1``). Keeping the doc path
    routed through this helper with the same inputs preserves its output exactly.
    """
    return {
        "ocr_confidence": _ocr_confidence(elements),
        "text_density": _text_density_value(text, page_count),
        "heading_rate": _heading_rate(elements, page_count),
        "table_success": _table_success(tables),
        "image_text_ratio": _image_text_ratio(elements, images),
        "unknown_element_ratio": _unknown_element_ratio(elements),
    }


def _iter_page_elements(page: Any) -> Iterable[Any]:
    """Yield one page's elements + tables + images (single-page ``_iter_elements``)."""
    for elem in getattr(page, "elements", []) or []:
        yield elem
    for tbl in getattr(page, "tables", []) or []:
        yield tbl
    for img in getattr(page, "images", []) or []:
        yield img


def _page_text(page: Any) -> str:
    """A page's prose text for text_density.

    Prefers the real ``PageContent.text_content`` (the combined per-page text);
    falls back to joining each element's ``content`` (what ``ExtractedElement``
    stores its text under, and what the test fakes provide).
    """
    tc = getattr(page, "text_content", None)
    if tc:
        return str(tc)
    parts = [
        str(getattr(elem, "content", "") or "")
        for elem in getattr(page, "elements", []) or []
        if getattr(elem, "content", None)
    ]
    return " ".join(parts)


def _text_density_value(text: str, page_count: int) -> float:
    """``len(text) / pages`` normalised by 1500 chars/page baseline.

    Scanned/image-only documents (or pages) have near-zero density; clean
    academic text comfortably exceeds 1500 chars/page and saturates to 1.0.
    """
    chars_per_page = len(text or "") / max(1, page_count)
    return _clamp01(chars_per_page / 1500.0)


def _heading_rate(elements: list[Any], page_count: int) -> float:
    """Heading density: ≥2 headings per page saturates to 1.0.

    A document with no headings at all (raw scanned text) is a strong
    signal that the structural recovery failed.
    """
    heading_count = sum(
        1 for elem in elements if _element_type_value(elem) in _HEADING_VALUES
    )
    headings_per_page = heading_count / max(1, page_count)
    return _clamp01(headings_per_page / 2.0)


def _table_success(tables: list[Any]) -> float:
    """Fraction of tables that parsed into non-empty rows.

    No tables ⇒ 1.0 (irrelevant signal for prose-only docs). Tables
    detected but with zero rows ⇒ 0.0 — Docling's structural pipeline
    misfired on every table, which is a strong indicator we should let
    MinerU try.
    """
    if not tables:
        return 1.0
    non_empty = sum(1 for tbl in tables if (getattr(tbl, "rows", None) or []))
    return _clamp01(non_empty / len(tables))


def _image_text_ratio(elements: list[Any], images: list[Any]) -> float:
    """1.0 - (image_count / max(1, text_count)), clamped to [0, 1].

    Many images relative to text suggests a scanned doc where Docling
    failed to OCR — exactly the case MinerU's pipeline backend tends
    to recover better.
    """
    text_count = sum(
        1 for elem in elements if _element_type_value(elem) in _TEXT_VALUES
    )
    image_count = len(images)
    if image_count == 0:
        return 1.0
    ratio = image_count / max(1, text_count)
    return _clamp01(1.0 - min(ratio, 1.0))


def _unknown_element_ratio(elements: list[Any]) -> float:
    """1.0 - (unknown_element_count / total_elements).

    An "unknown" element here is a generic TEXT block with no
    section_level and no bbox — Docling's signal for "I extracted some
    text but couldn't place it". A high unknown ratio means the layout
    pipeline punted on most of the document.
    """
    if not elements:
        return 1.0
    unknowns = 0
    for elem in elements:
        if _element_type_value(elem) != "text":
            continue
        if int(getattr(elem, "section_level", 0) or 0) != 0:
            continue
        if getattr(elem, "bbox", None) is not None:
            continue
        unknowns += 1
    return _clamp01(1.0 - (unknowns / len(elements)))


def _weighted_sum(signals: Dict[str, float]) -> float:
    """Aggregate signals using ``SIGNAL_WEIGHTS``."""
    total = 0.0
    for name, weight in SIGNAL_WEIGHTS.items():
        total += weight * float(signals.get(name, 0.0))
    return _clamp01(total)


def _clamp01(value: float) -> float:
    """Clamp to [0.0, 1.0] — defensive against rounding / negative inputs."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


__all__ = [
    "DEFAULT_THRESHOLD",
    "DoclingConfidenceScore",
    "PageConfidence",
    "SIGNAL_WEIGHTS",
    "score_docling_extraction",
    "score_docling_pages",
]
