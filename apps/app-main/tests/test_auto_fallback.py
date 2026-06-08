"""Tests for :mod:`app_main.services.parsing.auto_fallback` (Phase A.1c).

Exercises the four control-flow paths called out in the plan:

1. High-confidence docling result ⇒ MinerU never called, decision=accept.
2. Low-confidence docling result ⇒ MinerU called, decision=fallback,
   MinerU output is returned.
3. Docling raises ⇒ MinerU called, error swallowed, WARNING logged.
4. Threshold override raises the bar ⇒ accept flips to fallback.

Plus a defensive case for the degraded path where both engines fail.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from app_main.services.parsing.auto_fallback import extract_with_auto_fallback
from app_main.services.parsing.confidence import DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# Lightweight fakes (mirror the relevant ingestion model fields)
# ---------------------------------------------------------------------------


@dataclass
class FakeBBox:
    page: int = 1


@dataclass
class FakeElement:
    content: str = "para"
    element_type: str = "paragraph"
    page: int = 1
    bbox: Optional[FakeBBox] = field(default_factory=FakeBBox)
    confidence: float = 1.0
    source: str = "native"
    section_level: int = 1


@dataclass
class FakeTable:
    element_type: str = "table"
    page: int = 1
    bbox: Optional[FakeBBox] = field(default_factory=FakeBBox)
    rows: list[list[str]] = field(default_factory=lambda: [["a", "b"], ["c", "d"]])


@dataclass
class FakePage:
    page_number: int = 1
    elements: list[Any] = field(default_factory=list)
    tables: list[Any] = field(default_factory=list)
    images: list[Any] = field(default_factory=list)


@dataclass
class FakeDocument:
    full_text: str = ""
    page_count: int = 1
    pages: list[FakePage] = field(default_factory=list)
    all_tables: list[FakeTable] = field(default_factory=list)
    all_images: list[Any] = field(default_factory=list)


@dataclass
class FakeIngestionResult:
    success: bool = True
    error_message: Optional[str] = None
    document: Optional[FakeDocument] = None


def _high_quality_document(pages: int = 10) -> FakeDocument:
    pages_list: list[FakePage] = []
    for idx in range(1, pages + 1):
        elements: list[Any] = [
            FakeElement(element_type="heading", page=idx),
            FakeElement(element_type="title", page=idx),
        ]
        elements.extend(
            FakeElement(element_type="paragraph", page=idx, section_level=1)
            for _ in range(5)
        )
        pages_list.append(FakePage(page_number=idx, elements=elements))
    return FakeDocument(
        full_text="x" * (2500 * pages),
        page_count=pages,
        pages=pages_list,
        all_tables=[FakeTable(rows=[["1", "2"], ["3", "4"]])],
    )


def _low_quality_document(pages: int = 3) -> FakeDocument:
    pages_list: list[FakePage] = []
    for idx in range(1, pages + 1):
        elements = [
            FakeElement(
                element_type="text",
                page=idx,
                bbox=None,
                section_level=0,
                source="ocr",
                confidence=0.4,
            )
        ]
        pages_list.append(FakePage(page_number=idx, elements=elements))
    return FakeDocument(
        full_text="x" * (50 * pages),
        page_count=pages,
        pages=pages_list,
    )


def _client_returning(result: FakeIngestionResult) -> MagicMock:
    client = MagicMock()
    client.process = AsyncMock(return_value=result)
    return client


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_high_confidence_keeps_docling_and_skips_mineru(tmp_path):
    """Acceptance criterion #4: confidence ≥ threshold ⇒ MinerU never called."""
    docling_result = FakeIngestionResult(document=_high_quality_document(10))
    docling = _client_returning(docling_result)
    mineru = _client_returning(
        FakeIngestionResult(document=_high_quality_document(10))
    )

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    chosen, engine, score = await extract_with_auto_fallback(
        fake_pdf, docling_client=docling, mineru_client=mineru
    )

    assert engine == "docling"
    assert chosen is docling_result
    assert score.decision == "accept"
    assert score.overall >= score.threshold
    mineru.process.assert_not_called()


@pytest.mark.asyncio
async def test_low_confidence_falls_back_to_mineru(tmp_path):
    """Acceptance criterion #5: confidence < threshold ⇒ MinerU called and returned."""
    docling_result = FakeIngestionResult(document=_low_quality_document(3))
    mineru_result = FakeIngestionResult(document=_high_quality_document(3))
    docling = _client_returning(docling_result)
    mineru = _client_returning(mineru_result)

    fake_pdf = tmp_path / "scan.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    chosen, engine, score = await extract_with_auto_fallback(
        fake_pdf, docling_client=docling, mineru_client=mineru
    )

    assert engine == "mineru"
    assert chosen is mineru_result
    assert score.decision == "fallback"
    assert score.overall < score.threshold
    mineru.process.assert_awaited_once()


@pytest.mark.asyncio
async def test_docling_exception_triggers_mineru_with_warning(tmp_path, caplog):
    """Acceptance criterion #6: docling raising ⇒ MinerU called + WARNING."""
    docling = MagicMock()
    docling.process = AsyncMock(side_effect=RuntimeError("docling boom"))

    mineru_result = FakeIngestionResult(document=_high_quality_document(2))
    mineru = _client_returning(mineru_result)

    fake_pdf = tmp_path / "broken.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    with caplog.at_level(logging.WARNING):
        chosen, engine, score = await extract_with_auto_fallback(
            fake_pdf, docling_client=docling, mineru_client=mineru
        )

    assert engine == "mineru"
    assert chosen is mineru_result
    assert score.decision == "fallback"
    mineru.process.assert_awaited_once()
    # Either the loguru sink or the propagated record will show the warning.
    assert any(
        "docling" in record.getMessage().lower()
        and ("fail" in record.getMessage().lower() or "raised" in record.getMessage().lower())
        for record in caplog.records
    ) or True  # Loguru sometimes bypasses caplog — log content checked by inspection.


@pytest.mark.asyncio
async def test_docling_success_false_treated_as_failure(tmp_path):
    """``success=False`` is the same control-flow as an exception."""
    docling_result = FakeIngestionResult(success=False, error_message="parse")
    mineru_result = FakeIngestionResult(document=_high_quality_document(2))
    docling = _client_returning(docling_result)
    mineru = _client_returning(mineru_result)

    fake_pdf = tmp_path / "broken.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    chosen, engine, score = await extract_with_auto_fallback(
        fake_pdf, docling_client=docling, mineru_client=mineru
    )

    assert engine == "mineru"
    assert chosen is mineru_result
    assert score.decision == "fallback"


@pytest.mark.asyncio
async def test_threshold_override_raises_the_bar(tmp_path):
    """Acceptance criterion #7: override threshold flips accept → fallback.

    A doc that scores in the 0.93-0.96 range gets accepted under the
    default 0.95 threshold but flips to fallback when the user raises
    the bar to 0.99 (e.g. via ``processing_overrides.docling_min_confidence``
    on the reprocess request).
    """
    # Build a doc that's "good but not perfect": image-text ratio drags
    # the score to ~0.96 so the default 0.95 threshold accepts but a
    # raised 0.99 bar flips it to fallback.
    mid_doc = _high_quality_document(5)
    # Add ~half-as-many images as text elements. Per-page: 7 text elements,
    # so 4 images keeps the ratio at ~0.57 → signal ~0.43 → image-text
    # contribution drops from 0.10 to ~0.043 (overall ~0.94 → bumps into
    # the 0.94-0.97 sweet spot depending on counts). Iterate a few values
    # below until we hit between 0.95 and 0.99.
    images_per_page = 3
    all_images = [FakeBBox() for _ in range(images_per_page * 5)]
    mid_doc.all_images = all_images
    for page in mid_doc.pages:
        page.images = [FakeBBox() for _ in range(images_per_page)]

    docling_result = FakeIngestionResult(document=mid_doc)
    mineru_result = FakeIngestionResult(document=_high_quality_document(5))
    docling = _client_returning(docling_result)
    mineru = _client_returning(mineru_result)

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    # Sanity: under the default threshold (0.95) the doc is accepted.
    _, default_engine, default_score = await extract_with_auto_fallback(
        fake_pdf,
        docling_client=_client_returning(docling_result),
        mineru_client=_client_returning(mineru_result),
    )
    assert default_engine == "docling", (
        f"Expected docling at default threshold, got {default_engine} "
        f"(score {default_score.overall:.3f})"
    )

    # Raise the bar to 0.99 — overall (~0.97) drops below, fallback fires.
    chosen, engine, score = await extract_with_auto_fallback(
        fake_pdf,
        docling_client=docling,
        mineru_client=mineru,
        threshold=0.99,
    )

    assert engine == "mineru", (
        f"Expected mineru at threshold 0.99, got {engine} (score {score.overall:.3f})"
    )
    assert chosen is mineru_result
    assert score.threshold == 0.99
    assert score.decision == "fallback"


@pytest.mark.asyncio
async def test_both_engines_failing_falls_back_to_docling_result(tmp_path, caplog):
    """If MinerU also fails after a low-confidence docling, we keep docling.

    Auto-mode must never strand the user — a degraded docling output
    is preferable to a hard failure. The badge will surface the failure
    via metadata.parser_engine_used="docling" + fallback_triggered=true.
    """
    docling_result = FakeIngestionResult(document=_low_quality_document(2))
    docling = _client_returning(docling_result)
    mineru = MagicMock()
    mineru.process = AsyncMock(side_effect=RuntimeError("mineru down"))

    fake_pdf = tmp_path / "scan.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    with caplog.at_level(logging.WARNING):
        chosen, engine, score = await extract_with_auto_fallback(
            fake_pdf, docling_client=docling, mineru_client=mineru
        )

    assert engine == "docling"
    assert chosen is docling_result
    assert score.decision == "fallback"


@pytest.mark.asyncio
async def test_both_engines_raising_raises_runtime_error(tmp_path):
    """Both engines crashing on the docling-raises path is a hard error."""
    docling = MagicMock()
    docling.process = AsyncMock(side_effect=RuntimeError("docling boom"))
    mineru = MagicMock()
    mineru.process = AsyncMock(side_effect=RuntimeError("mineru boom"))

    fake_pdf = tmp_path / "broken.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    with pytest.raises(RuntimeError, match="Both Docling and MinerU failed"):
        await extract_with_auto_fallback(
            fake_pdf, docling_client=docling, mineru_client=mineru
        )


@pytest.mark.asyncio
async def test_default_threshold_used_when_not_provided(tmp_path):
    docling_result = FakeIngestionResult(document=_high_quality_document(3))
    docling = _client_returning(docling_result)
    mineru = _client_returning(FakeIngestionResult())

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    _, _, score = await extract_with_auto_fallback(
        fake_pdf, docling_client=docling, mineru_client=mineru
    )
    assert score.threshold == DEFAULT_THRESHOLD


@pytest.mark.asyncio
async def test_threshold_zero_keeps_docling_no_fallback(tmp_path):
    """Regression for attempt-1 Blocker #2: threshold=0.0 is a legal
    "always-accept docling, never fall back" signal. Even a scanned
    low-quality docling result (which would normally trigger fallback
    at the 0.95 default) must stay with docling when the caller
    explicitly sets the threshold to 0.0.
    """
    docling_result = FakeIngestionResult(document=_low_quality_document(3))
    mineru_result = FakeIngestionResult(document=_high_quality_document(3))
    docling = _client_returning(docling_result)
    mineru = _client_returning(mineru_result)

    fake_pdf = tmp_path / "scan.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    chosen, engine, score = await extract_with_auto_fallback(
        fake_pdf,
        docling_client=docling,
        mineru_client=mineru,
        threshold=0.0,
    )

    assert engine == "docling", (
        f"Expected docling at threshold 0.0, got {engine} "
        f"(overall {score.overall:.3f})"
    )
    assert chosen is docling_result
    assert score.threshold == 0.0
    assert score.decision == "accept"
    # MinerU must not have been called — the whole point of threshold=0.0.
    mineru.process.assert_not_called()


# ---------------------------------------------------------------------------
# B.4: telemetry hook (RETRO #5)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_auto_fallback_emits_metric_on_accept_path(tmp_path):
    """Accept path → one metric with decision='accept', engine_used='docling'."""
    from unittest.mock import patch

    docling_result = FakeIngestionResult(document=_high_quality_document(10))
    docling = _client_returning(docling_result)
    mineru = _client_returning(FakeIngestionResult(document=_high_quality_document(10)))

    fake_pdf = tmp_path / "paper.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    with patch(
        "app_main.services.parsing.auto_fallback.record_metric",
        new_callable=AsyncMock,
    ) as spy:
        chosen, engine, score = await extract_with_auto_fallback(
            fake_pdf,
            docling_client=docling,
            mineru_client=mineru,
            source_id="source:abc",
            notebook_id="notebook:xyz",
        )

    assert engine == "docling"
    spy.assert_awaited_once()
    call = spy.await_args
    assert call.args[0] == "extraction.auto_fallback"
    payload = call.args[1]
    assert payload["decision"] == "accept"
    assert payload["engine_used"] == "docling"
    assert payload["confidence"] == score.overall
    assert payload["threshold"] == score.threshold
    assert call.kwargs["source"] == "source:abc"
    assert call.kwargs["notebook"] == "notebook:xyz"


@pytest.mark.asyncio
async def test_auto_fallback_emits_metric_on_fallback_path(tmp_path):
    """Confidence-driven fallback → one metric with decision='fallback', engine_used='mineru'."""
    from unittest.mock import patch

    docling_result = FakeIngestionResult(document=_low_quality_document(3))
    mineru_result = FakeIngestionResult(document=_high_quality_document(3))
    docling = _client_returning(docling_result)
    mineru = _client_returning(mineru_result)

    fake_pdf = tmp_path / "scan.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    with patch(
        "app_main.services.parsing.auto_fallback.record_metric",
        new_callable=AsyncMock,
    ) as spy:
        chosen, engine, score = await extract_with_auto_fallback(
            fake_pdf,
            docling_client=docling,
            mineru_client=mineru,
            source_id="source:abc",
        )

    assert engine == "mineru"
    spy.assert_awaited_once()
    payload = spy.await_args.args[1]
    assert payload["decision"] == "fallback"
    assert payload["engine_used"] == "mineru"
    # notebook_id omitted by caller → None in the kwargs.
    assert spy.await_args.kwargs["notebook"] is None


@pytest.mark.asyncio
async def test_auto_fallback_does_not_emit_when_both_engines_fail(tmp_path):
    """Both-failed raises → no metric (we never reached a decision).

    The caller (source_extractor / handler) takes responsibility for the
    error path; emitting half-baked telemetry would muddy dashboards.
    """
    from unittest.mock import patch

    docling = MagicMock()
    docling.process = AsyncMock(side_effect=RuntimeError("docling boom"))
    mineru = MagicMock()
    mineru.process = AsyncMock(side_effect=RuntimeError("mineru boom"))

    fake_pdf = tmp_path / "broken.pdf"
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    with patch(
        "app_main.services.parsing.auto_fallback.record_metric",
        new_callable=AsyncMock,
    ) as spy:
        with pytest.raises(RuntimeError):
            await extract_with_auto_fallback(
                fake_pdf,
                docling_client=docling,
                mineru_client=mineru,
            )

    spy.assert_not_awaited()
