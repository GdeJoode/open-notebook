"""Tests for the markitdown parser engine + markdown→Document adapter (A.3)."""

from __future__ import annotations

from pathlib import Path

import app_main.services.parsing.markitdown_parser as mip
import pytest
from app_main.services.chunking import chunk_builder
from app_main.services.parsing.markitdown_parser import (
    MarkitdownNotInstalled,
    markdown_to_document,
    parse_with_markitdown,
    run_markitdown,
)
from ingestion.models import ElementType

_MD = """# Regio Deal Twente

Intro paragraph about brede welvaart.

## Achtergrond

First background paragraph.

Second background paragraph.

### Cijfers

```python
x = 1
```
"""


def test_markdown_to_document_builds_section_hierarchy():
    doc = markdown_to_document(_MD, source_path=Path("deal.md"))

    assert doc.title == "Regio Deal Twente"  # first level-1 heading
    assert doc.page_count == 1
    elements = doc.pages[0].elements

    # Title element
    title = elements[0]
    assert title.element_type == ElementType.TITLE
    assert title.section_level == 1

    # The "Second background paragraph" sits under Achtergrond (level 2).
    para = next(e for e in elements if e.content == "Second background paragraph.")
    assert para.section_path == ["Regio Deal Twente", "Achtergrond"]
    assert para.section_level == 2

    # The fenced code block is preserved as a CODE element under Cijfers.
    code = next(e for e in elements if e.element_type == ElementType.CODE)
    assert "x = 1" in code.content
    assert code.section_path[-1] == "Cijfers"

    # No spatial data — every bbox is None (markitdown has no geometry).
    assert all(e.bbox is None for e in elements)


def test_adapter_output_flows_through_chunk_builder():
    # The headline integration assertion: the adapter's ExtractedDocument is
    # consumable by the REAL chunk builder, so markitdown output uses the same
    # chunking/embedding/extraction path as docling — just without positions.
    doc = markdown_to_document(_MD, source_path=Path("deal.md"))
    chunks = chunk_builder.from_document(doc)

    assert len(chunks) >= 4
    # No spatial data → empty positions + has_spatial_data False.
    assert all(c["positions"] == [] for c in chunks)
    assert all(c["metadata"]["has_spatial_data"] is False for c in chunks)
    # Ordering is preserved and section metadata threaded.
    assert chunks[0]["text"] == "Regio Deal Twente"


def test_title_falls_back_to_stem_when_no_h1():
    doc = markdown_to_document("Just a paragraph.", source_path=Path("notes.md"))
    assert doc.title == "notes"


def test_skipped_heading_level_does_not_crash():
    # h1 then h3 (no h2) — the stack must fill the gap, not raise.
    doc = markdown_to_document(
        "# Top\n\n### Deep\n\nbody", source_path=Path("x.md")
    )
    body = next(e for e in doc.pages[0].elements if e.content == "body")
    assert body.section_path[0] == "Top"
    assert body.section_path[-1] == "Deep"


def test_parse_with_markitdown_is_fail_soft(monkeypatch):
    def boom(_path):
        raise RuntimeError("unreadable")

    monkeypatch.setattr(mip, "run_markitdown", boom)
    result = parse_with_markitdown(Path("broken.pdf"))
    assert result.success is False
    assert "unreadable" in (result.error_message or "")
    assert result.document is None


def test_parse_with_markitdown_success(monkeypatch):
    monkeypatch.setattr(mip, "run_markitdown", lambda _p: "# Doc\n\nbody")
    result = parse_with_markitdown(Path("ok.docx"))
    assert result.success is True
    assert result.document is not None
    assert result.document.title == "Doc"
    assert result.source_metadata.file_extension == ".docx"


def test_run_markitdown_raises_clear_error_when_absent():
    # markitdown is an OPTIONAL dep and is not installed in the test env, so the
    # lazy import must raise the actionable MarkitdownNotInstalled (not ImportError).
    pytest.importorskip  # noqa: B018 - just documenting the intent below
    try:
        import markitdown  # noqa: F401

        pytest.skip("markitdown is installed; the not-installed guard can't run")
    except ImportError:
        pass
    with pytest.raises(MarkitdownNotInstalled):
        run_markitdown(Path("whatever.pdf"))
