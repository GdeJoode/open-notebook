"""
Tests for app_main.services.parsing.mineru_layout_parser.

The parser ingests MinerU's structured output (content_list.json plus
optional middle.json + markdown) and rebuilds an ExtractedDocument. We
drive it with synthetic JSON fixtures that mirror the schema documented
in docs/tracks/A-mineru/MINERU_OUTPUT_SPIKE.md — no MinerU install
required.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ingestion.models import ElementType, ExtractedImage, ExtractedTable
from app_main.services.parsing import MineruLayoutParseError, parse_mineru_output


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _write_content_list(tmp_path: Path, records: list[dict]) -> Path:
    p = tmp_path / "foo_content_list.json"
    p.write_text(json.dumps(records), encoding="utf-8")
    return p


def _write_middle(tmp_path: Path, backend: str = "pipeline") -> Path:
    p = tmp_path / "foo_middle.json"
    p.write_text(
        json.dumps({"_backend": backend, "_version_name": "2.0.0", "pdf_info": []}),
        encoding="utf-8",
    )
    return p


def _source(tmp_path: Path) -> Path:
    src = tmp_path / "foo.pdf"
    src.write_bytes(b"%PDF-1.4 fake")
    return src


# ---------------------------------------------------------------------------
# Happy-path translation
# ---------------------------------------------------------------------------


def test_text_levels_map_to_title_heading_and_paragraph(tmp_path: Path) -> None:
    records = [
        {"type": "text", "text": "Doc title", "text_level": 1, "bbox": [100, 200, 900, 300], "page_idx": 0},
        {"type": "text", "text": "Section heading", "text_level": 2, "bbox": [100, 320, 900, 360], "page_idx": 0},
        {"type": "text", "text": "Body paragraph text.", "bbox": [100, 400, 900, 500], "page_idx": 0},
    ]
    content_list = _write_content_list(tmp_path, records)
    middle = _write_middle(tmp_path)
    src = _source(tmp_path)

    doc = parse_mineru_output(
        source_path=src,
        output_dir=tmp_path,
        content_list_path=content_list,
        middle_json_path=middle,
        processing_seconds=1.5,
    )

    assert doc.title == "Doc title"
    assert doc.page_count == 1
    assert len(doc.pages) == 1

    elems = doc.pages[0].elements
    assert [e.element_type for e in elems] == [
        ElementType.TITLE,
        ElementType.HEADING,
        ElementType.PARAGRAPH,
    ]
    assert elems[1].section_level == 2
    # Bbox normalised from 0-1000 to 0-1 with TOPLEFT origin and page=1.
    assert elems[0].bbox is not None
    assert elems[0].bbox.page == 1
    assert elems[0].bbox.x == pytest.approx(0.1)
    assert elems[0].bbox.y == pytest.approx(0.2)
    assert elems[0].bbox.width == pytest.approx(0.8)
    assert elems[0].bbox.height == pytest.approx(0.1)
    assert all(e.source == "mineru" for e in elems)


def test_full_markdown_is_loaded_when_present(tmp_path: Path) -> None:
    records = [
        {"type": "text", "text": "Hello world", "bbox": [0, 0, 1000, 50], "page_idx": 0},
    ]
    content_list = _write_content_list(tmp_path, records)
    md_path = tmp_path / "foo.md"
    md_path.write_text("# Foo\n\nHello world\n", encoding="utf-8")

    doc = parse_mineru_output(
        source_path=_source(tmp_path),
        output_dir=tmp_path,
        content_list_path=content_list,
        markdown_path=md_path,
    )
    assert doc.full_markdown.startswith("# Foo")
    assert "Hello world" in doc.full_text


def test_multi_page_grouping_and_page_count(tmp_path: Path) -> None:
    records = [
        {"type": "text", "text": "Page 0 line", "bbox": [0, 0, 100, 50], "page_idx": 0},
        {"type": "text", "text": "Page 2 line", "bbox": [0, 0, 100, 50], "page_idx": 2},
        {"type": "text", "text": "Page 2 line 2", "bbox": [0, 0, 100, 50], "page_idx": 2},
    ]
    content_list = _write_content_list(tmp_path, records)
    doc = parse_mineru_output(
        source_path=_source(tmp_path),
        output_dir=tmp_path,
        content_list_path=content_list,
    )
    assert doc.page_count == 3
    # We only build PageContent objects for pages that actually have elements.
    assert [p.page_number for p in doc.pages] == [1, 3]
    assert len(doc.pages[1].elements) == 2


def test_table_record_yields_extracted_table_with_rows_and_markdown(tmp_path: Path) -> None:
    html = (
        "<html><body><table>"
        "<tr><th>Site</th><th>P</th></tr>"
        "<tr><td>Pine Ck</td><td>1.2</td></tr>"
        "<tr><td>Bennett</td><td>0.4</td></tr>"
        "</table></body></html>"
    )
    records = [
        {
            "type": "table",
            "img_path": "images/abc.jpg",
            "table_caption": ["Caption text"],
            "table_footnote": ["Note text"],
            "table_body": html,
            "bbox": [62, 480, 946, 904],
            "page_idx": 0,
        }
    ]
    content_list = _write_content_list(tmp_path, records)
    doc = parse_mineru_output(
        source_path=_source(tmp_path),
        output_dir=tmp_path,
        content_list_path=content_list,
    )
    assert len(doc.all_tables) == 1
    table = doc.all_tables[0]
    assert isinstance(table, ExtractedTable)
    assert table.headers == ["Site", "P"]
    assert table.rows == [["Pine Ck", "1.2"], ["Bennett", "0.4"]]
    assert table.metadata.get("caption") == ["Caption text"]
    assert table.metadata.get("footnote") == ["Note text"]
    assert "html" in table.metadata
    assert table.markdown.startswith("| Site | P |")


def test_image_and_chart_records_yield_extracted_image_with_resolved_path(tmp_path: Path) -> None:
    records = [
        {
            "type": "image",
            "img_path": "images/img1.jpg",
            "image_caption": ["Figure 1."],
            "bbox": [62, 480, 946, 904],
            "page_idx": 1,
        },
        {
            "type": "chart",
            "img_path": "images/chart1.jpg",
            "image_caption": ["Chart 2."],
            "bbox": [62, 480, 946, 904],
            "page_idx": 1,
        },
    ]
    content_list = _write_content_list(tmp_path, records)
    doc = parse_mineru_output(
        source_path=_source(tmp_path),
        output_dir=tmp_path,
        content_list_path=content_list,
    )
    assert len(doc.all_images) == 2
    img, chart = doc.all_images
    assert isinstance(img, ExtractedImage)
    assert img.image_path == tmp_path / "images" / "img1.jpg"
    assert img.classification == "image"
    assert img.description == "Figure 1."
    assert chart.classification == "chart"


def test_equation_code_and_list_types_translated(tmp_path: Path) -> None:
    records = [
        {
            "type": "equation",
            "text": "$$E=mc^2$$",
            "text_format": "latex",
            "bbox": [0, 0, 100, 50],
            "page_idx": 0,
        },
        {
            "type": "code",
            "sub_type": "code",
            "code_body": "print('hi')",
            "bbox": [0, 100, 100, 200],
            "page_idx": 0,
        },
        {
            "type": "list",
            "sub_type": "text",
            "list_items": ["alpha", "beta", "gamma"],
            "bbox": [0, 300, 100, 400],
            "page_idx": 0,
        },
    ]
    content_list = _write_content_list(tmp_path, records)
    doc = parse_mineru_output(
        source_path=_source(tmp_path),
        output_dir=tmp_path,
        content_list_path=content_list,
    )
    elems = doc.pages[0].elements
    types = {e.element_type for e in elems}
    assert ElementType.FORMULA in types
    assert ElementType.CODE in types
    assert ElementType.LIST_ITEM in types
    list_elem = next(e for e in elems if e.element_type == ElementType.LIST_ITEM)
    assert "- alpha" in list_elem.content


def test_page_furniture_types_classified(tmp_path: Path) -> None:
    records = [
        {"type": "header", "text": "Top header", "bbox": [0, 0, 100, 30], "page_idx": 0},
        {"type": "footer", "text": "Bottom footer", "bbox": [0, 900, 100, 950], "page_idx": 0},
        {"type": "page_number", "text": "1", "bbox": [490, 950, 510, 980], "page_idx": 0},
        {"type": "page_footnote", "text": "fn1", "bbox": [0, 900, 100, 920], "page_idx": 0},
        {"type": "aside_text", "text": "side note", "bbox": [0, 200, 50, 250], "page_idx": 0},
    ]
    content_list = _write_content_list(tmp_path, records)
    doc = parse_mineru_output(
        source_path=_source(tmp_path),
        output_dir=tmp_path,
        content_list_path=content_list,
    )
    by_type = {e.element_type: e for e in doc.pages[0].elements}
    assert by_type[ElementType.HEADER].content == "Top header"
    assert by_type[ElementType.FOOTNOTE].content == "fn1"
    assert by_type[ElementType.CAPTION].content == "side note"
    # Both footer and page_number map to FOOTER — keep at least one.
    footer_contents = {
        e.content for e in doc.pages[0].elements if e.element_type == ElementType.FOOTER
    }
    assert "Bottom footer" in footer_contents
    assert "1" in footer_contents


def test_unknown_type_falls_back_to_text(tmp_path: Path) -> None:
    records = [
        {"type": "novel_type", "text": "Surprising content", "bbox": [0, 0, 100, 50], "page_idx": 0},
    ]
    content_list = _write_content_list(tmp_path, records)
    doc = parse_mineru_output(
        source_path=_source(tmp_path),
        output_dir=tmp_path,
        content_list_path=content_list,
    )
    elems = doc.pages[0].elements
    assert len(elems) == 1
    assert elems[0].element_type == ElementType.TEXT
    assert elems[0].metadata.get("mineru_type") == "novel_type"


# ---------------------------------------------------------------------------
# Bbox normalisation
# ---------------------------------------------------------------------------


def test_vlm_backend_uses_unit_normalisation(tmp_path: Path) -> None:
    records = [
        {"type": "text", "text": "Hello", "bbox": [0.1, 0.2, 0.9, 0.4], "page_idx": 0},
    ]
    content_list = _write_content_list(tmp_path, records)
    middle = _write_middle(tmp_path, backend="vlm")

    doc = parse_mineru_output(
        source_path=_source(tmp_path),
        output_dir=tmp_path,
        content_list_path=content_list,
        middle_json_path=middle,
    )
    bbox = doc.pages[0].elements[0].bbox
    assert bbox is not None
    assert bbox.x == pytest.approx(0.1)
    assert bbox.y == pytest.approx(0.2)
    assert bbox.width == pytest.approx(0.8)
    assert bbox.height == pytest.approx(0.2)


def test_missing_or_malformed_bbox_is_tolerated(tmp_path: Path) -> None:
    records = [
        {"type": "text", "text": "no bbox", "page_idx": 0},
        {"type": "text", "text": "bad bbox", "bbox": "not-a-list", "page_idx": 0},
        {"type": "text", "text": "short bbox", "bbox": [1, 2], "page_idx": 0},
    ]
    content_list = _write_content_list(tmp_path, records)
    doc = parse_mineru_output(
        source_path=_source(tmp_path),
        output_dir=tmp_path,
        content_list_path=content_list,
    )
    assert all(e.bbox is None for e in doc.pages[0].elements)


def test_unreadable_middle_json_defaults_to_pipeline_factor(tmp_path: Path) -> None:
    records = [
        {"type": "text", "text": "X", "bbox": [100, 200, 900, 300], "page_idx": 0},
    ]
    content_list = _write_content_list(tmp_path, records)
    middle = tmp_path / "foo_middle.json"
    middle.write_text("not json", encoding="utf-8")

    doc = parse_mineru_output(
        source_path=_source(tmp_path),
        output_dir=tmp_path,
        content_list_path=content_list,
        middle_json_path=middle,
    )
    bbox = doc.pages[0].elements[0].bbox
    assert bbox.x == pytest.approx(0.1)
    assert bbox.width == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_missing_content_list_raises(tmp_path: Path) -> None:
    with pytest.raises(MineruLayoutParseError, match="not found"):
        parse_mineru_output(
            source_path=_source(tmp_path),
            output_dir=tmp_path,
            content_list_path=tmp_path / "nope.json",
        )


def test_invalid_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "foo_content_list.json"
    p.write_text("definitely not json{", encoding="utf-8")
    with pytest.raises(MineruLayoutParseError, match="not valid JSON"):
        parse_mineru_output(
            source_path=_source(tmp_path),
            output_dir=tmp_path,
            content_list_path=p,
        )


def test_non_array_json_raises(tmp_path: Path) -> None:
    p = tmp_path / "foo_content_list.json"
    p.write_text(json.dumps({"oops": True}), encoding="utf-8")
    with pytest.raises(MineruLayoutParseError, match="JSON array"):
        parse_mineru_output(
            source_path=_source(tmp_path),
            output_dir=tmp_path,
            content_list_path=p,
        )


def test_empty_after_translation_raises(tmp_path: Path) -> None:
    records = [
        {"type": "text", "text": "", "bbox": [0, 0, 100, 50], "page_idx": 0},
        {"type": "list", "list_items": [], "bbox": [0, 0, 100, 50], "page_idx": 0},
    ]
    content_list = _write_content_list(tmp_path, records)
    with pytest.raises(MineruLayoutParseError, match="zero elements"):
        parse_mineru_output(
            source_path=_source(tmp_path),
            output_dir=tmp_path,
            content_list_path=content_list,
        )


def test_title_fallback_to_stem_when_no_h1(tmp_path: Path) -> None:
    records = [
        {"type": "text", "text": "Body only", "bbox": [0, 0, 100, 50], "page_idx": 0},
    ]
    content_list = _write_content_list(tmp_path, records)
    src = _source(tmp_path)
    doc = parse_mineru_output(
        source_path=src,
        output_dir=tmp_path,
        content_list_path=content_list,
    )
    assert doc.title == src.stem
