"""Tests for the parser engine dispatcher (Phase A.1b).

The dispatcher is a pure function — exhaustively cover every setting +
extension combination to satisfy the >=90% coverage acceptance criterion.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app_main.services.parsing.engine_dispatcher import (
    DEFAULT_MINERU_EXTENSIONS,
    select_parser_engine,
)


class TestSimpleSetting:
    def test_returns_simple_for_pdf(self):
        assert select_parser_engine("simple", Path("doc.pdf")) == "simple"

    def test_returns_simple_for_unknown_ext(self):
        assert select_parser_engine("simple", Path("notes.xyz")) == "simple"

    def test_returns_simple_with_no_extension(self):
        assert select_parser_engine("simple", Path("README")) == "simple"


class TestDoclingSetting:
    def test_returns_docling_for_pdf(self):
        assert select_parser_engine("docling", Path("doc.pdf")) == "docling"

    def test_returns_docling_for_html(self):
        assert select_parser_engine("docling", Path("page.html")) == "docling"

    def test_returns_docling_regardless_of_extension(self):
        # The docling setting is explicit — we trust the caller.
        assert select_parser_engine("docling", Path("img.png")) == "docling"


class TestMineruSetting:
    @pytest.mark.parametrize(
        "filename",
        [
            "paper.pdf",
            "PAPER.PDF",  # case insensitive
            "report.docx",
            "slides.pptx",
            "scan.png",
            "scan.jpg",
            "scan.jpeg",
            "old.doc",
        ],
    )
    def test_returns_mineru_for_supported_extension(self, filename: str):
        assert select_parser_engine("mineru", Path(filename)) == "mineru"

    @pytest.mark.parametrize(
        "filename",
        [
            "page.html",
            "page.htm",
            "notes.md",
            "data.csv",
            "logs.txt",
            "spreadsheet.xlsx",  # not in default mineru list
            "presentation.ppt",  # legacy ppt also excluded by default
            "noextension",
        ],
    )
    def test_falls_back_to_docling_for_unsupported_extension(self, filename: str):
        assert select_parser_engine("mineru", Path(filename)) == "docling"

    def test_respects_custom_mineru_extensions(self):
        # User shrinks the set to PDF only — docx no longer routes to mineru.
        assert (
            select_parser_engine(
                "mineru", Path("report.docx"), mineru_supported_extensions=[".pdf"]
            )
            == "docling"
        )

    def test_custom_extensions_without_dot_are_normalised(self):
        assert (
            select_parser_engine(
                "mineru", Path("notes.md"), mineru_supported_extensions=["md"]
            )
            == "mineru"
        )

    def test_empty_mineru_extensions_always_falls_back(self):
        # Edge: operator emptied the list — every file falls back to docling.
        assert (
            select_parser_engine(
                "mineru", Path("doc.pdf"), mineru_supported_extensions=[]
            )
            == "docling"
        )

    def test_none_extensions_uses_defaults(self):
        # Explicit None == "use built-in defaults".
        assert (
            select_parser_engine("mineru", Path("doc.pdf"), mineru_supported_extensions=None)
            == "mineru"
        )


class TestAutoSetting:
    """``auto`` resolves to docling here; the real fallback lives in
    ``app_main.services.parsing.auto_fallback`` (A.1c) and is invoked
    by SourceExtractor before consulting this dispatcher.

    Keeping the dispatcher pure means callers that don't care about
    auto-fallback (per-file routing, simple dispatcher tests) still
    get a single concrete engine back.
    """

    def test_auto_resolves_to_docling(self):
        assert select_parser_engine("auto", Path("doc.pdf")) == "docling"

    def test_auto_resolves_to_docling_for_image(self):
        assert select_parser_engine("auto", Path("scan.png")) == "docling"

    def test_auto_does_not_return_auto(self):
        # The dispatcher must always resolve to a concrete engine.
        resolved = select_parser_engine("auto", Path("a.pdf"))
        assert resolved in {"simple", "docling", "mineru"}


class TestDefensiveDefault:
    def test_unknown_setting_defaults_to_docling(self):
        # type: ignore — deliberate test of forward-compat fallback.
        assert select_parser_engine("bogus", Path("doc.pdf")) == "docling"  # type: ignore[arg-type]


class TestDefaultExtensions:
    def test_default_extensions_constant_includes_pdf(self):
        assert ".pdf" in DEFAULT_MINERU_EXTENSIONS

    def test_default_extensions_lowercase(self):
        assert all(ext == ext.lower() for ext in DEFAULT_MINERU_EXTENSIONS)
