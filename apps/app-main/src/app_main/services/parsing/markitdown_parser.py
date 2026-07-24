"""Markitdown parser engine (Track A.3).

A third, lightweight parser option alongside Docling (GPU/ML, spatially rich) and
MinerU (GPU service). Microsoft's `markitdown` is pure-Python and ML-free: it
converts Office / HTML / PDF / text files to plain Markdown. It carries **no
spatial data** — there are no bounding boxes or page geometry — so it is the
right pick for born-digital documents where Docling is overkill and page overlays
are not needed. Docling/MinerU remain the default for anything that benefits from
positions (the inspect-workspace overlay, the structure graph).

Dependency isolation
====================
`markitdown` is an OPTIONAL dependency (the ``markitdown`` extra), imported lazily
inside :func:`run_markitdown` so the base install never pays for it and an
environment without it degrades to a clear, actionable error rather than an
import-time crash. (An earlier idea — an ephemeral ``uv run --with`` subprocess —
was rejected: markitdown is light enough to be a normal optional import, and a
subprocess assumes ``uv`` on the runtime image, which the app container does not
guarantee.)

Impedance adapter
=================
The ingestion pipeline is built around Docling's page/element ``ExtractedDocument``
(``chunk_builder.from_document`` reads ``pages[].elements[].bbox`` etc.). Markitdown
yields flat Markdown, so :func:`markdown_to_document` adapts it: one page, one
element per Markdown block, with the section hierarchy reconstructed from ATX
headings and every ``bbox`` left ``None`` (``has_spatial_data`` is then False
downstream). The result flows through the identical chunking / embedding /
extraction path as the other engines.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from ingestion.models import (
    ElementType,
    ExtractedDocument,
    ExtractedElement,
    IngestionResult,
    PageContent,
    SourceMetadata,
    SourceType,
)
from loguru import logger

_ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


class MarkitdownNotInstalled(RuntimeError):
    """Raised when the optional ``markitdown`` dependency is unavailable."""


def run_markitdown(source_path: Path) -> str:
    """Convert a file to Markdown text via markitdown (lazy import).

    Raises:
        MarkitdownNotInstalled: when the optional dependency is not present —
            with the exact install hint, so the operator can enable the engine.
    """
    try:
        from markitdown import MarkItDown
    except ImportError as exc:  # pragma: no cover - exercised via the guard test
        raise MarkitdownNotInstalled(
            "The 'markitdown' parser engine requires the optional 'markitdown' "
            "dependency. Install it with: uv pip install 'markitdown[all]' "
            "(or add the 'markitdown' extra)."
        ) from exc

    result = MarkItDown().convert(str(source_path))
    # markitdown exposes the converted text as .text_content (newer) or .markdown.
    text = getattr(result, "text_content", None)
    if text is None:
        text = getattr(result, "markdown", "") or ""
    return text


def _blocks(markdown: str) -> List[tuple[str, str]]:
    """Split Markdown into ``(kind, text)`` blocks in document order.

    ``kind`` is ``"heading"`` (with the leading ``#``\\ s kept so the level is
    recoverable), ``"code"`` for a fenced block (fences kept), or ``"para"`` for a
    blank-line-delimited paragraph/list run. Whitespace-only runs are dropped.
    """
    blocks: List[tuple[str, str]] = []
    buf: List[str] = []

    def flush() -> None:
        if buf:
            text = "\n".join(buf).strip()
            if text:
                blocks.append(("para", text))
            buf.clear()

    lines = markdown.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if _FENCE.match(line):
            flush()
            fence = [line]
            i += 1
            while i < len(lines) and not _FENCE.match(lines[i]):
                fence.append(lines[i])
                i += 1
            if i < len(lines):  # closing fence
                fence.append(lines[i])
                i += 1
            blocks.append(("code", "\n".join(fence)))
            continue
        if _ATX_HEADING.match(line):
            flush()
            blocks.append(("heading", line.strip()))
            i += 1
            continue
        if not line.strip():
            flush()
            i += 1
            continue
        buf.append(line)
        i += 1
    flush()
    return blocks


def markdown_to_document(
    markdown: str,
    *,
    source_path: Path,
    title: Optional[str] = None,
) -> ExtractedDocument:
    """Adapt flat Markdown to an ``ExtractedDocument`` (no spatial data).

    Each Markdown block becomes one ``ExtractedElement`` on a single page; the
    section hierarchy is reconstructed from ATX headings (``#``…``######``). Every
    ``bbox`` is ``None`` — markitdown has no geometry, so downstream ``positions``
    are empty and ``has_spatial_data`` is False. ``title`` defaults to the first
    level-1 heading, else the file stem.
    """
    elements: List[ExtractedElement] = []
    stack: List[str] = []  # current heading texts, index = level-1
    doc_title = title

    for kind, text in _blocks(markdown):
        if kind == "heading":
            m = _ATX_HEADING.match(text)
            assert m is not None
            level = len(m.group(1))
            heading_text = m.group(2).strip()
            if not heading_text:
                continue
            # Reset the stack to this heading's depth, then push it.
            stack = stack[: level - 1]
            while len(stack) < level - 1:
                stack.append("")  # fill gaps if a level was skipped
            stack.append(heading_text)
            if doc_title is None and level == 1:
                doc_title = heading_text
            elements.append(
                ExtractedElement(
                    content=heading_text,
                    element_type=(
                        ElementType.TITLE if level == 1 else ElementType.HEADING
                    ),
                    page=1,
                    bbox=None,
                    section_path=list(stack),
                    section_level=level,
                )
            )
        else:
            elem_type = ElementType.CODE if kind == "code" else ElementType.PARAGRAPH
            elements.append(
                ExtractedElement(
                    content=text,
                    element_type=elem_type,
                    page=1,
                    bbox=None,
                    section_path=list(stack),
                    section_level=len(stack),
                )
            )

    resolved_title = doc_title or source_path.stem
    page = PageContent(page_number=1, elements=elements)
    full_text = "\n\n".join(e.content for e in elements)
    return ExtractedDocument(
        source_path=source_path,
        title=resolved_title,
        page_count=1,
        pages=[page],
        full_markdown=markdown,
        full_text=full_text,
        metadata={"parser_engine": "markitdown", "has_spatial_data": False},
    )


def parse_with_markitdown(source_path: Path) -> IngestionResult:
    """Parse a file via markitdown into an ``IngestionResult`` (fail-soft).

    On any error (missing dependency, unreadable/unsupported file) returns an
    ``IngestionResult`` with ``success=False`` and a message, mirroring the other
    engines' contract so ``SourceExtractor`` raises a uniform RuntimeError.
    """
    metadata = SourceMetadata(
        filename=source_path.name,
        source_type=SourceType.DOCUMENT,
        file_extension=source_path.suffix.lower(),
        page_count=1,
    )
    try:
        markdown = run_markitdown(source_path)
        document = markdown_to_document(markdown, source_path=source_path)
        logger.info(
            "markitdown: parsed {name} into {n} elements (no spatial data)",
            name=source_path.name,
            n=len(document.pages[0].elements) if document.pages else 0,
        )
        return IngestionResult(
            source_metadata=metadata, success=True, document=document
        )
    except Exception as exc:  # noqa: BLE001 — degrade to a failed result
        logger.warning("markitdown failed for {name}: {exc}", name=source_path.name, exc=exc)
        return IngestionResult(
            source_metadata=metadata, success=False, error_message=str(exc)
        )


__all__ = [
    "MarkitdownNotInstalled",
    "markdown_to_document",
    "parse_with_markitdown",
    "run_markitdown",
]
