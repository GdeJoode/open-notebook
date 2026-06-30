"""Pure parser-engine dispatcher (Phase A.1b).

Maps a ``parser_engine`` setting + file path to the concrete engine that
should actually run for this file. Kept as a pure function so it can be
unit-tested without I/O and reused by the auto-fallback orchestrator in
Phase A.1c.

The dispatcher is intentionally conservative: when the user picks MinerU
but the file extension is not in ``mineru_supported_extensions``, we fall
back to Docling and log at INFO. This matches the semantics from
``docs/tracks/A-mineru/plan.md`` Phase A.1b acceptance criterion 4.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional

from loguru import logger


ParserEngineSetting = Literal["simple", "docling", "mineru", "auto"]
ResolvedEngine = Literal["simple", "docling", "mineru"]


# Default extensions when the caller has no ContentSettings handy (rare; the
# typical call passes the configured list). Mirrors
# ContentSettings.mineru_supported_extensions in packages/shared/src/shared/models/settings.py.
DEFAULT_MINERU_EXTENSIONS: tuple[str, ...] = (
    ".pdf", ".docx", ".doc", ".pptx", ".png", ".jpg", ".jpeg",
)


# Document-style extensions the Docling pipeline (and therefore the "auto"
# confidence-fallback path) can handle. Audio / video and other non-document
# inputs skip the docling/mineru dispatch entirely — IngestionWorkflow picks
# the right pipeline (WhisperX etc.) for them. This set is the single source
# of truth for "is this a docling-routable file"; SourceExtractor imports it
# rather than keeping its own copy.
DOCLING_PARSEABLE_EXTENSIONS: frozenset[str] = frozenset(
    {
        ".pdf", ".docx", ".doc", ".xlsx", ".xls",
        ".pptx", ".ppt", ".html", ".htm", ".txt", ".md",
    }
)


def select_parser_engine(
    setting: ParserEngineSetting,
    path: Path,
    *,
    mineru_supported_extensions: Optional[Iterable[str]] = None,
) -> ResolvedEngine:
    """Pick the concrete engine for a file given the user's setting.

    Rules:
    1. ``"simple"`` -> ``"simple"`` (text-only extraction, no parser service).
    2. ``"docling"`` -> ``"docling"`` (current behaviour; default).
    3. ``"mineru"`` -> ``"mineru"`` when the extension is in the supported
       set; otherwise fall back to ``"docling"`` with an INFO log.
    4. ``"auto"`` -> ``"docling"`` for now. The confidence-driven fallback
       lands in Phase A.1c; routing it through the dispatcher means we can
       swap the implementation in one place. The dispatcher itself never
       returns ``"auto"`` — auto-mode resolves to an actual engine.

    The function is pure: it only reads ``path.suffix`` and the
    ``mineru_supported_extensions`` argument. No env vars, no filesystem
    access, no HTTP. Logging is incidental and easy to silence in tests.
    """
    extension = path.suffix.lower()
    allowed = _normalise_extensions(
        mineru_supported_extensions
        if mineru_supported_extensions is not None
        else DEFAULT_MINERU_EXTENSIONS
    )

    if setting == "simple":
        return "simple"

    if setting == "docling":
        return "docling"

    if setting == "mineru":
        if extension in allowed:
            return "mineru"
        logger.info(
            f"MinerU does not support {extension or '<no-extension>'}, "
            f"falling back to docling for {path.name}"
        )
        return "docling"

    if setting == "auto":
        # In Phase A.1b, "auto" defers to Docling. Phase A.1c will replace
        # this branch with the confidence-driven fallback. Keeping the
        # routing decision here (rather than in SourceExtractor) means the
        # confidence path can also reuse the extension guard above.
        return "docling"

    # Defensive default for forward-compat with future enum values.
    logger.warning(
        f"Unknown parser_engine setting {setting!r}; defaulting to docling"
    )
    return "docling"


@dataclass(frozen=True)
class ParserRoute:
    """The fully-resolved parsing decision for a single file.

    The single source of truth for *how* a document file gets parsed, so
    SourceExtractor no longer re-derives the ``"auto"`` decision separately
    from the dispatcher (the two used to be able to drift):

    - ``engine``: the concrete engine for the non-auto path
      (``simple`` / ``docling`` / ``mineru``), as ``select_parser_engine``
      resolves it.
    - ``use_auto_fallback``: True iff the confidence-driven Docling→MinerU
      fallback (A.1c) should run instead of a single fixed engine. Only ever
      True when the user picked ``"auto"`` AND the file is docling-parseable
      AND the extension didn't already route to MinerU.
    - ``is_docling_extension``: whether this is a document-style file the
      docling/mineru dispatch applies to (False for audio/video → WhisperX).
    """

    engine: ResolvedEngine
    use_auto_fallback: bool
    is_docling_extension: bool


def resolve_parser_route(
    setting: ParserEngineSetting,
    path: Path,
    *,
    mineru_supported_extensions: Optional[Iterable[str]] = None,
) -> ParserRoute:
    """Resolve the complete parsing decision for ``path`` in ONE place.

    This collapses the two formerly-independent decisions into a single
    function:

    1. The concrete engine (``select_parser_engine`` — unchanged, still pure
       and forward-compat: ``"auto"`` resolves to ``"docling"`` there).
    2. Whether to invoke the A.1c confidence fallback (``use_auto_fallback``),
       which previously lived inline in ``SourceExtractor._process_file`` and
       re-checked the raw ``"auto"`` string — a second decision site that
       could drift from the dispatcher.

    Non-document files (audio/video, anything outside
    ``DOCLING_PARSEABLE_EXTENSIONS``) never auto-fallback and resolve to
    ``"docling"`` (IngestionWorkflow routes them to the real pipeline).
    Behaviour is identical to the previous split logic.
    """
    is_docling_extension = path.suffix.lower() in DOCLING_PARSEABLE_EXTENSIONS

    if not is_docling_extension:
        # Audio/video and other non-document extensions skip the
        # docling/mineru dispatch entirely.
        return ParserRoute(
            engine="docling",
            use_auto_fallback=False,
            is_docling_extension=False,
        )

    engine = select_parser_engine(
        setting,
        path,
        mineru_supported_extensions=mineru_supported_extensions,
    )

    # Auto-fallback fires only when the user explicitly chose "auto", the
    # file is docling-parseable, and the dispatcher didn't already pick
    # MinerU (a MinerU-eligible extension under "auto" still resolves to
    # docling in select_parser_engine, so this is belt-and-braces).
    use_auto_fallback = setting == "auto" and engine != "mineru"

    return ParserRoute(
        engine=engine,
        use_auto_fallback=use_auto_fallback,
        is_docling_extension=True,
    )


def _normalise_extensions(extensions: Iterable[str]) -> frozenset[str]:
    """Lower-case + dot-prefix every entry, drop empties."""
    out: set[str] = set()
    for ext in extensions:
        if not ext:
            continue
        ext = ext.lower()
        if not ext.startswith("."):
            ext = "." + ext
        out.add(ext)
    return frozenset(out)


__all__ = [
    "DEFAULT_MINERU_EXTENSIONS",
    "DOCLING_PARSEABLE_EXTENSIONS",
    "ParserEngineSetting",
    "ParserRoute",
    "ResolvedEngine",
    "resolve_parser_route",
    "select_parser_engine",
]
