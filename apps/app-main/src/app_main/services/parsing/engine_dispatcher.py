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
    "ParserEngineSetting",
    "ResolvedEngine",
    "select_parser_engine",
]
