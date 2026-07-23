"""Reference extraction pipeline (Track V.1→V.3 chain) — chunks → references.

The thin composition of the three pure producer stages:

    V.1 :func:`locate_reference_region` — find the bibliography region in a
        source's chunks (or ``full_text`` fallback).
    V.2 :func:`segment_region` — split the region into individual entry strings.
    V.3 :func:`parse_reference` — parse each entry into a ``ParsedReference``.

:func:`extract_references` runs the chain end-to-end and returns the source's
``List[ParsedReference]`` — the Track V → U.3/V.4 boundary type. It is PURE (no
DB, no network, no LLM): the V.5 orchestration that persists these / feeds U.3's
``cites`` materialization is a LATER phase and is deliberately NOT wired here.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from shared.references.reference_parser import parse_reference
from shared.references.region_locator import ReferenceChunk, locate_reference_region
from shared.references.segmenter import AmbiguityResolver, segment_region
from shared.retrieval.cites_matching import ParsedReference


def extract_references(
    chunks: Sequence[ReferenceChunk],
    full_text: str = "",
    *,
    ambiguity_resolver: Optional[AmbiguityResolver] = None,
) -> List[ParsedReference]:
    """Extract a source's references from its chunks (+ optional ``full_text``).

    Chains V.1 → V.2 → V.3. When no reference region is located (a document with
    no bibliography), returns ``[]`` — never raises.

    Args:
        chunks: The source's persisted chunk projections (DB-free).
        full_text: The source's concatenated text (enables the V.1 fallback and
            span computation).
        ambiguity_resolver: Optional V.2 seam for a later LLM re-segmenter;
            ``None`` keeps the pipeline fully deterministic/offline.

    Returns:
        The list of parsed references (each carrying a verbatim ``raw_text``), in
        document order; ``[]`` when no region is found or it holds no entries.
    """
    region = locate_reference_region(chunks, full_text)
    if not region.found:
        return []

    entries = segment_region(region.text, ambiguity_resolver=ambiguity_resolver)
    return [parse_reference(entry) for entry in entries if entry.strip()]


__all__ = [
    "extract_references",
]
