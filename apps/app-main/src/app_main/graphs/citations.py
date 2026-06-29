"""Citation helpers for the answer graphs (Track X.2).

Derives a structured ``citations`` array from the provenance that Track X.1
attaches to retrieval hits (``source``/``chunk_id``/``physical_page``/
``section_path``). The citation set is the provenance of the context hits that
were actually fed to the LLM — a deterministic, defensible source set.

The X.3 faithfulness guard (membership-check against the retrieval set) is out
of scope here; we only emit citations from the context hits.

A citation is shaped::

    {"source": "source:...", "page": <int|None>, "chunk_id": "chunk:...|None",
     "section": "Chapter > Section" | None}

``page`` and ``chunk_id`` are ``None`` for page-less hits (insights, notes,
plain text, audio) — those still cite the ``source`` so the answer never lacks
attribution and never 500s on a missing page.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _section_to_str(section_path: Any) -> Optional[str]:
    """Render a ``section_path`` (list breadcrumb) to a readable string.

    Chunks carry ``section_path`` as a heading breadcrumb list
    (e.g. ``["Chapter 1", "Section 1.2"]``). We join it for both the prompt tag
    and the citation ``section`` field; ``None``/empty stays ``None``.
    """
    if not section_path:
        return None
    if isinstance(section_path, str):
        return section_path or None
    if isinstance(section_path, (list, tuple)):
        parts = [str(p).strip() for p in section_path if str(p).strip()]
        return " > ".join(parts) if parts else None
    return str(section_path)


def hit_to_citation(hit: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Build a single citation dict from a retrieval hit, or ``None``.

    The ``source`` anchor is taken from the hit's own ``source`` key (set by
    X.1 hydration) and falls back to the hit ``id``/``parent_id`` when the hit
    is itself a ``source:`` record. Returns ``None`` only when no source anchor
    can be determined (the hit is then uncitable and is skipped).
    """
    source = hit.get("source")
    if not source:
        # Fall back to the hit's own/parent id when it is a source record.
        for key in ("id", "parent_id"):
            candidate = hit.get(key)
            if isinstance(candidate, str) and candidate.startswith("source:"):
                source = candidate
                break
    if not source:
        return None

    return {
        "source": source,
        "page": hit.get("physical_page"),
        "chunk_id": hit.get("chunk_id"),
        "section": _section_to_str(hit.get("section_path")),
    }


def citations_from_hits(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Derive a de-duplicated citation list from retrieval hits.

    De-dup key is ``(source, page, chunk_id)`` so the same source cited at two
    different pages yields two citations, but a repeated chunk collapses to one.
    Order follows first appearance (retrieval rank).
    """
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for hit in hits:
        citation = hit_to_citation(hit)
        if citation is None:
            continue
        key = (citation["source"], citation["page"], citation["chunk_id"])
        if key in seen:
            continue
        seen.add(key)
        out.append(citation)
    return out


def format_citation_tag(hit: Dict[str, Any]) -> str:
    """A compact provenance tag for a context block, e.g.::

        [source: source:abc | p.7 | Methods > Results]

    Only the parts that exist are included; a page-less hit yields
    ``[source: source:abc]``. Used to prefix each result block in the prompt so
    the model can attribute claims to the exact source/page/section.
    """
    citation = hit_to_citation(hit)
    if citation is None:
        return ""
    parts = [f"source: {citation['source']}"]
    if citation["page"] is not None:
        parts.append(f"p.{citation['page']}")
    if citation["section"]:
        parts.append(citation["section"])
    return "[" + " | ".join(parts) + "]"


def merge_citations(
    citation_lists: List[List[Dict[str, Any]]],
) -> List[Dict[str, Any]]:
    """Flatten and de-duplicate citation lists (e.g. across sub-answers)."""
    flat: List[Dict[str, Any]] = []
    for lst in citation_lists:
        flat.extend(lst or [])
    return citations_from_hits(
        # Re-key through citations_from_hits' dedup by treating each citation as
        # a hit-shaped dict (its keys are a superset of what the helper reads).
        [
            {
                "source": c.get("source"),
                "physical_page": c.get("page"),
                "chunk_id": c.get("chunk_id"),
                "section_path": c.get("section"),
            }
            for c in flat
        ]
    )
