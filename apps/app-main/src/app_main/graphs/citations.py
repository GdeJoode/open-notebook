"""Citation helpers for the answer graphs (Track X.2 / X.3).

Derives a structured ``citations`` array from the provenance that Track X.1
attaches to retrieval hits (``source``/``chunk_id``/``physical_page``/
``section_path``). The citation set is the provenance of the context hits that
were actually fed to the LLM — a deterministic, defensible source set.

A citation is shaped::

    {"source": "source:...", "page": <int|None>, "chunk_id": "chunk:...|None",
     "section": "Chapter > Section" | None}

``page`` and ``chunk_id`` are ``None`` for page-less hits (insights, notes,
plain text, audio) — those still cite the ``source`` so the answer never lacks
attribution and never 500s on a missing page.

Track X.3 — faithfulness guard
==============================
The genuine hallucination risk is NOT the ``citations`` array (which is built
deterministically from the context hits, so it is ``⊆`` the retrieval set by
construction). The risk is the **inline ``[document_id, p.<page>]`` markers the
LLM writes into the answer prose** — the model can attribute a claim to a
source/page that was never in its context. ``guard_answer_citations`` parses and
validates those markers against the retrieval set; ``guard_citation_array`` is a
defensive membership safety-net for the array (a no-op on current output, but it
catches a future regression where citations stop being context-derived).

Both checks are **membership** checks (the cited source/page/chunk *was
retrieved*), NOT semantic-support checks (that the cited passage actually
supports the claim) — a semantic check would need a second LLM pass and is out
of scope. The guard only removes/flags unverifiable citations; it never
fabricates one.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple


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


# ---------------------------------------------------------------------------
# Track X.3 — faithfulness guard
# ---------------------------------------------------------------------------
#
# Inline-marker grammar (as instructed by the prompt templates):
#   [<document_id>]                 e.g. [source:abc] / [note:n1] / [insight:i2]
#   [<document_id>, p.<page>]       e.g. [source:abc, p.7]   (page is an int)
#
# A ``document_id`` is ``<type>:<id>`` where ``type`` is a word
# (``source``/``note``/``insight``/``source_insight``/...) and ``id`` is a token
# of word chars (SurrealDB ids are alphanumeric). We deliberately match only the
# strict bracketed forms so we never touch other bracketed prose the model may
# emit (e.g. ``[1]``, ``[see below]``) — least-destructive by construction.

_RECORD_ID = r"[A-Za-z_][A-Za-z0-9_]*:[A-Za-z0-9_]+"
# Group 1: the record id. Group 2 (optional): the page integer.
_MARKER_RE = re.compile(
    r"\["
    r"\s*(" + _RECORD_ID + r")\s*"
    r"(?:,\s*p\.\s*(\d+)\s*)?"
    r"\]"
)


def _retrieval_index(
    hits: List[Dict[str, Any]],
) -> Tuple[Set[str], Set[Tuple[str, int]]]:
    """Build the membership index for a retrieval set.

    Returns ``(ids, source_pages)`` where:

    - ``ids`` is every citable record id present in the context — each hit's own
      ``id``/``parent_id`` plus its ``source`` anchor. A page-less inline marker
      ``[id]`` is faithful iff ``id`` is in this set.
    - ``source_pages`` is the set of ``(source_id, physical_page)`` pairs that
      were actually retrieved. A paged inline marker ``[source, p.N]`` is
      faithful iff ``(source, N)`` is in this set — i.e. the model cited a page
      that genuinely appeared in its context for that source.
    """
    ids: Set[str] = set()
    source_pages: Set[Tuple[str, int]] = set()
    for hit in hits:
        for key in ("id", "parent_id", "source"):
            val = hit.get(key)
            if isinstance(val, str) and ":" in val:
                ids.add(val)
        source = hit.get("source")
        if not source:
            for key in ("id", "parent_id"):
                cand = hit.get(key)
                if isinstance(cand, str) and cand.startswith("source:"):
                    source = cand
                    break
        page = hit.get("physical_page")
        if isinstance(source, str) and isinstance(page, int):
            source_pages.add((source, page))
    return ids, source_pages


def parse_inline_markers(answer: str) -> List[Tuple[str, Optional[int], str]]:
    """Extract ``(record_id, page|None, raw_marker)`` for each inline citation.

    ``raw_marker`` is the exact matched substring (e.g. ``[source:abc, p.7]``) so
    a caller can strip *only* that marker without disturbing the surrounding
    answer text.
    """
    out: List[Tuple[str, Optional[int], str]] = []
    if not answer:
        return out
    for m in _MARKER_RE.finditer(answer):
        record_id = m.group(1)
        page = int(m.group(2)) if m.group(2) is not None else None
        out.append((record_id, page, m.group(0)))
    return out


def guard_answer_citations(
    answer: str,
    hits: List[Dict[str, Any]],
    *,
    strip: bool = False,
) -> Dict[str, Any]:
    """Validate the LLM's inline ``[id, p.X]`` markers against the retrieval set.

    This is the **primary** Track X.3 guard: it catches the real hallucination
    risk — a marker in the answer prose that attributes a claim to a source/page
    the model never actually saw.

    A marker is *faithful* (kept) when:

    - a paged marker ``[source, p.N]`` and ``(source, N)`` was retrieved, OR
    - a page-less marker ``[id]`` and ``id`` is a record present in the context.

    A paged marker whose ``(source, page)`` pair was NOT retrieved is
    **hallucinated** — even when the *source* was in context, the model invented
    a page for it. Likewise an ``[id]`` for a record absent from the context.

    By default the guard is **non-destructive**: it records every marker as kept
    or dropped and leaves ``answer`` untouched (``answer == text``). Pass
    ``strip=True`` to additionally remove *only* the hallucinated marker tokens
    from the prose (the surrounding sentence is never rewritten). We never
    fabricate or rewrite a citation.

    Returns::

        {
            "text": <answer, with hallucinated markers stripped iff strip=True>,
            "kept": [{"id", "page", "marker"}, ...],
            "dropped": [{"id", "page", "marker"}, ...],
            "kept_count": int,
            "dropped_count": int,
        }
    """
    ids, source_pages = _retrieval_index(hits)
    markers = parse_inline_markers(answer)

    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for record_id, page, raw in markers:
        if page is not None:
            faithful = (record_id, page) in source_pages
        else:
            faithful = record_id in ids
        entry = {"id": record_id, "page": page, "marker": raw}
        (kept if faithful else dropped).append(entry)

    text = answer
    if strip and dropped:
        # Remove only the exact hallucinated marker substrings, leaving the
        # surrounding prose intact. Collapse any doubled space the removal
        # leaves behind, but never touch sentence content.
        for entry in dropped:
            text = text.replace(entry["marker"], "")
        text = re.sub(r"[ \t]{2,}", " ", text)
        text = re.sub(r"\s+([.,;:])", r"\1", text)

    return {
        "text": text,
        "kept": kept,
        "dropped": dropped,
        "kept_count": len(kept),
        "dropped_count": len(dropped),
    }


def guard_citation_array(
    citations: List[Dict[str, Any]],
    hits: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Defensive membership safety-net for the structured ``citations`` array.

    Because X.2 builds the array from the context hits, every entry is ``⊆`` the
    retrieval set by construction, so this filter is a **no-op on current
    output** — we prove that in the tests. Its value is regression insurance: if
    a future change makes citations come from anywhere other than the context
    (e.g. parsed from the model's prose), an out-of-set entry is caught here.

    Only entries that *carry a ``chunk_id``* are membership-checked against the
    retrieved chunk ids (per the X.1 follow-up: a chunk-bearing citation makes a
    precise, checkable claim; a page-less source-only citation does not and is
    always kept). The check never fabricates — it only filters.

    Precision-first guard: when there is **no chunk-bearing retrieval set** to
    check against (e.g. the hits were not threaded into the final node, or the
    answer drew only on page-less context), the citations are already the trusted
    context-derived set and are returned **unchanged** — we never drop a valid
    citation merely because the membership index is empty.

    Returns ``{"citations", "dropped", "dropped_count"}``.
    """
    retrieved_chunks: Set[str] = {
        hit["chunk_id"]
        for hit in hits
        if isinstance(hit.get("chunk_id"), str)
    }
    if not retrieved_chunks:
        return {"citations": list(citations), "dropped": [], "dropped_count": 0}
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for citation in citations:
        chunk_id = citation.get("chunk_id")
        if isinstance(chunk_id, str) and chunk_id not in retrieved_chunks:
            dropped.append(citation)
            continue
        kept.append(citation)
    return {
        "citations": kept,
        "dropped": dropped,
        "dropped_count": len(dropped),
    }
