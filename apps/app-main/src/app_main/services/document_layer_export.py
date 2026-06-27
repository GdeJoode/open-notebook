"""Shared document-graph layer for the graph-native exporters (Track U.5).

Both the NetworkX (D.3) and JSONL (D.2) exporters project the same Track-B
*entity* graph (entity nodes + ``relation`` edges). Track U materialized a
second, document-centric layer on top of that graph:

* ``source`` nodes — the documents themselves.
* ``mentions`` edges (document -> entity) — the U.2 projection of
  ``entity.source_documents``, carrying the R.2 salience×rarity ``weight``.
* ``cites`` edges (document -> document) — the U.3 citation materialization,
  carrying ``confidence`` + the matched ``reference_text`` (0 live edges on the
  current corpus until Track V feeds references, but the export handles them so
  it's ready).

This module fetches that layer ONCE, scoped to a single notebook, and returns a
plain dataclass the two exporters render into their own wire formats. Keeping it
here (not inside either service) means the notebook→source scoping, the
edge-pruning rules, and the "why" metadata live in exactly one place — the two
exporters cannot drift.

Why gated behind a flag
=======================
The layer is additive and OFF by default (``ExportFilter.include_document_layer``).
With it off, neither exporter calls this module at all, so the entity-only output
is byte-for-byte identical to the pre-U.5 export — existing entity-graph
consumers are unaffected (the U.5 backward-compatibility contract).

Scoping + pruning rules
=======================
1. **Notebook scope.** ``mentions`` / ``cites`` are notebook-agnostic global
   edge tables. We resolve the notebook's sources via the ``reference`` edge
   (the same traversal the entity export opens with) and keep only edges whose
   endpoints fall inside that source set, so the document layer describes the
   SAME corpus slice as the entity layer.
2. **No dangling endpoints.** A ``mentions`` edge is kept only if its entity
   endpoint survived the entity filter (it is already a node in the graph). This
   guarantees the document layer never references an entity node the exporter
   dropped — the export stays internally consistent regardless of the filter.
   The source endpoint always becomes a ``source`` node (added by this layer).
3. **Source nodes.** Every notebook source that anchors a kept ``mentions`` or
   ``cites`` edge becomes a ``source`` node. Sources with no surviving edge are
   NOT injected — an isolated document node carries no graph information in an
   export (unlike the U.4 viz, where the user wants to *see* it). This keeps the
   export's added node count proportional to the edges it actually carries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from loguru import logger


@dataclass
class DocumentSourceNode:
    """A ``source`` (document) node added by the document layer."""

    id: str
    title: str


@dataclass
class DocumentMentionEdge:
    """A ``mentions`` (document -> entity) edge with its R.2 weight + why."""

    source_id: str
    entity_id: str
    weight: float
    concept_name: str
    concept_type: str
    document_frequency: int


@dataclass
class DocumentCitesEdge:
    """A ``cites`` (document -> document) edge with its U.3 confidence + why."""

    from_source_id: str
    to_source_id: str
    confidence: float
    reference_text: str
    match_method: str


@dataclass
class DocumentLayer:
    """The fetched, notebook-scoped document graph layer.

    ``source_nodes`` carries every document that anchors a kept edge;
    ``mentions`` / ``cites`` are the kept, scoped, dangling-free edges. All three
    are empty when the flag is off or the corpus has no document edges (the
    0-cites no-op corpus → ``cites == []`` cleanly).
    """

    source_nodes: List[DocumentSourceNode] = field(default_factory=list)
    mentions: List[DocumentMentionEdge] = field(default_factory=list)
    cites: List[DocumentCitesEdge] = field(default_factory=list)


async def fetch_document_layer(
    notebook_id: str,
    *,
    surviving_entity_ids: set[str],
    entity_repo: Any,
    source_repo: Any,
) -> DocumentLayer:
    """Fetch + scope + prune the document layer for one notebook.

    Args:
        notebook_id: The notebook being exported.
        surviving_entity_ids: Stringified ids of the entity nodes that made it
            into the graph (post-filter). A ``mentions`` edge to an entity NOT
            in this set is dropped so the layer never dangles.
        entity_repo: Provides ``load_mentions_edges()`` (U.2 seam).
        source_repo: Provides ``load_notebook_source_ids()`` (U.5 seam),
            ``load_cites_edges()`` (U.3 seam), ``get_titles_by_ids()`` (R.2 seam).

    Returns:
        A :class:`DocumentLayer`. Empty (not raised) on any repo failure — the
        document layer is strictly additive, so a failure to fetch it must never
        break the entity-graph export.
    """
    notebook_source_ids = set(
        await source_repo.load_notebook_source_ids(notebook_id)
    )
    if not notebook_source_ids:
        # No sources in this notebook → no document layer to add.
        return DocumentLayer()

    mentions = await _fetch_mentions(
        entity_repo,
        notebook_source_ids=notebook_source_ids,
        surviving_entity_ids=surviving_entity_ids,
    )
    cites = await _fetch_cites(
        source_repo,
        notebook_source_ids=notebook_source_ids,
    )

    # A source becomes a node iff it anchors a kept edge (see module docstring).
    anchoring_source_ids: set[str] = set()
    for m in mentions:
        anchoring_source_ids.add(m.source_id)
    for c in cites:
        anchoring_source_ids.add(c.from_source_id)
        anchoring_source_ids.add(c.to_source_id)

    source_nodes = await _build_source_nodes(source_repo, anchoring_source_ids)

    logger.info(
        "document layer: notebook={nb} sources={src} mentions={men} cites={cit}",
        nb=notebook_id,
        src=len(source_nodes),
        men=len(mentions),
        cit=len(cites),
    )
    return DocumentLayer(
        source_nodes=source_nodes,
        mentions=mentions,
        cites=cites,
    )


async def _fetch_mentions(
    entity_repo: Any,
    *,
    notebook_source_ids: set[str],
    surviving_entity_ids: set[str],
) -> List[DocumentMentionEdge]:
    """Load + scope + prune ``mentions`` edges.

    Kept iff the source endpoint is in the notebook AND the entity endpoint
    survived the entity filter (so the edge never dangles).
    """
    try:
        raw = await entity_repo.load_mentions_edges()
    except Exception as e:
        logger.error(f"document layer: load_mentions_edges failed: {e}")
        return []

    kept: List[DocumentMentionEdge] = []
    for row in raw or []:
        src = _opt_str(row.get("source"))
        tgt = _opt_str(row.get("target"))
        if not src or not tgt:
            continue
        if src not in notebook_source_ids:
            continue
        if tgt not in surviving_entity_ids:
            continue
        kept.append(
            DocumentMentionEdge(
                source_id=src,
                entity_id=tgt,
                weight=float(row.get("weight") or 0.0),
                concept_name=str(row.get("concept_name") or ""),
                concept_type=str(row.get("concept_type") or ""),
                document_frequency=int(row.get("document_frequency") or 0),
            )
        )
    return kept


async def _fetch_cites(
    source_repo: Any,
    *,
    notebook_source_ids: set[str],
) -> List[DocumentCitesEdge]:
    """Load + scope ``cites`` edges (intra-notebook only).

    Both endpoints must be notebook sources — a citation to a source outside the
    exported notebook is out of scope for this notebook's graph. Empty on the
    0-cites no-op corpus.
    """
    try:
        raw = await source_repo.load_cites_edges()
    except Exception as e:
        logger.error(f"document layer: load_cites_edges failed: {e}")
        return []

    kept: List[DocumentCitesEdge] = []
    for row in raw or []:
        src = _opt_str(row.get("source"))
        tgt = _opt_str(row.get("target"))
        if not src or not tgt:
            continue
        if src not in notebook_source_ids or tgt not in notebook_source_ids:
            continue
        kept.append(
            DocumentCitesEdge(
                from_source_id=src,
                to_source_id=tgt,
                confidence=float(row.get("confidence") or 0.0),
                reference_text=str(row.get("reference_text") or ""),
                match_method=str(row.get("match_method") or ""),
            )
        )
    return kept


async def _build_source_nodes(
    source_repo: Any,
    anchoring_source_ids: set[str],
) -> List[DocumentSourceNode]:
    """Hydrate the anchoring source ids to ``source`` nodes with titles.

    Titles are best-effort (a source may have none → empty string). Sorted by id
    for deterministic export output.
    """
    if not anchoring_source_ids:
        return []
    try:
        titles: Dict[str, Optional[str]] = await source_repo.get_titles_by_ids(
            sorted(anchoring_source_ids)
        )
    except Exception as e:
        logger.error(f"document layer: get_titles_by_ids failed: {e}")
        titles = {}
    return [
        DocumentSourceNode(id=sid, title=str(titles.get(sid) or ""))
        for sid in sorted(anchoring_source_ids)
    ]


def _opt_str(value: Any) -> Optional[str]:
    """Stringify a record-id-ish value, or ``None`` if falsy."""
    if value is None:
        return None
    s = str(value)
    return s or None


__all__ = [
    "DocumentSourceNode",
    "DocumentMentionEdge",
    "DocumentCitesEdge",
    "DocumentLayer",
    "fetch_document_layer",
]
