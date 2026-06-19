"""Builds the document structure graph from a source's chunks (Phase I.F).

Structure source — decision
---------------------------
The plan's first-choice source is the serialized ``DoclingDocument`` JSON
(``metadata["docling_document_json"]``). In this codebase that JSON is a
*transient* LangGraph-state artifact: it is produced inside the ingestion
workflow, consumed for chunk extraction, and discarded. It is NOT persisted to
``source.metadata`` and is NOT carried on ``ExtractionResult``, so it is not
available at the orchestration boundary where the structure graph is built.

The reliable, persisted structure source is therefore the ``chunk`` rows. Each
chunk carries the structure Docling produced:

  * ``order``                       reading order (0-indexed, monotonic)
  * ``element_type``                paragraph | table | picture | title | ...
  * ``metadata.section_path``       heading breadcrumb, e.g. ["Ch 1", "1.2"]
  * ``metadata.section_level``      nesting depth
  * ``positions``                   0-1 normalized bboxes (Phase I.C output)

So the builder derives the node tree from chunks:

  * one **section** doc_node per distinct ``section_path`` prefix (the heading
    hierarchy), and
  * one **leaf** doc_node per chunk (the element itself),

then wires ``parent_of`` (section tree + section→leaf), ``next_node`` (reading
order over leaves) and ``derived_from`` (chunk → its leaf doc_node).

When a real ``docling_document_json`` is available (passed explicitly by a
caller that still holds the live doc), :func:`compute_graph` accepts a
pre-extracted element list; otherwise it derives from chunks. Today every
production caller takes the chunk path.

Idempotency
-----------
``self_ref`` is deterministic for a given chunk set (section index by first
appearance, leaf index by chunk ``order``), and ``build`` deletes the source's
existing graph before re-inserting. Re-ingesting the same document therefore
produces an identical graph (AC6).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger
from surrealdb_service.connection import (
    ensure_record_id,
    execute_query,
    execute_transaction,
)

# RELATE inserts are batched to bound per-statement size on large documents
# (Risk 3: a 10K-page doc can produce 100K+ edges). 500 edges/batch matches the
# plan's data-volume mitigation.
RELATE_BATCH_SIZE = 500


@dataclass
class GraphNode:
    """A computed structure node, prior to persistence.

    ``self_ref`` is the stable per-source identifier ("#/sections/3",
    "#/chunks/12"). ``chunk_id`` is set only on leaf nodes and is the record id
    of the chunk this node was derived from (drives the ``derived_from`` edge
    and the UI bbox-highlight wiring).
    """

    self_ref: str
    element_type: str
    sequence: int
    text: Optional[str] = None
    page: Optional[int] = None
    level: Optional[int] = None
    bbox: Optional[List[float]] = None
    chunk_id: Optional[str] = None
    parent_ref: Optional[str] = None


@dataclass
class GraphEdge:
    """A computed edge between two ``self_ref`` endpoints (or chunk → node)."""

    source_ref: str
    target_ref: str


@dataclass
class ComputedGraph:
    """The full computed graph for one source, before persistence.

    Separated from persistence so the builder logic is unit-testable without a
    live SurrealDB: tests assert on node count / tree depth / chains / coverage
    against this structure directly.
    """

    nodes: List[GraphNode] = field(default_factory=list)
    parent_of: List[GraphEdge] = field(default_factory=list)
    next_node: List[GraphEdge] = field(default_factory=list)
    # derived_from edges carry a chunk record id as ``source_ref`` (not a
    # self_ref) and a leaf node self_ref as ``target_ref``.
    derived_from: List[GraphEdge] = field(default_factory=list)

    def node_by_ref(self, ref: str) -> Optional[GraphNode]:
        for node in self.nodes:
            if node.self_ref == ref:
                return node
        return None

    def max_depth(self) -> int:
        """Longest root-to-leaf path length in the ``parent_of`` tree.

        Returns the number of nodes on the deepest path (a single isolated node
        has depth 1). Used by tests to assert tree depth >= 3 (AC2).
        """
        if not self.nodes:
            return 0
        parent: Dict[str, Optional[str]] = {
            n.self_ref: n.parent_ref for n in self.nodes
        }

        def depth(ref: str, seen: frozenset[str]) -> int:
            if ref in seen:
                # Defensive: a cycle should never form, but never loop forever.
                return 0
            p = parent.get(ref)
            if p is None or p not in parent:
                return 1
            return 1 + depth(p, seen | {ref})

        return max(depth(n.self_ref, frozenset()) for n in self.nodes)


def _first_bbox(positions: Any) -> Optional[List[float]]:
    """Extract a [x1, y1, x2, y2] bbox from a chunk ``positions`` entry.

    Chunk positions are stored as ``[[page, x1, x2, y1, y2], ...]`` (note the
    x1,x2,y1,y2 interleave from ``chunk_builder``). The doc_node ``bbox`` is the
    plan's ``[x1, y1, x2, y2]`` order, so we reorder here. Returns ``None`` when
    no usable position exists.
    """
    if not positions or not isinstance(positions, (list, tuple)):
        return None
    first = positions[0]
    if not isinstance(first, (list, tuple)) or len(first) < 5:
        return None
    # stored: [page, x1, x2, y1, y2]  ->  bbox: [x1, y1, x2, y2]
    _page, x1, x2, y1, y2 = first[:5]
    try:
        return [float(x1), float(y1), float(x2), float(y2)]
    except (TypeError, ValueError):
        return None


def _page_of(chunk: Dict[str, Any]) -> Optional[int]:
    page = chunk.get("physical_page")
    if page is None:
        return None
    try:
        return int(page)
    except (TypeError, ValueError):
        return None


def _section_path(chunk: Dict[str, Any]) -> List[str]:
    """Read the heading breadcrumb off a chunk (lives under ``metadata``)."""
    meta = chunk.get("metadata") or {}
    path = meta.get("section_path")
    if not isinstance(path, (list, tuple)):
        return []
    return [str(p) for p in path if p not in (None, "")]


def compute_graph(chunks: Sequence[Dict[str, Any]]) -> ComputedGraph:
    """Compute the structure graph for a source from its persisted chunks.

    Pure function — no I/O. Builds:

      * a section node per distinct ``section_path`` prefix (heading tree), and
      * a leaf node per chunk,

    then ``parent_of`` (section tree + section→leaf), ``next_node`` (reading
    order over leaves) and ``derived_from`` (chunk → leaf).

    Chunks are expected to be dicts as returned by ``ChunkRepository`` /
    ``get_source_chunks`` (``id``, ``order``, ``element_type``, ``text``,
    ``physical_page``, ``positions``, ``metadata.section_path``). An empty input
    yields an empty graph.
    """
    graph = ComputedGraph()
    if not chunks:
        return graph

    # Stable reading order: sort by ``order`` (fall back to input order). The
    # leaf ``sequence`` and the next_node chain both follow this.
    def _order_key(chunk: Dict[str, Any]) -> int:
        value = chunk.get("order")
        try:
            return int(value) if value is not None else 0
        except (TypeError, ValueError):
            return 0

    ordered = sorted(chunks, key=_order_key)

    # ---- Section nodes (heading tree) ------------------------------------
    # Map a section-path prefix tuple -> its node self_ref. Sections are
    # numbered by first appearance so the refs are deterministic across
    # re-ingest of the same chunk set.
    section_ref: Dict[tuple[str, ...], str] = {}
    section_seq = 0

    def ensure_section(prefix: tuple[str, ...]) -> Optional[str]:
        """Create section nodes for ``prefix`` and all ancestors; return ref."""
        nonlocal section_seq
        if not prefix:
            return None
        if prefix in section_ref:
            return section_ref[prefix]
        parent_ref = ensure_section(prefix[:-1])
        ref = f"#/sections/{section_seq}"
        section_ref[prefix] = ref
        graph.nodes.append(
            GraphNode(
                self_ref=ref,
                element_type="section_header",
                sequence=section_seq,
                text=prefix[-1],
                level=len(prefix) - 1,
                parent_ref=parent_ref,
            )
        )
        if parent_ref is not None:
            graph.parent_of.append(GraphEdge(parent_ref, ref))
        section_seq += 1
        return ref

    # ---- Leaf nodes (one per chunk) -------------------------------------
    prev_leaf_ref: Optional[str] = None
    for seq, chunk in enumerate(ordered):
        chunk_id = chunk.get("id")
        path = _section_path(chunk)
        parent_section_ref = ensure_section(tuple(path)) if path else None

        leaf_ref = f"#/chunks/{chunk.get('order', seq)}"
        bbox = _first_bbox(chunk.get("positions"))
        node = GraphNode(
            self_ref=leaf_ref,
            element_type=str(chunk.get("element_type") or "paragraph"),
            sequence=seq,
            text=chunk.get("text"),
            page=_page_of(chunk),
            level=(len(path) if path else 0),
            bbox=bbox,
            chunk_id=str(chunk_id) if chunk_id is not None else None,
            parent_ref=parent_section_ref,
        )
        graph.nodes.append(node)

        if parent_section_ref is not None:
            graph.parent_of.append(GraphEdge(parent_section_ref, leaf_ref))

        if prev_leaf_ref is not None:
            graph.next_node.append(GraphEdge(prev_leaf_ref, leaf_ref))
        prev_leaf_ref = leaf_ref

        if chunk_id is not None:
            graph.derived_from.append(GraphEdge(str(chunk_id), leaf_ref))

    return graph


class DocGraphBuilder:
    """Persists the computed structure graph into SurrealDB (Phase I.F).

    Owns the DB seam. The pure graph computation lives in :func:`compute_graph`
    so it can be tested without a server; this class only handles the
    delete-rebuild persistence (idempotent per source) and batched RELATE
    inserts.
    """

    def __init__(self, config: Any = None) -> None:
        self.config = config

    async def build(
        self,
        source_id: str,
        chunks: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> ComputedGraph:
        """Delete and rebuild the structure graph for ``source_id``.

        Args:
            source_id: Source record id (e.g. ``"source:abc"``).
            chunks: The source's persisted chunk dicts. When ``None`` the
                builder loads them from the DB itself.

        Returns:
            The :class:`ComputedGraph` that was persisted (empty when the
            source has no chunks).
        """
        if chunks is None:
            chunks = await self._load_chunks(source_id)

        graph = compute_graph(chunks)

        await self._delete_existing(source_id)
        if not graph.nodes:
            logger.info(
                "doc_graph_builder: source {sid} produced no structure nodes",
                sid=source_id,
            )
            return graph

        ref_to_id = await self._insert_nodes(source_id, graph.nodes)
        await self._insert_parent_of(graph.parent_of, ref_to_id)
        await self._insert_next_node(graph.next_node, ref_to_id)
        await self._insert_derived_from(graph.derived_from, ref_to_id)

        logger.info(
            "doc_graph_builder: source {sid} -> {n} nodes, {p} parent_of, "
            "{x} next_node, {d} derived_from",
            sid=source_id,
            n=len(graph.nodes),
            p=len(graph.parent_of),
            x=len(graph.next_node),
            d=len(graph.derived_from),
        )
        return graph

    # ------------------------------------------------------------------
    # DB seam
    # ------------------------------------------------------------------

    async def _load_chunks(self, source_id: str) -> List[Dict[str, Any]]:
        rows = await execute_query(
            "SELECT id, order, element_type, text, physical_page, positions, "
            "metadata FROM chunk WHERE source = $source_id ORDER BY order ASC",
            {"source_id": ensure_record_id(source_id)},
            self.config,
        )
        return rows or []

    async def _delete_existing(self, source_id: str) -> None:
        """Delete the source's doc_nodes and all their incident edges.

        Deleting a doc_node does not automatically delete its RELATE rows, so we
        clear the edge tables first (matched via the node set), then the nodes.
        """
        sid = ensure_record_id(source_id)
        await execute_query(
            "DELETE parent_of WHERE in.source = $source_id "
            "OR out.source = $source_id;",
            {"source_id": sid},
            self.config,
        )
        await execute_query(
            "DELETE next_node WHERE in.source = $source_id "
            "OR out.source = $source_id;",
            {"source_id": sid},
            self.config,
        )
        await execute_query(
            "DELETE derived_from WHERE out.source = $source_id;",
            {"source_id": sid},
            self.config,
        )
        await execute_query(
            "DELETE doc_node WHERE source = $source_id;",
            {"source_id": sid},
            self.config,
        )

    async def _insert_nodes(
        self, source_id: str, nodes: Sequence[GraphNode]
    ) -> Dict[str, str]:
        """Bulk-insert doc_nodes; return a ``self_ref -> record id`` map.

        Inserted in batches so a very large document does not produce a single
        oversized INSERT statement.
        """
        sid = ensure_record_id(source_id)
        ref_to_id: Dict[str, str] = {}
        for start in range(0, len(nodes), RELATE_BATCH_SIZE):
            batch = nodes[start : start + RELATE_BATCH_SIZE]
            payload = [
                {
                    "source": sid,
                    "self_ref": n.self_ref,
                    "element_type": n.element_type,
                    "text": n.text,
                    "page": n.page,
                    "level": n.level,
                    "sequence": n.sequence,
                    "bbox": n.bbox,
                }
                for n in batch
            ]
            rows = await execute_query(
                "INSERT INTO doc_node $nodes RETURN id, self_ref",
                {"nodes": payload},
                self.config,
            )
            for row in rows or []:
                ref_to_id[str(row["self_ref"])] = str(row["id"])
        return ref_to_id

    async def _relate_batched(
        self,
        table: str,
        pairs: Sequence[tuple[str, str]],
    ) -> None:
        """RELATE ``in -> out`` rows for ``table`` in batches.

        ``pairs`` is a list of (in_record_id, out_record_id) string tuples. Each
        batch is one ``BEGIN ... COMMIT`` transaction so a partial failure rolls
        back rather than leaving a half-written edge set.
        """
        for start in range(0, len(pairs), RELATE_BATCH_SIZE):
            batch = pairs[start : start + RELATE_BATCH_SIZE]
            stmts = ["BEGIN TRANSACTION;"]
            params: Dict[str, Any] = {}
            for i, (in_id, out_id) in enumerate(batch):
                params[f"in_{i}"] = ensure_record_id(in_id)
                params[f"out_{i}"] = ensure_record_id(out_id)
                stmts.append(
                    f"RELATE $in_{i}->{table}->$out_{i};"
                )
            stmts.append("COMMIT TRANSACTION;")
            await execute_transaction("\n".join(stmts), params, self.config)

    async def _insert_parent_of(
        self, edges: Sequence[GraphEdge], ref_to_id: Dict[str, str]
    ) -> None:
        pairs = [
            (ref_to_id[e.source_ref], ref_to_id[e.target_ref])
            for e in edges
            if e.source_ref in ref_to_id and e.target_ref in ref_to_id
        ]
        await self._relate_batched("parent_of", pairs)

    async def _insert_next_node(
        self, edges: Sequence[GraphEdge], ref_to_id: Dict[str, str]
    ) -> None:
        pairs = [
            (ref_to_id[e.source_ref], ref_to_id[e.target_ref])
            for e in edges
            if e.source_ref in ref_to_id and e.target_ref in ref_to_id
        ]
        await self._relate_batched("next_node", pairs)

    async def _insert_derived_from(
        self, edges: Sequence[GraphEdge], ref_to_id: Dict[str, str]
    ) -> None:
        # ``source_ref`` here is a chunk record id, not a doc_node self_ref.
        pairs = [
            (e.source_ref, ref_to_id[e.target_ref])
            for e in edges
            if e.target_ref in ref_to_id
        ]
        await self._relate_batched("derived_from", pairs)
