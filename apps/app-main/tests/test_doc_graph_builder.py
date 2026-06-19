"""Tests for the document structure graph builder (Phase I.F).

These exercise the *builder logic* against chunk-derived fixtures with the
SurrealDB seam mocked (``execute_query`` / ``execute_transaction`` are patched to
capture the writes). There is no live SurrealDB in this environment, so the
DB-dependent ACs (migration apply/revert, live row counts, API latency) are
validated by inspection + the migration runner's discovery and are documented as
live-deferred in the phase self-review.

The six plan scenarios are covered:
  1. basic tree            -> TestComputeGraph.test_basic_tree*
  2. idempotency           -> TestIdempotency
  3. derived_from coverage -> TestDerivedFromCoverage
  4. empty document        -> TestEdgeCases.test_empty_document
  5. missing metadata      -> TestEdgeCases.test_missing_metadata
  6. missing docling json  -> TestEdgeCases.test_no_docling_json_falls_back_to_chunks
"""

from __future__ import annotations

from typing import Any, Dict, List

import pytest
from app_main.services.graph import doc_graph_builder as dgb
from app_main.services.graph.doc_graph_builder import (
    DocGraphBuilder,
    compute_graph,
)

# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def make_chunk(
    order: int,
    *,
    text: str = "body",
    element_type: str = "paragraph",
    section_path: List[str] | None = None,
    section_level: int = 0,
    page: int = 0,
    positions: List[List[float]] | None = None,
    chunk_id: str | None = None,
) -> Dict[str, Any]:
    """Build a chunk dict shaped like ``ChunkRepository`` / get_source_chunks."""
    return {
        "id": chunk_id if chunk_id is not None else f"chunk:c{order}",
        "order": order,
        "element_type": element_type,
        "text": text,
        "physical_page": page,
        "positions": positions if positions is not None else [],
        "metadata": {
            "section_path": section_path or [],
            "section_level": section_level,
        },
    }


def make_50_page_document() -> List[Dict[str, Any]]:
    """Synthesize a ~50-page doc with a 3-deep heading tree and 200+ chunks.

    Layout: 5 chapters, each with 2 sections, each with 2 subsections; each
    subsection holds several paragraph/table chunks spread across pages. The
    section_path on every leaf is 3 levels deep so the parent_of tree reaches
    depth 4 (root chapter -> section -> subsection -> leaf).
    """
    chunks: List[Dict[str, Any]] = []
    order = 0
    page = 0
    for ch in range(5):
        chapter = f"Chapter {ch + 1}"
        for sec in range(2):
            section = f"Section {ch + 1}.{sec + 1}"
            for sub in range(2):
                subsection = f"Subsection {ch + 1}.{sec + 1}.{sub + 1}"
                path = [chapter, section, subsection]
                # 11 leaf elements per subsection -> 5*2*2*11 = 220 leaf chunks,
                # plus 35 section nodes = 255 doc_nodes, comfortably over the
                # 200-row AC2 bar (a real 50-page doc has far more elements).
                for el in range(11):
                    etype = "table" if el == 10 else "paragraph"
                    chunks.append(
                        make_chunk(
                            order,
                            text=f"{subsection} para {el}",
                            element_type=etype,
                            section_path=path,
                            section_level=2,
                            page=page,
                            positions=[[page, 0.1, 0.9, 0.2, 0.3]],
                        )
                    )
                    order += 1
                    if order % 5 == 0:
                        page += 1
    return chunks


# ---------------------------------------------------------------------------
# DB-seam capture harness
# ---------------------------------------------------------------------------

class FakeDB:
    """Captures execute_query / execute_transaction calls for assertions.

    INSERT INTO doc_node returns synthetic ``id``s so the builder can map
    self_ref -> record id for the subsequent RELATE statements. Everything else
    returns an empty list. RELATE transactions are recorded so tests can count
    edges actually persisted.
    """

    def __init__(self) -> None:
        self.queries: List[tuple[str, Dict[str, Any]]] = []
        self.transactions: List[tuple[str, Dict[str, Any]]] = []
        self._node_counter = 0

    async def execute_query(self, query, params=None, config=None):
        self.queries.append((query, params or {}))
        if query.strip().startswith("INSERT INTO doc_node"):
            rows = []
            for node in (params or {}).get("nodes", []):
                self._node_counter += 1
                rows.append(
                    {
                        "id": f"doc_node:n{self._node_counter}",
                        "self_ref": node["self_ref"],
                    }
                )
            return rows
        return []

    async def execute_transaction(self, query, params=None, config=None):
        self.transactions.append((query, params or {}))
        return []

    def relate_count(self, table: str) -> int:
        """Total RELATE rows persisted for ``table`` across all batches."""
        total = 0
        for query, _params in self.transactions:
            total += query.count(f"->{table}->")
        return total


@pytest.fixture
def fake_db(monkeypatch: pytest.MonkeyPatch) -> FakeDB:
    db = FakeDB()
    monkeypatch.setattr(dgb, "execute_query", db.execute_query)
    monkeypatch.setattr(dgb, "execute_transaction", db.execute_transaction)
    # ensure_record_id is happy with our "table:id" synthetic ids; leave it.
    return db


# ===========================================================================
# 1. Basic tree (pure computation)
# ===========================================================================

class TestComputeGraph:

    def test_basic_tree_has_section_and_leaf_nodes(self):
        chunks = [
            make_chunk(0, section_path=["Intro"], text="hello"),
            make_chunk(1, section_path=["Intro"], text="world"),
        ]
        graph = compute_graph(chunks)

        section_nodes = [n for n in graph.nodes if n.element_type == "section_header"]
        leaf_nodes = [n for n in graph.nodes if n.element_type != "section_header"]
        assert len(section_nodes) == 1
        assert len(leaf_nodes) == 2
        # both leaves parent onto the single section
        assert all(n.parent_ref == section_nodes[0].self_ref for n in leaf_nodes)

    def test_duplicate_order_yields_distinct_leaf_refs(self):
        """Two chunks sharing an ``order`` must still get distinct leaf
        self_refs. A collision would violate the UNIQUE (source, self_ref)
        index and — via the best-effort ingestion hook — silently drop the
        whole structure graph. Leaf refs derive from the deterministic loop
        index, not the (untrusted) ``order`` field.
        """
        chunks = [
            make_chunk(0, section_path=["A"], text="first", chunk_id="chunk:a"),
            make_chunk(0, section_path=["A"], text="second", chunk_id="chunk:b"),
        ]
        graph = compute_graph(chunks)

        leaf_refs = [
            n.self_ref for n in graph.nodes if n.element_type != "section_header"
        ]
        assert len(leaf_refs) == 2
        assert len(set(leaf_refs)) == 2  # distinct — no UNIQUE-index collision
        # both chunks still get their derived_from edge
        assert {e.source_ref for e in graph.derived_from} == {
            "chunk:a",
            "chunk:b",
        }

    def test_next_node_chain_follows_reading_order(self):
        chunks = [
            make_chunk(2, section_path=["A"], text="third"),
            make_chunk(0, section_path=["A"], text="first"),
            make_chunk(1, section_path=["A"], text="second"),
        ]
        graph = compute_graph(chunks)

        # Leaves sorted by `order`: chunks/0 -> chunks/1 -> chunks/2
        chain = [(e.source_ref, e.target_ref) for e in graph.next_node]
        assert chain == [
            ("#/chunks/0", "#/chunks/1"),
            ("#/chunks/1", "#/chunks/2"),
        ]

    def test_deep_section_path_yields_depth_at_least_3(self):
        chunks = [
            make_chunk(
                0,
                section_path=["Chapter 1", "Section 1.1", "Sub 1.1.1"],
                section_level=2,
            )
        ]
        graph = compute_graph(chunks)
        # root section -> section -> subsection -> leaf = depth 4
        assert graph.max_depth() >= 4

    def test_bbox_reordered_to_x1y1x2y2(self):
        # stored position interleave is [page, x1, x2, y1, y2]
        chunks = [
            make_chunk(0, section_path=["A"], positions=[[3, 0.1, 0.7, 0.2, 0.6]])
        ]
        graph = compute_graph(chunks)
        leaf = next(n for n in graph.nodes if n.element_type != "section_header")
        assert leaf.bbox == [0.1, 0.2, 0.7, 0.6]  # x1, y1, x2, y2
        assert leaf.page == 0

    def test_50_page_document_exceeds_200_nodes_depth_3_with_chains(self):
        chunks = make_50_page_document()
        graph = compute_graph(chunks)

        assert len(chunks) >= 200
        # AC2: 200+ doc_node rows
        assert len(graph.nodes) >= 200
        # AC2: parent_of tree depth >= 3
        assert graph.max_depth() >= 3
        # AC2: next_node chains — one fewer than leaf count
        leaf_count = sum(1 for n in graph.nodes if n.element_type != "section_header")
        assert len(graph.next_node) == leaf_count - 1


# ===========================================================================
# 2. Idempotency (delete-rebuild produces identical computed graph)
# ===========================================================================

class TestIdempotency:

    def test_recompute_is_byte_identical(self):
        chunks = make_50_page_document()
        g1 = compute_graph(chunks)
        g2 = compute_graph(chunks)

        def signature(g):
            return (
                [(n.self_ref, n.element_type, n.parent_ref, n.sequence) for n in g.nodes],
                [(e.source_ref, e.target_ref) for e in g.parent_of],
                [(e.source_ref, e.target_ref) for e in g.next_node],
                [(e.source_ref, e.target_ref) for e in g.derived_from],
            )

        assert signature(g1) == signature(g2)

    @pytest.mark.asyncio
    async def test_build_deletes_before_inserting(self, fake_db: FakeDB):
        chunks = [make_chunk(0, section_path=["A"])]
        builder = DocGraphBuilder()

        await builder.build("source:s1", chunks=chunks)

        delete_idx = [
            i for i, (q, _) in enumerate(fake_db.queries)
            if q.strip().startswith("DELETE")
        ]
        insert_idx = [
            i for i, (q, _) in enumerate(fake_db.queries)
            if q.strip().startswith("INSERT INTO doc_node")
        ]
        assert delete_idx, "expected delete-existing queries before insert"
        assert insert_idx, "expected a doc_node insert"
        # every delete runs before the first node insert (delete-rebuild order)
        assert max(delete_idx) < min(insert_idx)

    @pytest.mark.asyncio
    async def test_build_twice_persists_same_edge_counts(self, fake_db: FakeDB):
        chunks = make_50_page_document()
        builder = DocGraphBuilder()

        g_first = await builder.build("source:s1", chunks=chunks)
        first_parent = fake_db.relate_count("parent_of")
        first_next = fake_db.relate_count("next_node")
        first_derived = fake_db.relate_count("derived_from")

        # reset capture and rebuild
        fake_db.transactions.clear()
        g_second = await builder.build("source:s1", chunks=chunks)

        assert len(g_first.nodes) == len(g_second.nodes)
        assert fake_db.relate_count("parent_of") == first_parent
        assert fake_db.relate_count("next_node") == first_next
        assert fake_db.relate_count("derived_from") == first_derived


# ===========================================================================
# 3. derived_from coverage (>= 90% of chunks linked)
# ===========================================================================

class TestDerivedFromCoverage:

    def test_every_chunk_with_id_gets_a_derived_from_edge(self):
        chunks = make_50_page_document()
        graph = compute_graph(chunks)

        linked_chunk_ids = {e.source_ref for e in graph.derived_from}
        all_chunk_ids = {str(c["id"]) for c in chunks}
        coverage = len(linked_chunk_ids) / len(all_chunk_ids)
        assert coverage >= 0.90
        # primary self_ref present for each link (target is a real leaf node)
        leaf_refs = {n.self_ref for n in graph.nodes}
        assert all(e.target_ref in leaf_refs for e in graph.derived_from)

    def test_chunks_without_id_are_skipped_not_errored(self):
        chunks = [
            make_chunk(0, section_path=["A"]),
            {**make_chunk(1, section_path=["A"]), "id": None},
        ]
        graph = compute_graph(chunks)
        # one linkable chunk (the id=None chunk produces a node but no edge)
        assert len(graph.derived_from) == 1
        assert len([n for n in graph.nodes if n.element_type != "section_header"]) == 2

    @pytest.mark.asyncio
    async def test_persisted_derived_from_count_matches_linkable_chunks(
        self, fake_db: FakeDB
    ):
        chunks = make_50_page_document()
        builder = DocGraphBuilder()
        await builder.build("source:s1", chunks=chunks)

        assert fake_db.relate_count("derived_from") == len(chunks)


# ===========================================================================
# 4/5/6. Edge cases
# ===========================================================================

class TestEdgeCases:

    def test_empty_document(self):
        graph = compute_graph([])
        assert graph.nodes == []
        assert graph.parent_of == []
        assert graph.next_node == []
        assert graph.derived_from == []
        assert graph.max_depth() == 0

    @pytest.mark.asyncio
    async def test_empty_document_skips_inserts(self, fake_db: FakeDB):
        builder = DocGraphBuilder()
        await builder.build("source:empty", chunks=[])

        # delete-existing still runs (idempotent clear) but no insert/relate
        assert not any(
            q.strip().startswith("INSERT INTO doc_node")
            for q, _ in fake_db.queries
        )
        assert fake_db.transactions == []

    def test_missing_metadata(self):
        # chunk dict with no 'metadata' key at all -> no section, flat leaves
        chunks = [
            {"id": "chunk:a", "order": 0, "element_type": "paragraph", "text": "x",
             "physical_page": 0, "positions": []},
            {"id": "chunk:b", "order": 1, "element_type": "paragraph", "text": "y",
             "physical_page": 0, "positions": []},
        ]
        graph = compute_graph(chunks)
        assert all(n.element_type != "section_header" for n in graph.nodes)
        assert len(graph.nodes) == 2
        assert graph.parent_of == []  # no headings -> no tree edges
        assert len(graph.next_node) == 1  # reading-order chain still built
        assert len(graph.derived_from) == 2

    def test_missing_section_path_key(self):
        chunks = [
            {"id": "chunk:a", "order": 0, "element_type": "paragraph",
             "text": "x", "physical_page": 0, "positions": [], "metadata": {}},
        ]
        graph = compute_graph(chunks)
        assert len(graph.nodes) == 1
        assert graph.nodes[0].parent_ref is None

    @pytest.mark.asyncio
    async def test_no_docling_json_falls_back_to_chunks(self, fake_db: FakeDB):
        """No docling_document_json anywhere — builder uses persisted chunks.

        This is the production path (the serialized DoclingDocument is never
        persisted). Passing chunks directly is exactly how the orchestrator
        invokes the builder; this asserts the chunk-derived graph is built and
        persisted without any docling JSON involvement.
        """
        chunks = [
            make_chunk(0, section_path=["Intro"], text="hello"),
            make_chunk(1, section_path=["Intro"], text="world"),
        ]
        builder = DocGraphBuilder()
        graph = await builder.build("source:s1", chunks=chunks)

        assert len(graph.nodes) == 3  # 1 section + 2 leaves
        assert fake_db.relate_count("derived_from") == 2

    @pytest.mark.asyncio
    async def test_build_loads_chunks_when_not_supplied(self, fake_db: FakeDB):
        """build(source_id) with chunks=None loads from the DB seam."""
        builder = DocGraphBuilder()
        # Patch the chunk-loading SELECT to return one chunk.
        original = fake_db.execute_query

        async def patched(query, params=None, config=None):
            if query.strip().startswith("SELECT") and "FROM chunk" in query:
                return [make_chunk(0, section_path=["A"], chunk_id="chunk:x")]
            return await original(query, params, config)

        fake_db.execute_query = patched  # type: ignore[assignment]
        dgb.execute_query = patched  # type: ignore[assignment]

        graph = await builder.build("source:s1")
        assert len(graph.nodes) == 2  # 1 section + 1 leaf


# ===========================================================================
# Batching (Risk 3 mitigation)
# ===========================================================================

class TestBatching:

    @pytest.mark.asyncio
    async def test_relate_inserts_are_batched(self, fake_db: FakeDB, monkeypatch):
        # Shrink the batch size so a modest fixture exercises multiple batches.
        monkeypatch.setattr(dgb, "RELATE_BATCH_SIZE", 50)
        chunks = make_50_page_document()
        builder = DocGraphBuilder()
        await builder.build("source:s1", chunks=chunks)

        # next_node has (leaves-1) edges; with batch 50 that's multiple txns.
        next_txns = [
            q for q, _ in fake_db.transactions if "->next_node->" in q
        ]
        assert len(next_txns) >= 2
        # no single transaction exceeds the batch cap
        assert all(q.count("->next_node->") <= 50 for q in next_txns)
