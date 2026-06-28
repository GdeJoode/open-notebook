"""MCP server for SurrealDB access.

Exposes the shared SurrealDB knowledge graph as MCP tools so every Claude Code
session that attaches this stdio server reads and writes the SAME memory
substrate (one database, not per-session state). The Track W.3 graph tools
(``search`` / ``get_node`` / ``related`` / ``cite`` / ``add_note``) sit
alongside the original low-level tools (``query_database`` / ``get_record`` /
``list_sources`` / ``search_similar`` / ``get_entity_graph``).

Auth: none — this is fine for the stdio-local transport (the only caller is the
local Claude Code process). If this is ever exposed over HTTP for remote
sessions, add authentication first.
"""

import argparse
import json
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories.graph import GraphRepository
from surrealdb_service.repositories.notebook import NoteRepository
from surrealdb_service.repositories.search import SearchRepository
from surrealdb_service.repositories.source import SourceRepository


def create_server() -> FastMCP:
    """Create and configure the SurrealDB MCP server."""
    mcp = FastMCP(
        name="surrealdb-service",
        instructions="SurrealDB knowledge-graph access for open-notebook.",
    )

    @mcp.tool()
    async def query_database(
        query: str, params: Optional[Dict[str, Any]] = None
    ) -> str:
        """Execute a SurrealQL query. Returns JSON results."""
        results = await execute_query(query, params)
        return json.dumps(results, default=str, indent=2)

    @mcp.tool()
    async def get_record(record_id: str) -> str:
        """Retrieve a record by full ID (e.g. 'source:abc123')."""
        rid = ensure_record_id(record_id)
        results = await execute_query("SELECT * FROM ONLY $id", {"id": rid})
        return json.dumps(results, default=str, indent=2)

    @mcp.tool()
    async def list_sources(
        limit: int = 20, offset: int = 0, order_by: str = "created_at DESC"
    ) -> str:
        """List sources with pagination."""
        repo = SourceRepository()
        sources = await repo.get_all(order_by=order_by, limit=limit, offset=offset)
        return json.dumps(
            [s.model_dump() for s in sources], default=str, indent=2
        )

    @mcp.tool()
    async def search_similar(
        embedding: List[float],
        results: int = 10,
        include_sources: bool = True,
        include_notes: bool = True,
        minimum_score: float = 0.2,
    ) -> str:
        """Vector similarity search across sources and notes.

        Takes a PRECOMPUTED query embedding (this server has no embedding model);
        see the ``search`` tool for a query-string entrypoint.
        """
        repo = SearchRepository()
        hits = await repo.vector_search(
            embedding, results, include_sources, include_notes, minimum_score
        )
        return json.dumps(hits, default=str, indent=2)

    # ------------------------------------------------------------------
    # Track W.3 — graph tools (shared memory substrate)
    # ------------------------------------------------------------------

    @mcp.tool()
    async def search(
        query: str,
        limit: int = 10,
        include_sources: bool = True,
        include_notes: bool = True,
        embedding: Optional[List[float]] = None,
        text_weight: float = 0.5,
        minimum_score: float = 0.2,
    ) -> str:
        """Search the corpus for chunks/notes matching ``query``, ranked.

        Embedding source (W.3 design decision): this surrealdb-service MCP layer
        has NO embedding model — ``search_similar`` already establishes the
        contract that the *caller* supplies any query vector (the embed step
        lives in app-main's local-Ollama pipeline, deliberately kept out of this
        package). So:

        - When ``embedding`` is supplied, this runs the full W.1 HYBRID search
          (BM25 ⊕ vector via Reciprocal Rank Fusion) — corroborated hits rank
          highest. This is the intended path for an embedding-capable caller.
        - When ``embedding`` is omitted, it falls back to LEXICAL-ONLY BM25
          (``fn::text_search``). This is still useful for keyword queries but is
          NOT semantic; pass an ``embedding`` for full hybrid ranking.

        Args:
            query: The search text (BM25 keyword side; also the text the caller
                would have embedded for the vector side).
            limit: Max results.
            include_sources: Search source chunks.
            include_notes: Search notes.
            embedding: Optional precomputed query vector (1024-dim local model).
                Present → hybrid; absent → lexical BM25 only.
            text_weight: RRF weight for the BM25 signal (0-1); vector gets the
                remainder. Only used on the hybrid path.
            minimum_score: Minimum cosine score for the vector side (hybrid path).

        Returns:
            JSON list of ranked hits. Hybrid hits carry ``_search_type`` /
            ``_combined_score`` / ``_text_rank`` / ``_vector_rank`` provenance.
        """
        repo = SearchRepository()
        if embedding:
            hits = await repo.hybrid_search(
                query,
                embedding,
                results=limit,
                include_sources=include_sources,
                include_notes=include_notes,
                minimum_score=minimum_score,
                text_weight=text_weight,
            )
        else:
            hits = await repo.text_search(
                query,
                results=limit,
                include_sources=include_sources,
                include_notes=include_notes,
            )
        return json.dumps(hits, default=str, indent=2)

    @mcp.tool()
    async def get_node(node_id: str) -> str:
        """Fetch any graph node by full record id (polymorphic).

        Works for any table — ``entity:...`` / ``source:...`` / ``note:...`` /
        ``notebook:...`` — via ``SELECT * FROM type::thing($id)``.

        Returns the record as JSON, or ``null`` when no row exists.
        """
        rid = ensure_record_id(node_id)
        results = await execute_query(
            "SELECT * FROM type::thing($id)", {"id": str(rid)}
        )
        record = results[0] if results else None
        return json.dumps(record, default=str, indent=2)

    @mcp.tool()
    async def related(
        node_id: str,
        edge_types: Optional[List[str]] = None,
        limit: int = 20,
    ) -> str:
        """List graph edges incident on a node (relation + mentions + cites).

        Returns every edge touching ``node_id`` as either endpoint, in a uniform
        ``{id, source, target, edge_type, metadata}`` shape. ``edge_type`` is one
        of ``relation`` (entity↔entity), ``mentions`` (source→entity), or
        ``cites`` (source→source).

        Args:
            node_id: Full record id (``entity:...`` / ``source:...``).
            edge_types: Optional subset of ``["relation", "mentions", "cites"]``
                to restrict to. ``None`` returns all.
            limit: Max edges returned (after loading; 0 or negative → all).

        Returns:
            JSON list of uniform edge dicts.
        """
        repo = GraphRepository()
        edges = await repo.load_all_edges(node_id, edge_types=edge_types)
        if limit and limit > 0:
            edges = edges[:limit]
        return json.dumps(edges, default=str, indent=2)

    @mcp.tool()
    async def cite(
        citing_source_id: str,
        cited_source_id: str,
        confidence: float = 1.0,
        reference_text: str = "",
    ) -> str:
        """Record a citation edge ``citing_source -> cites -> cited_source``.

        Idempotent: if an identical ``(citing, cited)`` ``cites`` edge already
        exists this is a no-op and reports ``"created": false`` — re-citing the
        same pair never duplicates the edge. Self-citations are refused.

        Args:
            citing_source_id: The citing ``source:...`` (edge ``in``).
            cited_source_id: The cited ``source:...`` (edge ``out``).
            confidence: Match confidence (1.0 for an exact/manual citation).
            reference_text: The verbatim cited reference (the "why").

        Returns:
            JSON ``{"created": bool, "citing": str, "cited": str, "edge": id?}``.
            ``created`` is False when the edge already existed or the pair was
            invalid (self-citation / bad id).
        """
        try:
            src = str(ensure_record_id(citing_source_id))
            tgt = str(ensure_record_id(cited_source_id))
        except Exception:
            return json.dumps(
                {
                    "created": False,
                    "citing": citing_source_id,
                    "cited": cited_source_id,
                    "reason": "invalid_id",
                },
                indent=2,
            )
        if src == tgt:
            return json.dumps(
                {
                    "created": False,
                    "citing": src,
                    "cited": tgt,
                    "reason": "self_citation",
                },
                indent=2,
            )

        # Idempotency: probe for an existing edge before writing. ``relate_cites``
        # itself is not dedup-aware (the U.3 service achieves idempotency by a
        # full clear+rebuild); a tool that writes one edge at a time must guard
        # the duplicate here so re-calling with the same pair is a clean no-op.
        existing = await execute_query(
            "SELECT id FROM cites "
            "WHERE in = type::thing($src) AND out = type::thing($tgt) LIMIT 1",
            {"src": src, "tgt": tgt},
        )
        if existing:
            return json.dumps(
                {
                    "created": False,
                    "citing": src,
                    "cited": tgt,
                    "edge": str(existing[0].get("id")),
                    "reason": "already_exists",
                },
                default=str,
                indent=2,
            )

        repo = SourceRepository()
        ok = await repo.relate_cites(
            src,
            tgt,
            confidence=float(confidence),
            reference_text=reference_text or "",
            match_method="mcp",
        )
        edge_id = None
        if ok:
            rows = await execute_query(
                "SELECT id FROM cites "
                "WHERE in = type::thing($src) AND out = type::thing($tgt) LIMIT 1",
                {"src": src, "tgt": tgt},
            )
            edge_id = str(rows[0].get("id")) if rows else None
        return json.dumps(
            {
                "created": bool(ok),
                "citing": src,
                "cited": tgt,
                "edge": edge_id,
            },
            default=str,
            indent=2,
        )

    @mcp.tool()
    async def add_note(
        title: str,
        content: str,
        notebook_id: Optional[str] = None,
        embedding: Optional[List[float]] = None,
    ) -> str:
        """Create a note in the shared graph, optionally linked to a notebook.

        The note is written to the same SurrealDB every session shares, so a note
        added here is immediately visible to any other connection. When
        ``notebook_id`` is given the note is linked to that notebook via the
        ``artifact`` edge.

        Args:
            title: Note title.
            content: Note body.
            notebook_id: Optional ``notebook:...`` to link the note into.
            embedding: Optional precomputed note embedding (this server does not
                embed; pass the vector if you have one).

        Returns:
            JSON ``{"id": note_id, "linked_notebook": notebook_id|null}``.
        """
        repo = NoteRepository()
        # ``note.embedding`` is a non-optional ``array<float>`` (migration 1)
        # with no DB default, so a note created without a vector must still
        # write ``[]`` — passing None would fail with "Found NONE for field
        # embedding". ``create_with_embedding`` only sets the field when the
        # arg is truthy, so default the empty vector here.
        note = await repo.create_with_embedding(
            {"title": title, "content": content, "embedding": embedding or []},
            embedding=embedding,
        )
        note_id = str(note.id)
        linked: Optional[str] = None
        if notebook_id:
            ok = await repo.add_to_notebook(note_id, notebook_id)
            if ok:
                linked = notebook_id
        return json.dumps(
            {"id": note_id, "linked_notebook": linked},
            default=str,
            indent=2,
        )

    @mcp.tool()
    async def get_entity_graph(entity_id: str) -> str:
        """Get an entity and its KG relationships.

        Reads the real ``relation`` edge (entity↔entity, migration 39). The
        outgoing/incoming neighbours are the entities on the far side of every
        ``relation`` edge incident on this entity.
        """
        rid = ensure_record_id(entity_id)
        results = await execute_query(
            "SELECT *, ->relation->entity AS outgoing, "
            "<-relation<-entity AS incoming "
            "FROM ONLY $id",
            {"id": rid},
        )
        return json.dumps(results, default=str, indent=2)

    return mcp


def main():
    """Run the MCP server."""
    parser = argparse.ArgumentParser(description="SurrealDB MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
    )
    parser.add_argument("--port", type=int, default=8200)
    args = parser.parse_args()

    server = create_server()
    transport_kwargs = {}
    if args.transport != "stdio":
        transport_kwargs["port"] = args.port
    server.run(transport=args.transport, **transport_kwargs)
