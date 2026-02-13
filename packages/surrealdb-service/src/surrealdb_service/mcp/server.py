"""MCP server for SurrealDB access."""

import argparse
import json
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP

from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.repositories.search import SearchRepository
from surrealdb_service.repositories.source import SourceRepository


def create_server() -> FastMCP:
    """Create and configure the SurrealDB MCP server."""
    mcp = FastMCP(
        name="surrealdb-service",
        instructions="SurrealDB database access for open-notebook.",
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
        """Vector similarity search across sources and notes."""
        repo = SearchRepository()
        hits = await repo.vector_search(
            embedding, results, include_sources, include_notes, minimum_score
        )
        return json.dumps(hits, default=str, indent=2)

    @mcp.tool()
    async def get_entity_graph(entity_id: str) -> str:
        """Get an entity and its relationships from the knowledge graph."""
        rid = ensure_record_id(entity_id)
        results = await execute_query(
            "SELECT *, ->relates_to->entity AS outgoing, "
            "<-relates_to<-entity AS incoming "
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
