"""Track W.4 — light end-to-end integration check: the MCP graph tools COMPOSE.

This is the closing-phase E2E for Track W. It proves the three READ tools chain
into a single navigation flow the way a Claude Code session would use them:

    search  (lexical/BM25, NO embedding — no model needed)
      -> related  (load every edge incident on a hit's node)
      -> get_node (fetch a neighbour node the edge points at)

Why this path:

- It exercises the *composition* contract (one tool's output id feeds the next
  tool's input), which is the whole point of the shared graph-memory substrate —
  per-tool roundtrips already exist in ``test_mcp_graph_tools_roundtrip.py``; this
  test is about the tools fitting together.
- It is deliberately MODEL-FREE: ``search`` is called WITHOUT an ``embedding`` so
  it runs the lexical ``fn::text_search`` (BM25) fallback path. No Ollama embed,
  no reranker, no 2 GB cross-encoder download. The rerank leg of the chain
  (``search?rerank`` / the ``services/reranker`` cross-encoder) is the
  OPERATOR-GATED smoke documented in
  ``docs/tracks/W-mcp-graph-memory/OPERATOR_GUIDE.md`` and is intentionally NOT
  run here (see the module-level note on the reranker below).
- It is READ-ONLY: it seeds a small isolated subgraph (unique-suffixed ids) and
  then only SELECTs through the tools. No ``cite`` / ``add_note`` writes.

Against a real migrated SurrealDB testcontainer (``@requires_docker``), via the
same ``mcp_global_db`` global-pool binding the W.3 roundtrip uses.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import pytest

import surrealdb_service.config as config_module
import surrealdb_service.connection as connection_module
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.mcp.server import create_server

pytestmark = pytest.mark.requires_docker


def _u(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _tool(server, name: str):
    """Return the underlying coroutine fn for a registered FastMCP tool."""
    return server._tool_manager.get_tool(name).fn


@pytest.fixture
async def mcp_global_db(live_surrealdb: SurrealDBConfig):
    """Point the GLOBAL config/pool at the container for the no-config tool path.

    The MCP tool fns build repos with the default (global) config, so bind
    ``_config`` to the container and reset the pool so it rebuilds against it.
    Mirrors ``test_mcp_graph_tools_roundtrip.py``; restores prior state on exit.
    """
    prev_config = config_module._config
    prev_pool = connection_module._pool
    config_module._config = live_surrealdb
    connection_module._pool = None
    try:
        yield live_surrealdb
    finally:
        connection_module._pool = prev_pool
        config_module._config = prev_config


async def _make_source(cfg: SurrealDBConfig, title: str, content: str) -> str:
    """Create a source whose chunk is full-text indexed (so BM25 can find it).

    ``fn::text_search`` reads the source FTS index built in migrations; a source
    needs a ``full_text`` field for the lexical path to return it.
    """
    rows = await execute_query(
        "CREATE source SET title = $t, full_text = $c;",
        {"t": title, "c": content},
        config=cfg,
    )
    return str(rows[0]["id"])


async def _make_entity(cfg: SurrealDBConfig, name: str, etype: str) -> str:
    rows = await execute_query(
        "CREATE entity SET canonical_name=$n, name=$n, entity_type=$t, "
        "embedding=[], confidence=0.9, source_documents=[], status='active';",
        {"n": name, "t": etype},
        config=cfg,
    )
    return str(rows[0]["id"])


async def test_search_related_get_node_compose_read_only(
    mcp_global_db: SurrealDBConfig,
) -> None:
    """search (BM25) -> related -> get_node compose into one navigation flow.

    No embedding, no reranker, no model — the lexical leg only. Proves a hit's
    node id flows into ``related``, and an edge endpoint flows into ``get_node``.
    """
    cfg = mcp_global_db
    server = create_server()
    search = _tool(server, "search")
    related = _tool(server, "related")
    get_node = _tool(server, "get_node")

    # A unique keyword so the lexical search hits exactly our seeded source.
    marker = _u("regiodeal")

    # Seed a tiny subgraph: a source that mentions an entity, and that entity
    # related to a neighbour entity. This is the chain search->related->get_node
    # will walk.
    src = await _make_source(
        cfg,
        title=f"Beleidsnota {marker}",
        content=(
            f"De {marker} richt zich op economische versterking van de regio "
            "en samenwerking tussen gemeenten."
        ),
    )
    programme = await _make_entity(cfg, _u("Regio Deal"), "organization")
    municipality = await _make_entity(cfg, _u("Gemeente"), "organization")

    # source -> mentions -> programme  (so `related(src)` surfaces the entity)
    await execute_query(
        f"RELATE {src}->mentions->{programme} SET weight=0.6, "
        "concept_name='Regio Deal', concept_type='organization', "
        "document_frequency=1;",
        None,
        config=cfg,
    )
    # programme -> relation -> municipality (so `related(programme)` has a hop)
    await execute_query(
        f"RELATE {programme}->relation->{municipality} SET "
        "relation_type='COLLABORATES_WITH', confidence=0.8, status='active', "
        "properties={};",
        None,
        config=cfg,
    )

    # --- Leg 1: search (LEXICAL BM25, embedding omitted) -----------------
    raw = await search(query=marker, limit=10, include_notes=False)
    hits = json.loads(raw)
    assert isinstance(hits, list)
    # A lexical hit carries the matched source id under ``id`` (title/full-text
    # branch) — ``text_search`` returns ``fn::text_search`` rows as-is. Our
    # uniquely-marked source must be among them.
    hit_ids = {str(h.get("id") or h.get("item_id")) for h in hits}
    assert src in hit_ids, f"lexical search did not return seeded source: {hits}"

    # --- Leg 2: related on the returned source node ----------------------
    src_edges = json.loads(await related(node_id=src))
    assert isinstance(src_edges, list)
    mentions_edges = [e for e in src_edges if e["edge_type"] == "mentions"]
    assert mentions_edges, f"related(source) missing mentions edge: {src_edges}"
    # The edge points at the programme entity — that id feeds the next hop.
    assert any(e["target"] == programme for e in mentions_edges), src_edges

    # --- Leg 3a: related on the entity reached via the mentions edge -----
    prog_edges = json.loads(await related(node_id=programme))
    relation_edges = [e for e in prog_edges if e["edge_type"] == "relation"]
    assert relation_edges, f"related(entity) missing relation edge: {prog_edges}"
    assert any(e["target"] == municipality for e in relation_edges), prog_edges

    # --- Leg 3b: get_node on a neighbour reached by traversal ------------
    node_raw = await get_node(node_id=municipality)
    node = json.loads(node_raw)
    assert node is not None, "get_node returned null for a traversed neighbour"
    assert str(node["id"]) == municipality
    assert node["entity_type"] == "organization"

    # Full chain: search(text) -> related(source).mentions.target == programme
    # -> related(programme).relation.target == municipality -> get_node(municipality).
    # Every step consumed the previous step's id; no model, no writes.


async def test_search_lexical_path_takes_no_embedding(
    mcp_global_db: SurrealDBConfig,
) -> None:
    """The composed search leg is genuinely model-free (no embedding argument).

    Guards the W.4 claim that the E2E exercises search WITHOUT the embedding
    pipeline: calling ``search`` with no ``embedding`` returns a JSON list and
    never touches the vector side (it would raise/empty if it tried to vector
    search a None embedding).
    """
    cfg = mcp_global_db
    server = create_server()
    search = _tool(server, "search")

    marker = _u("zorgakkoord")
    await _make_source(
        cfg,
        title=f"Nota {marker}",
        content=f"Het {marker} beschrijft afspraken over de zorg in de regio.",
    )

    raw = await search(query=marker, limit=5, include_notes=False)
    hits = json.loads(raw)
    assert isinstance(hits, list)
    # Lexical hits carry no hybrid provenance (no _combined_score / _vector_rank)
    # because the vector side was never run.
    for h in hits:
        assert "_vector_rank" not in h, f"unexpected vector provenance: {h}"
