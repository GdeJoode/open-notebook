"""Docker-backed roundtrip for the Track W.3 MCP graph tools.

Exercises the five new FastMCP tools (`search`, `get_node`, `related`, `cite`,
`add_note`) by calling their underlying coroutines directly (no external MCP
client needed) against a real migrated SurrealDB testcontainer. The tool fns
take NO config arg — they go through the global connection pool — so the
``mcp_global_db`` fixture points the global ``_config`` at the container and
resets the pool.

**Shared-substrate property (AC2).** The tool writes through the global pool;
the readback opens a genuinely INDEPENDENT connection via ``db_connection(cfg)``
(``_read_independent`` below). That helper constructs its OWN ``AsyncSurreal``
socket and never touches the global ``_pool`` (unlike ``execute_query``, whose
``get_pool(config)`` returns the cached singleton and ignores its ``config`` arg
once the pool exists). So a tool-written node being visible to that second
connection demonstrates two SEPARATE connections reading the SAME database — the
cross-session shared substrate — not merely durability on one connection.

Coverage:
- AC1: every tool returns valid JSON.
- AC2: ``related`` returns relation + mentions + cites edges, both endpoint
  directions; ``get_entity_graph`` reads ``relation`` (not the stale
  ``relates_to``); a tool-written node/edge is visible over an independent
  second connection.
- AC3: ``cite`` / ``add_note`` write, and re-``cite`` of the same pair is an
  idempotent no-op (no duplicate edge); self-citation refused.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, List

import pytest

import surrealdb_service.config as config_module
import surrealdb_service.connection as connection_module
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import (
    db_connection,
    execute_query,
    parse_record_ids,
)
from surrealdb_service.mcp.server import create_server

pytestmark = pytest.mark.requires_docker


def _u(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


async def _read_independent(
    cfg: SurrealDBConfig, query: str, params: dict[str, Any]
) -> List[dict[str, Any]]:
    """Run a SELECT over a connection INDEPENDENT of the global pool.

    ``db_connection(cfg)`` builds its own ``AsyncSurreal`` socket and signs in /
    uses the namespace+database itself — it never reads or mutates the global
    ``_pool``. This is the genuine "second session" the shared-substrate AC
    needs: the tool wrote through the global pool, this reads through a wholly
    separate connection to the same database. (``execute_query`` would NOT do —
    ``get_pool(config)`` hands back the cached singleton and ignores ``config``
    once the pool exists, so it would reuse the writer's own pool.)
    """
    async with db_connection(cfg) as conn:
        return parse_record_ids(await conn.query(query, params))


def _tool(server, name: str):
    """Return the underlying coroutine fn for a registered FastMCP tool."""
    return server._tool_manager.get_tool(name).fn


@pytest.fixture
async def mcp_global_db(live_surrealdb: SurrealDBConfig):
    """Bind the GLOBAL config/pool to the container for the no-config tool path.

    The MCP tool fns instantiate repos with the default (global) config, so we
    point ``_config`` at the container and reset the pool so it rebuilds against
    it. Restores the prior global state on teardown.
    """
    prev_config = config_module._config
    prev_pool = connection_module._pool
    config_module._config = live_surrealdb
    connection_module._pool = None
    try:
        yield live_surrealdb
    finally:
        # The pool we built here is bound to the test loop; drop it so the next
        # consumer rebuilds. Don't await close (loop teardown handles sockets).
        connection_module._pool = prev_pool
        config_module._config = prev_config


async def _make_source(cfg: SurrealDBConfig, title: str) -> str:
    rows = await execute_query(
        "CREATE source SET title = $t;", {"t": title}, config=cfg
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


# ----------------------------------------------------------------------
# related — relation + mentions + cites, both directions
# ----------------------------------------------------------------------


async def test_related_returns_all_three_edge_types_both_directions(
    mcp_global_db: SurrealDBConfig,
) -> None:
    cfg = mcp_global_db
    server = create_server()
    related = _tool(server, "related")

    # A hub entity with: an outgoing relation, an incoming relation, and an
    # incoming mentions edge. Plus a source that cites another (for cites cover).
    hub = await _make_entity(cfg, _u("Hub"), "organization")
    other = await _make_entity(cfg, _u("Other"), "person")
    src = await _make_source(cfg, _u("Doc"))

    # hub -> other (relation, hub is `in`)
    await execute_query(
        f"RELATE {hub}->relation->{other} SET relation_type='LEADS', "
        "confidence=0.8, status='active', properties={};",
        None, config=cfg,
    )
    # other -> hub (relation, hub is `out` — proves both-direction coverage)
    await execute_query(
        f"RELATE {other}->relation->{hub} SET relation_type='ADVISES', "
        "confidence=0.7, status='active', properties={};",
        None, config=cfg,
    )
    # src mentions hub (mentions, hub is `out`)
    await execute_query(
        f"RELATE {src}->mentions->{hub} SET weight=0.5, concept_name='Hub', "
        "concept_type='organization', document_frequency=1;",
        None, config=cfg,
    )

    out = await related(node_id=hub)
    edges = json.loads(out)
    assert isinstance(edges, list)

    by_type: dict[str, list[Any]] = {}
    for e in edges:
        by_type.setdefault(e["edge_type"], []).append(e)

    # Two relation edges (both directions), one mentions edge.
    assert len(by_type.get("relation", [])) == 2, edges
    assert len(by_type.get("mentions", [])) == 1, edges

    rel_dirs = {(e["source"], e["target"]) for e in by_type["relation"]}
    assert (hub, other) in rel_dirs  # hub as `in`
    assert (other, hub) in rel_dirs  # hub as `out`

    # cites cover: build a source->source citation and confirm `related` on the
    # citing source surfaces it.
    cited = await _make_source(cfg, _u("Cited"))
    await execute_query(
        f"RELATE {src}->cites->{cited} SET confidence=1.0, "
        "reference_text='ref', match_method='manual';",
        None, config=cfg,
    )
    src_edges = json.loads(await related(node_id=src))
    src_types = {e["edge_type"] for e in src_edges}
    assert "cites" in src_types
    assert "mentions" in src_types  # src mentions hub above


async def test_related_edge_types_filter_and_limit(
    mcp_global_db: SurrealDBConfig,
) -> None:
    cfg = mcp_global_db
    server = create_server()
    related = _tool(server, "related")

    a = await _make_entity(cfg, _u("A"), "organization")
    b = await _make_entity(cfg, _u("B"), "person")
    src = await _make_source(cfg, _u("S"))
    await execute_query(
        f"RELATE {a}->relation->{b} SET relation_type='X', confidence=0.8, "
        "status='active', properties={};",
        None, config=cfg,
    )
    await execute_query(
        f"RELATE {src}->mentions->{a} SET weight=0.5, concept_name='A', "
        "concept_type='organization', document_frequency=1;",
        None, config=cfg,
    )

    only_rel = json.loads(await related(node_id=a, edge_types=["relation"]))
    assert {e["edge_type"] for e in only_rel} == {"relation"}

    limited = json.loads(await related(node_id=a, limit=1))
    assert len(limited) == 1


# ----------------------------------------------------------------------
# get_node — polymorphic
# ----------------------------------------------------------------------


async def test_get_node_polymorphic(mcp_global_db: SurrealDBConfig) -> None:
    cfg = mcp_global_db
    server = create_server()
    get_node = _tool(server, "get_node")

    ent = await _make_entity(cfg, _u("Poly"), "organization")
    src = await _make_source(cfg, _u("PolySrc"))

    ent_node = json.loads(await get_node(node_id=ent))
    assert str(ent_node["id"]) == ent
    src_node = json.loads(await get_node(node_id=src))
    assert str(src_node["id"]) == src

    missing = json.loads(await get_node(node_id="entity:doesnotexist"))
    assert missing is None


# ----------------------------------------------------------------------
# get_entity_graph — reads `relation` (not stale `relates_to`)
# ----------------------------------------------------------------------


async def test_get_entity_graph_reads_relation(
    mcp_global_db: SurrealDBConfig,
) -> None:
    cfg = mcp_global_db
    server = create_server()
    get_entity_graph = _tool(server, "get_entity_graph")

    a = await _make_entity(cfg, _u("GA"), "organization")
    b = await _make_entity(cfg, _u("GB"), "person")
    await execute_query(
        f"RELATE {a}->relation->{b} SET relation_type='LEADS', confidence=0.8, "
        "status='active', properties={};",
        None, config=cfg,
    )

    out = json.loads(await get_entity_graph(entity_id=a))
    # ONLY query → a single dict.
    assert isinstance(out, dict)
    outgoing = [str(x) for x in (out.get("outgoing") or [])]
    assert b in outgoing, out

    # Falsifiability: the stale `relates_to` edge does not exist, so the old
    # query would have returned an empty neighbourhood. Prove no such table rows.
    stale = await execute_query(
        "SELECT count() AS c FROM relates_to GROUP ALL;", None, config=cfg
    )
    assert (stale[0]["c"] if stale else 0) == 0


# ----------------------------------------------------------------------
# cite — write + idempotent + shared substrate
# ----------------------------------------------------------------------


async def test_cite_writes_and_is_idempotent(
    mcp_global_db: SurrealDBConfig,
) -> None:
    cfg = mcp_global_db
    server = create_server()
    cite = _tool(server, "cite")

    citing = await _make_source(cfg, _u("Citing"))
    cited = await _make_source(cfg, _u("Cited"))

    first = json.loads(
        await cite(
            citing_source_id=citing,
            cited_source_id=cited,
            confidence=1.0,
            reference_text="ref text",
        )
    )
    assert first["created"] is True
    assert first["edge"]

    # Re-cite the SAME pair → no-op, no duplicate.
    second = json.loads(
        await cite(citing_source_id=citing, cited_source_id=cited)
    )
    assert second["created"] is False
    assert second["edge"] == first["edge"]

    # Shared substrate: read the edge over a genuinely INDEPENDENT connection
    # (its own AsyncSurreal socket, not the global pool the tool wrote through).
    # The edge being visible here proves a second session sees the same DB.
    rows = await _read_independent(
        cfg,
        "SELECT id, confidence, reference_text, match_method FROM cites "
        "WHERE in = type::thing($s) AND out = type::thing($t);",
        {"s": citing, "t": cited},
    )
    assert len(rows) == 1, "exactly one cites edge (idempotent)"
    assert rows[0]["match_method"] == "mcp"
    assert rows[0]["reference_text"] == "ref text"


async def test_cite_refuses_self_citation(
    mcp_global_db: SurrealDBConfig,
) -> None:
    cfg = mcp_global_db
    server = create_server()
    cite = _tool(server, "cite")
    s = await _make_source(cfg, _u("Solo"))
    res = json.loads(await cite(citing_source_id=s, cited_source_id=s))
    assert res["created"] is False
    assert res["reason"] == "self_citation"
    count = await execute_query(
        "SELECT count() AS c FROM cites WHERE in = type::thing($s) GROUP ALL;",
        {"s": s}, config=cfg,
    )
    assert (count[0]["c"] if count else 0) == 0


# ----------------------------------------------------------------------
# add_note — write + optional notebook link + shared substrate
# ----------------------------------------------------------------------


async def test_add_note_writes_and_links_notebook(
    mcp_global_db: SurrealDBConfig,
) -> None:
    cfg = mcp_global_db
    server = create_server()
    add_note = _tool(server, "add_note")

    nb_rows = await execute_query(
        "CREATE notebook SET name=$n;", {"n": _u("NB")}, config=cfg
    )
    notebook_id = str(nb_rows[0]["id"])

    title = _u("Note")
    res = json.loads(
        await add_note(
            title=title,
            content="hello world",
            notebook_id=notebook_id,
        )
    )
    note_id = res["id"]
    assert note_id.startswith("note:")
    assert res["linked_notebook"] == notebook_id

    # Shared substrate: a genuinely INDEPENDENT second connection (its own
    # AsyncSurreal socket via db_connection, NOT the writer's global pool) sees
    # the note + the artifact link — proving cross-connection visibility.
    note_rows = await _read_independent(
        cfg,
        "SELECT id, title, content FROM note WHERE id = type::thing($id);",
        {"id": note_id},
    )
    assert len(note_rows) == 1
    assert note_rows[0]["title"] == title
    assert note_rows[0]["content"] == "hello world"

    link = await _read_independent(
        cfg,
        "SELECT VALUE out FROM artifact WHERE in = type::thing($id);",
        {"id": note_id},
    )
    assert str(link[0]) == notebook_id


async def test_add_note_without_notebook(
    mcp_global_db: SurrealDBConfig,
) -> None:
    cfg = mcp_global_db
    server = create_server()
    add_note = _tool(server, "add_note")
    res = json.loads(await add_note(title=_u("Solo"), content="body only"))
    assert res["id"].startswith("note:")
    assert res["linked_notebook"] is None


async def test_shared_substrate_across_two_independent_connections(
    mcp_global_db: SurrealDBConfig,
) -> None:
    """AC2: a node written by a tool (global pool) is visible to a SECOND,
    independent connection — proving the shared substrate, not mere durability.

    The proof has two halves:

    1. The writer (the ``add_note`` tool) goes through the GLOBAL pool, and the
       reader (``db_connection(cfg)``) opens its OWN ``AsyncSurreal`` socket that
       never touches that pool. We assert the reader's connection object is NOT
       the pooled connection the writer used, so "the second session sees it" is
       a genuine cross-connection observation.
    2. That independent reader sees the just-written note.
    """
    cfg = mcp_global_db
    add_note = _tool(create_server(), "add_note")

    # Identify the global pool's own connection (the writer's side).
    pool = connection_module.get_pool()
    async with pool.acquire() as pooled_conn:
        writer_conn_id = id(pooled_conn)

    title = _u("Cross")
    note_id = json.loads(await add_note(title=title, content="cross-conn body"))[
        "id"
    ]

    # A genuinely independent connection: its own socket, distinct object.
    async with db_connection(cfg) as reader_conn:
        assert id(reader_conn) != writer_conn_id, (
            "readback connection must be independent of the global pool"
        )
        rows = parse_record_ids(
            await reader_conn.query(
                "SELECT id, title FROM note WHERE id = type::thing($id);",
                {"id": note_id},
            )
        )
    assert len(rows) == 1 and rows[0]["title"] == title


# ----------------------------------------------------------------------
# search — lexical (no embedding) returns JSON
# ----------------------------------------------------------------------


async def test_search_lexical_returns_json(
    mcp_global_db: SurrealDBConfig,
) -> None:
    server = create_server()
    search = _tool(server, "search")
    # No embedding → lexical BM25 path. Just assert it runs and returns a JSON
    # list (content correctness of fn::text_search is covered elsewhere).
    out = await search(query="regio deal", limit=5)
    parsed = json.loads(out)
    assert isinstance(parsed, list)


async def test_search_hybrid_with_embedding_returns_json(
    mcp_global_db: SurrealDBConfig,
) -> None:
    server = create_server()
    search = _tool(server, "search")
    # An arbitrary same-length vector exercises the hybrid RRF path end to end
    # (it won't match anything on an empty corpus, but must return a JSON list).
    vec: List[float] = [0.0] * 1024
    out = await search(query="beleid", limit=5, embedding=vec)
    parsed = json.loads(out)
    assert isinstance(parsed, list)
