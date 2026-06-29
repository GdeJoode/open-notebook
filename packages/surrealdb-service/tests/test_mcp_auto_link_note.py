"""Container tests for the MCP ``auto_link_note`` tool (Track Y.2 + NS.2).

The tool is the embedding-free MCP entrypoint to auto-link (option a): it ranks
related notes (``find_related_by_embedding``) AND the sources a note is about
(``find_related_sources_by_embedding``) and writes idempotent ``related_note``
(note→note) and ``note_about`` (note→source) edges directly, and REQUIRES the
note to already have an embedding — an unembedded note returns
``needs_embedding`` (this layer cannot embed). These tests exercise the
registered tool against a real SurrealDB container so the full repo path (cosine
+ RELATE) runs, not mocks.

Proven (Y.2 + NS.2 AC for the MCP surface):

* threshold: only candidates at/above ``min_similarity`` become edges;
* top-k: ``k`` caps the linked set;
* self-exclusion: the note never links to itself;
* idempotency: re-running yields the identical edge set (no duplicate rows);
* needs-embedding: an unembedded note → ``needs_embedding``, no edges, no crash;
* not_found / invalid_id: clean statuses, never an exception;
* NS.2 source pass: the same gate writes ``note_about`` edges; a single call
  reports BOTH note and source counts; source links are idempotent on re-run.

Mirrors ``test_mcp_graph_tools_roundtrip``: the tool fns take no config arg (they
go through the global connection pool), so the ``mcp_global_db`` fixture points
the global config/pool at the container.
"""

from __future__ import annotations

import json
import uuid

import pytest

import surrealdb_service.config as config_module
import surrealdb_service.connection as connection_module
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query
from surrealdb_service.mcp.server import create_server

pytestmark = pytest.mark.requires_docker


def _u(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:10]}"


def _tool(server, name: str):
    """Return the underlying coroutine fn for a registered FastMCP tool."""
    return server._tool_manager.get_tool(name).fn


@pytest.fixture
async def mcp_global_db(live_surrealdb: SurrealDBConfig):
    """Bind the GLOBAL config/pool to the container for the no-config tool path."""
    prev_config = config_module._config
    prev_pool = connection_module._pool
    config_module._config = live_surrealdb
    connection_module._pool = None
    try:
        yield live_surrealdb
    finally:
        connection_module._pool = prev_pool
        config_module._config = prev_config


async def _make_note(
    config: SurrealDBConfig, *, embedding: list[float], title: str | None = None
) -> str:
    rows = await execute_query(
        "CREATE note SET title = $t, content = $t, embedding = $e RETURN AFTER;",
        {"t": title or _u("note"), "e": embedding},
        config=config,
    )
    return str(rows[0]["id"])


async def _edge_count(config: SurrealDBConfig, from_id: str) -> int:
    rows = await execute_query(
        "SELECT count() AS c FROM related_note WHERE in = $f GROUP ALL;",
        {"f": ensure_record_id(from_id)},
        config=config,
    )
    return rows[0]["c"] if rows else 0


async def _targets(config: SurrealDBConfig, from_id: str) -> set[str]:
    rows = await execute_query(
        "SELECT out FROM related_note WHERE in = $f;",
        {"f": ensure_record_id(from_id)},
        config=config,
    )
    return {str(r["out"]) for r in rows}


async def _make_source(
    config: SurrealDBConfig, *, embedding: list[float], title: str | None = None
) -> str:
    rows = await execute_query(
        "CREATE source SET title = $t, embedding = $e RETURN AFTER;",
        {"t": title or _u("src"), "e": embedding},
        config=config,
    )
    return str(rows[0]["id"])


async def _source_targets(config: SurrealDBConfig, from_id: str) -> set[str]:
    rows = await execute_query(
        "SELECT out FROM note_about WHERE in = $f;",
        {"f": ensure_record_id(from_id)},
        config=config,
    )
    return {str(r["out"]) for r in rows}


async def _source_edge_count(config: SurrealDBConfig, from_id: str) -> int:
    rows = await execute_query(
        "SELECT count() AS c FROM note_about WHERE in = $f GROUP ALL;",
        {"f": ensure_record_id(from_id)},
        config=config,
    )
    return rows[0]["c"] if rows else 0


async def test_auto_link_note_threshold_and_self_exclusion(
    mcp_global_db: SurrealDBConfig,
) -> None:
    cfg = mcp_global_db
    fn = _tool(create_server(), "auto_link_note")

    q = await _make_note(cfg, embedding=[1.0, 0.0, 0.0])
    near = await _make_note(cfg, embedding=[0.99, 0.01, 0.0])  # ~1.0
    mid = await _make_note(cfg, embedding=[0.7, 0.7, 0.0])  # ~0.7
    far = await _make_note(cfg, embedding=[0.0, 1.0, 0.0])  # ~0.0

    out = json.loads(await fn(q, k=10, min_similarity=0.9))

    assert out["status"] == "linked"
    targets = set(out["linked_note_ids"])
    # near (~1.0) is above 0.9 → linked; mid (~0.7) and far (~0.0) are below the
    # threshold → never linked (whether they were ranked-and-dropped or pushed
    # out of the top-k by other notes in the shared DB, they must not be edges).
    assert near in targets
    assert mid not in targets and far not in targets
    assert q not in targets, "a note must never link to itself"
    # Every linked candidate was at/above the threshold (none slipped below), and
    # accounting balances: created + below_threshold + skipped == considered.
    assert out["created"] >= 1
    assert (
        out["created"] + out["below_threshold"] + out["skipped"]
        == out["candidates_considered"]
    )
    assert await _targets(cfg, q) == targets


async def test_auto_link_note_top_k_caps_links(
    mcp_global_db: SurrealDBConfig,
) -> None:
    cfg = mcp_global_db
    fn = _tool(create_server(), "auto_link_note")

    q = await _make_note(cfg, embedding=[1.0, 0.0, 0.0])
    for _ in range(5):
        await _make_note(cfg, embedding=[0.95, 0.05, 0.0])

    out = json.loads(await fn(q, k=2, min_similarity=0.5))
    assert out["status"] == "linked"
    assert out["created"] == 2, f"k=2 must cap links, got {out['created']}"
    assert await _edge_count(cfg, q) == 2


async def test_auto_link_note_idempotent_on_rerun(
    mcp_global_db: SurrealDBConfig,
) -> None:
    cfg = mcp_global_db
    fn = _tool(create_server(), "auto_link_note")

    q = await _make_note(cfg, embedding=[1.0, 0.0, 0.0])
    await _make_note(cfg, embedding=[0.97, 0.03, 0.0])
    await _make_note(cfg, embedding=[0.96, 0.04, 0.0])

    first = json.loads(await fn(q, k=10, min_similarity=0.8))
    targets_1 = await _targets(cfg, q)
    count_1 = await _edge_count(cfg, q)

    second = json.loads(await fn(q, k=10, min_similarity=0.8))
    targets_2 = await _targets(cfg, q)
    count_2 = await _edge_count(cfg, q)

    assert first["status"] == second["status"] == "linked"
    assert targets_1 == targets_2, "re-run changed the linked set"
    assert count_1 == count_2, "re-run duplicated edge rows (RELATE not idempotent)"


async def test_auto_link_note_needs_embedding(
    mcp_global_db: SurrealDBConfig,
) -> None:
    cfg = mcp_global_db
    fn = _tool(create_server(), "auto_link_note")

    # An unembedded note carries the strict-but-empty [] embedding.
    q = await _make_note(cfg, embedding=[])
    await _make_note(cfg, embedding=[1.0, 0.0, 0.0])

    out = json.loads(await fn(q, k=5, min_similarity=0.5))
    assert out["status"] == "needs_embedding"
    assert await _edge_count(cfg, q) == 0


async def test_auto_link_note_not_found_and_invalid_id(
    mcp_global_db: SurrealDBConfig,
) -> None:
    cfg = mcp_global_db
    fn = _tool(create_server(), "auto_link_note")

    missing = json.loads(await fn("note:does_not_exist_xyz", k=5))
    assert missing["status"] == "not_found"

    # A source id is a valid record id but the wrong table.
    src = await execute_query(
        "CREATE source SET title = $t RETURN AFTER;",
        {"t": _u("src")},
        config=cfg,
    )
    wrong = json.loads(await fn(str(src[0]["id"]), k=5))
    assert wrong["status"] == "invalid_id"


# ---------------------------------------------------------------------------
# NS.2 — the source pass: the tool also writes note_about edges and reports
# both counts; threshold/top-k/idempotency hold for sources too.
# ---------------------------------------------------------------------------


async def test_auto_link_note_links_sources_above_threshold(
    mcp_global_db: SurrealDBConfig,
) -> None:
    cfg = mcp_global_db
    fn = _tool(create_server(), "auto_link_note")

    q = await _make_note(cfg, embedding=[1.0, 0.0, 0.0])
    near = await _make_source(cfg, embedding=[0.99, 0.01, 0.0])  # ~1.0
    far = await _make_source(cfg, embedding=[0.0, 1.0, 0.0])  # ~0.0

    out = json.loads(await fn(q, k=10, min_similarity=0.9))

    assert out["status"] == "linked"
    src_targets = set(out["linked_source_ids"])
    assert near in src_targets, "highly-similar source must be linked"
    assert far not in src_targets, "below-threshold source must not be linked"
    assert out["source_links_created"] >= 1
    # Accounting balances for the source pass.
    assert (
        out["source_links_created"]
        + out["source_below_threshold"]
        + out["source_skipped_existing"]
        == out["source_candidates_considered"]
    )
    # The persisted note_about edges match the reported set.
    assert await _source_targets(cfg, q) == src_targets


async def test_auto_link_note_reports_both_note_and_source_counts(
    mcp_global_db: SurrealDBConfig,
) -> None:
    """A single tool call produces BOTH related_note and note_about edges and the
    summary carries both counters (no conflation)."""
    cfg = mcp_global_db
    fn = _tool(create_server(), "auto_link_note")

    q = await _make_note(cfg, embedding=[1.0, 0.0, 0.0])
    sim_note = await _make_note(cfg, embedding=[0.98, 0.02, 0.0])
    sim_src = await _make_source(cfg, embedding=[0.97, 0.03, 0.0])

    out = json.loads(await fn(q, k=10, min_similarity=0.8))

    assert out["status"] == "linked"
    assert sim_note in set(out["linked_note_ids"])
    assert sim_src in set(out["linked_source_ids"])
    # Both link types persisted.
    assert sim_note in await _targets(cfg, q)
    assert sim_src in await _source_targets(cfg, q)
    # Both counter families present.
    assert {
        "created",
        "source_links_created",
        "candidates_considered",
        "source_candidates_considered",
    } <= out.keys()


async def test_auto_link_note_source_top_k_caps(
    mcp_global_db: SurrealDBConfig,
) -> None:
    cfg = mcp_global_db
    fn = _tool(create_server(), "auto_link_note")

    q = await _make_note(cfg, embedding=[1.0, 0.0, 0.0])
    for _ in range(5):
        await _make_source(cfg, embedding=[0.95, 0.05, 0.0])

    out = json.loads(await fn(q, k=2, min_similarity=0.5))
    assert out["status"] == "linked"
    assert out["source_links_created"] == 2, "k=2 must cap source links"
    assert await _source_edge_count(cfg, q) == 2


async def test_auto_link_note_source_idempotent_on_rerun(
    mcp_global_db: SurrealDBConfig,
) -> None:
    cfg = mcp_global_db
    fn = _tool(create_server(), "auto_link_note")

    q = await _make_note(cfg, embedding=[1.0, 0.0, 0.0])
    await _make_source(cfg, embedding=[0.97, 0.03, 0.0])
    await _make_source(cfg, embedding=[0.96, 0.04, 0.0])

    json.loads(await fn(q, k=10, min_similarity=0.8))
    targets_1 = await _source_targets(cfg, q)
    count_1 = await _source_edge_count(cfg, q)

    json.loads(await fn(q, k=10, min_similarity=0.8))
    targets_2 = await _source_targets(cfg, q)
    count_2 = await _source_edge_count(cfg, q)

    assert targets_1 == targets_2, "re-run changed the linked source set"
    assert count_1 == count_2, "re-run duplicated note_about rows"
