"""Docker-backed round-trip for the Track OKF.4 MCP interchange tools.

Exercises ``export_okf`` and ``import_okf`` over the FastMCP tool interface
against a real migrated SurrealDB testcontainer, calling the tools' underlying
coroutines directly (no external MCP client). The tools delegate to the
app-main OKF export/import services (lazy-imported inside the tool body — the
one documented layering exception in ``mcp/server.py``); this package's tests
already reach app-main over the shared workspace venv (see
``test_entity_merge_roundtrip`` / ``test_relation_endpoint_resolution_roundtrip``),
so exercising that delegation here is consistent with the existing precedent.

The tool fns take NO config arg — they go through the GLOBAL connection pool and
the app-main DI factories (which build default-config repos) — so the
``mcp_global_db`` fixture points the global ``_config`` at the container and
resets the pool, exactly like the W.3 ``test_mcp_graph_tools_roundtrip`` suite.

Coverage:
- ``export_okf`` returns a SPEC-shaped ``{path: text}`` bundle + the ExportReport
  (incl. the OKF-D3 omitted-field ledger) for a seeded notebook.
- Round-trip: the exported bundle imported back MATCHES the same entities
  (deterministic ``(name, type)`` dedup — no duplication), asserted over a
  genuinely INDEPENDENT second connection (the shared-substrate precedent).
- ``import_okf`` accepts a base64 zip and reports the same counts.
- Typed error shapes (bad_request / malformed_bundle) never crash the tool.
"""

from __future__ import annotations

import base64
import io
import json
import uuid
import zipfile
from typing import Any, List

import pytest
import surrealdb_service.config as config_module
import surrealdb_service.connection as connection_module
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import db_connection, execute_query, parse_record_ids
from surrealdb_service.mcp.server import create_server

pytestmark = pytest.mark.requires_docker


def _u(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def _tool(server, name: str):
    """Return the underlying coroutine fn for a registered FastMCP tool."""
    return server._tool_manager.get_tool(name).fn


async def _read_independent(
    cfg: SurrealDBConfig, query: str, params: dict[str, Any]
) -> List[dict[str, Any]]:
    """Run a SELECT over a connection INDEPENDENT of the global pool.

    ``db_connection(cfg)`` builds its own ``AsyncSurreal`` socket — the genuine
    "second session" the shared-substrate assertion needs, matching the W.3
    ``test_mcp_graph_tools_roundtrip`` precedent.
    """
    async with db_connection(cfg) as conn:
        return parse_record_ids(await conn.query(query, params))


@pytest.fixture
async def mcp_global_db(live_surrealdb: SurrealDBConfig):
    """Bind the GLOBAL config/pool to the container for the no-config tool path.

    The MCP tool fns (and the app-main DI factories they call) instantiate repos
    with the default (global) config, so point ``_config`` at the container and
    reset the pool so it rebuilds against it. Restores prior global state on
    teardown. Identical to the W.3 graph-tools fixture.
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


async def _seed_notebook_with_graph(cfg: SurrealDBConfig, uid: str) -> str:
    """Seed a notebook whose sources own two related entities + a note.

    ``list_entities_for_notebook`` resolves notebook → sources via the
    ``reference`` edge, then entities whose ``source_documents`` intersect that
    set — so the entities must be owned by a source that is referenced by the
    notebook. Returns the notebook id.
    """
    nb_rows = await execute_query(
        "CREATE notebook SET name = $n;", {"n": _u("OKF")}, config=cfg
    )
    notebook_id = str(nb_rows[0]["id"])

    src_rows = await execute_query(
        "CREATE source SET title = $t;", {"t": f"Doc {uid}"}, config=cfg
    )
    source_id = str(src_rows[0]["id"])
    await execute_query(
        f"RELATE {source_id}->reference->{notebook_id};", None, config=cfg
    )

    a_rows = await execute_query(
        "CREATE entity SET canonical_name=$n, name=$n, entity_type='person', "
        "primary_type='Person', type_tags=['Person'], embedding=[], "
        "confidence=0.95, source_documents=[$src], status='active';",
        {"n": f"Ada {uid}", "src": source_id},
        config=cfg,
    )
    b_rows = await execute_query(
        "CREATE entity SET canonical_name=$n, name=$n, entity_type='person', "
        "primary_type='Person', type_tags=['Person'], embedding=[], "
        "confidence=0.95, source_documents=[$src], status='active';",
        {"n": f"Babbage {uid}", "src": source_id},
        config=cfg,
    )
    a_id, b_id = str(a_rows[0]["id"]), str(b_rows[0]["id"])
    await execute_query(
        f"RELATE {a_id}->relation->{b_id} SET relation_type='COLLABORATES', "
        "confidence=0.9, status='active', properties={};",
        None,
        config=cfg,
    )
    return notebook_id


# ----------------------------------------------------------------------
# export_okf — bundle shape + report
# ----------------------------------------------------------------------


async def test_export_okf_returns_bundle_and_report(
    mcp_global_db: SurrealDBConfig,
) -> None:
    cfg = mcp_global_db
    uid = uuid.uuid4().hex[:8]
    notebook_id = await _seed_notebook_with_graph(cfg, uid)

    export_okf = _tool(create_server(), "export_okf")
    out = json.loads(
        await export_okf(
            notebook_id=notebook_id,
            min_connections=0,
            min_confidence=0.0,
        )
    )

    assert out["status"] == "exported"
    bundle = out["bundle"]
    # SPEC bundle: reserved root index + the two entity concepts.
    assert "index.md" in bundle
    entity_files = [p for p in bundle if p.startswith("entities/")]
    assert len(entity_files) == 2

    report = out["report"]
    assert report["entities_written"] == 2
    # OKF-D3 omitted-field ledger surfaced (never silent).
    assert "entity.embedding" in report["metadata"]["omitted_fields"]


async def test_export_okf_invalid_filter_is_typed_error(
    mcp_global_db: SurrealDBConfig,
) -> None:
    export_okf = _tool(create_server(), "export_okf")
    out = json.loads(
        await export_okf(notebook_id="notebook:x", min_confidence=5.0)
    )
    assert out["status"] == "invalid_filter"


# ----------------------------------------------------------------------
# round-trip — export then import matches the same entities
# ----------------------------------------------------------------------


async def test_export_then_import_round_trip_matches_entities(
    mcp_global_db: SurrealDBConfig,
) -> None:
    cfg = mcp_global_db
    uid = uuid.uuid4().hex[:8]
    notebook_id = await _seed_notebook_with_graph(cfg, uid)

    server = create_server()
    export_okf = _tool(server, "export_okf")
    import_okf = _tool(server, "import_okf")

    exported = json.loads(
        await export_okf(
            notebook_id=notebook_id, min_connections=0, min_confidence=0.0
        )
    )
    bundle = exported["bundle"]

    # Import the bundle back. The two entities already exist under the same
    # (name, type) key → MATCHED, never duplicated (deterministic dedup).
    imported = json.loads(
        await import_okf(
            bundle=bundle, notebook_id=notebook_id, apply_dedup=False
        )
    )
    assert imported["status"] == "imported"
    assert imported["entities_matched"] == 2
    assert imported["entities_created"] == 0

    # Shared-substrate check: exactly one row per name over an INDEPENDENT
    # connection — the round-trip added no duplicate entities.
    for name in (f"Ada {uid}", f"Babbage {uid}"):
        rows = await _read_independent(
            cfg,
            "SELECT count() AS c FROM entity WHERE canonical_name = $n GROUP ALL;",
            {"n": name},
        )
        assert (rows[0]["c"] if rows else 0) == 1, name


async def test_import_okf_accepts_base64_zip(
    mcp_global_db: SurrealDBConfig,
) -> None:
    cfg = mcp_global_db
    uid = uuid.uuid4().hex[:8]
    notebook_id = await _seed_notebook_with_graph(cfg, uid)

    server = create_server()
    export_okf = _tool(server, "export_okf")
    import_okf = _tool(server, "import_okf")

    bundle = json.loads(
        await export_okf(
            notebook_id=notebook_id, min_connections=0, min_confidence=0.0
        )
    )["bundle"]

    # Re-zip the mapping and hand it in as base64 — the alternate input form.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for path, text in sorted(bundle.items()):
            z.writestr(path, text)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    imported = json.loads(
        await import_okf(
            zip_base64=b64, notebook_id=notebook_id, apply_dedup=False
        )
    )
    assert imported["status"] == "imported"
    assert imported["entities_matched"] == 2


# ----------------------------------------------------------------------
# typed error shapes — never crash the tool
# ----------------------------------------------------------------------


async def test_import_okf_bad_request_shapes(
    mcp_global_db: SurrealDBConfig,
) -> None:
    import_okf = _tool(create_server(), "import_okf")

    # Neither input.
    neither = json.loads(await import_okf())
    assert neither["status"] == "bad_request"

    # Both inputs.
    both = json.loads(
        await import_okf(bundle={"index.md": "x"}, zip_base64="AAAA")
    )
    assert both["status"] == "bad_request"

    # Undecodable base64.
    badb64 = json.loads(await import_okf(zip_base64="not valid base64 @@@"))
    assert badb64["status"] == "bad_request"


async def test_import_okf_malformed_zip_is_typed_error(
    mcp_global_db: SurrealDBConfig,
) -> None:
    import_okf = _tool(create_server(), "import_okf")
    # Valid base64, but the bytes are not a zip container.
    not_a_zip = base64.b64encode(b"plain text, not a zip").decode("ascii")
    out = json.loads(await import_okf(zip_base64=not_a_zip))
    assert out["status"] == "malformed_bundle"
