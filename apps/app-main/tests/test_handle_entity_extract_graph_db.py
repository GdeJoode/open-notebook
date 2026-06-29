"""PL.3 AC1/AC3: a successful auto-extract materializes the source's graph.

Drives the REAL ``handle_entity_extract`` against a live SurrealDB testcontainer
and asserts the PL.3 auto-chain's GRAPH link end-to-end with NO manual mentions
regenerate:

  * after extraction, this source's ``mentions`` edges exist (the document graph
    for the source is real) — AC1;
  * the read endpoint (``MentionsProjectionService.get_document_entity_edges``,
    source-scoped) returns them — AC1;
  * ``source.processing_stage`` advances ``extracted -> graphed -> complete`` — AC3.

We do NOT run the LLM extractor (heavy/flaky). Instead we mock
``run_extraction`` to PERSIST entities into the live DB exactly as a real
extraction would (canonical entities carrying this source in
``source_documents``), then let the handler's REAL post-extract advance
(PL.4: ``advance_source`` reads the ``extracted`` stage and runs the inline
GRAPH refresh) run against the container — so the seam under test
(extract success -> source-scoped refresh -> graphed/complete) is exercised
genuinely, end to end, against real SurrealDB RELATE/clear semantics. PL.4 is a
pure consolidation: this end-to-end result is identical to PL.3.

Gated behind ``@requires_docker``.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

import app_main.handlers as handlers
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.source import SourceRepository

pytestmark = pytest.mark.asyncio


def _u(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


async def _make_source(cfg: SurrealDBConfig) -> str:
    rows = await execute_query(
        "CREATE source SET title=$t RETURN VALUE id;",
        {"t": _u("src")},
        config=cfg,
    )
    return str(rows[0])


async def _make_entity(
    cfg: SurrealDBConfig, name: str, etype: str, source_docs: list[str]
) -> str:
    rows = await execute_query(
        "CREATE entity SET canonical_name=$n, name=$n, entity_type=$t, "
        "embedding=[], status='active', source_documents=$sd RETURN VALUE id;",
        {"n": name, "t": etype, "sd": source_docs},
        config=cfg,
    )
    return str(rows[0])


async def _stage(cfg: SurrealDBConfig, source_id: str) -> str:
    rows = await execute_query(
        "SELECT processing_stage FROM source WHERE id = type::thing($id);",
        {"id": source_id},
        config=cfg,
    )
    return rows[0]["processing_stage"]


@pytest.mark.requires_docker
async def test_auto_extract_materializes_source_graph_and_completes(
    live_surrealdb: SurrealDBConfig,
) -> None:
    cfg = live_surrealdb
    # Clean slate so edge/IDF counts are exact and isolated.
    await execute_query("DELETE mentions;", config=cfg)
    await execute_query("DELETE entity;", config=cfg)
    await execute_query("DELETE source;", config=cfg)

    # Two sources sharing a programme entity (df=2 -> survives R.6).
    sid = await _make_source(cfg)
    other = await _make_source(cfg)
    shared = await _make_entity(
        cfg, _u("Regio Deal"), "programme", [sid, other]
    )

    source_repo = SourceRepository(config=cfg)

    # Mock run_extraction to look like a real successful extraction (entities are
    # already persisted above, as run_extraction would have). The handler's REAL
    # post-extract graph refresh then runs against the container.
    async def _fake_run_extraction(*args, **kwargs):
        return {"source_id": sid, "entity_count": 1, "relation_count": 0}

    fake_service = type(
        "FakeSvc", (), {"run_extraction": staticmethod(_fake_run_extraction)}
    )()

    with patch(
        "app_main.dependencies.get_entity_extraction_service",
        new=lambda: fake_service,
    ), patch(
        "app_main.dependencies.get_source_repo",
        new=lambda: source_repo,
    ):
        result = await handlers.handle_entity_extract({"source_id": sid})

    assert result["success"] is True

    # AC1: this source's mentions edges exist with NO manual regenerate, and the
    # source-scoped read endpoint returns them.
    from app_main.services.mentions_projection_service import (
        MentionsProjectionService,
    )
    from surrealdb_service.repositories.entity import EntityRepository

    svc = MentionsProjectionService(
        entity_repo=EntityRepository(config=cfg), source_repo=source_repo
    )
    edges = await svc.get_document_entity_edges(source_id=sid)
    assert len(edges) == 1
    assert str(edges[0]["target"]) == shared
    assert float(edges[0]["weight"]) > 0.0

    # The refresh is source-scoped: the OTHER source got no edges (only sid
    # refreshed) even though it shares the concept.
    other_edges = await svc.get_document_entity_edges(source_id=other)
    assert other_edges == []

    # AC3: the stage advanced extracted -> graphed -> complete.
    assert await _stage(cfg, sid) == "complete"
