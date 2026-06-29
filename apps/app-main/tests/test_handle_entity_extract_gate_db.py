"""Track PL.2 AC2: the schema-review gate parks the AUTO-enqueued extract.

Drives the REAL ``handle_entity_extract`` against a live SurrealDB testcontainer
with a notebook whose ``NotebookSchema.review_required`` is set and that has no
accepted extensions — the exact gate condition. Asserts the auto-chained behaviour:

  * ``run_extraction`` raises ``SchemaReviewPendingError`` (the worker would park
    the job ``PAUSED_FOR_REVIEW``) — no crash;
  * ``source.processing_stage`` becomes ``awaiting_schema_review``;
  * ZERO ``entity`` rows are written for the source (nothing extracted until the
    user reviews).

This exercises the gate path end-to-end without an LLM: the review check in
``_run_multi_schema`` fires BEFORE any model call. Gated behind
``@requires_docker``.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

import pytest

import app_main.handlers as handlers
from app_main.services.entity_extraction_service import (
    EntityExtractionService,
    SchemaReviewPendingError,
)
from shared.models.notebook_schema import NotebookSchema
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.notebook_schema import NotebookSchemaRepository
from surrealdb_service.repositories.source import SourceRepository

pytestmark = pytest.mark.asyncio


def _u(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


async def _make_notebook(cfg: SurrealDBConfig) -> str:
    rows = await execute_query(
        "CREATE notebook SET name=$n RETURN VALUE id;",
        {"n": _u("nb")},
        config=cfg,
    )
    return str(rows[0])


async def _make_source_with_chunk(cfg: SurrealDBConfig, notebook_id: str) -> str:
    """Seed a source linked to the notebook, with one content chunk (so the
    extraction path has something to chew on AND so the gate is the only reason
    it stops)."""
    repo = SourceRepository(config=cfg)
    rows = await execute_query(
        "CREATE source SET title=$t RETURN VALUE id;",
        {"t": _u("src")},
        config=cfg,
    )
    sid = str(rows[0])
    await execute_query(
        "CREATE chunk SET source=type::thing($sid), text=$txt, order=0, "
        "physical_page=1, element_type='paragraph', is_content=true, "
        "metadata={}, positions=[], parent_chunk_ids=[], layer=0, "
        "is_raptor_node=false;",
        {"sid": sid, "txt": "Ada Lovelace worked with Charles Babbage."},
        config=cfg,
    )
    await repo.add_to_notebook(sid, notebook_id)
    return sid


async def _seed_review_required_schema(
    cfg: SurrealDBConfig, notebook_id: str
) -> None:
    """One review-required schema row for the notebook, no accepted extensions
    and one pending extension (the gate condition)."""
    repo = NotebookSchemaRepository(config=cfg)
    await repo.upsert(
        NotebookSchema(
            notebook=notebook_id,
            base_ontology="general",
            review_required=True,
            accepted_extensions=[],
            pending_extensions=[{"type_name": "Widget", "parent_type": "Thing"}],
        )
    )


async def _stage(cfg: SurrealDBConfig, source_id: str) -> str:
    rows = await execute_query(
        "SELECT processing_stage FROM source WHERE id = type::thing($id);",
        {"id": source_id},
        config=cfg,
    )
    return rows[0]["processing_stage"]


async def _entity_count_for_source(cfg: SurrealDBConfig, source_id: str) -> int:
    rows = await execute_query(
        "SELECT count() AS c FROM entity "
        "WHERE $sid IN source_documents GROUP ALL;",
        {"sid": source_id},
        config=cfg,
    )
    if not rows:
        return 0
    c = rows[0].get("c", 0)
    return int(c) if c is not None else 0


@pytest.mark.requires_docker
async def test_auto_extract_parks_on_schema_review_gate(
    live_surrealdb: SurrealDBConfig,
) -> None:
    cfg = live_surrealdb
    notebook_id = await _make_notebook(cfg)
    sid = await _make_source_with_chunk(cfg, notebook_id)
    await _seed_review_required_schema(cfg, notebook_id)

    # Sanity: the source starts at the default stage and has no entities.
    assert await _stage(cfg, sid) == "ingested"
    assert await _entity_count_for_source(cfg, sid) == 0

    # Real service + repo, both wired to the container config. The handler
    # resolves the notebook via get_source_repo().get_notebook_id and writes the
    # stage via the same repo, so one container-wired SourceRepository serves
    # both.
    source_repo = SourceRepository(config=cfg)
    service = EntityExtractionService(
        source_repo=source_repo,
        notebook_schema_repo=NotebookSchemaRepository(config=cfg),
    )

    with patch(
        "app_main.dependencies.get_entity_extraction_service",
        new=lambda: service,
    ), patch(
        "app_main.dependencies.get_source_repo",
        new=lambda: source_repo,
    ):
        # The gate raises SchemaReviewPendingError; the handler sets the stage,
        # then reraises (the worker would translate that into PAUSED_FOR_REVIEW).
        with pytest.raises(SchemaReviewPendingError):
            await handlers.handle_entity_extract(
                {"source_id": sid, "multi_schema_enabled": True}
            )

    # AC2: stage parked, and NO entities written (nothing extracted past the gate).
    assert await _stage(cfg, sid) == "awaiting_schema_review"
    assert await _entity_count_for_source(cfg, sid) == 0
