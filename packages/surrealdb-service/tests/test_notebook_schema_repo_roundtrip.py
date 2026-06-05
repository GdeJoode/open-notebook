"""Roundtrip tests for the Phase B.1b notebook_schema + pass1_results tables.

Boots the testcontainers SurrealDB harness (``live_surrealdb`` fixture from
``surrealdb_service.testing``), applies every migration up to and including
#45, and exercises both repositories end-to-end.

Acceptance criteria mapped to test cases:

* AC#1 (migration applies + idempotent) — covered indirectly by the harness
  itself, plus an explicit assertion that #45 is recorded in
  ``_sbl_migrations`` and that table-level DEFINE IF NOT EXISTS does not
  blow up on re-run.
* AC#2 (NotebookSchema upsert roundtrip) — ``test_notebook_schema_upsert_roundtrip``.
* AC#3 (Pass1Result record + list_by_source) — ``test_pass1_result_record_and_list``.
* AC#4 (UNIQUE notebook_schema.notebook) — ``test_notebook_schema_unique_index_rewrites``.
* AC#5 (regression: EntityExtractionService still works) — out-of-scope here,
  guarded by the pre-existing ``test_entity_roundtrip`` in
  ``test_migrations_roundtrip.py`` which we leave untouched.
"""

from __future__ import annotations

import uuid

import pytest

from shared.models import NotebookSchema, Pass1Result
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.notebook_schema import (
    NotebookSchemaRepository,
    Pass1ResultRepository,
)


def _unique(prefix: str) -> str:
    """Generate a collision-free SurrealDB id suffix."""
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


async def _create_notebook(config: SurrealDBConfig) -> str:
    """Create a real ``notebook`` row and return its record id.

    Both new tables hold ``record<notebook>`` fields, so SCHEMAFULL
    enforcement will reject writes that reference a non-existent notebook.
    """
    name = _unique("nb")
    rows = await execute_query(
        "CREATE notebook SET name = $name, description = 'b1b roundtrip';",
        {"name": name},
        config=config,
    )
    assert rows, "Expected CREATE notebook to return a row"
    return str(rows[0]["id"])


async def _create_source(config: SurrealDBConfig) -> str:
    """Create a real ``source`` row and return its record id."""
    title = _unique("src")
    rows = await execute_query(
        "CREATE source SET title = $title, full_text = 'lorem ipsum';",
        {"title": title},
        config=config,
    )
    assert rows, "Expected CREATE source to return a row"
    return str(rows[0]["id"])


# ---------------------------------------------------------------------------
# Smoke: migration 45 applied
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_45_recorded(live_surrealdb: SurrealDBConfig) -> None:
    """Migration 45 is recorded in ``_sbl_migrations``.

    Without this, every other test in this file is moot — they would skip
    silently because the tables wouldn't exist.
    """
    rows = await execute_query(
        "SELECT version FROM _sbl_migrations WHERE version = 45;",
        config=live_surrealdb,
    )
    assert rows, "Migration 45 not recorded — harness did not apply it"
    assert rows[0]["version"] == 45


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_migration_45_is_idempotent(live_surrealdb: SurrealDBConfig) -> None:
    """Re-executing migration 45 DDL is a no-op.

    The migration file uses ``IF NOT EXISTS`` on every DEFINE. Replaying
    those statements against a populated DB should succeed without errors
    and without disturbing existing rows.
    """
    from pathlib import Path

    here = Path(__file__).resolve()
    migration_path = None
    for ancestor in here.parents:
        candidate = ancestor / "migrations" / "45.surrealql"
        if candidate.is_file():
            migration_path = candidate
            break
    assert migration_path is not None, "Could not locate migrations/45.surrealql"

    body = migration_path.read_text(encoding="utf-8")
    # Strip SurrealQL line comments so they don't end up inside a single
    # query the runner sends.
    cleaned = "\n".join(
        line for line in body.splitlines() if not line.strip().startswith("--")
    )
    # Re-run every statement; IF NOT EXISTS makes this safe.
    await execute_query(cleaned, config=live_surrealdb)

    # And the version record is unchanged (still exactly one row for #45).
    rows = await execute_query(
        "SELECT count() AS n FROM _sbl_migrations WHERE version = 45 GROUP ALL;",
        config=live_surrealdb,
    )
    assert rows and rows[0]["n"] == 1


# ---------------------------------------------------------------------------
# NotebookSchemaRepository round-trip
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_notebook_schema_upsert_roundtrip(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """upsert → get_by_notebook returns an equivalent NotebookSchema."""
    notebook_id = await _create_notebook(live_surrealdb)
    repo = NotebookSchemaRepository(config=live_surrealdb)

    original = NotebookSchema(
        notebook=notebook_id,
        base_ontology="scholarly",
        coverage_pct=0.85,
        accepted_extensions=[
            {
                "extension_id": "ext-1",
                "type_name": "Methodology",
                "parent_type": "Concept",
                "properties": ["name"],
            }
        ],
        pending_extensions=[
            {"extension_id": "ext-2", "type_name": "Dataset"}
        ],
        review_required=True,
        metadata={"freeform": True},
    )
    record_id = await repo.upsert(original)
    assert record_id.startswith("notebook_schema:")

    fetched = await repo.get_by_notebook(notebook_id)
    assert fetched is not None
    assert fetched.notebook == notebook_id
    assert fetched.base_ontology == "scholarly"
    assert fetched.coverage_pct == pytest.approx(0.85)
    assert fetched.review_required is True
    assert fetched.metadata == {"freeform": True}
    assert len(fetched.accepted_extensions) == 1
    assert fetched.accepted_extensions[0]["type_name"] == "Methodology"
    assert len(fetched.pending_extensions) == 1
    assert fetched.pending_extensions[0]["extension_id"] == "ext-2"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_notebook_schema_unique_index_rewrites(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """A second upsert on the same notebook rewrites — UNIQUE is respected.

    Decision (documented for downstream callers): we chose
    **rewrite-semantic** for ``upsert`` rather than letting the UNIQUE
    index raise. The repository queries by ``notebook`` first and routes
    to UPDATE on hit, CREATE on miss. The DB-level UNIQUE constraint is
    still a hard guard for any caller that bypasses the repository.

    This test verifies:

    1. Second upsert returns the *same* record id as the first
       (proving UPDATE path, not CREATE).
    2. ``SELECT count()`` for the notebook still returns exactly 1 row.
    3. The second upsert's values overwrite the first.
    """
    notebook_id = await _create_notebook(live_surrealdb)
    repo = NotebookSchemaRepository(config=live_surrealdb)

    first_id = await repo.upsert(
        NotebookSchema(
            notebook=notebook_id,
            base_ontology="scholarly",
            coverage_pct=0.5,
        )
    )
    second_id = await repo.upsert(
        NotebookSchema(
            notebook=notebook_id,
            base_ontology="technical",
            coverage_pct=0.9,
            review_required=True,
        )
    )
    assert first_id == second_id, (
        "Second upsert should rewrite the same row (UNIQUE index "
        "would otherwise force a duplicate). Got distinct ids: "
        f"{first_id!r} vs {second_id!r}"
    )

    # And the DB itself agrees there is only one row.
    rows = await execute_query(
        "SELECT count() AS n FROM notebook_schema "
        "WHERE notebook = type::thing($nb) GROUP ALL;",
        {"nb": notebook_id},
        config=live_surrealdb,
    )
    assert rows and rows[0]["n"] == 1

    # And the rewrite took effect.
    fetched = await repo.get_by_notebook(notebook_id)
    assert fetched is not None
    assert fetched.base_ontology == "technical"
    assert fetched.coverage_pct == pytest.approx(0.9)
    assert fetched.review_required is True


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_notebook_schema_unique_index_blocks_direct_create(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """Direct CREATE bypassing the repo raises — the UNIQUE index is real.

    Defensive check: any callers that sidestep the repository and CREATE
    notebook_schema directly should hit the UNIQUE constraint. This proves
    the rewrite-semantic in the repo is the only safe path.
    """
    notebook_id = await _create_notebook(live_surrealdb)
    repo = NotebookSchemaRepository(config=live_surrealdb)
    await repo.upsert(
        NotebookSchema(notebook=notebook_id, base_ontology="scholarly")
    )

    with pytest.raises(Exception):
        await execute_query(
            "CREATE notebook_schema SET "
            "notebook = type::thing($nb), "
            "base_ontology = 'scholarly';",
            {"nb": notebook_id},
            config=live_surrealdb,
        )


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_notebook_schema_pending_extension_lifecycle(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """add → accept and add → reject move the extension correctly."""
    notebook_id = await _create_notebook(live_surrealdb)
    repo = NotebookSchemaRepository(config=live_surrealdb)
    await repo.upsert(
        NotebookSchema(notebook=notebook_id, base_ontology="scholarly")
    )

    # add
    await repo.add_pending_extension(
        notebook_id,
        {"extension_id": "ext-A", "type_name": "Methodology"},
    )
    await repo.add_pending_extension(
        notebook_id,
        {"extension_id": "ext-B", "type_name": "Dataset"},
    )
    after_add = await repo.get_by_notebook(notebook_id)
    assert after_add is not None
    assert {e["extension_id"] for e in after_add.pending_extensions} == {
        "ext-A",
        "ext-B",
    }

    # accept ext-A
    accepted = await repo.accept_pending_extension(notebook_id, "ext-A")
    assert accepted is True
    after_accept = await repo.get_by_notebook(notebook_id)
    assert after_accept is not None
    assert [e["extension_id"] for e in after_accept.pending_extensions] == ["ext-B"]
    assert [e["extension_id"] for e in after_accept.accepted_extensions] == ["ext-A"]

    # reject ext-B
    rejected = await repo.reject_pending_extension(notebook_id, "ext-B")
    assert rejected is True
    after_reject = await repo.get_by_notebook(notebook_id)
    assert after_reject is not None
    assert after_reject.pending_extensions == []
    assert [e["extension_id"] for e in after_reject.accepted_extensions] == ["ext-A"]

    # accepting/rejecting a non-existent id returns False
    assert await repo.accept_pending_extension(notebook_id, "missing") is False
    assert await repo.reject_pending_extension(notebook_id, "missing") is False


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_get_by_notebook_returns_none_when_absent(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """No row → ``get_by_notebook`` returns ``None`` instead of raising."""
    notebook_id = await _create_notebook(live_surrealdb)
    repo = NotebookSchemaRepository(config=live_surrealdb)
    assert await repo.get_by_notebook(notebook_id) is None


# ---------------------------------------------------------------------------
# Pass1ResultRepository round-trip
# ---------------------------------------------------------------------------


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_pass1_result_record_and_list(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """record → list_by_source returns the row; second insert returns both."""
    notebook_id = await _create_notebook(live_surrealdb)
    source_id = await _create_source(live_surrealdb)
    repo = Pass1ResultRepository(config=live_surrealdb)

    first_id = await repo.record(
        Pass1Result(
            source=source_id,
            notebook=notebook_id,
            schema_attempted="scholarly",
            detected_schema="scholarly",
            confidence_in_choice=0.78,
            coverage_pct=0.62,
            uncovered_concepts=[
                {"surface_form": "RAG", "suggested_type": "Methodology"}
            ],
            proposed_extensions=[
                {"extension_id": "ext-1", "type_name": "Methodology"}
            ],
            pass1_metadata={"llm_model": "gpt-4o"},
        )
    )
    assert first_id.startswith("pass1_results:")

    listed = await repo.list_by_source(source_id)
    assert len(listed) == 1
    result = listed[0]
    assert result.source == source_id
    assert result.notebook == notebook_id
    assert result.schema_attempted == "scholarly"
    assert result.detected_schema == "scholarly"
    assert result.confidence_in_choice == pytest.approx(0.78)
    assert result.coverage_pct == pytest.approx(0.62)
    assert result.uncovered_concepts == [
        {"surface_form": "RAG", "suggested_type": "Methodology"}
    ]
    assert result.proposed_extensions == [
        {"extension_id": "ext-1", "type_name": "Methodology"}
    ]
    assert result.pass1_metadata == {"llm_model": "gpt-4o"}

    # A second insert for the same source returns both, newest first.
    await repo.record(
        Pass1Result(
            source=source_id,
            notebook=notebook_id,
            schema_attempted="scholarly",
            detected_schema="technical",
            confidence_in_choice=0.4,
            coverage_pct=0.3,
        )
    )
    again = await repo.list_by_source(source_id)
    assert len(again) == 2
    # Newest first — second insert wins on detected_schema.
    assert again[0].detected_schema == "technical"
    assert again[1].detected_schema == "scholarly"


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_pass1_result_list_by_notebook(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """list_by_notebook returns rows across multiple sources in one notebook."""
    notebook_id = await _create_notebook(live_surrealdb)
    source_a = await _create_source(live_surrealdb)
    source_b = await _create_source(live_surrealdb)
    repo = Pass1ResultRepository(config=live_surrealdb)

    await repo.record(
        Pass1Result(
            source=source_a,
            notebook=notebook_id,
            schema_attempted="scholarly",
            detected_schema="scholarly",
        )
    )
    await repo.record(
        Pass1Result(
            source=source_b,
            notebook=notebook_id,
            schema_attempted="scholarly",
            detected_schema="technical",
        )
    )

    listed = await repo.list_by_notebook(notebook_id)
    assert len(listed) == 2
    assert {r.source for r in listed} == {source_a, source_b}


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_pass1_result_empty_lists_when_no_rows(
    live_surrealdb: SurrealDBConfig,
) -> None:
    """list_by_source / list_by_notebook return ``[]`` cleanly on no rows."""
    notebook_id = await _create_notebook(live_surrealdb)
    source_id = await _create_source(live_surrealdb)
    repo = Pass1ResultRepository(config=live_surrealdb)

    assert await repo.list_by_source(source_id) == []
    assert await repo.list_by_notebook(notebook_id) == []
