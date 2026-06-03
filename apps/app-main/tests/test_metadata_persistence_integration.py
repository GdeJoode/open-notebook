"""Integration tests for Source.metadata persistence (Phase A.1c attempt 2).

Verifies the things the unit-test layer in `test_source_processing_service.py`
hides because it mocks `source_repo.update` directly:

1. `SourceRepository.update()` actually accepts the four provenance keys
   under `metadata` and routes them through to the SurrealQL UPDATE
   payload (no field filtering / drop at the repository layer).
2. `Source.metadata` round-trips through the Pydantic model
   (construction → dict → reconstruction) without losing or coercing
   the per-signal sub-dict.
3. `migrations/43.surrealql` declares the field on the `source` table
   as a flexible object — so an accidental removal trips this test
   immediately and the SCHEMAFULL-rejection bug from attempt 1 cannot
   recur silently.

What this does NOT cover (acknowledged gap, deferred to A.3 Playwright):
a live SurrealDB round-trip that actually executes the migration +
INSERT/SELECT against the persisted row. The codebase has no
testcontainer / in-memory SurrealDB plumbing, and the pinned
`surrealdb-python` only accepts `ws://`/`http://` URLs. Phase A.3 will
drive a real PDF through the API + a live DB.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import AsyncMock, patch

import pytest

from shared.models import Source
from surrealdb_service.repositories import SourceRepository


_PROVENANCE_METADATA: Dict[str, Any] = {
    "parser_engine_used": "mineru",
    "extraction_confidence": 0.62,
    "extraction_confidence_signals": {
        "ocr_confidence": 0.80,
        "text_density": 0.55,
        "heading_rate": 0.45,
        "table_success": 1.0,
        "image_text_ratio": 0.40,
        "unknown_element_ratio": 0.85,
    },
    "extraction_fallback_triggered": True,
}


# ---------------------------------------------------------------------------
# Repository-layer round-trip
# ---------------------------------------------------------------------------


class TestSourceRepositoryMetadataPersistence:
    """The four provenance keys must survive ``SourceRepository.update``.

    The unit test in ``test_source_processing_service.py`` mocks
    ``source_repo.update`` to a side_effect lambda — it doesn't actually
    drive the real repository. If anything in the repo layer ever filters
    out unknown fields, the SCHEMAFULL bug from attempt 1 would
    re-surface. This test runs the real ``SourceRepository.update()``
    body with only ``execute_query`` patched, asserting the dict that
    ultimately reaches SurrealDB carries every provenance key.
    """

    @pytest.mark.asyncio
    async def test_update_passes_metadata_through_to_execute_query(self):
        """``SourceRepository.update`` must include the full metadata
        bag in the SurrealQL UPDATE payload (under the ``data`` parameter).
        Anything dropped here would silently fail in production even
        with migration #43 in place.
        """
        repo = SourceRepository()
        now = datetime.now(timezone.utc)

        async def fake_execute(query: str, params: Dict[str, Any], config: Any = None) -> List[Dict[str, Any]]:
            # Echo back a Source-shaped row from the params so the
            # repository constructs a valid Source on the return path.
            data = params["data"]
            return [
                {
                    "id": "source:test1",
                    "title": data.get("title", "Test"),
                    "topics": data.get("topics", []),
                    "full_text": data.get("full_text"),
                    "metadata": data.get("metadata", {}),
                    "created": now,
                    "updated": now,
                }
            ]

        # Patch execute_query in the module where BaseRepository.update
        # imports it (not via the public surrealdb_service namespace).
        with patch(
            "surrealdb_service.repositories.base.execute_query",
            side_effect=fake_execute,
        ) as mock_exec:
            updated = await repo.update(
                "source:test1",
                {
                    "title": "After auto-fallback",
                    "metadata": _PROVENANCE_METADATA,
                },
            )

        # 1. The UPDATE SQL went out to execute_query.
        assert mock_exec.await_count == 1
        sent_query, sent_params, *_ = mock_exec.await_args.args
        assert sent_query.startswith("UPDATE source:test1 MERGE")
        # 2. The metadata dict landed in the params unchanged.
        assert "metadata" in sent_params["data"]
        assert sent_params["data"]["metadata"] == _PROVENANCE_METADATA
        # 3. None of the four provenance keys was filtered.
        for key in (
            "parser_engine_used",
            "extraction_confidence",
            "extraction_confidence_signals",
            "extraction_fallback_triggered",
        ):
            assert key in sent_params["data"]["metadata"], (
                f"{key} was dropped by the repository layer"
            )
        # 4. The returned Source model has the same metadata.
        assert isinstance(updated, Source)
        assert updated.metadata == _PROVENANCE_METADATA

    @pytest.mark.asyncio
    async def test_update_with_empty_metadata_still_round_trips(self):
        """When the auto-fallback path didn't add provenance (e.g. raw
        text upload), we must still send an empty dict, not silently
        omit the key — otherwise legacy rows with stale metadata would
        be sticky.
        """
        repo = SourceRepository()
        now = datetime.now(timezone.utc)

        async def fake_execute(query: str, params: Dict[str, Any], config: Any = None) -> List[Dict[str, Any]]:
            return [
                {
                    "id": "source:test2",
                    "title": "Plain",
                    "topics": [],
                    "metadata": params["data"].get("metadata", {}),
                    "created": now,
                    "updated": now,
                }
            ]

        with patch(
            "surrealdb_service.repositories.base.execute_query",
            side_effect=fake_execute,
        ):
            updated = await repo.update("source:test2", {"metadata": {}})

        assert updated.metadata == {}


# ---------------------------------------------------------------------------
# Pydantic round-trip
# ---------------------------------------------------------------------------


class TestSourceMetadataModelRoundTrip:
    """The Pydantic model must preserve the nested signals dict losslessly."""

    def test_construction_then_model_dump_preserves_signals(self):
        src = Source(
            id="source:t",
            title="X",
            metadata=_PROVENANCE_METADATA,
        )
        dumped = src.model_dump()
        assert dumped["metadata"] == _PROVENANCE_METADATA
        # Pydantic must not flatten or strip nested floats.
        assert (
            dumped["metadata"]["extraction_confidence_signals"]
            == _PROVENANCE_METADATA["extraction_confidence_signals"]
        )

    def test_round_trip_through_model_dump_and_back(self):
        src = Source(id="source:t", title="X", metadata=_PROVENANCE_METADATA)
        reconstructed = Source(**src.model_dump())
        assert reconstructed.metadata == _PROVENANCE_METADATA


# ---------------------------------------------------------------------------
# Migration #43 static validation
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    """The repo root that holds ``migrations/`` (regardless of pytest cwd)."""
    here = Path(__file__).resolve()
    # apps/app-main/tests/<this>.py → up 3 levels for repo root.
    return here.parents[3]


class TestMigration43SourceMetadataField:
    """Migration #43 must declare ``metadata`` on the ``source`` table as
    a flexible object. If someone removes this migration thinking the
    table is schemaless, attempt 1's silent-drop bug returns. This test
    is the trip-wire.
    """

    def test_migration_43_exists_with_correct_declaration(self):
        migration_path = _repo_root() / "migrations" / "43.surrealql"
        assert migration_path.is_file(), (
            "migrations/43.surrealql is missing — without it the "
            "SCHEMAFULL source table will reject Source.metadata writes."
        )

        content = migration_path.read_text()
        normalised = " ".join(content.split())  # collapse whitespace/newlines

        # Required: declares the field on the source table.
        assert "DEFINE FIELD" in normalised
        assert "metadata" in normalised
        assert "ON TABLE source" in normalised
        # Required: FLEXIBLE so additive keys don't require new migrations.
        assert "FLEXIBLE" in normalised
        # Required: option<object> (or option <object>) so absent rows
        # remain valid.
        assert "option<object>" in normalised.replace(" ", "")
        # Required: idempotent (IF NOT EXISTS).
        assert "IF NOT EXISTS" in normalised

    def test_migration_43_down_removes_field(self):
        down_path = _repo_root() / "migrations" / "43_down.surrealql"
        assert down_path.is_file()

        content = down_path.read_text()
        normalised = " ".join(content.split())

        assert "REMOVE FIELD" in normalised
        assert "metadata" in normalised
        assert "ON TABLE source" in normalised
