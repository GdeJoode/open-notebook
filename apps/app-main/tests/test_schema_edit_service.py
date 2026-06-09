"""Unit tests for the Phase B.3b SchemaEditService.

Exercises each of the six edit ops in isolation against AsyncMock'd
repositories. The docker-gated end-to-end test for the underlying
``notebook_event`` table + ``excluded_types`` field lives in
``packages/surrealdb-service/tests/test_notebook_event_repo_roundtrip.py``.

Acceptance criteria mapped to test classes / cases:

* AC#1 (rename emits one event + accepted_extensions entry + GET returns
  renamed type) — :class:`TestRenameType`.
* AC#2 (merge/split/delete each pass equivalent assertions) —
  :class:`TestMergeTypes`, :class:`TestSplitType`, :class:`TestDeleteType`.
* AC#3 (idempotency — re-run = no state change + 0 new events) —
  one ``test_..._is_idempotent`` per op class.
* AC#6 (exactly one event row per successful op) — asserted on the mock
  ``event_repo.record`` call count in every "happy path" test.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest

from shared.models import NotebookSchema
from app_main.services.schema_edit_service import (
    NotebookSchemaNotFoundError,
    SchemaEditService,
    UnknownExtensionError,
)


NOTEBOOK_ID = "notebook:edit-fixture"


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _make_schema(
    *,
    accepted: Optional[List[Dict[str, Any]]] = None,
    pending: Optional[List[Dict[str, Any]]] = None,
    excluded: Optional[List[str]] = None,
) -> NotebookSchema:
    return NotebookSchema(
        notebook=NOTEBOOK_ID,
        base_ontology="scholarly",
        accepted_extensions=list(accepted or []),
        pending_extensions=list(pending or []),
        excluded_types=list(excluded or []),
    )


@pytest.fixture
def schema_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.get_by_notebook = AsyncMock(return_value=_make_schema())
    repo.upsert = AsyncMock(return_value=f"notebook_schema:{NOTEBOOK_ID}")
    return repo


@pytest.fixture
def event_repo() -> AsyncMock:
    repo = AsyncMock()
    repo.record = AsyncMock(return_value="notebook_event:1")
    return repo


@pytest.fixture
def service(schema_repo: AsyncMock, event_repo: AsyncMock) -> SchemaEditService:
    return SchemaEditService(schema_repo=schema_repo, event_repo=event_repo)


def _bind_state(
    schema_repo: AsyncMock,
    initial: NotebookSchema,
) -> List[NotebookSchema]:
    """Wire the mock so the second get_by_notebook reflects the upsert.

    The service calls get_by_notebook BEFORE the mutation (to load
    state) and AGAIN after the upsert (to return the refreshed view).
    The default AsyncMock return_value would return the same row both
    times — fine for assertion shape, but masks the state-rotation
    semantic the production code relies on. This helper mirrors the
    real DB by tracking the most-recently-upserted schema.
    """
    state: Dict[str, NotebookSchema] = {"current": initial}

    async def _get_current(_notebook_id: str) -> NotebookSchema:
        # Return a deep-ish copy so the service mutating its local copy
        # doesn't bleed back into the fixture row.
        return state["current"].model_copy(deep=True)

    async def _upsert(schema: NotebookSchema) -> str:
        state["current"] = schema.model_copy(deep=True)
        return f"notebook_schema:{NOTEBOOK_ID}"

    schema_repo.get_by_notebook = AsyncMock(side_effect=_get_current)
    schema_repo.upsert = AsyncMock(side_effect=_upsert)
    return [initial]


# ---------------------------------------------------------------------------
# accept_extension / reject_extension
# ---------------------------------------------------------------------------


class TestAcceptExtension:
    @pytest.mark.asyncio
    async def test_accept_moves_pending_to_accepted_and_emits_event(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ) -> None:
        _bind_state(
            schema_repo,
            _make_schema(
                pending=[
                    {"extension_id": "ext-1", "type_name": "Cohort"},
                    {"extension_id": "ext-2", "type_name": "Methodology"},
                ]
            ),
        )

        result = await service.accept_extension(NOTEBOOK_ID, "Cohort")

        assert {e["type_name"] for e in result.accepted_extensions} == {"Cohort"}
        assert {e["type_name"] for e in result.pending_extensions} == {
            "Methodology"
        }
        event_repo.record.assert_awaited_once()
        args, _ = event_repo.record.call_args
        assert args[0] == NOTEBOOK_ID
        assert args[1] == "schema_changed"
        assert args[2]["op"] == "accept_extension"
        assert args[2]["type_name"] == "Cohort"

    @pytest.mark.asyncio
    async def test_accept_is_idempotent_when_already_accepted(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ) -> None:
        _bind_state(
            schema_repo,
            _make_schema(
                accepted=[{"extension_id": "ext-1", "type_name": "Cohort"}]
            ),
        )

        result = await service.accept_extension(NOTEBOOK_ID, "Cohort")

        # Still exactly one accepted entry (no duplicate).
        assert len(result.accepted_extensions) == 1
        event_repo.record.assert_not_awaited()
        schema_repo.upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_accept_raises_when_extension_unknown(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ) -> None:
        _bind_state(schema_repo, _make_schema())

        with pytest.raises(UnknownExtensionError):
            await service.accept_extension(NOTEBOOK_ID, "Nope")

        event_repo.record.assert_not_awaited()


class TestRejectExtension:
    @pytest.mark.asyncio
    async def test_reject_drops_pending_and_emits_event(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ) -> None:
        _bind_state(
            schema_repo,
            _make_schema(
                pending=[{"extension_id": "ext-1", "type_name": "Cohort"}]
            ),
        )

        result = await service.reject_extension(NOTEBOOK_ID, "Cohort")

        assert result.pending_extensions == []
        event_repo.record.assert_awaited_once()
        args, _ = event_repo.record.call_args
        assert args[2] == {"op": "reject_extension", "type_name": "Cohort"}

    @pytest.mark.asyncio
    async def test_reject_is_idempotent_when_already_gone(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ) -> None:
        _bind_state(schema_repo, _make_schema())
        result = await service.reject_extension(NOTEBOOK_ID, "Cohort")
        assert result.pending_extensions == []
        event_repo.record.assert_not_awaited()
        schema_repo.upsert.assert_not_awaited()


# ---------------------------------------------------------------------------
# rename_type
# ---------------------------------------------------------------------------


class TestRenameType:
    @pytest.mark.asyncio
    async def test_rename_appends_synonym_entry_and_emits_event(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ) -> None:
        _bind_state(schema_repo, _make_schema())

        result = await service.rename_type(
            NOTEBOOK_ID, "Researcher", "ResearchFellow"
        )

        rename_entries = [
            e for e in result.accepted_extensions if e.get("op") == "rename"
        ]
        assert len(rename_entries) == 1
        entry = rename_entries[0]
        assert entry["old_name"] == "Researcher"
        assert entry["new_name"] == "ResearchFellow"
        # type_name mirrors the new name so the TTL exporter surfaces it.
        assert entry["type_name"] == "ResearchFellow"

        event_repo.record.assert_awaited_once()
        args, _ = event_repo.record.call_args
        assert args[2] == {
            "op": "rename",
            "old_name": "Researcher",
            "new_name": "ResearchFellow",
        }

    @pytest.mark.asyncio
    async def test_rename_is_idempotent_on_replay(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ) -> None:
        _bind_state(schema_repo, _make_schema())

        await service.rename_type(NOTEBOOK_ID, "Researcher", "ResearchFellow")
        await service.rename_type(NOTEBOOK_ID, "Researcher", "ResearchFellow")

        # Reload final state — only one rename entry expected.
        final = await schema_repo.get_by_notebook(NOTEBOOK_ID)
        rename_entries = [
            e for e in final.accepted_extensions if e.get("op") == "rename"
        ]
        assert len(rename_entries) == 1
        # Only one event emitted — the second call short-circuited.
        assert event_repo.record.await_count == 1

    @pytest.mark.asyncio
    async def test_rename_to_same_name_is_noop(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ) -> None:
        _bind_state(schema_repo, _make_schema())
        result = await service.rename_type(NOTEBOOK_ID, "X", "X")
        assert result.accepted_extensions == []
        event_repo.record.assert_not_awaited()


# ---------------------------------------------------------------------------
# merge_types
# ---------------------------------------------------------------------------


class TestMergeTypes:
    @pytest.mark.asyncio
    async def test_merge_records_entry_and_emits_event(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ) -> None:
        _bind_state(schema_repo, _make_schema())

        result = await service.merge_types(
            NOTEBOOK_ID,
            type_names=["Author", "Editor"],
            merged_name="Contributor",
        )

        merge_entries = [
            e for e in result.accepted_extensions if e.get("op") == "merge"
        ]
        assert len(merge_entries) == 1
        entry = merge_entries[0]
        assert entry["source_types"] == ["Author", "Editor"]
        assert entry["merged_name"] == "Contributor"
        assert entry["type_name"] == "Contributor"

        event_repo.record.assert_awaited_once()
        args, _ = event_repo.record.call_args
        assert args[2]["op"] == "merge"
        assert args[2]["merged_name"] == "Contributor"

    @pytest.mark.asyncio
    async def test_merge_is_idempotent_on_replay(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ) -> None:
        _bind_state(schema_repo, _make_schema())

        await service.merge_types(
            NOTEBOOK_ID, ["Author", "Editor"], "Contributor"
        )
        await service.merge_types(
            NOTEBOOK_ID, ["Editor", "Author"], "Contributor"
        )  # different order, same set

        final = await schema_repo.get_by_notebook(NOTEBOOK_ID)
        merge_entries = [
            e for e in final.accepted_extensions if e.get("op") == "merge"
        ]
        assert len(merge_entries) == 1
        assert event_repo.record.await_count == 1

    @pytest.mark.asyncio
    async def test_merge_rejects_single_type_list(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
    ) -> None:
        _bind_state(schema_repo, _make_schema())
        with pytest.raises(ValueError):
            await service.merge_types(NOTEBOOK_ID, ["X"], "Y")


# ---------------------------------------------------------------------------
# split_type
# ---------------------------------------------------------------------------


class TestSplitType:
    @pytest.mark.asyncio
    async def test_split_records_entry_and_emits_event(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ) -> None:
        _bind_state(schema_repo, _make_schema())

        result = await service.split_type(
            NOTEBOOK_ID,
            type_name="Cohort",
            into=["StudyCohort", "ControlCohort"],
            criterion="by trial role",
        )

        split_entries = [
            e for e in result.accepted_extensions if e.get("op") == "split"
        ]
        assert len(split_entries) == 1
        entry = split_entries[0]
        assert entry["source_type"] == "Cohort"
        assert entry["into"] == ["ControlCohort", "StudyCohort"]
        assert entry["criterion"] == "by trial role"
        # type_name mirrors source so existing tooling sees a class declaration.
        assert entry["type_name"] == "Cohort"

        event_repo.record.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_split_is_idempotent_on_replay(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ) -> None:
        _bind_state(schema_repo, _make_schema())

        await service.split_type(
            NOTEBOOK_ID, "Cohort", ["A", "B"], "by year"
        )
        await service.split_type(
            NOTEBOOK_ID, "Cohort", ["B", "A"], "by year"
        )  # same set + criterion

        final = await schema_repo.get_by_notebook(NOTEBOOK_ID)
        splits = [
            e for e in final.accepted_extensions if e.get("op") == "split"
        ]
        assert len(splits) == 1
        assert event_repo.record.await_count == 1

    @pytest.mark.asyncio
    async def test_split_different_criterion_emits_distinct_entry(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ) -> None:
        """A second split with a different criterion is NOT idempotent —
        it is a distinct operation (different user intent).
        """
        _bind_state(schema_repo, _make_schema())

        await service.split_type(
            NOTEBOOK_ID, "Cohort", ["A", "B"], "by year"
        )
        await service.split_type(
            NOTEBOOK_ID, "Cohort", ["A", "B"], "by cohort size"
        )

        final = await schema_repo.get_by_notebook(NOTEBOOK_ID)
        splits = [
            e for e in final.accepted_extensions if e.get("op") == "split"
        ]
        assert len(splits) == 2
        assert event_repo.record.await_count == 2

    @pytest.mark.asyncio
    async def test_split_rejects_single_target(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
    ) -> None:
        _bind_state(schema_repo, _make_schema())
        with pytest.raises(ValueError):
            await service.split_type(
                NOTEBOOK_ID, "Cohort", ["OnlyOne"], "x"
            )


# ---------------------------------------------------------------------------
# delete_type
# ---------------------------------------------------------------------------


class TestDeleteType:
    @pytest.mark.asyncio
    async def test_delete_appends_to_excluded_types(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ) -> None:
        _bind_state(schema_repo, _make_schema())

        result = await service.delete_type(NOTEBOOK_ID, "Methodology")

        assert "Methodology" in result.excluded_types
        event_repo.record.assert_awaited_once()
        args, _ = event_repo.record.call_args
        assert args[2] == {"op": "delete", "type_name": "Methodology"}

    @pytest.mark.asyncio
    async def test_delete_is_idempotent_on_replay(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ) -> None:
        _bind_state(schema_repo, _make_schema(excluded=["Methodology"]))

        result = await service.delete_type(NOTEBOOK_ID, "Methodology")

        assert result.excluded_types == ["Methodology"]  # no duplicate
        event_repo.record.assert_not_awaited()
        schema_repo.upsert.assert_not_awaited()


# ---------------------------------------------------------------------------
# Cross-cutting: missing schema row raises
# ---------------------------------------------------------------------------


class TestMissingSchemaRow:
    @pytest.mark.asyncio
    async def test_every_op_raises_when_schema_row_absent(
        self,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ) -> None:
        schema_repo.get_by_notebook = AsyncMock(return_value=None)
        service = SchemaEditService(schema_repo=schema_repo, event_repo=event_repo)

        with pytest.raises(NotebookSchemaNotFoundError):
            await service.accept_extension(NOTEBOOK_ID, "X")
        with pytest.raises(NotebookSchemaNotFoundError):
            await service.reject_extension(NOTEBOOK_ID, "X")
        with pytest.raises(NotebookSchemaNotFoundError):
            await service.rename_type(NOTEBOOK_ID, "X", "Y")
        with pytest.raises(NotebookSchemaNotFoundError):
            await service.merge_types(NOTEBOOK_ID, ["X", "Y"], "Z")
        with pytest.raises(NotebookSchemaNotFoundError):
            await service.split_type(
                NOTEBOOK_ID, "X", ["A", "B"], "criterion"
            )
        with pytest.raises(NotebookSchemaNotFoundError):
            await service.delete_type(NOTEBOOK_ID, "X")

        event_repo.record.assert_not_awaited()
