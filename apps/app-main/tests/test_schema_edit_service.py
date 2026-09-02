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


# ---------------------------------------------------------------------------
# reparent_type (Track N.4d.3)
# ---------------------------------------------------------------------------


class TestReparentType:
    """The write half of an accepted BROADER_THAN placement.

    The op only RECORDS the curator's decision; what makes it take effect is
    `ontology_manager.schema_projection`, tested against the real vocabulary in
    `packages/ontology-manager/tests/test_schema_projection.py`. The split is
    deliberate — a service that re-derived the placement at write time could
    disagree with what the curator was shown.
    """

    @pytest.mark.asyncio
    async def test_each_moved_type_gets_its_own_entry_and_one_event(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ):
        _bind_state(schema_repo, _make_schema())

        updated = await service.reparent_type(
            NOTEBOOK_ID, ["Article", "Report"], "Publication"
        )

        entries = [e for e in updated.accepted_extensions if e.get("op") == "reparent"]
        assert [e["type_name"] for e in entries] == ["Article", "Report"]
        assert {e["new_parent"] for e in entries} == {"Publication"}
        # `parent_type` mirrors `new_parent` so the existing extension tooling
        # (TTL export, schema browser) renders the entry without learning a new
        # field; `new_parent` stays authoritative.
        assert {e["parent_type"] for e in entries} == {"Publication"}

        assert event_repo.record.await_count == 1
        _nb, _type, payload = event_repo.record.await_args.args
        assert payload == {
            "op": "reparent",
            "type_names": ["Article", "Report"],
            "new_parent": "Publication",
        }

    @pytest.mark.asyncio
    async def test_re_running_the_same_move_changes_nothing(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ):
        _bind_state(schema_repo, _make_schema())

        first = await service.reparent_type(NOTEBOOK_ID, ["Article"], "Publication")
        before = list(first.accepted_extensions)
        event_repo.record.reset_mock()

        second = await service.reparent_type(NOTEBOOK_ID, ["Article"], "Publication")

        assert second.accepted_extensions == before
        assert event_repo.record.await_count == 0

    @pytest.mark.asyncio
    async def test_only_the_types_not_already_moved_are_recorded(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ):
        """A curator accepting an overlapping placement must not double-record —
        and the event must name what THIS call moved, not what it was asked to.
        """
        _bind_state(schema_repo, _make_schema())
        await service.reparent_type(NOTEBOOK_ID, ["Article"], "Publication")
        event_repo.record.reset_mock()

        updated = await service.reparent_type(
            NOTEBOOK_ID, ["Article", "Report"], "Publication"
        )

        entries = [e for e in updated.accepted_extensions if e.get("op") == "reparent"]
        assert [e["type_name"] for e in entries] == ["Article", "Report"]
        assert event_repo.record.await_args.args[2]["type_names"] == ["Report"]

    @pytest.mark.asyncio
    async def test_moving_a_type_again_appends_rather_than_rewrites(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
        event_repo: AsyncMock,
    ):
        """The earlier decision stays in the audit trail; the projection applies
        entries in order, so the LATEST parent is the one that takes effect.
        """
        _bind_state(schema_repo, _make_schema())
        await service.reparent_type(NOTEBOOK_ID, ["Article"], "Publication")

        updated = await service.reparent_type(NOTEBOOK_ID, ["Article"], "Report")

        entries = [e for e in updated.accepted_extensions if e.get("op") == "reparent"]
        assert [e["new_parent"] for e in entries] == ["Publication", "Report"]

    @pytest.mark.asyncio
    async def test_duplicate_names_in_one_call_are_recorded_once(
        self,
        service: SchemaEditService,
        schema_repo: AsyncMock,
    ):
        _bind_state(schema_repo, _make_schema())
        updated = await service.reparent_type(
            NOTEBOOK_ID, ["Article", "article", " Article "], "Publication"
        )
        entries = [e for e in updated.accepted_extensions if e.get("op") == "reparent"]
        assert len(entries) == 1

    @pytest.mark.asyncio
    async def test_a_type_cannot_be_moved_under_itself(
        self, service: SchemaEditService, schema_repo: AsyncMock
    ):
        _bind_state(schema_repo, _make_schema())
        with pytest.raises(ValueError, match="under itself"):
            await service.reparent_type(NOTEBOOK_ID, ["Article"], "article")

    @pytest.mark.asyncio
    async def test_a_blank_parent_is_rejected(
        self, service: SchemaEditService, schema_repo: AsyncMock
    ):
        _bind_state(schema_repo, _make_schema())
        with pytest.raises(ValueError, match="new_parent"):
            await service.reparent_type(NOTEBOOK_ID, ["Article"], "   ")

    @pytest.mark.asyncio
    async def test_no_usable_type_name_is_rejected(
        self, service: SchemaEditService, schema_repo: AsyncMock
    ):
        _bind_state(schema_repo, _make_schema())
        with pytest.raises(ValueError, match="at least one type name"):
            await service.reparent_type(NOTEBOOK_ID, ["", "  "], "Publication")

    @pytest.mark.asyncio
    async def test_a_reparent_does_not_make_an_unknown_extension_look_accepted(
        self, service: SchemaEditService, schema_repo: AsyncMock
    ):
        """A re-parent carries the moved type's name in `type_name` too, so an
        unguarded scan in `accept_extension` reads it as "already accepted" and
        turns a genuinely unknown extension into a silent no-op.
        """
        _bind_state(schema_repo, _make_schema())
        await service.reparent_type(NOTEBOOK_ID, ["Article"], "Publication")

        with pytest.raises(UnknownExtensionError):
            await service.accept_extension(NOTEBOOK_ID, "Article")

    @pytest.mark.asyncio
    async def test_a_missing_schema_row_raises(
        self, service: SchemaEditService, schema_repo: AsyncMock
    ):
        schema_repo.get_by_notebook = AsyncMock(return_value=None)
        with pytest.raises(NotebookSchemaNotFoundError):
            await service.reparent_type(NOTEBOOK_ID, ["Article"], "Publication")


def test_every_module_spells_the_reparent_discriminator_the_same_way():
    """N.4d.3 — six places decide what a re-parent entry is.

    Each spells the discriminator locally, on purpose: the router must not import
    a service symbol, and `ontology_manager` / `ontology_extraction` cannot import
    from `app_main` at all. The cost of that is silent divergence — a filter that
    quietly stops filtering, with every test still green because each side agrees
    with itself. This pins them together.
    """
    from ontology_manager.schema_projection import REPARENT_OP as projection_op

    from app_main.api.routers.schemas import _REPARENT_OP as router_op
    from app_main.services.entity_extraction_service import (
        _REPARENT_OP as extraction_op,
    )
    from app_main.services.schema_edit_service import REPARENT_OP as service_op

    assert service_op == router_op == extraction_op == projection_op == "reparent"

    # The two remaining sites use a bare literal because they filter a raw dict
    # in a comprehension; assert the literal is present in each source rather
    # than leaving them out of the pin entirely.
    from pathlib import Path

    import app_main.api.routers.sources_processing as sources_processing
    import ontology_extraction.prompts.pass2 as pass2

    for module in (pass2, sources_processing):
        source = Path(module.__file__).read_text(encoding="utf-8")
        assert f'"{service_op}"' in source, (
            f"{module.__name__} no longer mentions the reparent discriminator"
        )
