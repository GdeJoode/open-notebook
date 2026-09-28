"""The notebook's Pass-1 proposal queue: accept into the shared ontology, reject.

OE.5 retired the per-notebook edit ops (rename, merge, split, re-parent,
delete). Accepting a proposal publishes it to a shared ontology through the
ontology editor's gate (``OntologyEditService.publish_new_entity_type``).

* Unit tests, against AsyncMock'd repositories and a fake editor that keeps the
  real contract (a refusal raises ``OntologyEditError`` and stores nothing):
  what the queue does around a publish.
* ``default_ontology`` against the shipped ontologies.
* Against a real SurrealDB, through the registry's database-first read: an
  accepted proposal types as published; a blocking candidate and an open draft
  are refused, and nothing is stored — no version, no draft, the queue as it
  was.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest
from app_main.services.ontology_edit_service import OntologyEditError
from app_main.services.schema_edit_service import (
    FALLBACK_ONTOLOGY,
    NotebookSchemaNotFoundError,
    SchemaEditService,
    UnknownExtensionError,
    default_ontology,
)
from shared.models import NotebookSchema

NOTEBOOK_ID = "notebook:edit-fixture"


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
def editor() -> AsyncMock:
    """The editor's contract: returns the publish result, or raises
    ``OntologyEditError`` having stored nothing."""
    ed = AsyncMock()
    ed.publish_new_entity_type = AsyncMock(
        return_value={"name": "scholarly", "version": "1.0+t", "warnings": [],
                      "retype": {"changes": 0}}
    )
    return ed


@pytest.fixture
def service(schema_repo, event_repo, editor) -> SchemaEditService:
    return SchemaEditService(schema_repo=schema_repo, event_repo=event_repo,
                             ontology_edit=editor)


def _bind_state(schema_repo: AsyncMock, initial: NotebookSchema) -> Dict[str, NotebookSchema]:
    """get_by_notebook reflects the last upsert, as the database does."""
    state: Dict[str, NotebookSchema] = {"current": initial}

    async def _get_current(_notebook_id: str) -> NotebookSchema:
        return state["current"].model_copy(deep=True)

    async def _upsert(schema: NotebookSchema) -> str:
        state["current"] = schema.model_copy(deep=True)
        return f"notebook_schema:{NOTEBOOK_ID}"

    schema_repo.get_by_notebook = AsyncMock(side_effect=_get_current)
    schema_repo.upsert = AsyncMock(side_effect=_upsert)
    return state


# ---------------------------------------------------------------------------
# accept_extension
# ---------------------------------------------------------------------------


class TestAcceptExtension:
    @pytest.mark.asyncio
    async def test_accept_publishes_then_moves_the_proposal_out_of_the_queue(
        self, service, schema_repo, event_repo, editor
    ) -> None:
        _bind_state(schema_repo, _make_schema(pending=[
            {"extension_id": "ext-1", "type_name": "Cohort", "parent_type": "Group",
             "rationale": "a study group"},
            {"extension_id": "ext-2", "type_name": "Methodology"},
        ]))

        result, published = await service.accept_extension(NOTEBOOK_ID, "Cohort", "scholarly")

        editor.publish_new_entity_type.assert_awaited_once()
        args, kwargs = editor.publish_new_entity_type.await_args
        assert args[:3] == ("scholarly", "Cohort",
                            {"parent_type": "Group", "description": "a study group"})
        assert published["version"] == "1.0+t"
        assert [e["type_name"] for e in result.pending_extensions] == ["Methodology"]
        [accepted] = result.accepted_extensions
        assert (accepted["type_name"], accepted["published_to"], accepted["published_version"]) \
            == ("Cohort", "scholarly", "1.0+t")
        event_repo.record.assert_awaited_once()
        payload = event_repo.record.call_args.args[2]
        assert payload["op"] == "accept_extension"
        assert (payload["ontology"], payload["version"]) == ("scholarly", "1.0+t")

    @pytest.mark.asyncio
    async def test_the_curators_parent_replaces_the_proposed_one(
        self, service, schema_repo, editor
    ) -> None:
        _bind_state(schema_repo, _make_schema(pending=[
            {"type_name": "Cohort", "parent_type": "Group"}]))
        result, _ = await service.accept_extension(NOTEBOOK_ID, "Cohort", "scholarly", "Grant")
        assert editor.publish_new_entity_type.await_args.args[2]["parent_type"] == "Grant"
        assert result.accepted_extensions[0]["parent_type"] == "Grant"
        # An empty choice is "no parent", not "the proposed one".
        _bind_state(schema_repo, _make_schema(pending=[
            {"type_name": "Cohort", "parent_type": "Group"}]))
        await service.accept_extension(NOTEBOOK_ID, "Cohort", "scholarly", "")
        assert editor.publish_new_entity_type.await_args.args[2]["parent_type"] is None

    @pytest.mark.asyncio
    async def test_a_refused_publish_leaves_the_queue_as_it_was(
        self, service, schema_repo, event_repo, editor
    ) -> None:
        state = _bind_state(schema_repo, _make_schema(pending=[{"type_name": "Cohort"}]))
        editor.publish_new_entity_type.side_effect = OntologyEditError(
            422, {"message": "the new type has blocking findings", "findings": [{}]})
        with pytest.raises(OntologyEditError) as refused:
            await service.accept_extension(NOTEBOOK_ID, "Cohort", "scholarly")
        assert refused.value.status == 422
        schema_repo.upsert.assert_not_awaited()
        event_repo.record.assert_not_awaited()
        assert [e["type_name"] for e in state["current"].pending_extensions] == ["Cohort"]

    @pytest.mark.asyncio
    async def test_accept_is_idempotent_when_already_accepted(
        self, service, schema_repo, event_repo, editor
    ) -> None:
        _bind_state(schema_repo, _make_schema(
            accepted=[{"extension_id": "ext-1", "type_name": "Cohort"}]))
        result, published = await service.accept_extension(NOTEBOOK_ID, "Cohort", "scholarly")
        assert published is None
        assert len(result.accepted_extensions) == 1
        editor.publish_new_entity_type.assert_not_awaited()
        event_repo.record.assert_not_awaited()
        schema_repo.upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_accept_raises_when_extension_unknown(
        self, service, schema_repo, event_repo, editor
    ) -> None:
        _bind_state(schema_repo, _make_schema())
        with pytest.raises(UnknownExtensionError):
            await service.accept_extension(NOTEBOOK_ID, "Nope", "scholarly")
        editor.publish_new_entity_type.assert_not_awaited()
        event_repo.record.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_legacy_reparent_does_not_make_an_unknown_proposal_look_accepted(
        self, service, schema_repo, editor
    ) -> None:
        """A re-parent row stored before OE.5 carries the moved type's name."""
        _bind_state(schema_repo, _make_schema(accepted=[
            {"op": "reparent", "type_name": "Article", "new_parent": "Publication"}]))
        with pytest.raises(UnknownExtensionError):
            await service.accept_extension(NOTEBOOK_ID, "Article", "scholarly")


class TestRejectExtension:
    @pytest.mark.asyncio
    async def test_reject_drops_pending_and_emits_event(
        self, service, schema_repo, event_repo
    ) -> None:
        _bind_state(schema_repo, _make_schema(
            pending=[{"extension_id": "ext-1", "type_name": "Cohort"}]))
        result = await service.reject_extension(NOTEBOOK_ID, "Cohort")
        assert result.pending_extensions == []
        event_repo.record.assert_awaited_once()
        assert event_repo.record.call_args.args[2] == {
            "op": "reject_extension", "type_name": "Cohort"}

    @pytest.mark.asyncio
    async def test_reject_is_idempotent_when_already_gone(
        self, service, schema_repo, event_repo
    ) -> None:
        _bind_state(schema_repo, _make_schema())
        result = await service.reject_extension(NOTEBOOK_ID, "Cohort")
        assert result.pending_extensions == []
        event_repo.record.assert_not_awaited()
        schema_repo.upsert.assert_not_awaited()


class TestMissingSchemaRow:
    @pytest.mark.asyncio
    async def test_every_op_raises_when_schema_row_absent(
        self, service, schema_repo, event_repo, editor
    ) -> None:
        schema_repo.get_by_notebook = AsyncMock(return_value=None)
        with pytest.raises(NotebookSchemaNotFoundError):
            await service.accept_extension(NOTEBOOK_ID, "X", "scholarly")
        with pytest.raises(NotebookSchemaNotFoundError):
            await service.preview_accept(NOTEBOOK_ID, "X", "scholarly")
        with pytest.raises(NotebookSchemaNotFoundError):
            await service.reject_extension(NOTEBOOK_ID, "X")
        editor.publish_new_entity_type.assert_not_awaited()
        event_repo.record.assert_not_awaited()


def test_the_per_notebook_edit_ops_are_gone():
    for op in ("rename_type", "merge_types", "split_type", "reparent_type", "delete_type"):
        assert not hasattr(SchemaEditService, op), op


# ---------------------------------------------------------------------------
# default_ontology
# ---------------------------------------------------------------------------


class TestDefaultOntology:
    @pytest.fixture
    async def own(self):
        from app_main.services.ontology_preview_service import load_own_ontologies

        return await load_own_ontologies()

    @pytest.mark.asyncio
    async def test_a_parent_goes_to_the_ontology_that_introduces_it(self, own):
        defining = [n for n, o in own.items() if "Organization" in o.entity_types]
        assert len(defining) > 1 and "schema_core" in defining, defining
        assert default_ontology(own, "Organization") == "schema_core"
        assert default_ontology(own, "organization") == "schema_core"

    @pytest.mark.asyncio
    async def test_a_type_only_a_domain_ontology_defines(self, own):
        [only] = [n for n, o in own.items() if "RegioDeal" in o.entity_types]
        assert default_ontology(own, "RegioDeal") == only

    @pytest.mark.asyncio
    async def test_no_parent_or_an_unknown_one_goes_to_the_fallback(self, own):
        assert FALLBACK_ONTOLOGY in own
        assert default_ontology(own, None) == FALLBACK_ONTOLOGY
        assert default_ontology(own, "Nergens") == FALLBACK_ONTOLOGY


# ---------------------------------------------------------------------------
# PC.5: a rejection is a "no" that survives the next document
# ---------------------------------------------------------------------------


class TestARejectedTypeStaysRejectedThroughTheCuratorsOwnPath:
    """PC.5 (migration 84), on the path a curator's Reject button takes."""

    @pytest.mark.asyncio
    async def test_a_rejection_is_recorded_so_the_type_does_not_return(self):
        from surrealdb_service.repositories.notebook_schema import (
            NotebookSchemaRepository,
        )

        schema = _make_schema(pending=[])
        schema_repo = NotebookSchemaRepository()
        schema_repo.get_by_notebook = AsyncMock(return_value=schema)
        schema_repo.upsert = AsyncMock(return_value="notebook_schema:x")
        service = SchemaEditService(schema_repo, AsyncMock(), ontology_edit=AsyncMock())

        assert await schema_repo.merge_pending_extensions(
            NOTEBOOK_ID, [{"type_name": "Method"}]
        ) == 1
        after = await service.reject_extension(NOTEBOOK_ID, "Method")
        assert after.pending_extensions == []
        assert after.rejected_extensions == ["Method"]
        assert await schema_repo.merge_pending_extensions(
            NOTEBOOK_ID, [{"type_name": "METHOD"}]
        ) == 0
        assert schema.pending_extensions == []
        persisted = schema_repo.upsert.await_args_list[-1].args[0]
        assert persisted.rejected_extensions == ["Method"]

    @pytest.mark.asyncio
    async def test_rejecting_twice_records_once(self):
        from surrealdb_service.repositories.notebook_schema import (
            NotebookSchemaRepository,
        )

        schema = _make_schema(pending=[{"type_name": "Method"}])
        schema_repo = NotebookSchemaRepository()
        schema_repo.get_by_notebook = AsyncMock(return_value=schema)
        schema_repo.upsert = AsyncMock(return_value="notebook_schema:x")
        service = SchemaEditService(schema_repo, AsyncMock(), ontology_edit=AsyncMock())
        await service.reject_extension(NOTEBOOK_ID, "Method")
        await service.reject_extension(NOTEBOOK_ID, "Method")
        assert schema.rejected_extensions == ["Method"]


# ---------------------------------------------------------------------------
# Against a real SurrealDB
# ---------------------------------------------------------------------------


@pytest.fixture
async def live(live_surrealdb, monkeypatch):
    """The real editor and repositories on a real SurrealDB, with the registry
    reading the database first (the unit conftest keeps it on the YAML)."""
    import surrealdb_service.connection as conn
    from app_main.services.ontology_edit_service import OntologyEditService
    from ontology_manager.registry import OntologyRegistry, get_ontology_registry
    from surrealdb_service.repositories.notebook_event import NotebookEventRepository
    from surrealdb_service.repositories.notebook_schema import NotebookSchemaRepository

    from tests.test_ontology_edit_service import _LIST_ONTOLOGIES, _LOAD_FROM_DB

    monkeypatch.setattr(OntologyRegistry, "_load_from_db", _LOAD_FROM_DB)
    monkeypatch.setattr(OntologyRegistry, "list_ontologies", _LIST_ONTOLOGIES)
    original = conn._pool
    conn._pool = conn.ConnectionPool(config=live_surrealdb)
    registry = get_ontology_registry()
    registry.clear_cache()
    clean = ("DELETE ontology_draft; DELETE ontology_version; DELETE notebook_schema; "
             "DELETE notebook_event; DELETE notebook;")
    await conn.execute_query(clean, {})
    await conn.execute_query("CREATE notebook:oe5 CONTENT {name: 'oe5', description: ''};", {})
    repo = NotebookSchemaRepository()
    await repo.upsert(NotebookSchema(
        notebook="notebook:oe5", base_ontology="",
        pending_extensions=[
            {"extension_id": "pass1::streekfonds", "type_name": "Streekfonds",
             "parent_type": "Grant", "rationale": "a regional fund"},
            {"extension_id": "pass1::zwerfbegrip", "type_name": "Zwerfbegrip",
             "parent_type": "Nergens"},
        ],
    ))
    editor = OntologyEditService()
    yield SchemaEditService(repo, NotebookEventRepository(), ontology_edit=editor), editor, repo
    await conn.execute_query(clean, {})
    registry.clear_cache()
    await conn._pool.close_all()
    conn._pool = original


async def _typed(label):
    from app_main.services.entity_persistence_service import (
        _resolve_entity_type,
        load_typing_ontologies,
        typing_vocabulary,
    )

    return _resolve_entity_type(
        label, typing_vocabulary(None, await load_typing_ontologies())).entity_type


async def _stored():
    import surrealdb_service.connection as conn

    versions = await conn.execute_query("SELECT name, version FROM ontology_version;", {})
    drafts = await conn.execute_query("SELECT name FROM ontology_draft;", {})
    return versions, drafts


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_an_accepted_proposal_is_published_and_types_everywhere(live):
    service, _editor, repo = live
    assert await _typed("Streekfonds") == "concept"
    target = await service.accept_target("notebook:oe5", "Streekfonds")
    assert target["parent_type"] == "Grant"
    assert "deals" in target["ontologies"]
    assert target["default_ontology"] in target["ontologies"]
    preview = await service.preview_accept("notebook:oe5", "Streekfonds", "deals")
    assert preview["blocking"] is False and preview["draft_open"] is False
    assert await _stored() == ([], [])  # the preview wrote nothing

    row, published = await service.accept_extension("notebook:oe5", "Streekfonds", "deals")
    assert await _typed("Streekfonds") == "grant"
    assert [e["type_name"] for e in row.pending_extensions] == ["Zwerfbegrip"]
    assert row.accepted_extensions[-1]["published_version"] == published["version"]
    from ontology_manager.registry import get_ontology_registry

    active = [v["version"] for v in await get_ontology_registry().list_versions("deals")
              if v["is_active"]]
    assert active == [published["version"]]


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_a_blocking_proposal_is_refused_and_nothing_is_stored(live):
    service, _editor, repo = live
    preview = await service.preview_accept("notebook:oe5", "Zwerfbegrip", "deals")
    assert preview["blocking"] is True
    with pytest.raises(OntologyEditError) as refused:
        await service.accept_extension("notebook:oe5", "Zwerfbegrip", "deals")
    assert refused.value.status == 422
    assert any(f["severity"] == "block" for f in refused.value.detail["findings"])
    assert await _stored() == ([], [])
    row = await repo.get_by_notebook("notebook:oe5")
    assert [e["type_name"] for e in row.pending_extensions] == ["Streekfonds", "Zwerfbegrip"]
    assert row.accepted_extensions == []


@pytest.mark.requires_docker
@pytest.mark.asyncio
async def test_an_open_draft_refuses_the_accept_and_nothing_is_stored(live):
    service, editor, repo = live
    await editor.put_entity_type("deals", "Iets", {"parent_type": "Grant"})
    assert (await service.preview_accept("notebook:oe5", "Streekfonds", "deals"))["draft_open"]
    with pytest.raises(OntologyEditError) as refused:
        await service.accept_extension("notebook:oe5", "Streekfonds", "deals")
    assert refused.value.status == 409
    versions, drafts = await _stored()
    assert versions == [] and len(drafts) == 1
    assert "Streekfonds" not in (await editor.get_draft("deals"))["ontology"].entity_types
    row = await repo.get_by_notebook("notebook:oe5")
    assert [e["type_name"] for e in row.pending_extensions] == ["Streekfonds", "Zwerfbegrip"]
