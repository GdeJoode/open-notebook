"""Track PC.1 — the queue writers, without a database.

The roundtrip tests for these two methods need the testcontainers harness and
skip wherever docker is unavailable. A review measured the cost of leaving it
there: both methods could be reduced to no-ops — restoring the exact defect PC.1
exists to fix, an empty `pending_extensions` and a `coverage_pct` of 0.0 — with
every runnable test in both packages still green.

So they are exercised here against the REAL repository with only its two
collaborator calls mocked, the idiom `test_search_hybrid_fusion.py` already uses.
The dedup and validation logic is pure; it needs no container.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock

import pytest
from shared.models import NotebookSchema
from surrealdb_service.repositories.notebook_schema import NotebookSchemaRepository

NOTEBOOK = "notebook:pc1"


def _repo(schema: Optional[NotebookSchema]) -> NotebookSchemaRepository:
    """The real repository, with only its DB round-trips stubbed."""
    repo = NotebookSchemaRepository()
    repo.get_by_notebook = AsyncMock(return_value=schema)
    repo.upsert = AsyncMock(return_value="notebook_schema:pc1")
    return repo


def _schema(base_ontology: str = "deals", **kwargs: Any) -> NotebookSchema:
    return NotebookSchema(notebook=NOTEBOOK, base_ontology=base_ontology, **kwargs)


def _names(repo: NotebookSchemaRepository) -> List[str]:
    """The queue as it was written back."""
    written: NotebookSchema = repo.upsert.await_args.args[0]
    return [e["type_name"] for e in written.pending_extensions]


class TestMergePendingExtensions:
    @pytest.mark.asyncio
    async def test_proposals_are_queued_with_deterministic_ids(self):
        repo = _repo(_schema())
        added = await repo.merge_pending_extensions(
            NOTEBOOK,
            [
                {"type_name": "Method", "parent_type": "Concept"},
                {"type_name": "GrantFundingSource", "parent_type": "Organization"},
            ],
        )
        assert added == 2
        written: NotebookSchema = repo.upsert.await_args.args[0]
        assert [e["extension_id"] for e in written.pending_extensions] == [
            "pass1::method",
            "pass1::grantfundingsource",
        ]

    @pytest.mark.asyncio
    async def test_a_type_already_queued_is_not_queued_twice(self):
        """Proposals are per-DOCUMENT, the queue is per-NOTEBOOK."""
        repo = _repo(_schema(pending_extensions=[{"type_name": "Method"}]))
        assert await repo.merge_pending_extensions(
            NOTEBOOK, [{"type_name": "Method"}]
        ) == 0
        repo.upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_case_differences_do_not_create_a_second_row(self):
        """The model's capitalisation is not stable across documents — the live
        corpus held `Brede Welvaart` and `brede welvaart` as two graph rows for
        exactly this reason.
        """
        repo = _repo(_schema(pending_extensions=[{"type_name": "Method"}]))
        assert await repo.merge_pending_extensions(
            NOTEBOOK, [{"type_name": "  method  "}]
        ) == 0

    @pytest.mark.asyncio
    async def test_an_accepted_type_is_not_re_proposed(self):
        repo = _repo(_schema(accepted_extensions=[{"type_name": "Method"}]))
        assert await repo.merge_pending_extensions(
            NOTEBOOK, [{"type_name": "Method"}, {"type_name": "Tranche"}]
        ) == 1
        assert _names(repo) == ["Tranche"]

    @pytest.mark.asyncio
    async def test_an_excluded_type_is_not_re_proposed(self):
        """`excluded_types` is the curator's explicit soft-delete (B.3b). A
        review measured that without this the deleted type comes straight back
        the next time a document proposes it.
        """
        repo = _repo(_schema(excluded_types=["Method"]))
        assert await repo.merge_pending_extensions(
            NOTEBOOK, [{"type_name": "Method"}]
        ) == 0

    @pytest.mark.asyncio
    async def test_a_rejected_type_does_come_back(self):
        """Asserted so the behaviour is KNOWN, not discovered.

        `reject_pending_extension` drops the row and records nothing — there is
        no `rejected_extensions` field — so a rejected type returns the next time
        a document proposes it. Closing that needs a new field and a migration,
        which is a follow-up (PC.5); this test exists so the gap cannot be
        mistaken for a guarantee.

        Two reviews shaped this test. The first version could not fail at all: it
        merged one proposal into a pristine queue and asserted the count, with no
        rejection anywhere in it. This version walks the sequence — propose,
        reject, propose again — but through `reject_pending_extension`, which the
        second review pointed out has NO production caller: a curator's Reject
        button goes through `SchemaEditService.reject_extension`, keyed on
        `type_name`. So this pins the repository's own behaviour, and the guard
        that follows the curator's actual path lives in app-main, in
        `test_schema_edit_service.py::TestARejectedTypeComesBackThroughTheCuratorsOwnPath`.
        Whichever layer PC.5 records the "no" in, one of the two fails.
        """
        schema = _schema()
        repo = _repo(schema)

        assert await repo.merge_pending_extensions(
            NOTEBOOK, [{"type_name": "Method"}]
        ) == 1
        assert _names(repo) == ["Method"]

        assert await repo.reject_pending_extension(NOTEBOOK, "pass1::method") is True
        assert _names(repo) == []

        # The curator said no. The next document says Method again, and it is
        # back in the queue, because nothing recorded the no.
        assert await repo.merge_pending_extensions(
            NOTEBOOK, [{"type_name": "Method"}]
        ) == 1
        assert _names(repo) == ["Method"]

    # One test per refusal rule, because a review measured that bundling them
    # let two rules ride on a third: the control-character case was written
    # `"Bell\\x07"`, which is the six literal characters `B e l l \\ x 0 7` and
    # is refused by the BACKSLASH rule, so deleting the control-character rule
    # left the whole file green.

    @pytest.mark.asyncio
    async def test_a_name_with_a_slash_is_refused(self):
        """The accept/reject endpoints take the name as a bare path segment with
        no `:path` converter, so a `/` splits the path and the request 404s — the
        row would be queued and then be neither acceptable nor rejectable.
        """
        repo = _repo(_schema())
        added = await repo.merge_pending_extensions(
            NOTEBOOK,
            [
                {"type_name": "Grant/Funding Source"},
                {"type_name": "Perfectly Fine Name"},
            ],
        )
        assert added == 1
        assert _names(repo) == ["Perfectly Fine Name"]

    @pytest.mark.asyncio
    async def test_a_name_with_a_backslash_is_refused(self):
        repo = _repo(_schema())
        added = await repo.merge_pending_extensions(
            NOTEBOOK,
            [
                {"type_name": "Back\\slash"},
                {"type_name": "Perfectly Fine Name"},
            ],
        )
        assert added == 1
        assert _names(repo) == ["Perfectly Fine Name"]

    @pytest.mark.asyncio
    async def test_a_name_with_a_control_character_is_refused(self):
        """A REAL control character — `\\x07` as one byte, not as four source
        characters — and no backslash or slash anywhere in the name, so this case
        can only be carried by the control-character rule itself.
        """
        name = "Bell\x07Type"
        assert "\\" not in name and "/" not in name
        assert any(ord(ch) < 32 for ch in name)

        repo = _repo(_schema())
        added = await repo.merge_pending_extensions(
            NOTEBOOK,
            [
                {"type_name": name},
                {"type_name": "Perfectly Fine Name"},
            ],
        )
        assert added == 1
        assert _names(repo) == ["Perfectly Fine Name"]

    @pytest.mark.asyncio
    async def test_a_name_with_a_delete_character_is_refused(self):
        """`\\x7f` is the other half of the rule (`ord(ch) == 127`), which the
        `< 32` half does not cover.
        """
        name = "Del\x7fType"
        repo = _repo(_schema())
        added = await repo.merge_pending_extensions(
            NOTEBOOK,
            [{"type_name": name}, {"type_name": "Perfectly Fine Name"}],
        )
        assert added == 1
        assert _names(repo) == ["Perfectly Fine Name"]

    @pytest.mark.asyncio
    async def test_the_stored_name_matches_the_key_it_is_deduped_on(self):
        """An un-stripped name with a stripped key would let the row and its own
        dedup key disagree.
        """
        repo = _repo(_schema())
        await repo.merge_pending_extensions(NOTEBOOK, [{"type_name": "  Padded  "}])
        written: NotebookSchema = repo.upsert.await_args.args[0]
        entry = written.pending_extensions[0]
        assert entry["type_name"] == "Padded"
        assert entry["extension_id"] == "pass1::padded"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "batch",
        [
            ["not-a-dict"],
            [None],
            [{"no_type_name": 1}],
            [{"type_name": ""}],
            [{"type_name": "   "}],
        ],
    )
    async def test_unusable_proposals_are_skipped_not_raised(self, batch):
        repo = _repo(_schema())
        assert await repo.merge_pending_extensions(NOTEBOOK, batch) == 0
        repo.upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_nothing_is_written_when_nothing_was_added(self):
        """Vacuity guard for the assertions above: a write DOES happen when there
        is something to add, so `assert_not_awaited` means something.
        """
        repo = _repo(_schema())
        assert await repo.merge_pending_extensions(
            NOTEBOOK, [{"type_name": "Method"}]
        ) == 1
        repo.upsert.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_schema_row_is_reported_not_raised(self):
        repo = _repo(None)
        assert await repo.merge_pending_extensions(
            NOTEBOOK, [{"type_name": "Method"}]
        ) == 0
        repo.upsert.assert_not_awaited()


class TestSetCoveragePct:
    @pytest.mark.asyncio
    async def test_the_value_is_stored(self):
        repo = _repo(_schema())
        assert await repo.set_coverage_pct(NOTEBOOK, 0.72) is True
        written: NotebookSchema = repo.upsert.await_args.args[0]
        assert written.coverage_pct == pytest.approx(0.72)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "value, expected", [(87.0, 1.0), (-3.0, 0.0), (1.0, 1.0), (0.0, 0.0)]
    )
    async def test_out_of_range_values_are_clamped(self, value, expected):
        """The model constrains the field; a Pass-1 run reporting a percentage
        instead of a fraction must not make the write fail after the extraction
        already succeeded.
        """
        repo = _repo(_schema())
        await repo.set_coverage_pct(NOTEBOOK, value)
        written: NotebookSchema = repo.upsert.await_args.args[0]
        assert written.coverage_pct == pytest.approx(expected)

    @pytest.mark.asyncio
    async def test_no_schema_row_is_reported_not_raised(self):
        repo = _repo(None)
        assert await repo.set_coverage_pct(NOTEBOOK, 0.5) is False
        repo.upsert.assert_not_awaited()


class TestEnsureRow:
    """The row that HOLDS the queue, which nothing in production created.

    Measured on the live corpus before this landed: 17 `pass1_results` rows
    carrying 111 proposals across 79 distinct type names, and zero
    `notebook_schema` rows. Every writer was correctly reporting "no row" and
    doing nothing, which is why the queue looked empty rather than broken.
    """

    @pytest.mark.asyncio
    async def test_a_missing_row_is_created_with_the_given_base(self):
        repo = _repo(None)
        assert await repo.ensure_row(NOTEBOOK, "deals") is True
        written: NotebookSchema = repo.upsert.await_args.args[0]
        assert written.notebook == NOTEBOOK
        assert written.base_ontology == "deals"
        assert written.pending_extensions == []

    @pytest.mark.asyncio
    async def test_an_existing_row_is_left_alone(self):
        """It must not reset a curator's base ontology, accepted extensions or
        exclusions on every extraction — the whole point is "create if absent".
        """
        existing = _schema(
            base_ontology="scholarly",
            accepted_extensions=[{"type_name": "Method"}],
            excluded_types=["Noise"],
        )
        repo = _repo(existing)
        assert await repo.ensure_row(NOTEBOOK, "deals") is False
        repo.upsert.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_the_created_row_can_immediately_take_proposals(self):
        """The two halves compose: creating the row is only useful if the merge
        that follows it now finds one.
        """
        state = {"row": None}

        repo = NotebookSchemaRepository()

        async def _get(_notebook_id):
            return state["row"]

        async def _upsert(schema):
            state["row"] = schema
            return "notebook_schema:pc1"

        repo.get_by_notebook = AsyncMock(side_effect=_get)
        repo.upsert = AsyncMock(side_effect=_upsert)

        assert await repo.ensure_row(NOTEBOOK, "deals") is True
        added = await repo.merge_pending_extensions(
            NOTEBOOK, [{"type_name": "Method"}, {"type_name": "Tranche"}]
        )
        assert added == 2
        assert [e["type_name"] for e in state["row"].pending_extensions] == [
            "Method",
            "Tranche",
        ]
