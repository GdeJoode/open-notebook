"""Track N.4d.3 — the read half of D-N4-12, shown to a curator.

Run against the REAL shipped ontologies: the question these answer is "does a
placement over our own vocabulary say something true", and a stub of the loader
would make every assertion about the stub. The judge is stubbed, because the
question there is what the service does with a reply, not what a model says.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest
from app_main.services.type_placement_service import (
    DECIDED,
    NOT_ASKED,
    REFUSED,
    UNAVAILABLE,
    TypePlacementService,
)
from ontology_manager import get_ontology_manager
from shared.models.notebook_schema import NotebookSchema


def _loader():
    return get_ontology_manager().get_ontology


def _schema(accepted=None, base="deals"):
    return NotebookSchema(
        notebook="notebook:placement",
        base_ontology=base,
        accepted_extensions=list(accepted or []),
    )


def _service(caller=None):
    factory = None
    if caller is not None:
        factory = AsyncMock(return_value=caller)
    return TypePlacementService(
        ontology_loader=_loader(), llm_caller_factory=factory
    )


class TestTheVocabularyItJudgesAgainst:
    @pytest.mark.asyncio
    async def test_it_is_the_notebooks_forced_set(self):
        """base + affinity bundle, the same composition
        `_apply_notebook_schema_default` forces onto every extraction here.
        """
        report = await _service().placement_for(_schema(), "Tranche", "Deal")
        assert "deals" in report.vocabulary
        assert "policy_themes" in report.vocabulary

    @pytest.mark.asyncio
    async def test_a_schema_named_on_an_accepted_extension_is_included(self):
        report = await _service().placement_for(
            _schema([{"type_name": "X", "schema_name": "scholarly"}]),
            "Tranche",
            "Deal",
        )
        assert "scholarly" in report.vocabulary

    @pytest.mark.asyncio
    async def test_an_unknown_base_does_not_crash_the_report(self):
        report = await _service().placement_for(
            _schema(base="does_not_exist"), "Tranche", "Deal"
        )
        assert report.vocabulary == ()
        assert report.reason_code == "no_applied_schemas"

    @pytest.mark.asyncio
    async def test_a_type_accepted_a_moment_ago_is_already_in_the_vocabulary(self):
        """The forced set is PROJECTED with the notebook's own accepted edits, so
        a placement can name a parent the curator accepted on the previous click.
        Without the projection this reports PARENT_UNKNOWN.
        """
        report = await _service().placement_for(
            _schema([{"extension_id": "e1", "type_name": "Tranche", "parent_type": "Deal"}]),
            "Deeltranche",
            "Tranche",
        )
        assert report.verdict == "PLACED"
        assert report.parent == "Tranche"


class TestWhereItCanDisagreeWithTheRuntimeSet:
    """D-N4-13's two limitations, pinned as measurements rather than prose.

    The first draft of that decision claimed a placement can never contradict the
    runtime verdict, arguing from the forced set being a subset. Both halves of
    that were disproved, so both are asserted here — if either ever stops being
    true, the decision text should change with it.
    """

    @pytest.mark.asyncio
    async def test_adding_a_schema_can_turn_placed_into_duplicate(self):
        """Verdicts are NOT monotone in the applied set, which is what made the
        superset argument invalid: it licenses only monotone conclusions.
        """
        from ontology_manager.type_placement import place_proposed_type

        report = await _service().placement_for(_schema(), "ScholarlyArticle", "Deal")
        assert report.verdict == "PLACED"
        assert "scholarly" not in report.vocabulary

        manager = get_ontology_manager()
        runtime = [
            await manager.get_ontology(name)
            for name in ("deals", "policy_themes", "scholarly")
        ]
        assert place_proposed_type("ScholarlyArticle", "Deal", runtime).verdict == (
            "DUPLICATE"
        )

    @pytest.mark.asyncio
    async def test_an_empty_base_ontology_still_produces_a_report(self):
        """`_apply_notebook_schema_default` is gated on a truthy `base_ontology`,
        which the Regio-Deal notebooks leave empty — so there the forced set is
        not a subset of anything the runtime applies. This asserts THIS service's
        half: it still composes a set from the accepted extensions' schemas and
        places against it.

        The runtime half — that the gate really does force nothing — lives in
        `test_entity_extraction_service.py::TestRunMultiSchemaBody::
        test_an_empty_base_ontology_forces_no_schema`, which exercises the branch
        through `run_extraction`. A review measured that asserting it here read
        `service._apply_notebook_schema_default is not None`, which is true of any
        object carrying that attribute and could not fail; removing the gate left
        the whole 1670-test suite green.
        """
        notebook_schema = NotebookSchema(
            notebook="notebook:empty",
            base_ontology="",
            accepted_extensions=[{"type_name": "X", "schema_name": "scholarly"}],
        )
        report = await _service().placement_for(
            notebook_schema, "Preprint", "ScholarlyArticle"
        )
        assert report.vocabulary == ("scholarly",)
        assert report.verdict == "PLACED"


class TestTheDeterministicHalf:
    @pytest.mark.asyncio
    async def test_a_resolved_parent_places_the_type_and_bounds_its_candidates(self):
        report = await _service().placement_for(_schema(), "Tranche", "Deal")
        assert report.verdict == "PLACED"
        assert report.parent == "Deal"
        assert report.candidates, "a placed type must be offered its siblings"
        assert report.type_name not in report.candidates

    @pytest.mark.asyncio
    async def test_a_name_that_already_exists_is_a_duplicate_not_a_placement(self):
        report = await _service().placement_for(_schema(), "RegioDeal", "Deal")
        assert report.verdict == "DUPLICATE"
        assert report.candidates == ()

    @pytest.mark.asyncio
    async def test_an_unresolvable_parent_says_exactly_that(self):
        report = await _service().placement_for(_schema(), "Tranche", "NoSuchParent")
        assert report.verdict == "PARENT_UNKNOWN"
        assert report.candidates == ()


class TestTheJudge:
    @pytest.mark.asyncio
    async def test_no_caller_means_nobody_was_asked(self):
        report = await _service().placement_for(_schema(), "Tranche", "Deal")
        assert report.candidates and report.judge_status == NOT_ASKED
        assert report.judged is False
        assert report.selected == ()

    @pytest.mark.asyncio
    async def test_the_four_judge_states_are_distinguishable(self):
        """Three of the four carry an empty selection, so emptiness cannot be the
        discriminator. A review found this service reporting a REFUSED reply as
        "the judge moved nothing", which is what N.4d.4's gap loop would gate on.
        """
        async def raising(_s, _p, _m):
            raise RuntimeError("no route")

        async def refused(_s, _p, _m):
            return '["0"]'  # a top-level array; the parser will not use it

        async def moved_nothing(_s, _p, _m):
            return '{"move_under_proposal": []}'

        async def decided(_s, _p, _m):
            return '{"move_under_proposal": ["0"]}'

        cases = {
            NOT_ASKED: None,
            UNAVAILABLE: raising,
            REFUSED: refused,
            DECIDED: moved_nothing,
        }
        seen = {}
        for expected, caller in cases.items():
            report = await _service(caller).placement_for(_schema(), "Tranche", "Deal")
            assert report.candidates, "the placement must have asked something"
            assert report.judge_status == expected
            seen[expected] = report.selected

        # All four produced the same empty selection except the last, which is
        # the point: the status is the only thing that separates them.
        assert all(selected == () for selected in seen.values())

        chosen = await _service(decided).placement_for(_schema(), "Tranche", "Deal")
        assert chosen.judge_status == DECIDED and chosen.selected

    @pytest.mark.asyncio
    async def test_a_selection_names_the_types_it_chose(self):
        async def caller(_system, prompt, _model):
            # Choose the first candidate by its positional id, the way the
            # judge's own contract requires.
            assert "0." in prompt or "0)" in prompt or "[0]" in prompt or "0:" in prompt
            return '{"move_under_proposal": ["0"], "reasoning": "test"}'

        report = await _service(caller).placement_for(_schema(), "Tranche", "Deal")
        assert report.judged is True
        assert len(report.selected) == 1
        assert report.selected[0] in report.candidates

    @pytest.mark.asyncio
    async def test_a_judge_that_moves_nothing_is_still_a_decision(self):
        async def caller(_system, _prompt, _model):
            return '{"move_under_proposal": [], "reasoning": "none apply"}'

        report = await _service(caller).placement_for(_schema(), "Tranche", "Deal")
        assert report.judge_status == DECIDED
        assert report.judged is True
        assert report.selected == ()

    @pytest.mark.asyncio
    async def test_a_reply_the_parser_refuses_is_not_a_decision(self):
        async def caller(_system, _prompt, _model):
            return "I think Woondeal belongs under it."

        report = await _service(caller).placement_for(_schema(), "Tranche", "Deal")
        assert report.judge_status == REFUSED
        assert report.judged is False
        assert report.selected == ()
        # The refusal is explained rather than left as a bare flag.
        assert report.judge_evidence

    @pytest.mark.asyncio
    async def test_a_model_outage_leaves_the_deterministic_half_standing(self):
        async def caller(_system, _prompt, _model):
            raise RuntimeError("no route to model")

        report = await _service(caller).placement_for(_schema(), "Tranche", "Deal")
        assert report.judge_status == UNAVAILABLE
        assert report.judged is False
        assert report.selected == ()
        assert report.verdict == "PLACED"
        assert report.candidates

    @pytest.mark.asyncio
    async def test_the_judge_cannot_widen_the_candidate_set(self):
        """The fence N.4d.2 shipped, asserted through this service rather than
        assumed to be inherited by importing the parser.
        """
        async def caller(_system, _prompt, _model):
            return '{"move_under_proposal": ["0", "999", "Gemeente"]}'

        report = await _service(caller).placement_for(_schema(), "Tranche", "Deal")
        assert set(report.selected) <= set(report.candidates)
        assert len(report.selected) == 1

    @pytest.mark.asyncio
    async def test_an_unplaced_type_is_never_judged(self):
        """No resolved parent means no sibling set, so there is nothing to ask
        about — and the model must not be called at all.
        """
        caller = AsyncMock()
        report = await _service(caller).placement_for(
            _schema(), "Tranche", "NoSuchParent"
        )
        assert report.judge_status == NOT_ASKED
        caller.assert_not_awaited()
