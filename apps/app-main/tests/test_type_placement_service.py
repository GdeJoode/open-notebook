"""Track N.4d.3 — the read half of D-N4-12, shown to a curator.

Run against the REAL shipped ontologies: the question these answer is "does a
placement over our own vocabulary say something true", and a stub of the loader
would make every assertion about the stub. The judge is stubbed, because the
question there is what the service does with a reply, not what a model says.
"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from app_main.services.type_placement_service import TypePlacementService
from shared.models.notebook_schema import NotebookSchema


def _loader():
    from ontology_manager import get_ontology_manager

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
        """Distinct from a judge that looked and chose nothing: `judged` is False
        while `candidates` is non-empty, so a reader can tell the two apart.
        """
        report = await _service().placement_for(_schema(), "Tranche", "Deal")
        assert report.candidates and report.judged is False
        assert report.selected == ()

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
        assert report.judged is True
        assert report.selected == ()

    @pytest.mark.asyncio
    async def test_a_model_outage_leaves_the_deterministic_half_standing(self):
        async def caller(_system, _prompt, _model):
            raise RuntimeError("no route to model")

        report = await _service(caller).placement_for(_schema(), "Tranche", "Deal")
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
        assert report.judged is False
        caller.assert_not_awaited()
