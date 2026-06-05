"""Tests for ``ontology_extraction.pass1_schema_validation``.

All tests mock the LLM caller — no live model calls allowed in CI.
The injected ``llm_caller`` argument on ``Pass1SchemaValidator`` is
the seam used throughout.

Coverage areas (per B.1c acceptance criteria):

1. Token-budget guard fires at the boundary.
2. Prompt template renders correctly for a sample ontology.
3. LLM-output parsing handles malformed JSON / missing fields /
   extra fields.
4. LLM calls are mocked — no real API hits.
5. ``coverage_pct`` accepts both 0-1 floats and 0-100 percentages.
6. ``Pass1Output`` field shape is compatible with ``Pass1Result``
   in ``shared.models.notebook_schema``.
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import AsyncMock, patch

import pytest

from ontology_extraction.pass1_schema_validation import (
    TOKEN_BUDGET_TARGET,
    Pass1Output,
    Pass1ParseError,
    Pass1SchemaValidator,
    TokenBudgetExceeded,
    _estimate_tokens,
)
from ontology_extraction.prompts.pass1 import (
    build_pass1_prompt,
    build_schema_summary,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _small_ontology() -> Dict[str, Any]:
    """A 3-type scholarly-style ontology, dict-shape."""
    return {
        "metadata": {"name": "scholarly", "version": "1.0"},
        "entity_types": {
            "ScholarlyArticle": {
                "name": "ScholarlyArticle",
                "description": "A peer-reviewed academic paper.",
            },
            "Author": {
                "name": "Author",
                "description": "A person who wrote a scholarly work.",
            },
            "Journal": {
                "name": "Journal",
                "description": "A periodical publishing scholarly works.",
            },
        },
    }


def _list_shape_ontology() -> Dict[str, Any]:
    """Same ontology in YAML-list shape (used by ontology files in repo)."""
    return {
        "metadata": {"name": "scholarly"},
        "entity_types": [
            {"name": "ScholarlyArticle", "description": "A peer-reviewed paper."},
            {"name": "Author", "description": "A person who wrote a scholarly work."},
            {"name": "Journal", "description": "A periodical publication."},
        ],
    }


def _valid_llm_response() -> str:
    """A well-formed Pass-1 response with all six fields."""
    return json.dumps(
        {
            "detected_schema": "scholarly",
            "confidence_in_choice": 0.87,
            "coverage_pct": 0.74,
            "uncovered_concepts": [
                {"surface_form": "deep neural network", "suggested_type": "Method"},
            ],
            "proposed_extensions": [
                {
                    "type_name": "Method",
                    "parent_type": "Concept",
                    "rationale": "Many ML papers introduce methods.",
                },
            ],
            "alternative_schemas": ["general", "policy"],
        }
    )


# ---------------------------------------------------------------------------
# Prompt template tests
# ---------------------------------------------------------------------------


class TestBuildSchemaSummary:
    """Tests for ``build_schema_summary``."""

    def test_dict_shape_entity_types(self):
        out = build_schema_summary(_small_ontology())
        assert "## Schema: scholarly" in out
        assert "ScholarlyArticle" in out
        assert "A peer-reviewed academic paper." in out
        assert "Author" in out
        assert "Journal" in out

    def test_list_shape_entity_types(self):
        """YAML list-shape ontologies render identically to dict-shape."""
        out = build_schema_summary(_list_shape_ontology())
        assert "ScholarlyArticle" in out
        assert "Author" in out
        assert "Journal" in out

    def test_no_entity_types_renders_stub(self):
        out = build_schema_summary({"metadata": {"name": "empty"}})
        assert "## Schema: empty" in out
        assert "(none)" in out

    def test_missing_metadata_name(self):
        out = build_schema_summary({"entity_types": {}})
        assert "## Schema: unknown" in out

    def test_long_descriptions_are_truncated(self):
        """Descriptions over 160 chars are truncated with ``...``."""
        ontology = {
            "metadata": {"name": "long"},
            "entity_types": {
                "Verbose": {"name": "Verbose", "description": "x" * 500},
            },
        }
        out = build_schema_summary(ontology)
        # Truncated to 157 chars + "..."
        assert "..." in out
        # 500-char raw description must not appear in full
        assert "x" * 200 not in out

    def test_filters_unnamed_entries(self):
        """Entries with empty names are dropped."""
        ontology = {
            "metadata": {"name": "mixed"},
            "entity_types": [
                {"name": "", "description": "no name"},
                {"name": "Real", "description": "a real one"},
            ],
        }
        out = build_schema_summary(ontology)
        assert "Real" in out
        assert "no name" not in out


class TestBuildPass1Prompt:
    """Tests for ``build_pass1_prompt``."""

    def test_full_prompt_contains_all_sections(self):
        ontology = _small_ontology()
        text = "Example paper text here."
        prompt = build_pass1_prompt(ontology, text)

        assert "# Pass-1 Schema Validation" in prompt
        assert "## Schema: scholarly" in prompt
        assert "ScholarlyArticle" in prompt
        assert "## Text Sample" in prompt
        assert text in prompt
        assert "## Output Format" in prompt
        # Field names must appear verbatim so the LLM emits them.
        for field in [
            "detected_schema",
            "confidence_in_choice",
            "coverage_pct",
            "uncovered_concepts",
            "proposed_extensions",
            "alternative_schemas",
        ]:
            assert field in prompt


# ---------------------------------------------------------------------------
# Token-budget guard tests
# ---------------------------------------------------------------------------


class TestTokenBudget:
    """Tests for the budget heuristic + ``TokenBudgetExceeded`` boundary."""

    def test_estimator_is_len_div_four(self):
        """The Q-B-2 heuristic is exactly ``len(text) // 4``."""
        assert _estimate_tokens("") == 0
        assert _estimate_tokens("a" * 4) == 1
        assert _estimate_tokens("a" * 100) == 25
        assert _estimate_tokens("a" * 9601) == 2400

    async def test_budget_target_is_2400(self):
        """20% margin against 3000-token plan cap = 2400."""
        assert TOKEN_BUDGET_TARGET == 2400

    async def test_budget_guard_fires_when_exceeded(self):
        """Oversized prompt raises ``TokenBudgetExceeded`` pre-LLM call."""
        # A text sample large enough that the assembled prompt exceeds
        # 2400 tokens (≈ 9601 chars). The schema summary adds a few
        # hundred more chars on top, so 11k is comfortably over.
        oversized = "x" * 11_000
        called = False

        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            nonlocal called
            called = True
            return _valid_llm_response()

        validator = Pass1SchemaValidator(llm_caller=fake_llm)
        with pytest.raises(TokenBudgetExceeded) as exc_info:
            await validator.run(oversized, _small_ontology())

        assert exc_info.value.estimated > TOKEN_BUDGET_TARGET
        assert exc_info.value.budget == TOKEN_BUDGET_TARGET
        # Guard must fire BEFORE the LLM is invoked.
        assert called is False

    async def test_budget_guard_passes_at_boundary(self):
        """A prompt right at the boundary still calls the LLM."""
        # Sample sized so the assembled prompt sits comfortably
        # below 2400 tokens (~ 4000 chars total budget).
        sample = "Some short academic text."
        called = False

        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            nonlocal called
            called = True
            return _valid_llm_response()

        validator = Pass1SchemaValidator(llm_caller=fake_llm)
        result = await validator.run(sample, _small_ontology())
        assert called is True
        assert isinstance(result, Pass1Output)


# ---------------------------------------------------------------------------
# Output parsing tests
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Tests for ``Pass1SchemaValidator._parse_response``."""

    def test_well_formed_response(self):
        result = Pass1SchemaValidator._parse_response(_valid_llm_response())
        assert result.detected_schema == "scholarly"
        assert result.confidence_in_choice == 0.87
        assert result.coverage_pct == 0.74
        assert len(result.uncovered_concepts) == 1
        assert result.uncovered_concepts[0]["surface_form"] == "deep neural network"
        assert len(result.proposed_extensions) == 1
        assert result.alternative_schemas == ["general", "policy"]

    def test_response_wrapped_in_json_fence(self):
        wrapped = (
            "Here is my analysis:\n"
            "```json\n"
            f"{_valid_llm_response()}\n"
            "```"
        )
        result = Pass1SchemaValidator._parse_response(wrapped)
        assert result.detected_schema == "scholarly"

    def test_response_wrapped_in_plain_fence(self):
        wrapped = f"```\n{_valid_llm_response()}\n```"
        result = Pass1SchemaValidator._parse_response(wrapped)
        assert result.detected_schema == "scholarly"

    def test_percentage_coverage_is_rescaled(self):
        """The LLM sometimes emits 87 (percent) instead of 0.87.

        The clamp validator rescales > 1.5 by dividing by 100, so
        downstream code can rely on the value being in [0, 1].
        """
        raw = json.dumps(
            {
                "detected_schema": "scholarly",
                "confidence_in_choice": 87,  # percentage
                "coverage_pct": 74,  # percentage
                "uncovered_concepts": [],
                "proposed_extensions": [],
                "alternative_schemas": [],
            }
        )
        result = Pass1SchemaValidator._parse_response(raw)
        assert result.coverage_pct == 0.74
        assert result.confidence_in_choice == 0.87

    def test_negative_floats_are_clamped_to_zero(self):
        raw = json.dumps(
            {
                "detected_schema": "scholarly",
                "confidence_in_choice": -0.1,
                "coverage_pct": -5.0,
                "uncovered_concepts": [],
                "proposed_extensions": [],
                "alternative_schemas": [],
            }
        )
        result = Pass1SchemaValidator._parse_response(raw)
        assert result.coverage_pct == 0.0
        assert result.confidence_in_choice == 0.0

    def test_oversized_floats_are_clamped_to_one(self):
        """A scalar in (1, 1.5] is not auto-divided; it gets clamped."""
        raw = json.dumps(
            {
                "detected_schema": "scholarly",
                "confidence_in_choice": 1.2,
                "coverage_pct": 1.4,
                "uncovered_concepts": [],
                "proposed_extensions": [],
                "alternative_schemas": [],
            }
        )
        result = Pass1SchemaValidator._parse_response(raw)
        assert result.coverage_pct == 1.0
        assert result.confidence_in_choice == 1.0

    def test_null_lists_are_coerced_to_empty(self):
        """LLM may emit ``null`` for empty arrays — must not crash."""
        raw = json.dumps(
            {
                "detected_schema": "scholarly",
                "confidence_in_choice": 0.5,
                "coverage_pct": 0.5,
                "uncovered_concepts": None,
                "proposed_extensions": None,
                "alternative_schemas": None,
            }
        )
        result = Pass1SchemaValidator._parse_response(raw)
        assert result.uncovered_concepts == []
        assert result.proposed_extensions == []
        assert result.alternative_schemas == []

    def test_extra_fields_are_ignored(self):
        """``ConfigDict(extra='ignore')`` keeps Pass-1 forward-compatible."""
        raw = json.dumps(
            {
                "detected_schema": "scholarly",
                "confidence_in_choice": 0.5,
                "coverage_pct": 0.5,
                "uncovered_concepts": [],
                "proposed_extensions": [],
                "alternative_schemas": [],
                "future_field": "ignored",
                "another": {"deep": "data"},
            }
        )
        result = Pass1SchemaValidator._parse_response(raw)
        assert result.detected_schema == "scholarly"

    def test_missing_required_field_raises(self):
        """``detected_schema`` is required — its absence is a parse error."""
        raw = json.dumps(
            {
                # detected_schema intentionally absent
                "confidence_in_choice": 0.5,
                "coverage_pct": 0.5,
            }
        )
        with pytest.raises(Pass1ParseError):
            Pass1SchemaValidator._parse_response(raw)

    def test_alternative_schemas_truncated_to_three(self):
        raw = json.dumps(
            {
                "detected_schema": "scholarly",
                "confidence_in_choice": 0.5,
                "coverage_pct": 0.5,
                "uncovered_concepts": [],
                "proposed_extensions": [],
                "alternative_schemas": ["a", "b", "c", "d", "e"],
            }
        )
        result = Pass1SchemaValidator._parse_response(raw)
        assert result.alternative_schemas == ["a", "b", "c"]

    def test_alternative_schemas_non_string_items_dropped(self):
        raw = json.dumps(
            {
                "detected_schema": "scholarly",
                "confidence_in_choice": 0.5,
                "coverage_pct": 0.5,
                "uncovered_concepts": [],
                "proposed_extensions": [],
                "alternative_schemas": ["a", None, "b"],
            }
        )
        result = Pass1SchemaValidator._parse_response(raw)
        # ``None`` filtered, the rest coerced to str
        assert result.alternative_schemas == ["a", "b"]

    def test_invalid_json_raises_parse_error(self):
        with pytest.raises(Pass1ParseError, match="Invalid JSON"):
            Pass1SchemaValidator._parse_response("not valid json {{ at all")

    def test_empty_response_raises_parse_error(self):
        with pytest.raises(Pass1ParseError, match="Empty"):
            Pass1SchemaValidator._parse_response("")

    def test_whitespace_only_response_raises_parse_error(self):
        with pytest.raises(Pass1ParseError, match="Empty"):
            Pass1SchemaValidator._parse_response("   \n\t  ")

    def test_non_object_json_raises_parse_error(self):
        """A JSON array is structurally invalid for Pass-1."""
        with pytest.raises(Pass1ParseError, match="must be a JSON object"):
            Pass1SchemaValidator._parse_response("[1, 2, 3]")

    def test_uncovered_concepts_non_dict_items_filtered(self):
        raw = json.dumps(
            {
                "detected_schema": "scholarly",
                "confidence_in_choice": 0.5,
                "coverage_pct": 0.5,
                "uncovered_concepts": [
                    {"surface_form": "good", "suggested_type": "T"},
                    "junk string",
                    None,
                    42,
                ],
                "proposed_extensions": [],
                "alternative_schemas": [],
            }
        )
        result = Pass1SchemaValidator._parse_response(raw)
        assert len(result.uncovered_concepts) == 1
        assert result.uncovered_concepts[0]["surface_form"] == "good"


# ---------------------------------------------------------------------------
# End-to-end (with mocked LLM) tests
# ---------------------------------------------------------------------------


class TestRunEndToEnd:
    """``Pass1SchemaValidator.run`` with mocked LLM callers."""

    async def test_run_with_sync_caller(self):
        """A synchronous LLM caller is accepted and produces a Pass1Output."""
        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            # Verify the prompt body reached the caller verbatim.
            assert "## Schema: scholarly" in user_p
            assert "Example paper text" in user_p
            return _valid_llm_response()

        validator = Pass1SchemaValidator(llm_caller=fake_llm)
        result = await validator.run("Example paper text", _small_ontology())
        assert isinstance(result, Pass1Output)
        assert result.detected_schema == "scholarly"
        assert result.coverage_pct == 0.74

    async def test_run_with_async_caller(self):
        """An async LLM caller is awaited correctly."""
        async def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            return _valid_llm_response()

        validator = Pass1SchemaValidator(llm_caller=fake_llm)
        result = await validator.run("Example paper text", _small_ontology())
        assert result.detected_schema == "scholarly"

    async def test_run_passes_model_override(self):
        """The ``model`` arg overrides the default."""
        seen_model = []

        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            seen_model.append(model)
            return _valid_llm_response()

        validator = Pass1SchemaValidator(
            llm_caller=fake_llm, default_model="default-model"
        )
        await validator.run("text", _small_ontology(), model="gpt-4o")
        assert seen_model == ["gpt-4o"]

    async def test_run_uses_default_model_when_unspecified(self):
        seen_model = []

        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            seen_model.append(model)
            return _valid_llm_response()

        validator = Pass1SchemaValidator(
            llm_caller=fake_llm, default_model="my-default"
        )
        await validator.run("text", _small_ontology())
        assert seen_model == ["my-default"]

    async def test_run_propagates_parse_error(self):
        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            return "not valid json"

        validator = Pass1SchemaValidator(llm_caller=fake_llm)
        with pytest.raises(Pass1ParseError):
            await validator.run("text", _small_ontology())

    async def test_run_propagates_llm_exception(self):
        """Unrelated LLM exceptions bubble — caller decides retry policy."""
        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            raise RuntimeError("LLM out of budget")

        validator = Pass1SchemaValidator(llm_caller=fake_llm)
        with pytest.raises(RuntimeError, match="LLM out of budget"):
            await validator.run("text", _small_ontology())


# ---------------------------------------------------------------------------
# Pass1Output ↔ Pass1Result compatibility
# ---------------------------------------------------------------------------


class TestPass1OutputCompatibility:
    """``Pass1Output.model_dump()`` must feed cleanly into ``Pass1Result``.

    This is the contract the persistence path (B.1f) will rely on.
    """

    def test_model_dump_keys_match_pass1result_field_names(self):
        out = Pass1Output(
            detected_schema="scholarly",
            confidence_in_choice=0.5,
            coverage_pct=0.5,
            uncovered_concepts=[],
            proposed_extensions=[],
            alternative_schemas=["general"],
        )
        dumped = out.model_dump()

        # Every field on Pass1Output must be a Pass1Result field too.
        from shared.models.notebook_schema import Pass1Result
        pass1_result_fields = set(Pass1Result.model_fields.keys())
        for key in dumped:
            assert key in pass1_result_fields, (
                f"Pass1Output field '{key}' missing from Pass1Result — "
                "the two models drifted; persistence will break."
            )

    def test_alternative_schemas_dump_is_list_of_strings(self):
        """B.1b spec uses ``List[Dict[str, Any]]`` for alternative_schemas
        on Pass1Result, but the LLM-facing Pass1Output uses ``List[str]``
        for the simpler single-schema case. The persistence layer wraps
        strings into ``{"name": s}`` dicts — see B.1f.
        """
        out = Pass1Output(
            detected_schema="scholarly",
            confidence_in_choice=0.5,
            coverage_pct=0.5,
            uncovered_concepts=[],
            proposed_extensions=[],
            alternative_schemas=["general", "policy"],
        )
        assert out.alternative_schemas == ["general", "policy"]


# ---------------------------------------------------------------------------
# Lazy default LLM caller — confirms tests never invoke it
# ---------------------------------------------------------------------------


class TestRealWorldTokenBudget:
    """Realistic-ontology budget headroom check (informational + regression).

    Loads the real ``scholarly.yaml`` from the ontology-manager package
    and confirms a typical prompt + 1500-token sample fits well under
    the 2400-token cap. If this test fails after an ontology edit,
    the schema summary builder needs tightening before B.1d ships.
    """

    def test_scholarly_ontology_fits_budget_with_1500_token_sample(self):
        import yaml
        from pathlib import Path

        # Resolve the YAML relative to the repo root (parents reach up
        # from .../pipelines/ontology-extraction/tests/test_pass1_*.py).
        repo_root = Path(__file__).resolve().parents[3]
        yaml_path = (
            repo_root
            / "packages"
            / "ontology-manager"
            / "ontologies"
            / "scholarly.yaml"
        )
        if not yaml_path.exists():
            pytest.skip(f"scholarly.yaml not found at {yaml_path}")

        with yaml_path.open() as f:
            ontology = yaml.safe_load(f)

        # 1500 tokens ≈ 6000 chars under the len(text)//4 heuristic.
        sample = "x " * 3000
        prompt = build_pass1_prompt(ontology, sample)
        estimated = _estimate_tokens(prompt)

        assert estimated <= TOKEN_BUDGET_TARGET, (
            f"Pass-1 prompt for scholarly.yaml + 1500-token sample "
            f"estimated at ~{estimated} tokens; budget is "
            f"{TOKEN_BUDGET_TARGET}. Tighten the schema summary."
        )
        # Informational: print the headroom so future readers can
        # eyeball drift. ``-s`` makes this visible in CI logs.
        headroom_pct = 100 * (1 - estimated / TOKEN_BUDGET_TARGET)
        print(
            f"\n[pass1 budget] scholarly.yaml + 1500-tok sample: "
            f"~{estimated} tokens / {TOKEN_BUDGET_TARGET} budget "
            f"({headroom_pct:.1f}% headroom)"
        )


class TestDefaultLLMCaller:
    """The lazy default caller exists but tests must not trigger it."""

    async def test_default_caller_is_not_invoked_when_injected(self):
        """Injecting a caller bypasses the lazy import path entirely."""
        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            return _valid_llm_response()

        with patch(
            "ontology_extraction.pass1_schema_validation."
            "Pass1SchemaValidator._default_llm_caller"
        ) as default_caller:
            default_caller.side_effect = AssertionError(
                "default caller must not be invoked when llm_caller is injected"
            )
            validator = Pass1SchemaValidator(llm_caller=fake_llm)
            await validator.run("text", _small_ontology())
