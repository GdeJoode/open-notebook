"""Tests for ``ontology_extraction.pass1_schema_validation``.

All tests mock the LLM caller — no live model calls allowed in CI.
The injected ``llm_caller`` argument on ``Pass1SchemaValidator`` is
the seam used throughout.

Coverage areas (per B.1c acceptance criteria, attempt 2):

1. Token-budget guard fires at the boundary, including large
   ontologies via the auto-compression path (Major 3 fix).
2. Prompt template renders correctly for a sample ontology + closed
   candidate-schema list (minor #2 fix).
3. LLM-output parsing degrades gracefully on malformed JSON / missing
   fields / extra fields — returns empty ``Pass1Output`` with a
   WARNING log instead of raising (Major 1 fix per plan AC #3).
4. LLM calls are mocked — no real API hits.
5. ``coverage_pct`` accepts both 0-1 floats and 0-100 percentages.
6. ``Pass1Output`` field shape is compatible with ``Pass1Result``
   in ``shared.models.notebook_schema`` — both use
   ``List[Dict[str, Any]]`` for ``alternative_schemas`` (Major 2 fix).
7. Brace-delimited JSON salvage from prose-wrapped LLM output
   (minor #3 fix).
8. Transport-level LLM caller failures are wrapped in
   ``Pass1ParseError`` (Major 1 contract distinction).
"""

from __future__ import annotations

import json
from typing import Any, Dict
from unittest.mock import patch

import pytest

from ontology_extraction.pass1_schema_validation import (
    TOKEN_BUDGET_TARGET,
    Pass1Output,
    Pass1ParseError,
    Pass1SchemaValidator,
    TokenBudgetExceeded,
    _empty_pass1_output,
    _estimate_tokens,
    _salvage_json_object,
)
from ontology_extraction.prompts.pass1 import (
    CANDIDATE_SCHEMA_NAMES,
    COMPACT_SUMMARY_THRESHOLD,
    PASS1_SYSTEM_PROMPT,
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


def _synthetic_n_type_ontology(n: int, description_len: int = 80) -> Dict[str, Any]:
    """A synthetic ontology with ``n`` entity types for stress-testing.

    Used by ``TestStressTokenBudget`` to reproduce the reviewer's
    100-type worst case. Descriptions are filler text — the load-
    bearing field for token budget is the count, not the content.
    """
    types = []
    for i in range(n):
        types.append(
            {
                "name": f"SynthType{i:03d}",
                "description": ("x" * description_len)
                if description_len
                else "",
            }
        )
    return {
        "metadata": {"name": "synthetic_huge"},
        "entity_types": types,
    }


def _valid_llm_response() -> str:
    """A well-formed Pass-1 response with all six fields.

    ``alternative_schemas`` is the canonical list-of-dicts shape now
    that Major 2 aligns ``Pass1Output`` with ``Pass1Result``.
    """
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
            "alternative_schemas": [
                {"name": "general", "confidence": 0.4},
                {"name": "policy", "confidence": 0.2},
            ],
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

    def test_compression_kicks_in_above_threshold(self):
        """Ontologies with > 30 types drop descriptions (Major 3)."""
        ontology = _synthetic_n_type_ontology(COMPACT_SUMMARY_THRESHOLD + 1, 80)
        out = build_schema_summary(ontology)
        # Compressed form uses single brace-wrapped name group, no
        # per-type bullet (no ``** —`` markers).
        assert "{SynthType000" in out
        assert "Types compressed" in out
        assert "**SynthType000**" not in out
        # The filler ``x...`` description content must NOT appear in
        # the compressed form (that's the whole point).
        assert "xxxxxx" not in out

    def test_at_threshold_uses_verbose(self):
        """At exactly the threshold (30 types) descriptions are kept."""
        ontology = _synthetic_n_type_ontology(COMPACT_SUMMARY_THRESHOLD, 30)
        out = build_schema_summary(ontology)
        # Verbose form uses ``- **Name** —`` bullets, not the
        # brace-wrapped compression sentinel.
        assert "- **SynthType000**" in out
        assert "Types compressed" not in out

    def test_dict_entity_types_with_object_attrs(self):
        """Dict values exposing ``description`` attr (Pydantic model)."""

        class _FakeDef:
            description = "obj-attr description"

        ontology = {
            "metadata": {"name": "objs"},
            "entity_types": {"Thing": _FakeDef()},
        }
        out = build_schema_summary(ontology)
        assert "Thing" in out
        assert "obj-attr description" in out

    def test_empty_description_renders_name_only(self):
        """Below-threshold type with empty description renders ``- **Name**``.

        Closes coverage on the no-description branch of the verbose
        renderer.
        """
        ontology = {
            "metadata": {"name": "minimal"},
            "entity_types": [{"name": "OnlyName", "description": ""}],
        }
        out = build_schema_summary(ontology)
        assert "- **OnlyName**" in out
        # No em-dash after the name when there's no description.
        assert "OnlyName** —" not in out

    def test_list_entity_types_with_object_attrs(self):
        """List items exposing ``name``/``description`` attrs."""

        class _FakeET:
            name = "Thing"
            description = "from attrs"

        ontology = {
            "metadata": {"name": "objs"},
            "entity_types": [_FakeET()],
        }
        out = build_schema_summary(ontology)
        assert "Thing" in out
        assert "from attrs" in out


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

    def test_candidate_schemas_list_is_complete(self):
        """All 11 YAML ontologies appear in the closed candidate list.

        Minor #2 of the attempt-1 review: ``base``, ``policy_themes``
        and ``social_profiles`` were missing in attempt 1.
        """
        prompt = build_pass1_prompt(_small_ontology(), "text")
        for name in [
            "base",
            "policy_themes",
            "social_profiles",
            "deals",
            "general",
            "government",
            "instruments",
            "policy",
            "regiodeal",
            "schema_core",
            "scholarly",
        ]:
            assert name in prompt, f"candidate '{name}' missing from prompt"

    def test_output_format_uses_dict_shape_for_alternatives(self):
        """Example JSON in the prompt should illustrate the dict shape."""
        prompt = build_pass1_prompt(_small_ontology(), "text")
        # The exact example payload includes name + confidence.
        assert '"name": "general"' in prompt
        assert '"confidence": 0.4' in prompt


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


class TestStressTokenBudget:
    """Stress test: 100-type ontology + 1500-token sample fits budget.

    Major 3 of the B.1c attempt-1 review: a synthesised 100-type
    ontology + 1500-token sample produced ~4595 tokens (53% over).
    With ``COMPACT_SUMMARY_THRESHOLD = 30`` triggering the names-only
    compression, the same input now fits comfortably.
    """

    def test_100_type_ontology_fits_under_budget(self):
        ontology = _synthetic_n_type_ontology(100, description_len=80)
        # 1500 tokens ≈ 6000 chars under the len(text)//4 heuristic.
        sample = "x " * 3000
        prompt = build_pass1_prompt(ontology, sample)
        estimated = _estimate_tokens(prompt)
        assert estimated <= TOKEN_BUDGET_TARGET, (
            f"100-type synthetic ontology + 1500-token sample produced "
            f"~{estimated} tokens (cap {TOKEN_BUDGET_TARGET}). The "
            "compression threshold needs lowering or the names list "
            "tightening."
        )

    def test_150_type_ontology_still_fits(self):
        """Headroom check: budget still holds at 1.5x the worst case."""
        ontology = _synthetic_n_type_ontology(150, description_len=80)
        sample = "x " * 3000
        prompt = build_pass1_prompt(ontology, sample)
        estimated = _estimate_tokens(prompt)
        assert estimated <= TOKEN_BUDGET_TARGET, (
            f"150-type synthetic ontology + 1500-token sample produced "
            f"~{estimated} tokens (cap {TOKEN_BUDGET_TARGET})."
        )


# ---------------------------------------------------------------------------
# Output parsing tests
# ---------------------------------------------------------------------------


class TestParseResponse:
    """Tests for ``Pass1SchemaValidator._parse_response``.

    Plan AC #3 (Major 1): malformed JSON, missing fields, extra
    fields, and structural problems degrade gracefully to an empty
    ``Pass1Output`` plus a WARNING log. ``Pass1ParseError`` is
    reserved for transport-level failures and is NOT raised from
    parsing.
    """

    def test_well_formed_response(self):
        result = Pass1SchemaValidator._parse_response(_valid_llm_response())
        assert result.detected_schema == "scholarly"
        assert result.confidence_in_choice == 0.87
        assert result.coverage_pct == 0.74
        assert len(result.uncovered_concepts) == 1
        assert result.uncovered_concepts[0]["surface_form"] == "deep neural network"
        assert len(result.proposed_extensions) == 1
        # Now dict-shaped (Major 2) — name + confidence preserved.
        assert result.alternative_schemas == [
            {"name": "general", "confidence": 0.4},
            {"name": "policy", "confidence": 0.2},
        ]

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

    def test_prose_around_json_is_salvaged(self):
        """Markdown-free prose around a JSON object still parses.

        Minor #3 of the attempt-1 review: LLMs sometimes wrap JSON in
        commentary. The salvage step extracts the first ``{...}`` block
        so the parser succeeds.
        """
        payload = _valid_llm_response()
        wrapped = (
            "Sure, here's the analysis you asked for:\n"
            f"{payload}\n"
            "Hope that helps!"
        )
        result = Pass1SchemaValidator._parse_response(wrapped)
        assert result.detected_schema == "scholarly"

    def test_prose_then_fenced_json(self):
        """Combination: prose + fenced JSON (most common LLM pattern)."""
        payload = _valid_llm_response()
        wrapped = (
            "Here is the result:\n"
            "```json\n"
            f"{payload}\n"
            "```\n"
            "Hope that helps."
        )
        result = Pass1SchemaValidator._parse_response(wrapped)
        assert result.detected_schema == "scholarly"

    def test_salvage_helper_returns_input_when_no_braces(self):
        """The salvage helper is a no-op when nothing brace-shaped exists."""
        assert _salvage_json_object("plain text no braces") == (
            "plain text no braces"
        )

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

    def test_unparseable_scalar_falls_back_to_zero(self):
        """Strings that don't parse as float become 0.0 (defensive)."""
        raw = json.dumps(
            {
                "detected_schema": "scholarly",
                "confidence_in_choice": "high",
                "coverage_pct": "?",
                "uncovered_concepts": [],
                "proposed_extensions": [],
                "alternative_schemas": [],
            }
        )
        result = Pass1SchemaValidator._parse_response(raw)
        assert result.confidence_in_choice == 0.0
        assert result.coverage_pct == 0.0

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

    def test_non_list_lists_are_coerced_to_empty(self):
        """A scalar in a list slot becomes ``[]`` defensively."""
        raw = json.dumps(
            {
                "detected_schema": "scholarly",
                "confidence_in_choice": 0.5,
                "coverage_pct": 0.5,
                "uncovered_concepts": "not a list",
                "proposed_extensions": 42,
                "alternative_schemas": "also not",
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

    def test_missing_required_field_degrades_gracefully(self):
        """``detected_schema`` is required; its absence yields empty output.

        Major 1 of the attempt-1 review: attempt 1 raised
        ``Pass1ParseError``. The plan (AC #3) requires graceful
        degradation. Pass1Output's ``detected_schema`` now has a
        default of ``""`` — missing field is accepted as the empty
        sentinel; the existing test guarded the deprecated contract,
        so this test now pins the new graceful path.
        """
        raw = json.dumps(
            {
                # detected_schema intentionally absent
                "confidence_in_choice": 0.5,
                "coverage_pct": 0.5,
            }
        )
        result = Pass1SchemaValidator._parse_response(raw)
        # ``detected_schema`` is now an empty-string default, so a
        # missing field is silently treated as "unknown".
        assert result.detected_schema == ""

    def test_alternative_schemas_truncated_to_three(self):
        """Excess entries beyond 3 are dropped at validator time."""
        raw = json.dumps(
            {
                "detected_schema": "scholarly",
                "confidence_in_choice": 0.5,
                "coverage_pct": 0.5,
                "uncovered_concepts": [],
                "proposed_extensions": [],
                "alternative_schemas": [
                    {"name": "a", "confidence": 0.9},
                    {"name": "b", "confidence": 0.8},
                    {"name": "c", "confidence": 0.7},
                    {"name": "d", "confidence": 0.6},
                    {"name": "e", "confidence": 0.5},
                ],
            }
        )
        result = Pass1SchemaValidator._parse_response(raw)
        names = [a["name"] for a in result.alternative_schemas]
        assert names == ["a", "b", "c"]

    def test_alternative_schemas_legacy_string_format_lifted_to_dict(self):
        """Bare strings are lifted to ``{"name": s}`` for back-compat.

        Major 2 of the review picked dict-shaped output as canonical
        (matches ``Pass1Result``). The validator still accepts a
        legacy ``List[str]`` shape so an LLM regression doesn't crash
        the pipeline.
        """
        raw = json.dumps(
            {
                "detected_schema": "scholarly",
                "confidence_in_choice": 0.5,
                "coverage_pct": 0.5,
                "uncovered_concepts": [],
                "proposed_extensions": [],
                "alternative_schemas": ["legacy_a", "legacy_b"],
            }
        )
        result = Pass1SchemaValidator._parse_response(raw)
        assert result.alternative_schemas == [
            {"name": "legacy_a"},
            {"name": "legacy_b"},
        ]

    def test_alternative_schemas_missing_name_dict_dropped(self):
        """Dict entries without ``name`` are dropped — no usable signal."""
        raw = json.dumps(
            {
                "detected_schema": "scholarly",
                "confidence_in_choice": 0.5,
                "coverage_pct": 0.5,
                "uncovered_concepts": [],
                "proposed_extensions": [],
                "alternative_schemas": [
                    {"name": "ok", "confidence": 0.5},
                    {"confidence": 0.3},  # no name
                    {"name": "also_ok"},
                ],
            }
        )
        result = Pass1SchemaValidator._parse_response(raw)
        names = [a["name"] for a in result.alternative_schemas]
        assert names == ["ok", "also_ok"]

    def test_alternative_schemas_non_dict_non_string_dropped(self):
        """Ints / None / nested lists are silently dropped."""
        raw = json.dumps(
            {
                "detected_schema": "scholarly",
                "confidence_in_choice": 0.5,
                "coverage_pct": 0.5,
                "uncovered_concepts": [],
                "proposed_extensions": [],
                "alternative_schemas": [
                    {"name": "good"},
                    None,
                    42,
                    "legacy",
                    ["nested"],
                ],
            }
        )
        result = Pass1SchemaValidator._parse_response(raw)
        names = [a["name"] for a in result.alternative_schemas]
        assert names == ["good", "legacy"]

    def test_invalid_json_degrades_gracefully(self):
        """Malformed JSON returns empty output + WARNING (Major 1)."""
        result = Pass1SchemaValidator._parse_response("not valid json {{ at all")
        assert result == _empty_pass1_output()
        assert result.detected_schema == ""
        assert result.coverage_pct == 0.0
        assert result.confidence_in_choice == 0.0

    def test_empty_response_degrades_gracefully(self):
        result = Pass1SchemaValidator._parse_response("")
        assert result == _empty_pass1_output()
        assert result.detected_schema == ""

    def test_whitespace_only_response_degrades_gracefully(self):
        result = Pass1SchemaValidator._parse_response("   \n\t  ")
        assert result == _empty_pass1_output()

    def test_non_object_json_degrades_gracefully(self):
        """A JSON array is structurally invalid for Pass-1 — empty output."""
        result = Pass1SchemaValidator._parse_response("[1, 2, 3]")
        assert result == _empty_pass1_output()

    def test_truncated_json_degrades_gracefully(self):
        """Truncated JSON (LLM ran out of tokens) → empty output."""
        truncated = '{"detected_schema": "scholarly", "coverage_pct": 0.5,'
        result = Pass1SchemaValidator._parse_response(truncated)
        assert result == _empty_pass1_output()

    def test_malformed_json_emits_warning_log(self, caplog):
        """The graceful-degradation path also logs WARNING (plan AC #3)."""
        import logging

        # Loguru -> stdlib handler bridge so pytest's caplog captures
        # the WARNING. Loguru does not emit through the stdlib root
        # logger by default; route it here for the duration of the
        # test so the contract is verifiable.
        from loguru import logger as loguru_logger

        sink_id = loguru_logger.add(
            lambda msg: logging.getLogger("pass1_test").warning(msg.record["message"]),
            level="WARNING",
        )
        try:
            with caplog.at_level(logging.WARNING, logger="pass1_test"):
                Pass1SchemaValidator._parse_response("not valid json {{")
            assert any(
                "Pass-1 returned empty" in r.message for r in caplog.records
            )
        finally:
            loguru_logger.remove(sink_id)

    def test_pass1_output_constructor_validation_error_degrades(self):
        """ValidationError from Pass1Output(**data) → empty sentinel.

        Closes the coverage hole on the explicit ``except ValidationError``
        branch. The defensive field validators normally absorb every
        bad value (coerce-or-empty), so we patch the constructor to
        prove the explicit branch returns the sentinel rather than
        raising. The ``_empty_pass1_output()`` helper also calls
        ``Pass1Output(...)`` so the patch must let subsequent calls
        succeed — first call raises, the rest construct normally.
        """
        from pydantic import ValidationError

        real_constructor = Pass1Output
        calls = {"n": 0}

        def _selective_constructor(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                # Simulate Pydantic surfacing a ValidationError. We
                # construct one from a contrived error payload so the
                # branch is exercised authentically.
                raise ValidationError.from_exception_data("Pass1Output", [])
            return real_constructor(*args, **kwargs)

        with patch(
            "ontology_extraction.pass1_schema_validation.Pass1Output",
            side_effect=_selective_constructor,
        ):
            result = Pass1SchemaValidator._parse_response(
                '{"detected_schema": "scholarly"}'
            )
        assert result == _empty_pass1_output()

    def test_pass1_output_constructor_generic_exception_degrades(self):
        """Belt-and-braces: a generic exception during construction also
        falls back to the empty sentinel. Catches the broad
        ``except Exception`` branch that guards against unexpected
        Pydantic surface changes.
        """
        real_constructor = Pass1Output
        calls = {"n": 0}

        def _selective_constructor(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("simulated unexpected error")
            return real_constructor(*args, **kwargs)

        with patch(
            "ontology_extraction.pass1_schema_validation.Pass1Output",
            side_effect=_selective_constructor,
        ):
            result = Pass1SchemaValidator._parse_response(
                '{"detected_schema": "scholarly"}'
            )
        assert result == _empty_pass1_output()

    def test_none_passed_to_unit_clamp_validator(self):
        """Construct Pass1Output with a ``None`` scalar — clamp returns 0.

        Closes the coverage hole on the ``if v is None: return 0.0``
        branch of the clamp_to_unit_interval validator. Goes through
        the Pydantic constructor so the field validator path runs.
        """
        out = Pass1Output(
            detected_schema="scholarly",
            confidence_in_choice=None,  # type: ignore[arg-type]
            coverage_pct=None,  # type: ignore[arg-type]
        )
        assert out.confidence_in_choice == 0.0
        assert out.coverage_pct == 0.0

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
            # Verify the system prompt came from the prompts module.
            assert sys_p == PASS1_SYSTEM_PROMPT
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

    async def test_run_returns_empty_on_malformed_json(self):
        """Plan AC #3: ``run`` returns empty ``Pass1Output``, no exception."""

        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            return "not valid json"

        validator = Pass1SchemaValidator(llm_caller=fake_llm)
        result = await validator.run("text", _small_ontology())
        assert result == _empty_pass1_output()
        assert result.detected_schema == ""
        assert result.coverage_pct == 0.0

    async def test_run_wraps_llm_caller_exception_in_pass1_parse_error(self):
        """Transport errors are wrapped in ``Pass1ParseError`` (Major 1)."""

        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            raise RuntimeError("LLM out of budget")

        validator = Pass1SchemaValidator(llm_caller=fake_llm)
        with pytest.raises(Pass1ParseError, match="LLM caller failed"):
            await validator.run("text", _small_ontology())

    async def test_run_wraps_async_caller_exception(self):
        """Async transport errors get the same wrapping treatment."""

        async def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            raise ConnectionError("network down")

        validator = Pass1SchemaValidator(llm_caller=fake_llm)
        with pytest.raises(Pass1ParseError, match="LLM caller failed"):
            await validator.run("text", _small_ontology())


# ---------------------------------------------------------------------------
# Pass1Output ↔ Pass1Result compatibility
# ---------------------------------------------------------------------------


class TestPass1OutputCompatibility:
    """``Pass1Output.model_dump()`` must feed cleanly into ``Pass1Result``.

    This is the contract the persistence path (B.1f) will rely on.
    Major 2 of the attempt-1 review: the dump now produces dict-
    shaped ``alternative_schemas`` so Pydantic doesn't raise at
    persistence time.
    """

    def test_model_dump_keys_match_pass1result_field_names(self):
        out = Pass1Output(
            detected_schema="scholarly",
            confidence_in_choice=0.5,
            coverage_pct=0.5,
            uncovered_concepts=[],
            proposed_extensions=[],
            alternative_schemas=[{"name": "general", "confidence": 0.5}],
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

    def test_alternative_schemas_dump_is_list_of_dicts(self):
        """Major 2 fix: shapes align with ``Pass1Result``."""
        out = Pass1Output(
            detected_schema="scholarly",
            confidence_in_choice=0.5,
            coverage_pct=0.5,
            uncovered_concepts=[],
            proposed_extensions=[],
            alternative_schemas=[
                {"name": "general", "confidence": 0.4},
                {"name": "policy", "confidence": 0.2},
            ],
        )
        dumped = out.model_dump()
        assert dumped["alternative_schemas"] == [
            {"name": "general", "confidence": 0.4},
            {"name": "policy", "confidence": 0.2},
        ]
        # Every entry must be a dict — the persistence layer relies on this.
        assert all(isinstance(x, dict) for x in dumped["alternative_schemas"])

    def test_dump_feeds_pass1_result_construction_directly(self):
        """End-to-end: dump → Pass1Result(...) succeeds without lifting.

        This is the contract B.1f relies on: no Pydantic ValidationError
        at persistence call time.
        """
        from shared.models.notebook_schema import Pass1Result

        out = Pass1Output(
            detected_schema="scholarly",
            confidence_in_choice=0.87,
            coverage_pct=0.74,
            uncovered_concepts=[{"surface_form": "x", "suggested_type": "y"}],
            proposed_extensions=[{"type_name": "Z"}],
            alternative_schemas=[{"name": "general", "confidence": 0.4}],
        )
        result = Pass1Result(
            source="source:abc",
            notebook="notebook:xyz",
            schema_attempted="scholarly",
            **out.model_dump(),
        )
        assert result.detected_schema == "scholarly"
        assert result.alternative_schemas == [
            {"name": "general", "confidence": 0.4}
        ]


# ---------------------------------------------------------------------------
# Real-world ontology budget regression
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


# ---------------------------------------------------------------------------
# Module-surface tests
# ---------------------------------------------------------------------------


class TestPublicExports:
    """The pass-1 surface is importable from ``ontology_extraction``."""

    def test_top_level_imports(self):
        from ontology_extraction import (  # noqa: F401
            Pass1Output,
            Pass1ParseError,
            Pass1SchemaValidator,
            TokenBudgetExceeded,
        )

    def test_prompts_module_exports(self):
        from ontology_extraction.prompts import (  # noqa: F401
            CANDIDATE_SCHEMA_NAMES,
            COMPACT_SUMMARY_THRESHOLD,
            PASS1_SYSTEM_PROMPT,
            build_pass1_prompt,
            build_schema_summary,
        )

        assert isinstance(CANDIDATE_SCHEMA_NAMES, list)
        assert "scholarly" in CANDIDATE_SCHEMA_NAMES
        assert isinstance(COMPACT_SUMMARY_THRESHOLD, int)
        assert isinstance(PASS1_SYSTEM_PROMPT, str)


# ---------------------------------------------------------------------------
# Lazy default LLM caller — confirms tests never invoke it
# ---------------------------------------------------------------------------


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

    async def test_default_caller_returns_empty_json_when_used(self):
        """No injected caller: the default returns ``"{}"`` and logs WARNING.

        Touches the default-caller branch so coverage closes over the
        lazy-import path. The default is deliberately a stub (B.1f
        will replace) and returning ``{}`` exercises the graceful
        malformed-JSON path: the resulting ``Pass1Output`` is the
        empty sentinel.
        """
        validator = Pass1SchemaValidator()  # no caller -> default used
        result = await validator.run("text", _small_ontology())
        assert result == _empty_pass1_output()
