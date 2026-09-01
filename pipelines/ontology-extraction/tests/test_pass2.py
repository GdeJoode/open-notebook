"""Tests for ``ontology_extraction.pass2_typed_extraction``.

All tests mock the LLM caller — no live model calls. The injected
``llm_caller`` argument on ``run_pass2`` is the seam, mirroring
B.1c's pattern.

Coverage areas (per B.1d plan acceptance criteria):

1. ``run_pass2`` with ``accepted_extensions=[]`` produces an
   ``ExtractionResult`` for a 3-chunk fixture (AC #1).
2. With a non-empty extensions list, the prompt body carries the
   extension definition and an LLM response using the extension
   label produces an ``ExtractedEntity(label=<extension>)`` (AC #2).
3. Every entity AND every relation in the output has a non-null
   ``confidence`` (AC #3 — the B4 confidence-everywhere invariant).
4. Malformed LLM JSON degrades to an empty per-chunk result with a
   WARNING log (AC #4 — mirrors B.1c).
5. Token budget enforced before the LLM call (AC #5).
6. Empty chunks list returns an empty ``ExtractionResult`` with no
   LLM calls (AC #6).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from ontology_manager.schema import (
    EntityTypeDefinition,
    Ontology,
    OntologyMetadata,
    RelationshipTypeDefinition,
)

from ontology_extraction.pass2_typed_extraction import (
    TOKEN_BUDGET_TARGET,
    Pass2ParseError,
    Pass2TokenBudgetExceeded,
    _clamp_confidence,
    _estimate_tokens,
    _parse_chunk_response,
    _salvage_json_object,
    run_pass2,
)
from ontology_extraction.prompts.pass2 import (
    COMPACT_RELATION_THRESHOLD,
    COMPACT_TYPE_THRESHOLD,
    PASS2_SYSTEM_PROMPT,
    build_pass2_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _small_ontology() -> Ontology:
    """A 3-type scholarly-style ontology."""
    return Ontology(
        metadata=OntologyMetadata(name="scholarly", version="1.0"),
        entity_types={
            "Researcher": EntityTypeDefinition(
                name="Researcher",
                description="A person who conducts research.",
            ),
            "Paper": EntityTypeDefinition(
                name="Paper",
                description="A peer-reviewed publication.",
            ),
            "Institution": EntityTypeDefinition(
                name="Institution",
                description="A research-performing organisation.",
            ),
        },
        relationship_types={
            "AUTHORED": RelationshipTypeDefinition(
                name="AUTHORED",
                description="A researcher authored a paper.",
                domain=["Researcher"],
                range=["Paper"],
            ),
            "AFFILIATED_WITH": RelationshipTypeDefinition(
                name="AFFILIATED_WITH",
                description="A researcher is affiliated with an institution.",
                domain=["Researcher"],
                range=["Institution"],
            ),
        },
    )


def _synthetic_large_ontology(n: int) -> Ontology:
    """A synthetic ``n``-type ontology for compression/stress tests."""
    types = {
        f"SynthType{i:03d}": EntityTypeDefinition(
            name=f"SynthType{i:03d}",
            description=("x" * 80),
        )
        for i in range(n)
    }
    return Ontology(
        metadata=OntologyMetadata(name="synthetic_huge", version="1.0"),
        entity_types=types,
        relationship_types={},
    )


def _make_chunk(text: str, chunk_id: Optional[str] = None) -> Dict[str, Any]:
    """Tiny helper so tests don't repeat the dict-shape boilerplate."""
    return {"text": text, "id": chunk_id}


def _three_chunks() -> List[Dict[str, Any]]:
    return [
        _make_chunk("Alice is a researcher at MIT.", "c-1"),
        _make_chunk("Bob authored 'Quantum 101'.", "c-2"),
        _make_chunk("Carol works on graph theory.", "c-3"),
    ]


def _valid_chunk_response(label: str = "Researcher") -> str:
    """A well-formed Pass-2 response with one entity and one relation."""
    return json.dumps(
        {
            "entities": [
                {
                    "text": "Alice",
                    "label": label,
                    "confidence": 0.92,
                    "properties": {"affiliation": "MIT"},
                }
            ],
            "relations": [
                {
                    "source": "Alice",
                    "target": "MIT",
                    "type": "AFFILIATED_WITH",
                    "confidence": 0.88,
                    "properties": {},
                }
            ],
        }
    )


# ---------------------------------------------------------------------------
# Prompt-builder tests
# ---------------------------------------------------------------------------


class TestBuildPass2Prompt:
    """Tests for ``build_pass2_prompt`` shape and section presence."""

    def test_full_prompt_contains_all_sections(self):
        prompt = build_pass2_prompt(_small_ontology(), "Alice is a researcher.")
        assert "# Pass-2 Typed Extraction" in prompt
        assert "**scholarly**" in prompt
        assert "## Entity Types" in prompt
        # All three types must be visible verbatim so the LLM knows the
        # closed vocabulary.
        assert "Researcher" in prompt
        assert "Paper" in prompt
        assert "Institution" in prompt
        assert "## Relationship Types" in prompt
        assert "AUTHORED" in prompt
        assert "AFFILIATED_WITH" in prompt
        assert "## Text" in prompt
        assert "Alice is a researcher." in prompt
        assert "## Output Format" in prompt
        # The B4 confidence contract must be visible in the rules.
        assert "confidence" in prompt

    def test_no_extensions_omits_extension_section(self):
        """Empty list = no ``Accepted Extension Types`` heading."""
        prompt = build_pass2_prompt(_small_ontology(), "text", [])
        assert "Accepted Extension Types" not in prompt

    def test_none_extensions_omits_extension_section(self):
        """``None`` is equivalent to ``[]`` — back-compat with single-arg calls."""
        prompt = build_pass2_prompt(_small_ontology(), "text", None)
        assert "Accepted Extension Types" not in prompt

    def test_extension_definition_injected(self):
        """Non-empty extensions appear as their own section."""
        extensions = [
            {
                "type_name": "EarlyCareerResearcher",
                "parent_type": "Researcher",
                "rationale": "Distinguishes junior from senior researchers.",
            }
        ]
        prompt = build_pass2_prompt(_small_ontology(), "text", extensions)
        assert "## Accepted Extension Types" in prompt
        assert "EarlyCareerResearcher" in prompt
        assert "parent: Researcher" in prompt
        assert "junior from senior" in prompt

    def test_extension_without_parent_renders(self):
        extensions = [{"type_name": "Foo", "rationale": "test"}]
        prompt = build_pass2_prompt(_small_ontology(), "text", extensions)
        assert "Foo" in prompt
        assert "no parent" in prompt

    def test_extension_without_rationale_renders(self):
        extensions = [{"type_name": "Foo", "parent_type": "Bar"}]
        prompt = build_pass2_prompt(_small_ontology(), "text", extensions)
        assert "Foo" in prompt
        assert "parent: Bar" in prompt

    def test_extension_with_empty_type_name_dropped(self):
        """Defensive: missing ``type_name`` means no usable label."""
        extensions = [
            {"type_name": "", "parent_type": "Bar"},
            {"type_name": "Valid", "parent_type": "X"},
        ]
        prompt = build_pass2_prompt(_small_ontology(), "text", extensions)
        assert "Valid" in prompt
        # No empty label rendered as ** ** anywhere
        assert "** ** (" not in prompt

    def test_long_extension_rationale_truncated(self):
        extensions = [
            {
                "type_name": "Foo",
                "parent_type": "Bar",
                "rationale": "x" * 500,
            }
        ]
        prompt = build_pass2_prompt(_small_ontology(), "text", extensions)
        assert "..." in prompt
        # 500-char raw rationale must not appear in full
        assert "x" * 200 not in prompt

    def test_format_accepted_extensions_filters_sentinels(self):
        """B.3c attempt-2 — defence-in-depth at the prompt layer.

        The resume sentinel
        (``{type_name: "_resumed_without_extensions",
        is_resume_sentinel: True}``) is appended by the resume endpoint
        to satisfy the review-gate predicate. It is infrastructure, not
        ontology — letting it through renders
        ``_resumed_without_extensions`` into the LLM prompt as a
        first-class type and pollutes Pass-2 output.

        Primary filter lives in ``entity_extraction_service``; this
        layer is defence-in-depth. If the upstream filter ever
        regresses, the LLM still never sees the leak.

        Real types in the same list MUST survive — the filter is
        sentinel-specific, not a wholesale drop.
        """
        extensions = [
            {
                "type_name": "RealType",
                "parent_type": "Researcher",
                "rationale": "Legitimate extension.",
            },
            {
                "type_name": "_resumed_without_extensions",
                "is_resume_sentinel": True,
                "created_at": "2026-06-09T12:00:00Z",
            },
        ]
        prompt = build_pass2_prompt(_small_ontology(), "text", extensions)
        # Sentinel must not leak into the prompt under any spelling.
        assert "_resumed_without_extensions" not in prompt
        assert "is_resume_sentinel" not in prompt
        # The real type comes through and the section renders.
        assert "RealType" in prompt
        assert "## Accepted Extension Types" in prompt

    def test_format_accepted_extensions_filters_reparents(self):
        """N.4d.3 — the same defence, for the other entry kind.

        A re-parent records that an EXISTING type moved under a different parent.
        Rendered here it would appear under "Accepted Extension Types" with the
        instruction to treat it as a first-class ADDITION to the schema — telling
        the LLM a base-ontology type is a curator's new type. The move itself is
        applied to the ontology this prompt already renders its vocabulary from
        (``ontology_manager.schema_projection``), so nothing is lost by skipping
        it.

        Real extensions in the same list must survive — the filter is
        reparent-specific.
        """
        extensions = [
            {
                "type_name": "RealType",
                # Deliberately NOT parented on the re-parented type: the two
                # names must be independent, or "the moved type is absent" would
                # hold for a reason unrelated to the filter.
                "parent_type": "Institution",
                "rationale": "Legitimate extension.",
            },
            {
                "reparent_id": "reparent::Researcher->Institution",
                "op": "reparent",
                "type_name": "Researcher",
                "new_parent": "Institution",
                "parent_type": "Institution",
            },
        ]
        prompt = build_pass2_prompt(_small_ontology(), "text", extensions)
        section = prompt.split("## Accepted Extension Types", 1)[1]
        assert "RealType" in section
        assert "Researcher" not in section, (
            "a re-parent was rendered as an accepted extension type"
        )

    def test_format_accepted_extensions_omits_section_when_only_reparents(self):
        """The empty-section guard has to see through the new filter too."""
        extensions = [
            {
                "reparent_id": "reparent::Researcher->Organization",
                "op": "reparent",
                "type_name": "Researcher",
                "new_parent": "Organization",
            },
        ]
        prompt = build_pass2_prompt(_small_ontology(), "text", extensions)
        assert "Accepted Extension Types" not in prompt

    def test_format_accepted_extensions_omits_section_when_all_sentinels(self):
        """When the only extensions in the list are sentinels, the
        ``Accepted Extension Types`` section is omitted entirely —
        matching the behaviour for an empty / ``None`` list. Without
        this guard the prompt would render an empty section header
        with no bullets, which is confusing to the LLM.
        """
        extensions = [
            {
                "type_name": "_resumed_without_extensions",
                "is_resume_sentinel": True,
                "created_at": "2026-06-09T12:00:00Z",
            },
        ]
        prompt = build_pass2_prompt(_small_ontology(), "text", extensions)
        assert "Accepted Extension Types" not in prompt
        assert "_resumed_without_extensions" not in prompt

    def test_long_descriptions_truncated_in_entity_types(self):
        """Entity descriptions over 160 chars truncate with ``...``."""
        ontology = Ontology(
            metadata=OntologyMetadata(name="long", version="1.0"),
            entity_types={
                "Verbose": EntityTypeDefinition(
                    name="Verbose",
                    description="x" * 500,
                )
            },
        )
        prompt = build_pass2_prompt(ontology, "text")
        assert "Verbose" in prompt
        # Truncation marker present.
        assert "..." in prompt
        # Full 500-char description must not appear.
        assert "x" * 200 not in prompt

    def test_long_relationship_descriptions_truncated(self):
        """Relationship descriptions over 100 chars truncate."""
        ontology = Ontology(
            metadata=OntologyMetadata(name="long", version="1.0"),
            entity_types={},
            relationship_types={
                "REL": RelationshipTypeDefinition(
                    name="REL",
                    description="y" * 300,
                )
            },
        )
        prompt = build_pass2_prompt(ontology, "text")
        assert "REL" in prompt
        assert "..." in prompt

    def test_relationship_without_description_renders_head_only(self):
        ontology = Ontology(
            metadata=OntologyMetadata(name="x", version="1.0"),
            entity_types={},
            relationship_types={
                "REL": RelationshipTypeDefinition(name="REL"),
            },
        )
        prompt = build_pass2_prompt(ontology, "text")
        assert "REL" in prompt
        # Head-only form has no trailing em-dash content.
        assert "REL** (Any → Any) — " not in prompt

    def test_entity_type_without_description_renders_name_only(self):
        """Below-threshold type with empty description renders ``- **Name**``.

        Closes coverage on the no-description branch of the verbose
        renderer (mirrors a similar Pass-1 test).
        """
        ontology = Ontology(
            metadata=OntologyMetadata(name="minimal", version="1.0"),
            entity_types={
                "OnlyName": EntityTypeDefinition(name="OnlyName"),
            },
        )
        prompt = build_pass2_prompt(ontology, "text")
        assert "- **OnlyName**" in prompt
        # No em-dash after the name when there's no description.
        assert "OnlyName** —" not in prompt

    def test_no_entity_types_renders_stub(self):
        ontology = Ontology(
            metadata=OntologyMetadata(name="empty", version="1.0"),
            entity_types={},
        )
        prompt = build_pass2_prompt(ontology, "text")
        assert "(no entity types defined)" in prompt

    def test_no_relationship_types_renders_stub(self):
        ontology = Ontology(
            metadata=OntologyMetadata(name="ents-only", version="1.0"),
            entity_types={
                "Foo": EntityTypeDefinition(name="Foo", description="bar"),
            },
        )
        prompt = build_pass2_prompt(ontology, "text")
        assert "(no relationship types defined)" in prompt

    def test_compression_kicks_in_above_type_threshold(self):
        """Ontologies with > 30 types drop descriptions (mirrors Pass-1)."""
        ontology = _synthetic_large_ontology(COMPACT_TYPE_THRESHOLD + 1)
        prompt = build_pass2_prompt(ontology, "text")
        assert "{SynthType000" in prompt
        assert "Types compressed" in prompt
        # The filler ``xxxxxx`` description content must NOT appear in
        # the compressed form (that's the whole point).
        assert "xxxxxx" not in prompt

    def test_no_compression_at_threshold(self):
        """Exactly at the threshold descriptions are kept."""
        ontology = _synthetic_large_ontology(COMPACT_TYPE_THRESHOLD)
        prompt = build_pass2_prompt(ontology, "text")
        assert "- **SynthType000**" in prompt
        assert "Types compressed" not in prompt

    def test_relationship_compression_kicks_in(self):
        """Same compression threshold for relationship types."""
        rel_types = {
            f"REL{i:03d}": RelationshipTypeDefinition(name=f"REL{i:03d}")
            for i in range(COMPACT_RELATION_THRESHOLD + 1)
        }
        ontology = Ontology(
            metadata=OntologyMetadata(name="rel-heavy", version="1.0"),
            entity_types={},
            relationship_types=rel_types,
        )
        prompt = build_pass2_prompt(ontology, "text")
        assert "{REL000" in prompt
        assert "Relationship types compressed" in prompt


# ---------------------------------------------------------------------------
# Token-budget guard tests
# ---------------------------------------------------------------------------


class TestTokenBudget:
    """Tests for the budget heuristic + ``Pass2TokenBudgetExceeded`` boundary."""

    def test_estimator_is_len_div_four(self):
        assert _estimate_tokens("") == 0
        assert _estimate_tokens("a" * 4) == 1
        assert _estimate_tokens("a" * 100) == 25

    def test_budget_target_is_2800(self):
        # Legacy default budget. Raised 2400 → 2800 in N.1 to reserve
        # room for the exhaustive-extraction prompt overhead (~+290 tokens).
        assert TOKEN_BUDGET_TARGET == 2800

    async def test_budget_guard_fires_pre_llm(self):
        """Oversized chunk raises ``Pass2TokenBudgetExceeded`` before LLM."""
        # AC #5: synthetic 10K-token chunk → 40K chars, far above budget.
        oversized = "x" * 40_000
        called = False

        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            nonlocal called
            called = True
            return _valid_chunk_response()

        with pytest.raises(Pass2TokenBudgetExceeded) as exc_info:
            await run_pass2(
                [_make_chunk(oversized, "huge")],
                _small_ontology(),
                accepted_extensions=[],
                llm_caller=fake_llm,
            )

        assert exc_info.value.estimated > TOKEN_BUDGET_TARGET
        assert exc_info.value.budget == TOKEN_BUDGET_TARGET
        # Guard fires BEFORE the LLM is invoked — the canary stays cold.
        assert called is False

    async def test_budget_guard_passes_at_normal_size(self):
        """A reasonable chunk + small ontology passes the budget guard."""
        sample = "Alice is a researcher at MIT."
        called = False

        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            nonlocal called
            called = True
            return _valid_chunk_response()

        await run_pass2(
            [_make_chunk(sample, "c1")],
            _small_ontology(),
            accepted_extensions=[],
            llm_caller=fake_llm,
        )
        assert called is True

    async def test_context_budget_admits_big_window(self):
        """Track M: a context-derived ``token_budget`` admits a packed window
        that the legacy fixed 2400 cap would reject (no false-positive guard
        for a big-context model)."""
        # ~5000-token window: rejected by the default 2400, admitted by a
        # 200K context budget.
        big_window = "word " * 4000  # ~20K chars -> ~5000 tokens
        called = False

        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            nonlocal called
            called = True
            return _valid_chunk_response()

        # Default budget rejects it.
        with pytest.raises(Pass2TokenBudgetExceeded):
            await run_pass2(
                [_make_chunk(big_window, "big")],
                _small_ontology(),
                accepted_extensions=[],
                llm_caller=fake_llm,
            )
        assert called is False

        # A context-derived budget admits it.
        await run_pass2(
            [_make_chunk(big_window, "big")],
            _small_ontology(),
            accepted_extensions=[],
            llm_caller=fake_llm,
            token_budget=200_000,
        )
        assert called is True


class TestStressTokenBudget:
    """A 100-type ontology + a 1500-token chunk + 3 extensions fits."""

    def test_realistic_headroom(self):
        ontology = _synthetic_large_ontology(100)
        # 1500 tokens ≈ 6000 chars under len(text) // 4.
        chunk = "x " * 3000
        extensions = [
            {
                "type_name": f"Ext{i}",
                "parent_type": "SynthType000",
                "rationale": "test extension",
            }
            for i in range(3)
        ]
        prompt = build_pass2_prompt(ontology, chunk, extensions)
        estimated = _estimate_tokens(prompt)
        assert estimated <= TOKEN_BUDGET_TARGET, (
            f"100-type ontology + 1500-token chunk + 3 extensions "
            f"produced ~{estimated} tokens (budget {TOKEN_BUDGET_TARGET})."
        )


# ---------------------------------------------------------------------------
# Confidence clamp tests
# ---------------------------------------------------------------------------


class TestClampConfidence:
    """``_clamp_confidence`` is the load-bearing helper for the B4 invariant."""

    def test_none_returns_zero(self):
        assert _clamp_confidence(None) == 0.0

    def test_unparseable_string_returns_zero(self):
        assert _clamp_confidence("high") == 0.0

    def test_percentage_rescaled(self):
        """A scalar > 1.5 is treated as a percentage and divided by 100."""
        assert _clamp_confidence(87) == 0.87

    def test_negative_clamped_to_zero(self):
        assert _clamp_confidence(-0.5) == 0.0

    def test_oversized_clamped_to_one(self):
        """A value in (1, 1.5] is clamped (no auto-rescale)."""
        assert _clamp_confidence(1.4) == 1.0

    def test_in_range_passes_through(self):
        assert _clamp_confidence(0.5) == 0.5
        assert _clamp_confidence(0.0) == 0.0
        assert _clamp_confidence(1.0) == 1.0


# ---------------------------------------------------------------------------
# Output parsing tests
# ---------------------------------------------------------------------------


class TestParseChunkResponse:
    """Tests for ``_parse_chunk_response`` deterministic parsing."""

    def test_well_formed_response(self):
        result = _parse_chunk_response(_valid_chunk_response())
        assert len(result.entities) == 1
        assert result.entities[0].text == "Alice"
        assert result.entities[0].label == "Researcher"
        assert result.entities[0].confidence == 0.92
        assert len(result.relations) == 1
        assert result.relations[0].source_entity == "Alice"
        assert result.relations[0].target_entity == "MIT"
        assert result.relations[0].relation_type == "AFFILIATED_WITH"
        assert result.relations[0].confidence == 0.88

    def test_response_wrapped_in_json_fence(self):
        wrapped = f"Here's the result:\n```json\n{_valid_chunk_response()}\n```"
        result = _parse_chunk_response(wrapped)
        assert len(result.entities) == 1
        assert result.entities[0].text == "Alice"

    def test_response_wrapped_in_plain_fence(self):
        wrapped = f"```\n{_valid_chunk_response()}\n```"
        result = _parse_chunk_response(wrapped)
        assert len(result.entities) == 1

    def test_prose_around_json_salvaged(self):
        wrapped = f"Sure: {_valid_chunk_response()} Hope that helps!"
        result = _parse_chunk_response(wrapped)
        assert len(result.entities) == 1

    def test_legacy_subject_predicate_object_shape(self):
        """Back-compat: the legacy LLMExtractor uses subject/predicate/object."""
        raw = json.dumps(
            {
                "entities": [
                    {"name": "Alice", "entity_type": "Researcher", "confidence": 0.9}
                ],
                "relations": [
                    {
                        "subject": "Alice",
                        "object": "MIT",
                        "predicate": "AFFILIATED_WITH",
                        "confidence": 0.85,
                    }
                ],
            }
        )
        result = _parse_chunk_response(raw)
        assert len(result.entities) == 1
        assert result.entities[0].text == "Alice"
        assert result.entities[0].label == "Researcher"
        assert len(result.relations) == 1
        assert result.relations[0].source_entity == "Alice"

    def test_empty_text_entity_dropped(self):
        raw = json.dumps(
            {
                "entities": [
                    {"text": "", "label": "Researcher", "confidence": 0.5},
                    {"text": "Alice", "label": "Researcher", "confidence": 0.9},
                ],
                "relations": [],
            }
        )
        result = _parse_chunk_response(raw)
        assert len(result.entities) == 1
        assert result.entities[0].text == "Alice"

    def test_relation_missing_endpoint_dropped(self):
        raw = json.dumps(
            {
                "entities": [],
                "relations": [
                    {"source": "Alice", "type": "AFFILIATED_WITH"},
                    {
                        "source": "Alice",
                        "target": "MIT",
                        "type": "AFFILIATED_WITH",
                        "confidence": 0.8,
                    },
                ],
            }
        )
        result = _parse_chunk_response(raw)
        assert len(result.relations) == 1

    def test_missing_confidence_defaults_to_zero(self):
        """The B4 invariant: confidence is always non-null on output."""
        raw = json.dumps(
            {
                "entities": [
                    {"text": "Alice", "label": "Researcher"},
                ],
                "relations": [],
            }
        )
        result = _parse_chunk_response(raw)
        # Confidence is mandatory in the model; default to 0.0 on
        # malformed output rather than dropping the entity entirely.
        assert result.entities[0].confidence == 0.0

    def test_non_dict_entity_silently_dropped(self):
        raw = json.dumps(
            {
                "entities": ["just a string", None, 42, {"text": "Real", "label": "X"}],
                "relations": [],
            }
        )
        result = _parse_chunk_response(raw)
        assert len(result.entities) == 1
        assert result.entities[0].text == "Real"

    def test_non_dict_relation_silently_dropped(self):
        raw = json.dumps(
            {
                "entities": [],
                "relations": [
                    "string",
                    None,
                    {"source": "A", "target": "B", "type": "X", "confidence": 0.5},
                ],
            }
        )
        result = _parse_chunk_response(raw)
        assert len(result.relations) == 1

    def test_non_list_entities_treated_as_empty(self):
        raw = json.dumps(
            {
                "entities": "not a list",
                "relations": [],
            }
        )
        result = _parse_chunk_response(raw)
        assert len(result.entities) == 0

    def test_non_list_relations_treated_as_empty(self):
        raw = json.dumps(
            {
                "entities": [],
                "relations": 42,
            }
        )
        result = _parse_chunk_response(raw)
        assert len(result.relations) == 0

    def test_invalid_json_degrades_gracefully(self):
        """AC #4: malformed JSON → empty result + parse_error metadata."""
        result = _parse_chunk_response("not valid json {{ at all")
        assert len(result.entities) == 0
        assert len(result.relations) == 0
        assert result.metadata.get("parse_error") == "invalid_json"

    def test_empty_response_degrades_gracefully(self):
        result = _parse_chunk_response("")
        assert len(result.entities) == 0
        assert result.metadata.get("parse_error") == "empty_response"

    def test_whitespace_only_response_degrades_gracefully(self):
        result = _parse_chunk_response("   \n\t  ")
        assert result.metadata.get("parse_error") == "empty_response"

    def test_non_object_json_degrades_gracefully(self):
        """A JSON array is structurally invalid → empty result."""
        result = _parse_chunk_response("[1, 2, 3]")
        assert result.metadata.get("parse_error") == "non_object_json"

    def test_salvage_helper_returns_input_when_no_braces(self):
        assert _salvage_json_object("plain text") == "plain text"

    def test_percentage_confidence_rescaled_on_entity(self):
        raw = json.dumps(
            {
                "entities": [
                    {"text": "Alice", "label": "X", "confidence": 87},
                ],
                "relations": [],
            }
        )
        result = _parse_chunk_response(raw)
        assert result.entities[0].confidence == 0.87

    def test_malformed_json_emits_warning_log(self, caplog):
        """AC #4 contract: graceful degradation also logs WARNING."""
        import logging

        from loguru import logger as loguru_logger

        sink_id = loguru_logger.add(
            lambda msg: logging.getLogger("pass2_test").warning(msg.record["message"]),
            level="WARNING",
        )
        try:
            with caplog.at_level(logging.WARNING, logger="pass2_test"):
                _parse_chunk_response("not valid json")
            assert any(
                "Pass-2 returned empty result" in r.message for r in caplog.records
            )
        finally:
            loguru_logger.remove(sink_id)


# ---------------------------------------------------------------------------
# run_pass2 end-to-end (mocked LLM) tests
# ---------------------------------------------------------------------------


class TestRunPass2:
    """End-to-end ``run_pass2`` behaviour with injected LLM callers."""

    async def test_three_chunks_with_no_extensions(self):
        """AC #1: three-chunk fixture produces an ExtractionResult."""
        seen_prompts: List[str] = []

        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            assert sys_p == PASS2_SYSTEM_PROMPT
            seen_prompts.append(user_p)
            return _valid_chunk_response()

        result = await run_pass2(
            _three_chunks(),
            _small_ontology(),
            accepted_extensions=[],
            llm_caller=fake_llm,
        )

        assert len(seen_prompts) == 3
        # Each prompt body includes the chunk text verbatim.
        assert "Alice is a researcher at MIT." in seen_prompts[0]
        # Three chunks × 1 entity each → 3 entities total.
        assert len(result.entities) == 3
        assert len(result.relations) == 3
        # Confidence present and non-null on EVERY entity AND relation
        # (AC #3 — B4 invariant).
        for e in result.entities:
            assert e.confidence is not None
            assert 0.0 <= e.confidence <= 1.0
        for r in result.relations:
            assert r.confidence is not None
            assert 0.0 <= r.confidence <= 1.0
        # Metadata is populated for downstream observability.
        assert result.metadata["chunk_count"] == 3
        assert result.metadata["total_entities"] == 3
        assert result.metadata["total_relations"] == 3
        assert result.metadata["extensions_count"] == 0
        assert result.metadata["ontology_name"] == "scholarly"
        # Source-chunk tagging preserves chunk lineage downstream.
        assert {e.source_chunk_id for e in result.entities} == {"c-1", "c-2", "c-3"}

    async def test_extension_injected_and_used(self):
        """AC #2: extension label flows through to the ExtractedEntity."""
        extensions = [
            {
                "type_name": "EarlyCareerResearcher",
                "parent_type": "Researcher",
                "rationale": "Distinguishes junior researchers.",
            }
        ]

        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            # AC #2: the extension must be visible in the prompt body
            # so the LLM knows the additional label exists.
            assert "EarlyCareerResearcher" in user_p
            assert "parent: Researcher" in user_p
            # Mock returns an entity with the extension label.
            return json.dumps(
                {
                    "entities": [
                        {
                            "text": "Xavier Chen",
                            "label": "EarlyCareerResearcher",
                            "confidence": 0.85,
                            "properties": {},
                        }
                    ],
                    "relations": [],
                }
            )

        result = await run_pass2(
            [_make_chunk("Xavier Chen is a junior researcher.", "c-ext")],
            _small_ontology(),
            accepted_extensions=extensions,
            llm_caller=fake_llm,
        )

        assert len(result.entities) == 1
        assert result.entities[0].label == "EarlyCareerResearcher"
        assert result.entities[0].confidence == 0.85
        assert result.metadata["extensions_count"] == 1

    async def test_confidence_present_on_every_element(self):
        """AC #3: every entity AND every relation carries confidence.

        Mock returns a richer payload across multiple chunks to make
        the loop assertion meaningful.
        """
        payloads = [
            json.dumps(
                {
                    "entities": [
                        {"text": "Alice", "label": "Researcher", "confidence": 0.9},
                        {"text": "Attention Paper", "label": "Paper", "confidence": 0.7},
                    ],
                    "relations": [
                        {
                            "source": "Alice",
                            "target": "Attention Paper",
                            "type": "AUTHORED",
                            "confidence": 0.85,
                        }
                    ],
                }
            ),
            json.dumps(
                {
                    "entities": [
                        {"text": "Carol", "label": "Researcher", "confidence": 0.6},
                    ],
                    "relations": [
                        {
                            "source": "Carol",
                            "target": "MIT",
                            "type": "AFFILIATED_WITH",
                            "confidence": 0.5,
                        }
                    ],
                }
            ),
        ]
        call_idx = {"n": 0}

        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            i = call_idx["n"]
            call_idx["n"] += 1
            return payloads[i]

        result = await run_pass2(
            [_make_chunk("text 1", "c-1"), _make_chunk("text 2", "c-2")],
            _small_ontology(),
            accepted_extensions=[],
            llm_caller=fake_llm,
        )

        assert len(result.entities) == 3
        assert len(result.relations) == 2

        # The B4 invariant — explicit per-element check.
        for e in result.entities:
            assert e.confidence is not None, f"entity {e.text!r} missing confidence"
            assert isinstance(e.confidence, float)
            assert 0.0 <= e.confidence <= 1.0
        for r in result.relations:
            assert r.confidence is not None, (
                f"relation {r.source_entity!r}->{r.target_entity!r} missing confidence"
            )
            assert isinstance(r.confidence, float)
            assert 0.0 <= r.confidence <= 1.0

    async def test_malformed_response_per_chunk_degrades(self):
        """AC #4: a malformed chunk response yields empty entities + relations
        for THAT chunk but does not interrupt the run."""

        responses = iter(
            [
                _valid_chunk_response(),  # chunk 1 succeeds
                "not valid json {{",  # chunk 2 malformed
                _valid_chunk_response(),  # chunk 3 succeeds
            ]
        )

        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            return next(responses)

        result = await run_pass2(
            _three_chunks(),
            _small_ontology(),
            accepted_extensions=[],
            llm_caller=fake_llm,
        )

        # 2 chunks × 1 entity each + 0 from the malformed chunk = 2.
        assert len(result.entities) == 2
        assert len(result.relations) == 2
        assert result.metadata["parse_failures"] == 1

    async def test_empty_chunks_list_returns_empty(self):
        """AC #6: empty list → empty result, no LLM calls."""
        called = False

        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            nonlocal called
            called = True
            return _valid_chunk_response()

        result = await run_pass2(
            [],
            _small_ontology(),
            accepted_extensions=[],
            llm_caller=fake_llm,
        )

        assert called is False
        assert len(result.entities) == 0
        assert len(result.relations) == 0
        assert result.metadata["chunk_count"] == 0
        assert result.metadata["total_entities"] == 0
        assert result.metadata["total_relations"] == 0

    async def test_chunk_with_blank_text_skipped(self):
        """Whitespace-only chunks don't trigger LLM calls."""
        call_count = {"n": 0}

        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            call_count["n"] += 1
            return _valid_chunk_response()

        chunks = [
            _make_chunk("   \n  ", "blank"),
            _make_chunk("real content", "real"),
        ]
        await run_pass2(
            chunks,
            _small_ontology(),
            accepted_extensions=[],
            llm_caller=fake_llm,
        )
        # Only the real chunk hit the LLM.
        assert call_count["n"] == 1

    async def test_async_llm_caller(self):
        """Async callers are awaited correctly (mirrors Pass-1)."""

        async def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            return _valid_chunk_response()

        result = await run_pass2(
            [_make_chunk("text", "c1")],
            _small_ontology(),
            accepted_extensions=[],
            llm_caller=fake_llm,
        )
        assert len(result.entities) == 1

    async def test_llm_caller_transport_error_wrapped(self):
        """Transport errors → ``Pass2ParseError`` (mirrors Pass-1's contract)."""

        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            raise RuntimeError("LLM out of budget")

        with pytest.raises(Pass2ParseError, match="LLM caller failed"):
            await run_pass2(
                [_make_chunk("text", "c1")],
                _small_ontology(),
                accepted_extensions=[],
                llm_caller=fake_llm,
            )

    async def test_async_caller_transport_error_wrapped(self):
        async def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            raise ConnectionError("network down")

        with pytest.raises(Pass2ParseError, match="LLM caller failed"):
            await run_pass2(
                [_make_chunk("text", "c1")],
                _small_ontology(),
                accepted_extensions=[],
                llm_caller=fake_llm,
            )

    async def test_model_arg_threaded_through(self):
        seen_model: List[str] = []

        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            seen_model.append(model)
            return _valid_chunk_response()

        await run_pass2(
            [_make_chunk("text", "c1")],
            _small_ontology(),
            accepted_extensions=[],
            llm_caller=fake_llm,
            model="gpt-4o-mini",
        )
        assert seen_model == ["gpt-4o-mini"]

    async def test_default_model_when_unspecified(self):
        seen_model: List[str] = []

        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            seen_model.append(model)
            return _valid_chunk_response()

        await run_pass2(
            [_make_chunk("text", "c1")],
            _small_ontology(),
            accepted_extensions=[],
            llm_caller=fake_llm,
        )
        assert seen_model == ["default"]

    async def test_none_extensions_treated_as_empty(self):
        """Calling without ``accepted_extensions`` is the back-compat path."""

        def fake_llm(sys_p: str, user_p: str, model: str) -> str:
            assert "Accepted Extension Types" not in user_p
            return _valid_chunk_response()

        result = await run_pass2(
            [_make_chunk("text", "c1")],
            _small_ontology(),
            llm_caller=fake_llm,
        )
        assert result.metadata["extensions_count"] == 0

    async def test_default_llm_caller_resolves_when_unset(self):
        """``llm_caller=None`` falls back to the lazy default (logs canary).

        The lazy default returns ``{}`` so the chunk produces zero
        entities + relations — exactly the safe-degradation path.
        """

        async def stub_caller(sys_p: str, user_p: str, model: str) -> str:
            return "{}"

        with patch(
            "ontology_extraction.pass2_typed_extraction._default_llm_caller",
            return_value=stub_caller,
        ) as mock_default:
            result = await run_pass2(
                [_make_chunk("text", "c1")],
                _small_ontology(),
                accepted_extensions=[],
                llm_caller=None,
            )

        mock_default.assert_called_once()
        assert len(result.entities) == 0
        assert len(result.relations) == 0


# ---------------------------------------------------------------------------
# Public API smoke test
# ---------------------------------------------------------------------------


class TestPublicAPI:
    """The plan's AC #7: imports from the package root must work."""

    def test_run_pass2_exported(self):
        from ontology_extraction import run_pass2 as exported_run_pass2

        assert exported_run_pass2 is run_pass2

    def test_token_budget_exceeded_exported(self):
        from ontology_extraction import Pass2TokenBudgetExceeded as exported

        assert exported is Pass2TokenBudgetExceeded

    def test_pass2_parse_error_exported(self):
        from ontology_extraction import Pass2ParseError as exported

        assert exported is Pass2ParseError
