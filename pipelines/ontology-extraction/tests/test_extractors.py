"""Tests for ontology_extraction.extractors.llm_extractor module.

Only the _parse_response method is tested here since it is the only
deterministic path that does not require a live LLM or ontology-manager.
"""

import json

from ontology_extraction.extractors.llm_extractor import LLMExtractor


class TestLLMExtractorParseResponse:
    """Tests for LLMExtractor._parse_response deterministic parsing logic."""

    def test_parse_valid_json(self):
        """Valid JSON with entities and relationships is parsed correctly."""
        extractor = LLMExtractor(confidence_threshold=0.5)
        response = json.dumps(
            {
                "entities": [
                    {"name": "John Doe", "entity_type": "Person", "confidence": 0.9},
                    {"name": "Acme Corp", "entity_type": "Organization", "confidence": 0.8},
                ],
                "relationships": [
                    {
                        "subject": "John Doe",
                        "predicate": "WORKS_AT",
                        "object": "Acme Corp",
                        "confidence": 0.7,
                    },
                ],
                "concepts": [],
                "claims": [],
            }
        )
        result = extractor._parse_response(response)

        assert len(result.entities) == 2
        assert result.entities[0].text == "John Doe"
        assert result.entities[0].label == "Person"
        assert result.entities[0].confidence == 0.9
        assert result.entities[1].text == "Acme Corp"
        assert result.entities[1].label == "Organization"

        assert len(result.relations) == 1
        assert result.relations[0].source_entity == "John Doe"
        assert result.relations[0].target_entity == "Acme Corp"
        assert result.relations[0].relation_type == "WORKS_AT"
        assert result.relations[0].confidence == 0.7

    def test_parse_markdown_wrapped_json(self):
        """JSON wrapped in ```json ... ``` code blocks is extracted and parsed."""
        extractor = LLMExtractor()
        response = (
            "Here is the extraction:\n"
            "```json\n"
            '{\n'
            '    "entities": [{"name": "Test Entity", "entity_type": "Thing", "confidence": 0.9}],\n'
            '    "relationships": []\n'
            "}\n"
            "```"
        )
        result = extractor._parse_response(response)

        assert len(result.entities) == 1
        assert result.entities[0].text == "Test Entity"
        assert result.entities[0].label == "Thing"
        assert len(result.relations) == 0

    def test_parse_plain_code_block(self):
        """JSON wrapped in plain ``` ... ``` (no language tag) is parsed."""
        extractor = LLMExtractor()
        response = (
            "Result:\n"
            "```\n"
            '{"entities": [{"name": "Foo", "entity_type": "Bar", "confidence": 0.8}], "relationships": []}\n'
            "```"
        )
        result = extractor._parse_response(response)

        assert len(result.entities) == 1
        assert result.entities[0].text == "Foo"

    def test_parse_entities_only_no_relationships(self):
        """Response with entities but no relationships key defaults to empty relations."""
        extractor = LLMExtractor()
        response = json.dumps(
            {
                "entities": [
                    {"name": "Solo Entity", "entity_type": "Concept", "confidence": 0.85},
                ],
            }
        )
        result = extractor._parse_response(response)

        assert len(result.entities) == 1
        assert result.entities[0].text == "Solo Entity"
        assert len(result.relations) == 0

    def test_parse_invalid_json_returns_empty_result_with_error(self):
        """Completely invalid JSON produces an empty result with parse_error metadata."""
        extractor = LLMExtractor()
        result = extractor._parse_response("not valid json at all")

        assert len(result.entities) == 0
        assert len(result.relations) == 0
        assert "parse_error" in result.metadata

    def test_parse_empty_string(self):
        """Empty string produces an empty result with parse_error metadata."""
        extractor = LLMExtractor()
        result = extractor._parse_response("")

        assert len(result.entities) == 0
        assert "parse_error" in result.metadata

    def test_confidence_filtering_above_threshold(self):
        """Entities and relations below confidence threshold are excluded."""
        extractor = LLMExtractor(confidence_threshold=0.8)
        response = json.dumps(
            {
                "entities": [
                    {"name": "High", "entity_type": "X", "confidence": 0.9},
                    {"name": "Low", "entity_type": "X", "confidence": 0.3},
                    {"name": "Exact", "entity_type": "X", "confidence": 0.8},
                ],
                "relationships": [
                    {
                        "subject": "A",
                        "predicate": "REL",
                        "object": "B",
                        "confidence": 0.95,
                    },
                    {
                        "subject": "C",
                        "predicate": "REL",
                        "object": "D",
                        "confidence": 0.5,
                    },
                ],
            }
        )
        result = extractor._parse_response(response)

        # Only High (0.9) and Exact (0.8 >= 0.8) should pass
        assert len(result.entities) == 2
        entity_names = [e.text for e in result.entities]
        assert "High" in entity_names
        assert "Exact" in entity_names
        assert "Low" not in entity_names

        # Only the 0.95 relation should pass
        assert len(result.relations) == 1
        assert result.relations[0].confidence == 0.95

    def test_confidence_filtering_zero_threshold_keeps_all(self):
        """A confidence threshold of 0.0 keeps all entities."""
        extractor = LLMExtractor(confidence_threshold=0.0)
        response = json.dumps(
            {
                "entities": [
                    {"name": "A", "entity_type": "X", "confidence": 0.01},
                    {"name": "B", "entity_type": "X", "confidence": 0.0},
                ],
                "relationships": [],
            }
        )
        result = extractor._parse_response(response)

        assert len(result.entities) == 2

    def test_metadata_includes_concept_and_claim_counts(self):
        """Successful parsing records concept_count and claim_count in metadata."""
        extractor = LLMExtractor()
        response = json.dumps(
            {
                "entities": [],
                "relationships": [],
                "concepts": [{"name": "c1"}, {"name": "c2"}],
                "claims": [{"text": "cl1"}],
            }
        )
        result = extractor._parse_response(response)

        assert result.metadata["concept_count"] == 2
        assert result.metadata["claim_count"] == 1

    def test_missing_confidence_defaults_to_one(self):
        """Entities without a confidence key default to 1.0."""
        extractor = LLMExtractor(confidence_threshold=0.5)
        response = json.dumps(
            {
                "entities": [
                    {"name": "NoConf", "entity_type": "Y"},
                ],
                "relationships": [],
            }
        )
        result = extractor._parse_response(response)

        assert len(result.entities) == 1
        assert result.entities[0].confidence == 1.0

    def test_missing_entity_type_defaults_to_unknown(self):
        """Entities without entity_type get label UNKNOWN."""
        extractor = LLMExtractor()
        response = json.dumps(
            {
                "entities": [
                    {"name": "Mystery"},
                ],
                "relationships": [],
            }
        )
        result = extractor._parse_response(response)

        assert len(result.entities) == 1
        assert result.entities[0].label == "UNKNOWN"

    def test_missing_predicate_defaults_to_related_to(self):
        """Relations without predicate get relation_type RELATED_TO."""
        extractor = LLMExtractor()
        response = json.dumps(
            {
                "entities": [],
                "relationships": [
                    {"subject": "A", "object": "B"},
                ],
            }
        )
        result = extractor._parse_response(response)

        assert len(result.relations) == 1
        assert result.relations[0].relation_type == "RELATED_TO"

    def test_entity_properties_preserved(self):
        """Extra properties on entities are passed through."""
        extractor = LLMExtractor()
        response = json.dumps(
            {
                "entities": [
                    {
                        "name": "Prop",
                        "entity_type": "X",
                        "confidence": 0.9,
                        "properties": {"role": "CEO", "age": 42},
                    },
                ],
                "relationships": [],
            }
        )
        result = extractor._parse_response(response)

        assert result.entities[0].properties == {"role": "CEO", "age": 42}


class TestLLMExtractorInit:
    """Tests for LLMExtractor constructor defaults."""

    def test_default_model_and_threshold(self):
        """Default constructor sets model to 'default' and threshold to 0.5."""
        extractor = LLMExtractor()
        assert extractor._llm_model == "default"
        assert extractor._confidence_threshold == 0.5
        # B.1f: caller defaults to None (legacy "silent empty" path,
        # but now with an explicit WARNING canary).
        assert extractor._llm_caller is None

    def test_custom_model_and_threshold(self):
        """Custom values are stored correctly."""
        extractor = LLMExtractor(llm_model="gpt-4o", confidence_threshold=0.9)
        assert extractor._llm_model == "gpt-4o"
        assert extractor._confidence_threshold == 0.9


class TestLLMExtractorCallerWiring:
    """B.1f: LLMExtractor accepts an injected LLM caller and dispatches
    through it. Pins that the pre-B.1f broken import path (``LLMManager``
    + ``manager.generate``) is gone and that the extractor wires to a
    real caller without ImportError."""

    def test_constructs_with_async_caller_no_import_error(self):
        """The constructor binds an async caller without touching
        ``llm_manager.manager`` (the pre-B.1f import path). Pin via
        attribute identity — if a future refactor accidentally
        re-introduces a module-level import the test stays green only
        if the binding survives."""

        async def caller(system: str, user: str, model: str) -> str:
            return "{}"

        extractor = LLMExtractor(llm_caller=caller)
        # Caller bound, not None, ready to dispatch.
        assert extractor._llm_caller is caller

    def test_extract_dispatches_to_injected_caller(self):
        """End-to-end: injected caller's output flows through
        _parse_response. Verifies the wire is genuinely connected, not
        just stored."""

        import asyncio

        from ontology_manager.schema import (
            EntityTypeDefinition,
            Ontology,
            OntologyMetadata,
        )

        async def caller(system: str, user: str, model: str) -> str:
            # Verify the caller is invoked with both prompts and the
            # model id — pins the LLMCaller contract.
            assert "Extract knowledge" in user
            assert model == "default"
            return json.dumps(
                {
                    "entities": [
                        {
                            "name": "Pinned Entity",
                            "entity_type": "Thing",
                            "confidence": 0.95,
                        }
                    ],
                    "relationships": [],
                }
            )

        extractor = LLMExtractor(llm_caller=caller, confidence_threshold=0.5)
        ontology = Ontology(
            metadata=OntologyMetadata(name="t", version="1.0"),
            entity_types={
                "Thing": EntityTypeDefinition(
                    name="Thing", description="anything"
                )
            },
        )

        result = asyncio.run(extractor.extract("Some text", ontology))

        # The injected caller's JSON travelled through _parse_response
        # — pin via entity content rather than just count, so any
        # short-circuit return path is detected.
        assert len(result.entities) == 1
        assert result.entities[0].text == "Pinned Entity"
        assert result.entities[0].confidence == 0.95

    def test_extract_no_caller_returns_empty_without_import_error(self):
        """Backwards-compat: no caller wired → empty result + WARNING.
        Critically, no ``ImportError`` for ``llm_manager.manager`` —
        the pre-B.1f code path is gone."""
        import asyncio

        from ontology_manager.schema import (
            EntityTypeDefinition,
            Ontology,
            OntologyMetadata,
        )

        extractor = LLMExtractor()  # no llm_caller
        ontology = Ontology(
            metadata=OntologyMetadata(name="t", version="1.0"),
            entity_types={
                "Thing": EntityTypeDefinition(name="Thing", description="x")
            },
        )

        # The whole extract() call must complete without raising.
        result = asyncio.run(extractor.extract("Text", ontology))
        assert result.entities == []
        assert result.relations == []

    def test_pre_b1f_import_path_no_longer_referenced(self):
        """Regression guard: the broken
        ``from llm_manager.manager import LLMManager`` import is gone.

        Compile the module AST and inspect import statements +
        function calls — checking source text would match the
        docstring's historical mention of the bad path.
        """
        import ast
        import inspect

        from ontology_extraction.extractors import llm_extractor as mod

        tree = ast.parse(inspect.getsource(mod))

        # Walk imports — no module attempts to bring ``LLMManager`` in.
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name != "LLMManager", (
                        f"Pre-B.1f broken import path resurfaced: "
                        f"from {node.module} import {alias.name}"
                    )
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert "LLMManager" not in alias.name

        # Walk calls — no ``X.generate(...)`` where X is named ``manager``
        # (the non-existent method on the old class).
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "generate"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "manager"
            ):
                raise AssertionError(
                    "Pre-B.1f broken method call ``manager.generate(...)`` "
                    "resurfaced — the ModelManager API is achat_complete."
                )
