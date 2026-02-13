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

    def test_custom_model_and_threshold(self):
        """Custom values are stored correctly."""
        extractor = LLMExtractor(llm_model="gpt-4o", confidence_threshold=0.9)
        assert extractor._llm_model == "gpt-4o"
        assert extractor._confidence_threshold == 0.9
