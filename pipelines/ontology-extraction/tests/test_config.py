"""Tests for ontology_extraction.config module."""

from ontology_extraction.config import ExtractionConfig


class TestExtractionConfig:
    """Tests for the ExtractionConfig dataclass."""

    def test_defaults(self):
        """All default values are applied when no arguments given."""
        config = ExtractionConfig()
        assert config.ontology_name == "general"
        assert config.llm_model == "default"
        assert config.max_entities_per_chunk == 50
        assert config.max_relations_per_chunk == 30
        assert config.batch_size == 10
        assert config.include_concepts is True
        assert config.include_claims is True
        assert config.confidence_threshold == 0.5

    def test_custom_config(self):
        """Provided values override defaults while others remain."""
        config = ExtractionConfig(ontology_name="scholarly", batch_size=5)
        assert config.ontology_name == "scholarly"
        assert config.batch_size == 5
        # Unset fields keep their defaults
        assert config.llm_model == "default"
        assert config.confidence_threshold == 0.5

    def test_all_fields_overridden(self):
        """Every field can be set explicitly."""
        config = ExtractionConfig(
            ontology_name="medical",
            llm_model="gpt-4",
            max_entities_per_chunk=100,
            max_relations_per_chunk=60,
            batch_size=20,
            include_concepts=False,
            include_claims=False,
            confidence_threshold=0.9,
        )
        assert config.ontology_name == "medical"
        assert config.llm_model == "gpt-4"
        assert config.max_entities_per_chunk == 100
        assert config.max_relations_per_chunk == 60
        assert config.batch_size == 20
        assert config.include_concepts is False
        assert config.include_claims is False
        assert config.confidence_threshold == 0.9

    def test_confidence_threshold_boundary_values(self):
        """Confidence threshold accepts boundary float values."""
        zero = ExtractionConfig(confidence_threshold=0.0)
        assert zero.confidence_threshold == 0.0

        one = ExtractionConfig(confidence_threshold=1.0)
        assert one.confidence_threshold == 1.0

    def test_is_dataclass_instance(self):
        """ExtractionConfig is a proper dataclass."""
        import dataclasses

        assert dataclasses.is_dataclass(ExtractionConfig)
        config = ExtractionConfig()
        assert dataclasses.is_dataclass(config)
