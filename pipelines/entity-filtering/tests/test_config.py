"""Tests for FilteringConfig dataclass defaults and custom values."""

from entity_filtering.config import FilteringConfig


class TestFilteringConfig:
    def test_defaults(self):
        config = FilteringConfig()
        assert config.min_entity_length == 2
        assert config.dedup_enabled is True
        assert config.dedup_similarity_threshold == 0.85
        assert config.edge_prediction_enabled is False
        assert config.treekg_enabled is False
        assert config.raptor_enabled is False
        assert config.custom_noise_patterns == []
        assert config.custom_reclassification_rules == {}
        assert config.strip_articles is True
        assert config.custom_articles == []
        assert config.normalize_whitespace is True

    def test_custom_config(self):
        config = FilteringConfig(
            min_entity_length=3,
            custom_noise_patterns=[r"^\d+$"],
            dedup_similarity_threshold=0.9,
        )
        assert config.min_entity_length == 3
        assert len(config.custom_noise_patterns) == 1
        assert config.dedup_similarity_threshold == 0.9

    def test_custom_reclassification_rules(self):
        rules = {r"^Prof\.\s": "PERSON", r"^Dr\.\s": "PERSON"}
        config = FilteringConfig(custom_reclassification_rules=rules)
        assert len(config.custom_reclassification_rules) == 2
        assert config.custom_reclassification_rules[r"^Prof\.\s"] == "PERSON"

    def test_disable_deduplication(self):
        config = FilteringConfig(dedup_enabled=False)
        assert config.dedup_enabled is False

    def test_enable_optional_stages(self):
        config = FilteringConfig(
            edge_prediction_enabled=True,
            treekg_enabled=True,
            raptor_enabled=True,
        )
        assert config.edge_prediction_enabled is True
        assert config.treekg_enabled is True
        assert config.raptor_enabled is True

    def test_custom_articles(self):
        config = FilteringConfig(custom_articles=["De ", "Het "])
        assert len(config.custom_articles) == 2
        assert "De " in config.custom_articles
