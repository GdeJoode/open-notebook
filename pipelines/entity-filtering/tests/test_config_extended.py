"""Tests for the extended sub-config dataclasses added in Phase 1.

Validates that all new sub-configs have correct defaults, that
FilteringConfig includes them with the right factory defaults, and
that constructing FilteringConfig() with or without sub-configs
remains backward-compatible.
"""

from entity_filtering.config import (
    EdgePredictionConfig,
    EmbeddingDedupConfig,
    FilteringConfig,
    FuzzyDedupConfig,
    KGResolutionConfig,
    LLMVerificationConfig,
    OntologyValidationConfig,
    SemanticConfig,
    SyntacticConfig,
)


# ------------------------------------------------------------------
# Individual sub-config defaults
# ------------------------------------------------------------------


class TestSyntacticConfigDefaults:
    def test_all_features_disabled_by_default(self):
        cfg = SyntacticConfig()
        assert cfg.remove_diacritics is False
        assert cfg.ocr_cleanup_enabled is False
        assert cfg.html_strip_enabled is False
        assert cfg.page_number_filter is False

    def test_ocr_artifact_patterns_empty_by_default(self):
        cfg = SyntacticConfig()
        assert cfg.ocr_artifact_patterns == []

    def test_custom_values(self):
        cfg = SyntacticConfig(
            remove_diacritics=True,
            ocr_cleanup_enabled=True,
            ocr_artifact_patterns=[r"\f", r"\u00ad"],
            html_strip_enabled=True,
            page_number_filter=True,
        )
        assert cfg.remove_diacritics is True
        assert cfg.ocr_cleanup_enabled is True
        assert len(cfg.ocr_artifact_patterns) == 2
        assert cfg.html_strip_enabled is True
        assert cfg.page_number_filter is True


class TestFuzzyDedupConfigDefaults:
    def test_disabled_by_default(self):
        cfg = FuzzyDedupConfig()
        assert cfg.enabled is False

    def test_algorithm_defaults(self):
        cfg = FuzzyDedupConfig()
        assert cfg.algorithm == "levenshtein"
        assert cfg.similarity_threshold == 0.85
        assert cfg.phonetic_algorithm == "none"
        assert cfg.phonetic_weight == 0.2
        assert cfg.max_candidates_per_entity == 10

    def test_custom_values(self):
        cfg = FuzzyDedupConfig(
            enabled=True,
            algorithm="jaro_winkler",
            similarity_threshold=0.9,
            phonetic_algorithm="soundex",
            phonetic_weight=0.3,
            max_candidates_per_entity=20,
        )
        assert cfg.enabled is True
        assert cfg.algorithm == "jaro_winkler"
        assert cfg.similarity_threshold == 0.9
        assert cfg.phonetic_algorithm == "soundex"
        assert cfg.phonetic_weight == 0.3
        assert cfg.max_candidates_per_entity == 20


class TestEmbeddingDedupConfigDefaults:
    def test_disabled_by_default(self):
        cfg = EmbeddingDedupConfig()
        assert cfg.enabled is False

    def test_defaults(self):
        cfg = EmbeddingDedupConfig()
        assert cfg.similarity_threshold == 0.90
        assert cfg.k_candidates == 5
        assert cfg.embedding_model is None
        assert cfg.use_faiss is True


class TestSemanticConfigDefaults:
    def test_all_features_disabled_by_default(self):
        cfg = SemanticConfig()
        assert cfg.entity_linking_enabled is False
        assert cfg.contextual_clustering_enabled is False

    def test_defaults(self):
        cfg = SemanticConfig()
        assert cfg.linking_provider == "none"
        assert cfg.linking_confidence_threshold == 0.7


class TestKGResolutionConfigDefaults:
    def test_disabled_by_default(self):
        cfg = KGResolutionConfig()
        assert cfg.enabled is False

    def test_defaults(self):
        cfg = KGResolutionConfig()
        assert cfg.match_strategy == "cascade"
        assert cfg.fuzzy_threshold == 0.85
        assert cfg.semantic_threshold == 0.90
        assert cfg.max_candidates == 100
        # PC.2: a fuzzy match no longer writes an alias by itself. See
        # `test_alias_policy.py` for the policy and the reason; this line only
        # records that the default moved deliberately.
        assert cfg.register_aliases is False
        assert cfg.mark_new_entities is True
        assert cfg.use_alias_table is True


class TestOntologyValidationConfigDefaults:
    def test_disabled_by_default(self):
        cfg = OntologyValidationConfig()
        assert cfg.enabled is False

    def test_defaults(self):
        cfg = OntologyValidationConfig()
        assert cfg.strict_mode is False
        assert cfg.filter_invalid_entities is True
        assert cfg.filter_invalid_relations is True
        assert cfg.graph_centrality_enabled is False
        assert cfg.centrality_min_score == 0.01


class TestLLMVerificationConfigDefaults:
    def test_disabled_by_default(self):
        cfg = LLMVerificationConfig()
        assert cfg.enabled is False

    def test_defaults(self):
        cfg = LLMVerificationConfig()
        assert cfg.verify_triples is False
        assert cfg.schema_alignment_enabled is False
        assert cfg.self_correction_enabled is False
        assert cfg.self_correction_max_iterations == 2
        assert cfg.llm_model is None


class TestEdgePredictionConfigDefaults:
    def test_disabled_by_default(self):
        cfg = EdgePredictionConfig()
        assert cfg.enabled is False

    def test_defaults(self):
        cfg = EdgePredictionConfig()
        assert cfg.cosine_weight == 0.6
        assert cfg.adamic_adar_weight == 0.3
        assert cfg.common_ancestors_weight == 0.1
        assert cfg.similarity_threshold == 0.5
        assert cfg.k_neighbors == 10


# ------------------------------------------------------------------
# FilteringConfig integration with sub-configs
# ------------------------------------------------------------------


class TestFilteringConfigSubConfigs:
    def test_default_construction_succeeds(self):
        """Constructing FilteringConfig() must not raise."""
        config = FilteringConfig()
        assert config is not None

    def test_existing_fields_unchanged(self):
        """Sub-configs do not break existing FilteringConfig fields."""
        config = FilteringConfig()
        assert config.min_entity_length == 2
        assert config.dedup_enabled is True
        assert config.dedup_similarity_threshold == 0.85
        assert config.edge_prediction_enabled is False
        assert config.treekg_enabled is False
        assert config.raptor_enabled is False
        assert config.strip_articles is True
        assert config.normalize_whitespace is True
        assert config.custom_noise_patterns == []
        assert config.custom_articles == []
        assert config.custom_reclassification_rules == {}

    def test_sub_configs_are_correct_types(self):
        config = FilteringConfig()
        assert isinstance(config.syntactic, SyntacticConfig)
        assert isinstance(config.fuzzy_dedup, FuzzyDedupConfig)
        assert isinstance(config.embedding_dedup, EmbeddingDedupConfig)
        assert isinstance(config.semantic, SemanticConfig)
        assert isinstance(config.kg_resolution, KGResolutionConfig)
        assert isinstance(config.ontology_validation, OntologyValidationConfig)
        assert isinstance(config.llm_verification, LLMVerificationConfig)
        assert isinstance(config.edge_prediction, EdgePredictionConfig)

    def test_sub_configs_all_disabled_by_default(self):
        """Every extended stage is opt-in -- disabled out of the box."""
        config = FilteringConfig()
        assert config.syntactic.remove_diacritics is False
        assert config.syntactic.html_strip_enabled is False
        assert config.syntactic.page_number_filter is False
        assert config.syntactic.ocr_cleanup_enabled is False
        assert config.fuzzy_dedup.enabled is False
        assert config.embedding_dedup.enabled is False
        assert config.semantic.entity_linking_enabled is False
        assert config.kg_resolution.enabled is False
        assert config.ontology_validation.enabled is False
        assert config.llm_verification.enabled is False
        assert config.edge_prediction.enabled is False

    def test_sub_configs_can_be_passed_at_construction(self):
        syn = SyntacticConfig(remove_diacritics=True)
        fuzzy = FuzzyDedupConfig(enabled=True, similarity_threshold=0.9)
        config = FilteringConfig(syntactic=syn, fuzzy_dedup=fuzzy)
        assert config.syntactic.remove_diacritics is True
        assert config.fuzzy_dedup.enabled is True
        assert config.fuzzy_dedup.similarity_threshold == 0.9

    def test_sub_configs_are_independent_instances(self):
        """Each FilteringConfig gets its own sub-config instances."""
        config_a = FilteringConfig()
        config_b = FilteringConfig()
        assert config_a.syntactic is not config_b.syntactic
        assert config_a.fuzzy_dedup is not config_b.fuzzy_dedup

    def test_sub_configs_accessible_after_construction(self):
        config = FilteringConfig()
        config.syntactic.remove_diacritics = True
        config.fuzzy_dedup.enabled = True
        assert config.syntactic.remove_diacritics is True
        assert config.fuzzy_dedup.enabled is True
