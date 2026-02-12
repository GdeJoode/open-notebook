"""
Configuration for the entity filtering pipeline.

All domain-specific behavior is injected via config, not hardcoded.
Custom noise patterns, reclassification rules, and articles to strip
are all provided as configuration values.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class FilteringConfig:
    """Configuration for the entity filtering pipeline.

    Attributes:
        min_entity_length: Minimum character length for an entity to be kept.
        custom_noise_patterns: Additional regex patterns to treat as noise.
        strip_articles: Whether to strip leading articles during normalization.
        custom_articles: Extra leading articles to strip (language-specific).
        normalize_whitespace: Collapse multiple whitespace characters.
        custom_reclassification_rules: Mapping of regex pattern to new label.
        dedup_enabled: Whether to run deduplication.
        dedup_similarity_threshold: Threshold for considering two entities
            as duplicates (0.0-1.0).
        edge_prediction_enabled: Whether to run edge prediction scoring.
        treekg_enabled: Whether to run TreeKG summarization.
        raptor_enabled: Whether to run RAPTOR summarization.
    """

    # Noise filtering
    min_entity_length: int = 2
    custom_noise_patterns: List[str] = field(default_factory=list)

    # Normalization
    strip_articles: bool = True
    custom_articles: List[str] = field(default_factory=list)
    normalize_whitespace: bool = True

    # Reclassification
    custom_reclassification_rules: Dict[str, str] = field(default_factory=dict)

    # Deduplication
    dedup_enabled: bool = True
    dedup_similarity_threshold: float = 0.85

    # Edge scoring
    edge_prediction_enabled: bool = False

    # Summarization (TreeKG/RAPTOR)
    treekg_enabled: bool = False
    raptor_enabled: bool = False
