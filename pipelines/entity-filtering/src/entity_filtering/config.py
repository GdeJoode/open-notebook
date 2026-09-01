"""
Configuration for the entity filtering pipeline.

All domain-specific behavior is injected via config, not hardcoded.
Custom noise patterns, reclassification rules, and articles to strip
are all provided as configuration values.

Sub-configs group related settings for each pipeline stage. All extended
stages default to disabled so the pipeline remains backward-compatible.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class SyntacticConfig:
    """Syntactic pre-processing options applied before any matching.

    Attributes:
        remove_diacritics: Normalize accented characters via unicodedata NFD
            (e.g. e with acute -> e, u with diaeresis -> u).
        ocr_cleanup_enabled: Remove common OCR artifacts from entity text.
        ocr_artifact_patterns: Regex patterns matching OCR noise.
        html_strip_enabled: Strip residual HTML tags from entity text.
        page_number_filter: Discard entities that are bare page numbers.
    """

    remove_diacritics: bool = False
    ocr_cleanup_enabled: bool = False
    ocr_artifact_patterns: List[str] = field(default_factory=list)
    html_strip_enabled: bool = False
    page_number_filter: bool = False


@dataclass
class FuzzyDedupConfig:
    """Fuzzy string-matching deduplication settings.

    Attributes:
        enabled: Whether to run fuzzy deduplication.
        algorithm: Distance algorithm -- "levenshtein" or "jaro_winkler".
        similarity_threshold: Minimum similarity (0-1) to consider a match.
        phonetic_algorithm: Phonetic code -- "soundex", "metaphone", or "none".
        phonetic_weight: Blend weight for phonetic similarity (0-1).
        max_candidates_per_entity: Upper bound on comparisons per entity.
        auto_merge_threshold: K.5 review-band split — pairs scoring at or above
            this are auto-merge candidates. Tuned conservatively HIGH (0.93) so
            only near-identical surface forms (typos/OCR noise) auto-merge; the
            uncertain band below it is queued for human review, never silently
            merged. ``None`` means "use ``similarity_threshold``" (back-compat).
        review_threshold: K.5 review-band floor — pairs scoring in
            ``[review_threshold, auto_merge_threshold)`` are REVIEW candidates
            (queued, never auto-applied); pairs below it are rejected. ``None``
            falls back to ``similarity_threshold``.
    """

    enabled: bool = False
    algorithm: str = "levenshtein"
    similarity_threshold: float = 0.85
    phonetic_algorithm: str = "none"
    phonetic_weight: float = 0.2
    max_candidates_per_entity: int = 10
    # K.5 review-band thresholds (consumed by candidate_dedup_service). Tuned
    # over the must_not_merge corpus: 0.93/0.86 keeps the fuzzy near-miss pairs
    # (``Regio Deal Groningen`` ↔ ``Regio Deal Drenthe`` ≈ 0.83) below the review
    # floor while catching the OCR-typo class (``Koninkrijksrelaties`` ↔
    # ``Koninkrijksreiaties`` ≈ 0.94) as an auto-merge. Edit distance alone
    # cannot separate a 1-char typo from a 1-char *discriminator* in a long
    # shared-prefix name (``Bergen NH``/``Bergen NB`` ≈ 0.94), so the threshold
    # is NOT the sole over-merge guard: candidate_dedup_service applies a
    # deterministic discriminator guard that demotes such pairs to REVIEW
    # regardless of score. See ``CandidateDedupService._is_discriminator_difference``.
    auto_merge_threshold: Optional[float] = 0.93
    review_threshold: Optional[float] = 0.86


@dataclass
class EmbeddingDedupConfig:
    """Embedding-based (semantic) deduplication settings.

    Attributes:
        enabled: Whether to run embedding deduplication.
        similarity_threshold: Cosine similarity threshold for merging.
        k_candidates: Number of nearest neighbours to consider.
        embedding_model: Model identifier for the embedding provider.
            ``None`` means use the pipeline default.
        use_faiss: Use FAISS for approximate nearest-neighbour search.
        auto_merge_threshold: K.5 review-band split for embedding similarity —
            pairs at/above this auto-merge. Default 0.95 (embeddings are noisier
            than exact string matches, so the auto bar is higher than fuzzy's).
        review_threshold: K.5 review-band floor for embedding similarity — pairs
            in ``[review_threshold, auto_merge_threshold)`` are review candidates.
    """

    enabled: bool = False
    similarity_threshold: float = 0.90
    k_candidates: int = 5
    embedding_model: Optional[str] = None
    use_faiss: bool = True
    # K.5 review-band thresholds (consumed by candidate_dedup_service).
    auto_merge_threshold: Optional[float] = 0.95
    review_threshold: Optional[float] = 0.90


@dataclass
class SemanticConfig:
    """Semantic enrichment and entity linking settings.

    Attributes:
        entity_linking_enabled: Whether to link entities to a knowledge base.
        linking_provider: Provider name (e.g. "wikidata", "none").
        linking_confidence_threshold: Minimum confidence for accepting a link.
        contextual_clustering_enabled: Cluster entities by surrounding context.
    """

    entity_linking_enabled: bool = False
    linking_provider: str = "none"
    linking_confidence_threshold: float = 0.7
    contextual_clustering_enabled: bool = False


@dataclass
class KGResolutionConfig:
    """Knowledge-graph entity resolution settings.

    Attributes:
        enabled: Whether to resolve against an existing KG.
        match_strategy: Resolution strategy -- "cascade", "fuzzy", or "semantic".
        fuzzy_threshold: Fuzzy similarity threshold for candidate matching.
        semantic_threshold: Embedding similarity threshold for candidate matching.
        max_candidates: Maximum KG candidates evaluated per entity.
        register_aliases: Store matched surface forms as aliases.
        mark_new_entities: Flag entities not found in the KG as new.
        use_alias_table: Consult the alias table during resolution.
        centrality_aware: Adjust matching thresholds based on KG entity importance.
        centrality_strictness: Amount to raise thresholds for important entities.
        importance_threshold: Entity weight above which it is considered important.
    """

    enabled: bool = False
    match_strategy: str = "cascade"
    fuzzy_threshold: float = 0.85
    semantic_threshold: float = 0.90
    max_candidates: int = 100
    register_aliases: bool = True
    mark_new_entities: bool = True
    use_alias_table: bool = True
    centrality_aware: bool = False
    centrality_strictness: float = 0.05
    importance_threshold: float = 5.0


@dataclass
class ConceptAlignmentConfig:
    """Concept-level alignment settings (Track N.4).

    Classifies the entities KG resolution marked ``is_new`` as RELATED_TO
    something the graph already holds, or NOVEL, instead of leaving them floating.
    Emits no relations: subsumption is planned to move to the type boundary in
    N.4d (D-N4-12), and is currently handled nowhere.

    Attributes:
        enabled: Whether to classify novel concepts at all.
        judge_enabled: Let the LLM-judge arbitrate the ambiguous RELATED/NOVEL
            band (D4, default ON). Without it that band resolves to NOVEL.
        related_floor: Cosine below which nothing is close enough to relate.
        match_ceiling: Cosine at/above which similarity alone implies RELATED_TO
            (mirror of ``KGResolutionConfig.semantic_threshold``).
        max_candidates: Rows per type fetch. The query is ``LIMIT``-capped and
            unordered, so this is an arbitrary sample — the verdicts disclose it.
        min_inner_tokens: Precision guard for the alias-candidate signal.
    """

    enabled: bool = False
    judge_enabled: bool = True
    related_floor: float = 0.75
    match_ceiling: float = 0.90
    max_candidates: int = 100
    min_inner_tokens: int = 2


@dataclass
class OntologyValidationConfig:
    """Ontology and graph-structure validation settings.

    Attributes:
        enabled: Whether to validate entities/relations against an ontology.
        strict_mode: Reject entities not present in the ontology type system.
        filter_invalid_entities: Drop entities with invalid types.
        filter_invalid_relations: Drop relations with invalid predicates.
        graph_centrality_enabled: Compute centrality scores and filter low-signal nodes.
        centrality_min_score: Minimum centrality score for retaining an entity.
        outlier_detection_enabled: Classify entities by extraction centrality + KG status.
        outlier_centrality_low: Centrality score below which an entity is "low".
    """

    enabled: bool = False
    strict_mode: bool = False
    filter_invalid_entities: bool = True
    filter_invalid_relations: bool = True
    graph_centrality_enabled: bool = False
    centrality_min_score: float = 0.01
    outlier_detection_enabled: bool = False
    outlier_centrality_low: float = 0.05


@dataclass
class LLMVerificationConfig:
    """LLM-based verification and self-correction settings.

    Attributes:
        enabled: Whether to use an LLM for entity/relation verification.
        verify_triples: Ask the LLM to verify extracted triples.
        schema_alignment_enabled: Use the LLM to align types with the schema.
        self_correction_enabled: Allow the LLM to iteratively refine results.
        self_correction_max_iterations: Max self-correction rounds.
        llm_model: Model identifier for the LLM provider.
            ``None`` means use the pipeline default.
    """

    enabled: bool = False
    verify_triples: bool = False
    schema_alignment_enabled: bool = False
    self_correction_enabled: bool = False
    self_correction_max_iterations: int = 2
    llm_model: Optional[str] = None


@dataclass
class SemanticBlockingConfig:
    """UMAP + HDBSCAN semantic blocking settings.

    Attributes:
        enabled: Whether to run semantic blocking before pairwise dedup.
        umap_n_components: UMAP target dimensions (lower = faster, less precise).
        min_cluster_size: Minimum entities to form a block (2 = allow pairs).
        min_samples: HDBSCAN density parameter (1 = permissive).
    """

    enabled: bool = False
    umap_n_components: int = 20
    min_cluster_size: int = 2
    min_samples: int = 1


@dataclass
class LLMMatcherConfig:
    """LLM-based entity matching settings.

    Attributes:
        enabled: Whether to run LLM matching on candidate pairs.
        model: Ollama model name for matching.
        base_url: Ollama API endpoint.
        confidence_threshold: Minimum LLM confidence to accept a match.
        timeout: HTTP timeout per LLM request in seconds.
        agentic_enabled: Enable iterative reasoning for uncertain matches.
        agentic_lower_threshold: Below this → reject without iteration.
        agentic_upper_threshold: Above this → accept without iteration.
        agentic_max_iterations: Max extra context-fetch rounds (1-2 recommended).
    """

    enabled: bool = False
    model: str = "qwen3.5:35b-a3b"
    base_url: str = "http://localhost:11434"
    confidence_threshold: float = 0.7
    timeout: int = 60
    agentic_enabled: bool = False
    agentic_lower_threshold: float = 0.3
    agentic_upper_threshold: float = 0.8
    agentic_max_iterations: int = 2


@dataclass
class IncrementalResolutionConfig:
    """Incremental entity resolution settings.

    Assigns new entities to existing clusters based on embedding
    similarity and periodically repairs cluster quality.

    Attributes:
        enabled: Whether to run incremental cluster resolution.
        similarity_threshold: Minimum cosine similarity to assign to a cluster.
        coherence_threshold: Minimum internal coherence before splitting.
        merge_threshold: Maximum inter-cluster similarity before merging.
        repair_enabled: Run cluster repair after resolution.
    """

    enabled: bool = False
    similarity_threshold: float = 0.85
    coherence_threshold: float = 0.70
    merge_threshold: float = 0.92
    repair_enabled: bool = True


@dataclass
class OrphanConnectorConfig:
    """Orphan-connector (B.5a) settings.

    The orphan-connector runs after dedup but before persistence. It
    detects entities that ended up with zero relations, proposes new
    relations via chunk co-occurrence, and asks an injected LLM caller
    to confirm or deny each proposal.

    Attributes:
        enabled: Whether to run the orphan connector. Default ``True``
            so the pipeline picks up orphans automatically; flip to
            ``False`` for evaluation runs that want to measure orphan
            counts before reconnection.
        max_proposals_per_orphan: Cap on candidate partners per orphan
            to bound the LLM-call count on busy sources.
        min_confidence: Discard confirmations below this LLM-reported
            confidence. Conservative default — matches the
            extraction-time confidence threshold used elsewhere in the
            pipeline.
    """

    enabled: bool = True
    max_proposals_per_orphan: int = 3
    min_confidence: float = 0.6


@dataclass
class EdgePredictionConfig:
    """Edge (relation) prediction and scoring settings.

    Attributes:
        enabled: Whether to run edge prediction scoring.
        cosine_weight: Weight for cosine similarity signal.
        adamic_adar_weight: Weight for Adamic-Adar index signal.
        common_ancestors_weight: Weight for common-ancestors signal.
        similarity_threshold: Minimum combined score to emit a predicted edge.
        k_neighbors: Number of neighbours for graph-based signals.
    """

    enabled: bool = False
    cosine_weight: float = 0.6
    adamic_adar_weight: float = 0.3
    common_ancestors_weight: float = 0.1
    similarity_threshold: float = 0.5
    k_neighbors: int = 10


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
        syntactic: Syntactic pre-processing sub-config.
        fuzzy_dedup: Fuzzy deduplication sub-config.
        embedding_dedup: Embedding deduplication sub-config.
        semantic: Semantic enrichment sub-config.
        kg_resolution: Knowledge-graph resolution sub-config.
        ontology_validation: Ontology validation sub-config.
        llm_verification: LLM verification sub-config.
        edge_prediction: Edge prediction sub-config.
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

    # Extended sub-configs (all disabled by default)
    syntactic: SyntacticConfig = field(default_factory=SyntacticConfig)
    fuzzy_dedup: FuzzyDedupConfig = field(default_factory=FuzzyDedupConfig)
    embedding_dedup: EmbeddingDedupConfig = field(
        default_factory=EmbeddingDedupConfig
    )
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    kg_resolution: KGResolutionConfig = field(
        default_factory=KGResolutionConfig
    )
    concept_alignment: ConceptAlignmentConfig = field(
        default_factory=ConceptAlignmentConfig
    )
    ontology_validation: OntologyValidationConfig = field(
        default_factory=OntologyValidationConfig
    )
    llm_verification: LLMVerificationConfig = field(
        default_factory=LLMVerificationConfig
    )
    semantic_blocking: SemanticBlockingConfig = field(
        default_factory=SemanticBlockingConfig
    )
    llm_matcher: LLMMatcherConfig = field(
        default_factory=LLMMatcherConfig
    )
    incremental_resolution: IncrementalResolutionConfig = field(
        default_factory=IncrementalResolutionConfig
    )
    edge_prediction: EdgePredictionConfig = field(
        default_factory=EdgePredictionConfig
    )
    orphan_connector: OrphanConnectorConfig = field(
        default_factory=OrphanConnectorConfig
    )
