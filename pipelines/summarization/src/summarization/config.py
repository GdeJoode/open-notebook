"""
Configuration for the summarization pipeline.

All strategy-specific behavior is injected via config dataclasses.
Sub-configs group related settings for each strategy.
"""

import os
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """LLM provider settings shared by all strategies."""

    model_name: str = field(
        default_factory=lambda: os.getenv("SUMMARIZATION_MODEL", "llama3.1:8b-instruct-q4_0")
    )
    provider: str = field(
        default_factory=lambda: os.getenv("SUMMARIZATION_PROVIDER", "ollama")
    )
    base_url: str = field(
        default_factory=lambda: os.getenv("SUMMARIZATION_BASE_URL", "http://localhost:11434")
    )
    temperature: float = 0.3
    max_tokens: int = 3000
    timeout: int = 300
    num_ctx: int = field(
        default_factory=lambda: int(os.getenv("SUMMARIZATION_NUM_CTX", "32768"))
    )


@dataclass
class RaptorConfig:
    """RAPTOR-specific clustering and tree-building settings."""

    max_layers: int = 5
    min_chunks_for_clustering: int = 3
    reduction_dimension: int = 10
    # Raised from 0.1 → 0.5: GMM soft-assignment only adds a chunk to clusters
    # where prob ≥ threshold. At 0.1 nearly every chunk lands in 3-5 clusters,
    # making them inhoud-identiek; at 0.5 only confident assignments count, so
    # clusters become distinct topics instead of overlapping mush.
    cluster_threshold: float = 0.5
    # Raised from 1500 → 4000: 1500 chars (~375 tokens) truncated long clusters
    # to their first 2-3 member chunks, dropping minority topics inside a cluster.
    # 4000 (~1000 tokens) fits most clusters whole, fed into a model with
    # summarization_max_tokens=2000 output budget — still well within num_ctx.
    max_tokens_per_cluster: int = 4000
    # Raised from 500 → 2000: a cluster summary of <400 words drops half the topics.
    summarization_max_tokens: int = 2000
    use_pca_fallback: bool = False


@dataclass
class TreeKGConfig:
    """TreeKG section summarization settings."""

    min_content_length: int = 100
    max_content_length: int = 8000
    summarization_max_tokens: int = 500


@dataclass
class NaiveConfig:
    """Naive single-pass / chunk-and-combine settings."""

    max_input_length: int = 12000
    chunk_overlap: int = 200
    combine_summaries: bool = True
    summarization_max_tokens: int = 1000


@dataclass
class MapReduceConfig:
    """Map-Reduce strategy settings."""

    batch_size: int = 10
    reduce_strategy: str = "tree"  # "tree" or "sequential"


@dataclass
class RefineConfig:
    """Refine / iterative strategy settings."""

    max_refinement_passes: int = 3


@dataclass
class WalkingTreeConfig:
    """Walking Tree strategy settings."""

    topic_threshold: float = 0.3
    max_branch_depth: int = 4


@dataclass
class ExtractiveAbstractiveConfig:
    """Extractive-Abstractive strategy settings."""

    extractor: str = "textrank"
    top_k_sentences: int = 10


@dataclass
class SkeletonConfig:
    """Skeleton-of-Thought strategy settings."""

    max_skeleton_points: int = 8
    expansion_tokens: int = 200


@dataclass
class ChainOfDensityConfig:
    """Chain-of-Density enhancer settings."""

    num_density_rounds: int = 5
    max_summary_words: int = 200


@dataclass
class SelfCorrectionConfig:
    """Self-Correction enhancer settings."""

    critic_model: str = ""  # empty = same as main model
    max_correction_rounds: int = 3


@dataclass
class LinkedEntityConfig:
    """Linked Entity strategy settings."""

    entity_types: list = field(default_factory=lambda: ["PERSON", "ORG", "GPE", "CONCEPT"])
    min_entity_mentions: int = 2


@dataclass
class SummarizationConfig:
    """Top-level configuration for the summarization pipeline.

    Attributes:
        strategy: Which strategy to use (see SummarizationStrategy enum).
        llm: Shared LLM provider settings.
        raptor: RAPTOR-specific settings.
        treekg: TreeKG-specific settings.
        naive: Naive strategy settings.
        map_reduce: Map-Reduce strategy settings.
        refine: Refine strategy settings.
        walking_tree: Walking Tree strategy settings.
        extractive_abstractive: Extractive-Abstractive strategy settings.
        skeleton: Skeleton-of-Thought strategy settings.
        chain_of_density: Chain-of-Density enhancer settings.
        self_correction: Self-Correction enhancer settings.
        linked_entity: Linked Entity strategy settings.
    """

    strategy: str = "naive"

    # Prompt customization (optional — overrides default system message / appends to prompts)
    system_prompt: str = ""
    output_instructions: str = ""

    llm: LLMConfig = field(default_factory=LLMConfig)
    raptor: RaptorConfig = field(default_factory=RaptorConfig)
    treekg: TreeKGConfig = field(default_factory=TreeKGConfig)
    naive: NaiveConfig = field(default_factory=NaiveConfig)
    map_reduce: MapReduceConfig = field(default_factory=MapReduceConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    walking_tree: WalkingTreeConfig = field(default_factory=WalkingTreeConfig)
    extractive_abstractive: ExtractiveAbstractiveConfig = field(
        default_factory=ExtractiveAbstractiveConfig
    )
    skeleton: SkeletonConfig = field(default_factory=SkeletonConfig)
    chain_of_density: ChainOfDensityConfig = field(default_factory=ChainOfDensityConfig)
    self_correction: SelfCorrectionConfig = field(default_factory=SelfCorrectionConfig)
    linked_entity: LinkedEntityConfig = field(default_factory=LinkedEntityConfig)
