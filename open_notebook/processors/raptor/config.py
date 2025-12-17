"""
RAPTOR Configuration

Defines configuration options for RAPTOR tree building.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass
class RaptorConfig:
    """Configuration for RAPTOR tree building."""

    # Tree structure
    max_layers: int = 5
    """Maximum depth of the summary tree (typically 3-5 is sufficient)."""

    min_chunks_for_clustering: int = 3
    """Minimum number of chunks required to perform clustering."""

    # Clustering parameters
    cluster_method: Literal["gmm", "kmeans"] = "gmm"
    """Clustering algorithm: 'gmm' (Gaussian Mixture) or 'kmeans'."""

    reduction_dimension: int = 10
    """Target dimensionality for UMAP/PCA reduction before clustering."""

    cluster_threshold: float = 0.1
    """Probability threshold for soft clustering assignment (GMM only)."""

    max_tokens_per_cluster: int = 3500
    """Maximum tokens in a cluster before re-clustering."""

    # Summarization parameters
    summarization_max_tokens: int = 200
    """Maximum tokens for generated summaries."""

    summarization_model: Optional[str] = None
    """Model ID for summarization (uses app default if None)."""

    summarization_prompt: str = """Write a concise summary of the following text passages, capturing the key themes, main arguments, and important details. Focus on synthesizing the information rather than listing points.

Text passages:
{context}

Summary:"""
    """Prompt template for summarization. Use {context} placeholder."""

    # Processing options
    use_multithreading: bool = False
    """Enable multithreaded processing (not recommended for async code)."""

    verbose: bool = False
    """Enable verbose logging."""

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_layers < 1:
            raise ValueError("max_layers must be at least 1")
        if self.min_chunks_for_clustering < 2:
            raise ValueError("min_chunks_for_clustering must be at least 2")
        if not 0 < self.cluster_threshold < 1:
            raise ValueError("cluster_threshold must be between 0 and 1")
        if self.reduction_dimension < 2:
            raise ValueError("reduction_dimension must be at least 2")
