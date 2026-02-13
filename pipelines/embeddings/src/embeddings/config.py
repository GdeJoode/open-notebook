"""
Embedding pipeline configuration.
"""

from dataclasses import dataclass


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding pipeline."""
    batch_size: int = 10           # Chunks per embedding API call
    max_concurrent: int = 5        # Parallel embedding tasks
    retry_attempts: int = 2
    retry_delay: float = 1.0
    # Text splitting fallback (when no structural chunks exist)
    fallback_chunk_size: int = 1000
    fallback_chunk_overlap: int = 200
