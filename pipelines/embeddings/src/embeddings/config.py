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
    # Context-length guard (R.0b). Local embedding models have a fixed,
    # often small, context window (e.g. mxbai-embed-large ~512 tokens). The
    # Ollama legacy /api/embeddings endpoint used by esperanto IGNORES the
    # `truncate` option and hard-fails ("input length exceeds the context
    # length") on over-long input, so we guard length client-side.
    #
    # ``max_input_chars`` is a conservative char cap applied before every
    # embed call. 2048 ≈ 512 tokens × 4 chars/token for Latin text. It is a
    # heuristic, not a guarantee (token density varies by script: dense CJK
    # can exceed 512 tokens in far fewer chars), so over-long input that the
    # cap does not catch is handled adaptively by halving and retrying — see
    # ``EmbeddingService._embed_with_retry``. Truncation is tail-truncation
    # (keep the head), which is acceptable for retrieval embeddings.
    max_input_chars: int = 2048
    # Whether to adaptively shrink + retry a text that still triggers a
    # context-length error after the char cap. Off would mean a hard failure;
    # we keep it on so every chunk reliably gets a vector.
    truncate_on_context_error: bool = True
