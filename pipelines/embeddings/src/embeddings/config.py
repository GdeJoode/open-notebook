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
    # Fragmenten boven 512 tokens (F2-08, ADR 0008): AFKAPPEN, niet opdelen.
    # mxbai-embed-large draait in llama-server met ``-c 512``; één fragment
    # krijgt één vector, de kop blijft en de staart valt weg.
    #
    # ``max_input_chars`` is de eerste grens, vóór elke aanroep: 1 536 tekens
    # = 512 tokens × 3 tekens per token, ruim voor Nederlands met de Engelse
    # WordPiece van mxbai. Weigert llama-server een tekst toch wegens lengte,
    # dan halveert de dienst hem tot hij past
    # (``EmbeddingService._embed_one_with_shrink``). Een tekst van ten hoogste
    # ``ZEKER_TEKENS`` (510) past altijd: WordPiece maakt per teken hooguit
    # één token, plus [CLS] en [SEP].
    max_input_chars: int = 1536
    # Whether to adaptively shrink + retry a text that still triggers a
    # context-length error after the char cap. Off would mean a hard failure;
    # we keep it on so every chunk reliably gets a vector.
    truncate_on_context_error: bool = True
