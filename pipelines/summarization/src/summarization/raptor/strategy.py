"""RAPTOR recursive clustering and summarization strategy."""

from typing import Any, Callable, Coroutine, List

from summarization.config import SummarizationConfig
from summarization.models.result import ChunkInput, SummarizationResult
from summarization.strategies.base import BaseSummarizationStrategy

EmbeddingFn = Callable[[List[str]], Coroutine[Any, Any, List[List[float]]]]


class RaptorStrategy(BaseSummarizationStrategy):
    """Recursive Abstractive Processing for Tree-Organized Retrieval — NOT YET IMPLEMENTED.

    Clusters chunks by embedding similarity, summarizes each cluster,
    then repeats recursively to build a hierarchical summary tree.
    Requires an embedding function for clustering.
    """

    def __init__(
        self,
        config: SummarizationConfig,
        embedding_fn: EmbeddingFn,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._embedding_fn = embedding_fn

    async def summarize(self, chunks: List[ChunkInput]) -> SummarizationResult:
        raise NotImplementedError(
            "RaptorStrategy is not yet implemented. "
            "See docs/SUMMARIZATION_APPROACHES.md"
        )
