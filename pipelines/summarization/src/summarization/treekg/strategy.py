"""TreeKG structure-aware hierarchical summarization strategy."""

from typing import List

from summarization.models.result import ChunkInput, SummarizationResult
from summarization.strategies.base import BaseSummarizationStrategy


class TreeKGStrategy(BaseSummarizationStrategy):
    """Structure-aware hierarchical summarization — NOT YET IMPLEMENTED.

    Leverages document section structure to build a summary tree that
    mirrors the document's heading hierarchy. Summarizes leaf sections
    first, then aggregates upward.
    """

    async def summarize(self, chunks: List[ChunkInput]) -> SummarizationResult:
        raise NotImplementedError(
            "TreeKGStrategy is not yet implemented. "
            "See docs/SUMMARIZATION_APPROACHES.md"
        )
