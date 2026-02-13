"""Walking Tree summarization — linear tree with topic segmentation."""

from typing import List

from summarization.models.result import ChunkInput, SummarizationResult
from summarization.strategies.base import BaseSummarizationStrategy


class WalkingTreeStrategy(BaseSummarizationStrategy):
    """Linear tree with topic segmentation — NOT YET IMPLEMENTED.

    Segments the document by topic boundaries, then builds a tree where
    each branch is a coherent topic segment. Preserves reading order.
    """

    async def summarize(self, chunks: List[ChunkInput]) -> SummarizationResult:
        raise NotImplementedError(
            "WalkingTreeStrategy is not yet implemented. "
            "See docs/SUMMARIZATION_APPROACHES.md"
        )
