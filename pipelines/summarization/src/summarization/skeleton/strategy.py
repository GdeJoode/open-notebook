"""Skeleton-of-Thought summarization — skeleton outline then point-wise expansion.

Reference: Ning et al. (2023), "Skeleton-of-Thought: Large Language Models
Can Do Parallel Decoding"
"""

from typing import List

from summarization.models.result import ChunkInput, SummarizationResult
from summarization.strategies.base import BaseSummarizationStrategy


class SkeletonStrategy(BaseSummarizationStrategy):
    """Skeleton outline then point-wise expansion — NOT YET IMPLEMENTED.

    Generates a skeleton (outline of key points), then expands each point
    into a paragraph. The skeleton prevents drift and ensures coverage.
    """

    async def summarize(self, chunks: List[ChunkInput]) -> SummarizationResult:
        raise NotImplementedError(
            "SkeletonStrategy is not yet implemented. "
            "See docs/SUMMARIZATION_APPROACHES.md"
        )
