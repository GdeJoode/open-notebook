"""Extractive-Abstractive summarization — TextRank extraction then LLM rewrite."""

from typing import List

from summarization.models.result import ChunkInput, SummarizationResult
from summarization.strategies.base import BaseSummarizationStrategy


class ExtractiveAbstractiveStrategy(BaseSummarizationStrategy):
    """TextRank extraction then LLM rewrite — NOT YET IMPLEMENTED.

    First extracts the most important sentences using graph-based ranking,
    then rewrites them into a fluent abstractive summary via LLM.
    """

    async def summarize(self, chunks: List[ChunkInput]) -> SummarizationResult:
        raise NotImplementedError(
            "ExtractiveAbstractiveStrategy is not yet implemented. "
            "See docs/SUMMARIZATION_APPROACHES.md"
        )
