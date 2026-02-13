"""Linked Entity summarization — entity-anchored mini-summaries."""

from typing import List

from summarization.models.result import ChunkInput, SummarizationResult
from summarization.strategies.base import BaseSummarizationStrategy


class LinkedEntityStrategy(BaseSummarizationStrategy):
    """Entity-anchored mini-summaries — NOT YET IMPLEMENTED.

    For each significant entity in the document, generates a focused
    summary of how that entity is discussed, its relationships, and
    key facts. Creates an entity-indexed summary map.
    """

    async def summarize(self, chunks: List[ChunkInput]) -> SummarizationResult:
        raise NotImplementedError(
            "LinkedEntityStrategy is not yet implemented. "
            "See docs/SUMMARIZATION_APPROACHES.md"
        )
