"""Hybrid RAPTOR+TreeKG — structure-first then semantic clustering."""

from typing import List

from summarization.models.result import ChunkInput, SummarizationResult
from summarization.strategies.base import BaseSummarizationStrategy


class HybridStrategy(BaseSummarizationStrategy):
    """Structure-first then semantic clustering — NOT YET IMPLEMENTED.

    Uses TreeKG's structural hierarchy for the first pass, then applies
    RAPTOR's clustering within each section to discover sub-themes.
    """

    async def summarize(self, chunks: List[ChunkInput]) -> SummarizationResult:
        raise NotImplementedError(
            "HybridStrategy is not yet implemented. "
            "See docs/SUMMARIZATION_APPROACHES.md"
        )
