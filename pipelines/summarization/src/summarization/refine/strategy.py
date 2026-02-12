"""Refine summarization — sequential running-summary refinement."""

from typing import List

from summarization.models.result import ChunkInput, SummarizationResult
from summarization.strategies.base import BaseSummarizationStrategy


class RefineStrategy(BaseSummarizationStrategy):
    """Sequential running-summary refinement — NOT YET IMPLEMENTED.

    Processes chunks sequentially, refining a running summary. Each step
    sees the previous summary + next chunk, producing an updated summary.
    """

    async def summarize(self, chunks: List[ChunkInput]) -> SummarizationResult:
        raise NotImplementedError(
            "RefineStrategy is not yet implemented. "
            "See docs/SUMMARIZATION_APPROACHES.md"
        )
