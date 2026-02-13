"""DPR-Abs — query-focused dense retrieval + abstractive summarization."""

from typing import List

from summarization.models.result import ChunkInput, SummarizationResult
from summarization.strategies.base import BaseSummarizationStrategy


class DprAbsStrategy(BaseSummarizationStrategy):
    """Query-focused dense retrieval + summarization — NOT YET IMPLEMENTED.

    Uses dense retrieval to select the most relevant chunks for a given
    query, then summarizes only those chunks.
    """

    async def summarize(self, chunks: List[ChunkInput]) -> SummarizationResult:
        raise NotImplementedError(
            "DprAbsStrategy is not yet implemented. "
            "See docs/SUMMARIZATION_APPROACHES.md"
        )
