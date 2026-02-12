"""Map-Reduce summarization — parallel map per chunk, then reduce partials."""

from typing import List

from summarization.models.result import ChunkInput, SummarizationResult
from summarization.strategies.base import BaseSummarizationStrategy


class MapReduceStrategy(BaseSummarizationStrategy):
    """Parallel map-per-chunk then reduce partials — NOT YET IMPLEMENTED.

    Maps each chunk through the LLM independently, then reduces the
    partial summaries into a final summary. Supports multi-stage reduce.
    """

    async def summarize(self, chunks: List[ChunkInput]) -> SummarizationResult:
        raise NotImplementedError(
            "MapReduceStrategy is not yet implemented. "
            "See docs/SUMMARIZATION_APPROACHES.md"
        )
