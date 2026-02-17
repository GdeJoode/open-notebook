"""Naive single-pass LLM summarization strategy."""

from typing import List

from summarization.models.result import ChunkInput, SummarizationResult
from summarization.strategies.base import BaseSummarizationStrategy


class NaiveLLMStrategy(BaseSummarizationStrategy):
    """Single-pass LLM summarization — NOT YET IMPLEMENTED.

    Concatenates all chunks (up to a token limit), sends them to the LLM
    in a single prompt, and returns the resulting summary. For documents
    exceeding the context window, chunks are split and partial summaries
    are combined in a second pass.
    """

    async def summarize(self, chunks: List[ChunkInput]) -> SummarizationResult:
        raise NotImplementedError(
            "NaiveLLMStrategy is not yet implemented. "
            "See docs/SUMMARIZATION_APPROACHES.md"
        )
