"""Self-Correction enhancer — summarizer + critic agent loop."""

from typing import List

from summarization.enhancers.base import BaseEnhancer
from summarization.models.result import ChunkInput, SummarizationResult


class SelfCorrectionEnhancer(BaseEnhancer):
    """Two-agent fact-checking and correction loop — NOT YET IMPLEMENTED.

    A Critic agent evaluates the summary for factual consistency,
    completeness, and coherence, then provides corrections.
    """

    async def enhance(
        self, result: SummarizationResult, chunks: List[ChunkInput]
    ) -> SummarizationResult:
        raise NotImplementedError(
            "SelfCorrectionEnhancer is not yet implemented. "
            "See docs/SUMMARIZATION_APPROACHES.md"
        )
