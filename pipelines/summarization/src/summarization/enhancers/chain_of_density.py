"""Chain-of-Density enhancer — iterative summary densification.

Reference: Adams et al. (2023), "From Sparse to Dense: GPT-4 Summarization
with Chain of Density Prompting"
"""

from typing import List

from summarization.enhancers.base import BaseEnhancer
from summarization.models.result import ChunkInput, SummarizationResult


class ChainOfDensityEnhancer(BaseEnhancer):
    """Iteratively densify a summary over multiple rounds — NOT YET IMPLEMENTED.

    Each round adds missing salient entities while keeping the word count
    roughly constant, producing increasingly information-dense summaries.
    """

    async def enhance(
        self, result: SummarizationResult, chunks: List[ChunkInput]
    ) -> SummarizationResult:
        raise NotImplementedError(
            "ChainOfDensityEnhancer is not yet implemented. "
            "See docs/SUMMARIZATION_APPROACHES.md"
        )
