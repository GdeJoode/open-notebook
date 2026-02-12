"""Gist Tokens enhancer — compress summaries for efficient reuse.

Reference: Mu et al. (2023), "Learning to Compress Prompts with Gist Tokens"
"""

from typing import List

from summarization.enhancers.base import BaseEnhancer
from summarization.models.result import ChunkInput, SummarizationResult


class GistTokensEnhancer(BaseEnhancer):
    """Compress summaries into compact gist representations — NOT YET IMPLEMENTED.

    Produces a compressed token representation that captures essential
    information for downstream retrieval and LLM processing.
    """

    async def enhance(
        self, result: SummarizationResult, chunks: List[ChunkInput]
    ) -> SummarizationResult:
        raise NotImplementedError(
            "GistTokensEnhancer is not yet implemented. "
            "See docs/SUMMARIZATION_APPROACHES.md"
        )
