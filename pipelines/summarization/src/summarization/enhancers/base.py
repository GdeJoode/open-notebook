"""Abstract base class for summarization enhancers."""

from abc import ABC, abstractmethod
from typing import List

from summarization.config import SummarizationConfig
from summarization.models.result import ChunkInput, SummarizationResult


class BaseEnhancer(ABC):
    """Wraps a SummarizationResult to enhance quality.

    Enhancers are composable post-processing layers that improve
    a strategy's output (e.g. densification, fact-checking, compression).
    """

    def __init__(self, config: SummarizationConfig) -> None:
        self.config = config

    @abstractmethod
    async def enhance(
        self, result: SummarizationResult, chunks: List[ChunkInput]
    ) -> SummarizationResult:
        """Enhance a summarization result.

        Args:
            result: The output from a summarization strategy.
            chunks: The original input chunks (for reference/fact-checking).

        Returns:
            An enhanced SummarizationResult.
        """
        ...
