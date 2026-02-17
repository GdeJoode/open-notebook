"""Base class for all summarization strategies."""

from abc import ABC, abstractmethod
from typing import List

from summarization.config import SummarizationConfig
from summarization.models.result import ChunkInput, SummarizationResult


class BaseSummarizationStrategy(ABC):
    """Abstract base for summarization strategies.

    Subclasses implement ``summarize()`` which receives ordered chunks
    and returns a :class:`SummarizationResult`.
    """

    def __init__(self, config: SummarizationConfig, **kwargs) -> None:
        self.config = config

    @abstractmethod
    async def summarize(self, chunks: List[ChunkInput]) -> SummarizationResult:
        """Run the strategy on the provided chunks.

        Args:
            chunks: Ordered list of input chunks.

        Returns:
            SummarizationResult with document summary, nodes, and metadata.
        """
        ...
