"""
Summarization workflow — config-driven strategy selection and execution.
"""

from typing import Any, Callable, Coroutine, List, Optional

from loguru import logger

from summarization.config import SummarizationConfig
from summarization.models.result import ChunkInput, SummarizationResult, SummarizationStrategy
from summarization.naive.strategy import NaiveLLMStrategy
from summarization.raptor.strategy import RaptorStrategy
from summarization.strategies.base import BaseSummarizationStrategy
from summarization.treekg.strategy import TreeKGStrategy

# Type alias for RAPTOR embedding callback
EmbeddingFn = Callable[[List[str]], Coroutine[Any, Any, List[List[float]]]]


class SummarizationWorkflow:
    """Config-driven orchestrator that selects and runs a summarization strategy."""

    def __init__(
        self,
        config: Optional[SummarizationConfig] = None,
        embedding_fn: Optional[EmbeddingFn] = None,
    ) -> None:
        self.config = config or SummarizationConfig()
        self._embedding_fn = embedding_fn

    def _get_strategy(self) -> BaseSummarizationStrategy:
        """Instantiate the strategy indicated by config.strategy."""
        name = self.config.strategy.lower()

        if name == SummarizationStrategy.NAIVE.value:
            return NaiveLLMStrategy(self.config)
        elif name == SummarizationStrategy.TREEKG.value:
            return TreeKGStrategy(self.config)
        elif name == SummarizationStrategy.RAPTOR.value:
            if self._embedding_fn is None:
                raise ValueError(
                    "RAPTOR strategy requires an embedding_fn. "
                    "Pass it to SummarizationWorkflow()."
                )
            return RaptorStrategy(self.config, embedding_fn=self._embedding_fn)

        # --- Standalone strategy stubs ---
        elif name == SummarizationStrategy.MAP_REDUCE.value:
            from summarization.map_reduce.strategy import MapReduceStrategy

            return MapReduceStrategy(self.config)
        elif name == SummarizationStrategy.REFINE.value:
            from summarization.refine.strategy import RefineStrategy

            return RefineStrategy(self.config)
        elif name == SummarizationStrategy.WALKING_TREE.value:
            from summarization.walking_tree.strategy import WalkingTreeStrategy

            return WalkingTreeStrategy(self.config)
        elif name == SummarizationStrategy.EXTRACTIVE_ABSTRACTIVE.value:
            from summarization.extractive_abstractive.strategy import (
                ExtractiveAbstractiveStrategy,
            )

            return ExtractiveAbstractiveStrategy(self.config)
        elif name == SummarizationStrategy.SKELETON.value:
            from summarization.skeleton.strategy import SkeletonStrategy

            return SkeletonStrategy(self.config)

        # --- Cross-cutting strategy stubs ---
        elif name == SummarizationStrategy.HYBRID.value:
            from summarization.hybrid.strategy import HybridStrategy

            return HybridStrategy(self.config)
        elif name == SummarizationStrategy.DPR_ABS.value:
            from summarization.dpr_abs.strategy import DprAbsStrategy

            return DprAbsStrategy(self.config)
        elif name == SummarizationStrategy.LINKED_ENTITY.value:
            from summarization.linked_entity.strategy import LinkedEntityStrategy

            return LinkedEntityStrategy(self.config)

        # --- Enhancement layers as primary strategy stubs ---
        elif name == SummarizationStrategy.CHAIN_OF_DENSITY.value:
            raise NotImplementedError(
                "ChainOfDensity as primary strategy is not yet implemented. "
                "See docs/SUMMARIZATION_APPROACHES.md"
            )
        elif name == SummarizationStrategy.SELF_CORRECTION.value:
            raise NotImplementedError(
                "SelfCorrection as primary strategy is not yet implemented. "
                "See docs/SUMMARIZATION_APPROACHES.md"
            )
        elif name == SummarizationStrategy.GIST_TOKENS.value:
            raise NotImplementedError(
                "GistTokens as primary strategy is not yet implemented. "
                "See docs/SUMMARIZATION_APPROACHES.md"
            )
        else:
            raise ValueError(
                f"Unknown strategy '{self.config.strategy}'. "
                f"Choose from: {[s.value for s in SummarizationStrategy]}"
            )

    async def process(self, chunks: List[ChunkInput]) -> SummarizationResult:
        """Run the configured summarization strategy.

        Args:
            chunks: Ordered list of input chunks.

        Returns:
            SummarizationResult from the selected strategy.
        """
        strategy = self._get_strategy()
        strategy_name = self.config.strategy
        logger.info(
            f"Starting summarization workflow: strategy={strategy_name}, "
            f"chunks={len(chunks)}"
        )

        result = await strategy.summarize(chunks)

        logger.info(
            f"Summarization complete: {result.node_count} nodes, "
            f"{result.num_layers} layers"
        )
        return result
