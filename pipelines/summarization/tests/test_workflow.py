"""Tests for SummarizationWorkflow — the config-driven strategy dispatcher.

Mocks every concrete strategy at the import site so we only verify the
dispatch logic, not the implementations (those have their own tests in
``test_strategies.py`` / ``test_naive_strategy.py``).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from summarization.config import SummarizationConfig
from summarization.models.result import (
    SummarizationResult,
    SummarizationStrategy,
)
from summarization.workflow import SummarizationWorkflow


def _empty_result() -> SummarizationResult:
    """Minimal SummarizationResult for AsyncMock returns."""
    return SummarizationResult(
        strategy=SummarizationStrategy.NAIVE,
        nodes=[],
        root_node_ids=[],
    )


class TestStrategyDispatch:
    """SummarizationWorkflow._get_strategy() should pick the right class
    based on config.strategy.
    """

    def test_naive_strategy_selected(self):
        cfg = SummarizationConfig(strategy="naive")
        wf = SummarizationWorkflow(config=cfg)

        strategy = wf._get_strategy()

        from summarization.naive.strategy import NaiveLLMStrategy

        assert isinstance(strategy, NaiveLLMStrategy)

    def test_treekg_strategy_selected(self):
        cfg = SummarizationConfig(strategy="treekg")
        wf = SummarizationWorkflow(config=cfg)

        strategy = wf._get_strategy()

        from summarization.treekg.strategy import TreeKGStrategy

        assert isinstance(strategy, TreeKGStrategy)

    def test_raptor_requires_embedding_fn(self):
        cfg = SummarizationConfig(strategy="raptor")
        wf = SummarizationWorkflow(config=cfg, embedding_fn=None)

        with pytest.raises(ValueError, match="RAPTOR strategy requires"):
            wf._get_strategy()

    def test_raptor_with_embedding_fn(self):
        cfg = SummarizationConfig(strategy="raptor")
        embed_fn = AsyncMock()
        wf = SummarizationWorkflow(config=cfg, embedding_fn=embed_fn)

        strategy = wf._get_strategy()

        from summarization.raptor.strategy import RaptorStrategy

        assert isinstance(strategy, RaptorStrategy)

    def test_map_reduce_strategy_selected(self):
        cfg = SummarizationConfig(strategy="map_reduce")
        wf = SummarizationWorkflow(config=cfg)

        strategy = wf._get_strategy()

        from summarization.map_reduce.strategy import MapReduceStrategy

        assert isinstance(strategy, MapReduceStrategy)

    def test_refine_strategy_selected(self):
        cfg = SummarizationConfig(strategy="refine")
        wf = SummarizationWorkflow(config=cfg)

        strategy = wf._get_strategy()

        from summarization.refine.strategy import RefineStrategy

        assert isinstance(strategy, RefineStrategy)

    def test_strategy_name_is_case_insensitive(self):
        cfg = SummarizationConfig(strategy="NAIVE")
        wf = SummarizationWorkflow(config=cfg)

        strategy = wf._get_strategy()

        from summarization.naive.strategy import NaiveLLMStrategy

        assert isinstance(strategy, NaiveLLMStrategy)


class TestStrategyDispatchErrors:
    """Error paths in the dispatcher."""

    def test_unknown_strategy_raises_value_error(self):
        cfg = SummarizationConfig(strategy="not-a-real-strategy")
        wf = SummarizationWorkflow(config=cfg)

        with pytest.raises(ValueError, match="Unknown strategy"):
            wf._get_strategy()

    def test_chain_of_density_primary_not_implemented(self):
        cfg = SummarizationConfig(strategy="chain_of_density")
        wf = SummarizationWorkflow(config=cfg)

        with pytest.raises(NotImplementedError, match="ChainOfDensity"):
            wf._get_strategy()

    def test_self_correction_primary_not_implemented(self):
        cfg = SummarizationConfig(strategy="self_correction")
        wf = SummarizationWorkflow(config=cfg)

        with pytest.raises(NotImplementedError, match="SelfCorrection"):
            wf._get_strategy()


class TestProcessDelegation:
    """The public process() method should instantiate the strategy and
    forward chunks to its summarize() method.
    """

    @pytest.mark.asyncio
    async def test_process_calls_strategy_summarize(self, monkeypatch):
        cfg = SummarizationConfig(strategy="naive")
        wf = SummarizationWorkflow(config=cfg)

        fake_strategy = MagicMock()
        fake_strategy.summarize = AsyncMock(return_value=_empty_result())
        monkeypatch.setattr(wf, "_get_strategy", lambda: fake_strategy)

        result = await wf.process(chunks=[])

        fake_strategy.summarize.assert_awaited_once_with([])
        assert result.strategy == SummarizationStrategy.NAIVE

    @pytest.mark.asyncio
    async def test_process_forwards_chunks(self, monkeypatch):
        from summarization.models.result import ChunkInput

        cfg = SummarizationConfig(strategy="naive")
        wf = SummarizationWorkflow(config=cfg)

        fake_strategy = MagicMock()
        fake_strategy.summarize = AsyncMock(return_value=_empty_result())
        monkeypatch.setattr(wf, "_get_strategy", lambda: fake_strategy)

        chunks = [
            ChunkInput(text="first"),
            ChunkInput(text="second"),
        ]
        await wf.process(chunks=chunks)

        fake_strategy.summarize.assert_awaited_once_with(chunks)


class TestWorkflowDefaults:
    """Constructor defaults shouldn't surprise callers."""

    def test_default_config_when_none_passed(self):
        wf = SummarizationWorkflow()
        assert wf.config is not None
        assert wf.config.strategy == "naive"

    def test_embedding_fn_optional(self):
        wf = SummarizationWorkflow()
        assert wf._embedding_fn is None
