"""Tests for NaiveLLMStrategy."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from summarization.config import LLMConfig, NaiveConfig, SummarizationConfig
from summarization.models.result import ChunkInput, SummarizationStrategy
from summarization.naive.strategy import NaiveLLMStrategy


def _make_config(**overrides) -> SummarizationConfig:
    naive_overrides = {}
    llm_overrides = {}
    for k, v in overrides.items():
        if k.startswith("naive_"):
            naive_overrides[k[6:]] = v
        elif k.startswith("llm_"):
            llm_overrides[k[4:]] = v
    return SummarizationConfig(
        strategy="naive",
        llm=LLMConfig(**llm_overrides),
        naive=NaiveConfig(**naive_overrides),
    )


def _make_chunks(texts: list[str]) -> list[ChunkInput]:
    return [
        ChunkInput(text=t, chunk_id=f"chunk:{i}", order=i)
        for i, t in enumerate(texts)
    ]


class TestNaiveStrategyInit:
    def test_creates_from_config(self):
        config = _make_config()
        strategy = NaiveLLMStrategy(config)
        assert strategy._naive_config.max_input_length == 12000

    def test_custom_config(self):
        config = _make_config(naive_max_input_length=5000, naive_chunk_overlap=100)
        strategy = NaiveLLMStrategy(config)
        assert strategy._naive_config.max_input_length == 5000
        assert strategy._naive_config.chunk_overlap == 100


class TestNaiveStrategySplit:
    def test_single_window_when_fits(self):
        config = _make_config(naive_max_input_length=1000)
        strategy = NaiveLLMStrategy(config)
        windows = strategy._split_into_windows("short text")
        assert len(windows) == 1
        assert windows[0] == "short text"

    def test_multiple_windows_when_exceeds(self):
        config = _make_config(naive_max_input_length=100, naive_chunk_overlap=20)
        strategy = NaiveLLMStrategy(config)
        text = "a" * 250
        windows = strategy._split_into_windows(text)
        assert len(windows) >= 3
        # Each window should be max 100 chars
        for w in windows:
            assert len(w) <= 100

    def test_overlap_preserved(self):
        config = _make_config(naive_max_input_length=100, naive_chunk_overlap=20)
        strategy = NaiveLLMStrategy(config)
        text = "a" * 150
        windows = strategy._split_into_windows(text)
        assert len(windows) == 2
        # Second window starts at 80, ends at 150
        assert len(windows[1]) == 70


class TestNaiveStrategySummarize:
    @pytest.mark.asyncio
    async def test_empty_chunks_returns_empty(self):
        config = _make_config()
        strategy = NaiveLLMStrategy(config)
        result = await strategy.summarize([])
        assert result.strategy == SummarizationStrategy.NAIVE
        assert result.document_summary == ""
        assert result.num_input_chunks == 0

    @pytest.mark.asyncio
    async def test_single_pass_when_fits(self):
        config = _make_config(naive_max_input_length=10000)
        strategy = NaiveLLMStrategy(config)

        with patch.object(strategy, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "This is a summary."
            chunks = _make_chunks(["Hello world.", "Second paragraph."])
            result = await strategy.summarize(chunks)

        assert result.strategy == SummarizationStrategy.NAIVE
        assert result.document_summary == "This is a summary."
        assert result.num_layers == 1
        assert result.num_input_chunks == 2
        assert len(result.summary_nodes) == 1
        mock_llm.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_multi_pass_with_combine(self):
        config = _make_config(
            naive_max_input_length=50,
            naive_chunk_overlap=10,
            naive_combine_summaries=True,
        )
        strategy = NaiveLLMStrategy(config)

        call_count = 0

        async def mock_llm(prompt):
            nonlocal call_count
            call_count += 1
            if "Combine them" in prompt:
                return "Final combined summary."
            return f"Partial summary {call_count}."

        with patch.object(strategy, "_call_llm", side_effect=mock_llm):
            chunks = _make_chunks(["a" * 40, "b" * 40])
            result = await strategy.summarize(chunks)

        assert result.document_summary == "Final combined summary."
        assert result.num_layers == 2
        assert result.num_input_chunks == 2
        # Window summaries + final combined node
        assert len(result.summary_nodes) >= 2
        assert any(n.layer == 1 for n in result.summary_nodes)

    @pytest.mark.asyncio
    async def test_multi_pass_without_combine(self):
        config = _make_config(
            naive_max_input_length=50,
            naive_chunk_overlap=10,
            naive_combine_summaries=False,
        )
        strategy = NaiveLLMStrategy(config)

        async def mock_llm(prompt):
            return "A partial summary."

        with patch.object(strategy, "_call_llm", side_effect=mock_llm):
            chunks = _make_chunks(["a" * 40, "b" * 40])
            result = await strategy.summarize(chunks)

        assert result.num_layers == 1
        assert result.document_summary == "A partial summary."
        assert all(n.layer == 0 for n in result.summary_nodes)

    @pytest.mark.asyncio
    async def test_chunk_ids_preserved(self):
        config = _make_config(naive_max_input_length=10000)
        strategy = NaiveLLMStrategy(config)

        with patch.object(strategy, "_call_llm", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Summary."
            chunks = _make_chunks(["text"])
            result = await strategy.summarize(chunks)

        assert "chunk:0" in result.summary_nodes[0].source_chunk_ids
