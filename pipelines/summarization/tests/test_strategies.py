"""Tests for TreeKG, MapReduce, Refine, and RAPTOR strategies."""

from unittest.mock import AsyncMock, patch

import pytest

from summarization.config import (
    LLMConfig,
    MapReduceConfig,
    RaptorConfig,
    RefineConfig,
    SummarizationConfig,
    TreeKGConfig,
)
from summarization.models.result import ChunkInput, SummarizationStrategy


def _make_config(strategy: str, **overrides) -> SummarizationConfig:
    return SummarizationConfig(strategy=strategy, **overrides)


def _make_chunks(texts: list[str], section_paths: list[list[str]] | None = None) -> list[ChunkInput]:
    chunks = []
    for i, text in enumerate(texts):
        sp = section_paths[i] if section_paths else None
        chunks.append(ChunkInput(
            text=text,
            chunk_id=f"chunk:{i}",
            order=i,
            section_path=sp,
            section_level=len(sp) if sp else 0,
        ))
    return chunks


# ---------------------------------------------------------------
# TreeKG
# ---------------------------------------------------------------


class TestTreeKGStrategy:
    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        from summarization.treekg.strategy import TreeKGStrategy

        config = _make_config("treekg")
        strategy = TreeKGStrategy(config)
        result = await strategy.summarize([])
        assert result.strategy == SummarizationStrategy.TREEKG
        assert result.document_summary == ""
        assert result.num_input_chunks == 0

    @pytest.mark.asyncio
    async def test_single_chunk(self):
        from summarization.treekg.strategy import TreeKGStrategy

        config = _make_config("treekg")
        strategy = TreeKGStrategy(config)

        with patch.object(strategy, "_call_llm", new_callable=AsyncMock) as mock:
            mock.return_value = "Summary of single chunk."
            result = await strategy.summarize(_make_chunks(["Some text."]))

        # Single short chunk may be returned as-is or summarized
        assert result.document_summary is not None
        assert result.num_input_chunks == 1

    @pytest.mark.asyncio
    async def test_uses_section_path(self):
        from summarization.treekg.strategy import TreeKGStrategy

        config = _make_config("treekg")
        strategy = TreeKGStrategy(config)

        chunks = _make_chunks(
            ["Intro text", "Method details", "Results here"],
            section_paths=[["Intro"], ["Methods"], ["Results"]],
        )

        call_count = 0

        async def mock_llm(prompt, max_tokens=500):
            nonlocal call_count
            call_count += 1
            return f"Summary {call_count}"

        with patch.object(strategy, "_call_llm", side_effect=mock_llm):
            result = await strategy.summarize(chunks)

        assert result.strategy == SummarizationStrategy.TREEKG
        assert result.document_summary is not None
        assert result.num_layers >= 1
        # Should have section-level summaries + root
        assert len(result.summary_nodes) >= 3

    @pytest.mark.asyncio
    async def test_no_section_path_fallback(self):
        from summarization.treekg.strategy import TreeKGStrategy

        config = _make_config("treekg")
        strategy = TreeKGStrategy(config)

        chunks = _make_chunks(["Text A", "Text B"])  # No section paths

        with patch.object(strategy, "_call_llm", new_callable=AsyncMock) as mock:
            mock.return_value = "Flat summary."
            result = await strategy.summarize(chunks)

        assert result.document_summary is not None


# ---------------------------------------------------------------
# MapReduce
# ---------------------------------------------------------------


class TestMapReduceStrategy:
    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        from summarization.map_reduce.strategy import MapReduceStrategy

        config = _make_config("map_reduce")
        strategy = MapReduceStrategy(config)
        result = await strategy.summarize([])
        assert result.strategy == SummarizationStrategy.MAP_REDUCE
        assert result.document_summary == ""

    @pytest.mark.asyncio
    async def test_single_chunk(self):
        from summarization.map_reduce.strategy import MapReduceStrategy

        config = _make_config("map_reduce")
        strategy = MapReduceStrategy(config)

        with patch.object(strategy, "_call_llm", new_callable=AsyncMock) as mock:
            mock.return_value = "Mapped summary."
            result = await strategy.summarize(_make_chunks(["Text."]))

        assert result.document_summary == "Mapped summary."
        assert result.num_layers >= 1

    @pytest.mark.asyncio
    async def test_tree_reduce(self):
        from summarization.map_reduce.strategy import MapReduceStrategy

        config = _make_config("map_reduce", map_reduce=MapReduceConfig(
            batch_size=2, reduce_strategy="tree",
        ))
        strategy = MapReduceStrategy(config)

        call_count = 0

        async def mock_llm(prompt, max_tokens=500):
            nonlocal call_count
            call_count += 1
            return f"Summary {call_count}"

        with patch.object(strategy, "_call_llm", side_effect=mock_llm):
            result = await strategy.summarize(_make_chunks(["A", "B", "C", "D"]))

        assert result.strategy == SummarizationStrategy.MAP_REDUCE
        assert result.document_summary is not None
        # Map: 4 calls + Reduce rounds
        assert call_count >= 4

    @pytest.mark.asyncio
    async def test_sequential_reduce(self):
        from summarization.map_reduce.strategy import MapReduceStrategy

        config = _make_config("map_reduce", map_reduce=MapReduceConfig(
            reduce_strategy="sequential",
        ))
        strategy = MapReduceStrategy(config)

        with patch.object(strategy, "_call_llm", new_callable=AsyncMock) as mock:
            mock.return_value = "Combined."
            result = await strategy.summarize(_make_chunks(["A", "B"]))

        assert result.document_summary is not None


# ---------------------------------------------------------------
# Refine
# ---------------------------------------------------------------


class TestRefineStrategy:
    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        from summarization.refine.strategy import RefineStrategy

        config = _make_config("refine")
        strategy = RefineStrategy(config)
        result = await strategy.summarize([])
        assert result.strategy == SummarizationStrategy.REFINE
        assert result.document_summary == ""

    @pytest.mark.asyncio
    async def test_single_chunk(self):
        from summarization.refine.strategy import RefineStrategy

        config = _make_config("refine")
        strategy = RefineStrategy(config)

        with patch.object(strategy, "_call_llm", new_callable=AsyncMock) as mock:
            mock.return_value = "Initial summary."
            result = await strategy.summarize(_make_chunks(["First chunk."]))

        assert result.document_summary == "Initial summary."

    @pytest.mark.asyncio
    async def test_sequential_refinement(self):
        from summarization.refine.strategy import RefineStrategy

        config = _make_config("refine", refine=RefineConfig(max_refinement_passes=0))
        strategy = RefineStrategy(config)

        call_count = 0

        async def mock_llm(prompt, max_tokens=500):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return "Summary of chunk 1."
            return f"Refined summary v{call_count}."

        with patch.object(strategy, "_call_llm", side_effect=mock_llm):
            result = await strategy.summarize(_make_chunks(["Chunk 1", "Chunk 2", "Chunk 3"]))

        assert result.strategy == SummarizationStrategy.REFINE
        assert result.document_summary is not None
        assert call_count == 3  # 1 initial + 2 refinements

    @pytest.mark.asyncio
    async def test_additional_polish_passes(self):
        from summarization.refine.strategy import RefineStrategy

        config = _make_config("refine", refine=RefineConfig(max_refinement_passes=2))
        strategy = RefineStrategy(config)

        call_count = 0

        async def mock_llm(prompt, max_tokens=500):
            nonlocal call_count
            call_count += 1
            return f"Version {call_count}."

        with patch.object(strategy, "_call_llm", side_effect=mock_llm):
            result = await strategy.summarize(_make_chunks(["A", "B"]))

        # 1 initial + 1 refine + up to 2 polish passes
        assert call_count >= 3


# ---------------------------------------------------------------
# RAPTOR
# ---------------------------------------------------------------


class TestRaptorStrategy:
    @pytest.mark.asyncio
    async def test_empty_chunks(self):
        from summarization.raptor.strategy import RaptorStrategy

        mock_embed = AsyncMock(return_value=[])
        config = _make_config("raptor")
        strategy = RaptorStrategy(config, embedding_fn=mock_embed)
        result = await strategy.summarize([])
        assert result.strategy == SummarizationStrategy.RAPTOR
        assert result.document_summary == ""

    @pytest.mark.asyncio
    async def test_single_chunk(self):
        from summarization.raptor.strategy import RaptorStrategy

        mock_embed = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        config = _make_config("raptor")
        strategy = RaptorStrategy(config, embedding_fn=mock_embed)

        with patch.object(strategy, "_call_llm", new_callable=AsyncMock) as mock:
            mock.return_value = "Single summary."
            result = await strategy.summarize(_make_chunks(["Only chunk."]))

        assert result.document_summary == "Single summary."

    @pytest.mark.asyncio
    async def test_clustering_and_layers(self):
        from summarization.raptor.strategy import RaptorStrategy

        # Return distinct embeddings for 4 chunks
        embeddings = [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.9, 0.1],
        ]
        embed_call = 0

        async def mock_embed(texts):
            nonlocal embed_call
            embed_call += 1
            # First call: 4 texts. Subsequent: fewer (summaries)
            return embeddings[:len(texts)]

        config = _make_config("raptor", raptor=RaptorConfig(
            max_layers=3, min_chunks_for_clustering=2,
        ))
        strategy = RaptorStrategy(config, embedding_fn=mock_embed)

        call_count = 0

        async def mock_llm(prompt, max_tokens=200):
            nonlocal call_count
            call_count += 1
            return f"Cluster summary {call_count}."

        with patch.object(strategy, "_call_llm", side_effect=mock_llm):
            result = await strategy.summarize(_make_chunks(["A", "B", "C", "D"]))

        assert result.strategy == SummarizationStrategy.RAPTOR
        assert result.document_summary is not None
        assert result.num_layers >= 1
        assert call_count >= 1  # At least one cluster summarization

    @pytest.mark.asyncio
    async def test_embedding_failure_fallback(self):
        from summarization.raptor.strategy import RaptorStrategy

        async def failing_embed(texts):
            raise RuntimeError("Embedding service down")

        config = _make_config("raptor")
        strategy = RaptorStrategy(config, embedding_fn=failing_embed)

        with patch.object(strategy, "_call_llm", new_callable=AsyncMock) as mock:
            mock.return_value = "Fallback summary."
            result = await strategy.summarize(_make_chunks(["Text A", "Text B"]))

        # Should still produce a result (fallback to concatenation)
        assert result.document_summary is not None
