"""Map-Reduce summarization strategy.

Implements a two-phase approach: first *map* each chunk to a summary in
parallel, then *reduce* the partial summaries into a single document
summary.  Two reduce modes are supported:

- **tree**: recursively combine batches of partial summaries until one
  summary remains (log-depth reduction).
- **sequential**: concatenate all map summaries and combine in a single
  LLM call.
"""

import asyncio
import uuid
from typing import List

from esperanto import LanguageModel
from loguru import logger

from summarization.config import SummarizationConfig
from summarization.models.result import (
    ChunkInput,
    SummarizationResult,
    SummarizationStrategy,
    SummaryNode,
)
from summarization.strategies.base import BaseSummarizationStrategy


class MapReduceStrategy(BaseSummarizationStrategy):
    """Parallel map-per-chunk then reduce partials.

    Each chunk is independently summarized (map phase), then the partial
    summaries are combined into a final document summary (reduce phase).
    """

    def __init__(self, config: SummarizationConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self._llm_config = config.llm
        self._mr_config = config.map_reduce

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _get_model(self, max_tokens: int) -> LanguageModel:
        return LanguageModel(
            model_name=self._llm_config.model_name,
            api_base=self._llm_config.base_url,
            temperature=self._llm_config.temperature,
            max_tokens=max_tokens,
        )

    async def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        model = self._get_model(max_tokens)
        messages = [{"role": "user", "content": prompt}]
        response = await model.achat(messages)
        return response.text

    # ------------------------------------------------------------------
    # Map phase
    # ------------------------------------------------------------------

    async def _map_chunk(self, chunk: ChunkInput, index: int, total: int) -> str:
        """Summarize a single chunk (map step)."""
        prompt = (
            f"Summarize the following text (chunk {index + 1} of {total}). "
            "Capture the key information concisely.\n\n"
            f"TEXT:\n{chunk.text}"
        )
        return await self._call_llm(prompt)

    # ------------------------------------------------------------------
    # Reduce phase
    # ------------------------------------------------------------------

    async def _reduce_tree(
        self,
        summaries: List[str],
        all_nodes: List[SummaryNode],
        layer: int,
    ) -> tuple[str, List[SummaryNode], int]:
        """Recursively combine batches of summaries (tree reduce).

        Returns ``(final_text, updated_nodes, max_layer)``.
        """
        batch_size = self._mr_config.batch_size
        if len(summaries) <= 1:
            return summaries[0] if summaries else "", all_nodes, layer

        batches: List[List[str]] = []
        for i in range(0, len(summaries), batch_size):
            batches.append(summaries[i : i + batch_size])

        logger.info(
            f"Tree reduce: {len(summaries)} summaries -> "
            f"{len(batches)} batches at layer {layer}"
        )

        new_summaries: List[str] = []
        for batch in batches:
            combined = "\n\n".join(
                f"[Part {i + 1}]: {s}" for i, s in enumerate(batch)
            )
            prompt = (
                "Combine the following partial summaries into a single "
                "coherent summary that preserves all key information.\n\n"
                f"{combined}"
            )
            reduced = await self._call_llm(prompt)
            new_summaries.append(reduced)

            node = SummaryNode(
                text=reduced,
                layer=layer,
                source_chunk_ids=[],
                children_ids=[],
                node_id=str(uuid.uuid4()),
                metadata={"reduce_layer": layer, "batch_size": len(batch)},
            )
            all_nodes.append(node)

        if len(new_summaries) == 1:
            return new_summaries[0], all_nodes, layer

        return await self._reduce_tree(new_summaries, all_nodes, layer + 1)

    async def _reduce_sequential(
        self,
        summaries: List[str],
        all_nodes: List[SummaryNode],
    ) -> tuple[str, List[SummaryNode], int]:
        """Concatenate all map summaries and combine in one LLM call.

        Returns ``(final_text, updated_nodes, layer)``.
        """
        combined = "\n\n".join(
            f"[Section {i + 1}]: {s}" for i, s in enumerate(summaries)
        )
        prompt = (
            "The following are summaries of consecutive sections of a "
            "document. Combine them into a single coherent summary that "
            "captures all key themes and conclusions.\n\n"
            f"{combined}"
        )
        final = await self._call_llm(prompt)

        node = SummaryNode(
            text=final,
            layer=1,
            source_chunk_ids=[],
            children_ids=[n.node_id or "" for n in all_nodes],
            node_id=str(uuid.uuid4()),
            metadata={"reduce_strategy": "sequential"},
        )
        all_nodes.append(node)

        return final, all_nodes, 1

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def summarize(self, chunks: List[ChunkInput]) -> SummarizationResult:
        """Run Map-Reduce summarization on the provided chunks.

        Steps:
        1. **Map**: Summarize each chunk independently (layer 0 nodes).
        2. **Reduce**: Combine map summaries into a document summary using
           the configured reduce strategy (``tree`` or ``sequential``).
        """
        if not chunks:
            return SummarizationResult(
                strategy=SummarizationStrategy.MAP_REDUCE,
                document_summary="",
                num_input_chunks=0,
            )

        # Single chunk — no map/reduce needed
        if len(chunks) == 1:
            text = chunks[0].text.strip()
            prompt = (
                "Provide a concise summary of the following text.\n\n"
                f"TEXT:\n{text}"
            )
            summary = await self._call_llm(prompt)
            node = SummaryNode(
                text=summary,
                layer=0,
                source_chunk_ids=[chunks[0].chunk_id or "0"],
                node_id=str(uuid.uuid4()),
            )
            return SummarizationResult(
                strategy=SummarizationStrategy.MAP_REDUCE,
                document_summary=summary,
                summary_nodes=[node],
                num_layers=1,
                num_input_chunks=1,
            )

        logger.info(
            f"MapReduce strategy: {len(chunks)} chunks, "
            f"reduce_strategy={self._mr_config.reduce_strategy}"
        )

        # --- Map phase (concurrent) ---
        total = len(chunks)
        map_tasks = [
            self._map_chunk(chunk, i, total) for i, chunk in enumerate(chunks)
        ]
        map_results = await asyncio.gather(*map_tasks)

        all_nodes: List[SummaryNode] = []
        map_summaries: List[str] = []
        for i, (chunk, summary_text) in enumerate(zip(chunks, map_results)):
            map_summaries.append(summary_text)
            all_nodes.append(
                SummaryNode(
                    text=summary_text,
                    layer=0,
                    source_chunk_ids=[chunk.chunk_id or str(chunk.order)],
                    node_id=str(uuid.uuid4()),
                    metadata={"map_index": i},
                )
            )

        logger.info(f"Map phase complete: {len(map_summaries)} partial summaries")

        # --- Reduce phase ---
        if self._mr_config.reduce_strategy == "sequential":
            final_text, all_nodes, max_layer = await self._reduce_sequential(
                map_summaries, all_nodes
            )
        else:
            # Default: tree reduce
            final_text, all_nodes, max_layer = await self._reduce_tree(
                map_summaries, all_nodes, layer=1
            )

        return SummarizationResult(
            strategy=SummarizationStrategy.MAP_REDUCE,
            document_summary=final_text,
            summary_nodes=all_nodes,
            num_layers=max_layer + 1,
            num_input_chunks=len(chunks),
            metadata={
                "reduce_strategy": self._mr_config.reduce_strategy,
                "batch_size": self._mr_config.batch_size,
            },
        )
