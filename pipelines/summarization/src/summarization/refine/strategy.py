"""Refine summarization strategy.

Processes chunks sequentially, building a running summary that is
iteratively refined as each new chunk is incorporated.  After all chunks
are consumed, optional additional refinement passes polish the final
summary for coherence and completeness.
"""

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


class RefineStrategy(BaseSummarizationStrategy):
    """Sequential running-summary refinement.

    Starts with the first chunk as the initial summary, then iteratively
    refines by feeding (current_summary + next_chunk) to the LLM.  After
    all chunks are processed, up to ``max_refinement_passes`` additional
    passes polish the accumulated summary.
    """

    def __init__(self, config: SummarizationConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self._llm_config = config.llm
        self._refine_config = config.refine

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
    # Core algorithm
    # ------------------------------------------------------------------

    async def summarize(self, chunks: List[ChunkInput]) -> SummarizationResult:
        """Run Refine summarization on the provided chunks.

        Algorithm:
        1. Summarize the first chunk to create the initial running summary.
        2. For each subsequent chunk, send ``(current_summary + chunk)``
           to the LLM with a refinement prompt.
        3. After all chunks, perform up to ``max_refinement_passes``
           additional polishing passes on the accumulated summary.
        4. Refinement steps are recorded as layer-0 nodes; the final
           polished summary is a layer-1 node.
        """
        if not chunks:
            return SummarizationResult(
                strategy=SummarizationStrategy.REFINE,
                document_summary="",
                num_input_chunks=0,
            )

        # Filter out empty chunks
        non_empty = [c for c in chunks if c.text.strip()]
        if not non_empty:
            return SummarizationResult(
                strategy=SummarizationStrategy.REFINE,
                document_summary="",
                num_input_chunks=len(chunks),
            )

        logger.info(
            f"Refine strategy: {len(non_empty)} non-empty chunks, "
            f"max_refinement_passes={self._refine_config.max_refinement_passes}"
        )

        all_nodes: List[SummaryNode] = []

        # --- Step 1: initial summary from first chunk ---
        first_chunk = non_empty[0]
        prompt = (
            "Provide a comprehensive summary of the following text. "
            "Focus on the key ideas, arguments, and conclusions.\n\n"
            f"TEXT:\n{first_chunk.text}"
        )
        current_summary = await self._call_llm(prompt)

        all_nodes.append(
            SummaryNode(
                text=current_summary,
                layer=0,
                source_chunk_ids=[first_chunk.chunk_id or str(first_chunk.order)],
                node_id=str(uuid.uuid4()),
                metadata={"step": "initial", "chunk_index": 0},
            )
        )

        # --- Step 2: refine with each subsequent chunk ---
        for i, chunk in enumerate(non_empty[1:], start=1):
            prompt = (
                "You have an existing summary of a document so far. "
                "A new section of the document is provided below. "
                "Refine and update the summary to incorporate the new "
                "information. Keep the summary coherent and comprehensive.\n\n"
                f"EXISTING SUMMARY:\n{current_summary}\n\n"
                f"NEW SECTION (part {i + 1} of {len(non_empty)}):\n{chunk.text}"
            )
            current_summary = await self._call_llm(prompt)

            all_nodes.append(
                SummaryNode(
                    text=current_summary,
                    layer=0,
                    source_chunk_ids=[chunk.chunk_id or str(chunk.order)],
                    children_ids=[all_nodes[-1].node_id or ""],
                    node_id=str(uuid.uuid4()),
                    metadata={"step": "refine", "chunk_index": i},
                )
            )

        logger.info(
            f"Refinement pass through all chunks complete "
            f"({len(non_empty)} iterations)"
        )

        # --- Step 3: optional additional refinement passes ---
        max_passes = self._refine_config.max_refinement_passes
        for pass_num in range(max_passes):
            prompt = (
                "Review and improve the following document summary. "
                "Ensure it is coherent, well-structured, and captures "
                "all key themes. Remove any redundancy and improve "
                f"clarity. (Refinement pass {pass_num + 1} of {max_passes})\n\n"
                f"SUMMARY:\n{current_summary}"
            )
            refined = await self._call_llm(prompt)

            # Stop early if the refinement did not meaningfully change
            if refined.strip() == current_summary.strip():
                logger.info(
                    f"Refinement pass {pass_num + 1} produced no change; "
                    "stopping early."
                )
                break

            current_summary = refined
            all_nodes.append(
                SummaryNode(
                    text=current_summary,
                    layer=0,
                    source_chunk_ids=[],
                    children_ids=[all_nodes[-1].node_id or ""],
                    node_id=str(uuid.uuid4()),
                    metadata={"step": "polish", "pass": pass_num + 1},
                )
            )

        # --- Build final layer-1 node ---
        final_node = SummaryNode(
            text=current_summary,
            layer=1,
            source_chunk_ids=[
                c.chunk_id or str(c.order) for c in non_empty
            ],
            children_ids=[n.node_id or "" for n in all_nodes],
            node_id=str(uuid.uuid4()),
            metadata={"step": "final"},
        )
        all_nodes.append(final_node)

        return SummarizationResult(
            strategy=SummarizationStrategy.REFINE,
            document_summary=current_summary,
            summary_nodes=all_nodes,
            num_layers=2,
            num_input_chunks=len(chunks),
            metadata={
                "refinement_passes": max_passes,
                "total_steps": len(all_nodes),
            },
        )
