"""Naive single-pass LLM summarization strategy."""

import uuid
from typing import List

from esperanto import AIFactory
from loguru import logger

from summarization.config import SummarizationConfig
from summarization.models.result import (
    ChunkInput,
    SummarizationResult,
    SummarizationStrategy,
    SummaryNode,
)
from summarization.strategies.base import BaseSummarizationStrategy


class NaiveLLMStrategy(BaseSummarizationStrategy):
    """Single-pass LLM summarization.

    Concatenates all chunks (up to a token limit), sends them to the LLM
    in a single prompt, and returns the resulting summary. For documents
    exceeding the context window, chunks are split and partial summaries
    are combined in a second pass.
    """

    def __init__(self, config: SummarizationConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self._llm_config = config.llm
        self._naive_config = config.naive

    def _get_model(self):
        """Return the LanguageModel: app-injected (J.4 routed) when present, else
        constructed from the env-driven LLMConfig (back-compat ollama path)."""
        if self.injected_model is not None:
            return self.injected_model
        return AIFactory.create_language(
            self._llm_config.provider,
            model_name=self._llm_config.model_name,
            config={
                "base_url": self._llm_config.base_url,
                "temperature": self._llm_config.temperature,
                "max_tokens": self._naive_config.summarization_max_tokens,
                "num_ctx": self._llm_config.num_ctx,
            },
        )

    def _split_into_windows(self, text: str) -> List[str]:
        """Split text into overlapping windows that fit within max_input_length."""
        max_len = self._naive_config.max_input_length
        overlap = self._naive_config.chunk_overlap

        if len(text) <= max_len:
            return [text]

        windows = []
        start = 0
        while start < len(text):
            end = start + max_len
            windows.append(text[start:end])
            start = end - overlap
            if start >= len(text):
                break

        return windows

    async def _call_llm(self, prompt: str) -> str:
        """Send a prompt to the LLM and return the response text."""
        model = self._get_model()
        prompt = self.apply_output_instructions(prompt)
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        response = await model.achat_complete(messages)
        return self.normalize_response(response.content)

    async def summarize(self, chunks: List[ChunkInput]) -> SummarizationResult:
        """Run naive summarization on the provided chunks."""
        if not chunks:
            return SummarizationResult(
                strategy=SummarizationStrategy.NAIVE,
                document_summary="",
                num_input_chunks=0,
            )

        # Concatenate all chunk texts
        full_text = "\n\n".join(c.text for c in chunks if c.text.strip())
        chunk_ids = [c.chunk_id for c in chunks if c.chunk_id]

        logger.info(
            f"Naive strategy: {len(chunks)} chunks, "
            f"{len(full_text)} chars total"
        )

        windows = self._split_into_windows(full_text)
        summary_nodes: List[SummaryNode] = []

        if len(windows) == 1:
            # Single pass — fits in one window
            prompt = (
                "Provide a comprehensive summary of the following text. "
                "Focus on the key ideas, arguments, and conclusions.\n\n"
                f"TEXT:\n{windows[0]}"
            )
            summary_text = await self._call_llm(prompt)

            summary_nodes.append(
                SummaryNode(
                    text=summary_text,
                    layer=0,
                    source_chunk_ids=chunk_ids,
                    node_id=str(uuid.uuid4()),
                )
            )

            return SummarizationResult(
                strategy=SummarizationStrategy.NAIVE,
                document_summary=summary_text,
                summary_nodes=summary_nodes,
                num_layers=1,
                num_input_chunks=len(chunks),
            )

        # Multiple windows — summarize each, then combine
        logger.info(f"Text split into {len(windows)} windows")
        partial_summaries: List[str] = []

        for i, window in enumerate(windows):
            prompt = (
                f"Provide a summary of this section (part {i + 1} of "
                f"{len(windows)}) of a larger document.\n\n"
                f"TEXT:\n{window}"
            )
            partial = await self._call_llm(prompt)
            partial_summaries.append(partial)

            summary_nodes.append(
                SummaryNode(
                    text=partial,
                    layer=0,
                    source_chunk_ids=chunk_ids,
                    node_id=str(uuid.uuid4()),
                    metadata={"window_index": i},
                )
            )

        if self._naive_config.combine_summaries:
            # Second pass: combine partial summaries
            combined_text = "\n\n".join(
                f"[Section {i + 1}]: {s}"
                for i, s in enumerate(partial_summaries)
            )
            combine_prompt = (
                "The following are summaries of different sections of the "
                "same document. Combine them into a single coherent summary "
                "that captures the key themes and conclusions.\n\n"
                f"{combined_text}"
            )
            final_summary = await self._call_llm(combine_prompt)

            summary_nodes.append(
                SummaryNode(
                    text=final_summary,
                    layer=1,
                    source_chunk_ids=chunk_ids,
                    children_ids=[n.node_id or "" for n in summary_nodes],
                    node_id=str(uuid.uuid4()),
                    metadata={"combined": True},
                )
            )

            return SummarizationResult(
                strategy=SummarizationStrategy.NAIVE,
                document_summary=final_summary,
                summary_nodes=summary_nodes,
                num_layers=2,
                num_input_chunks=len(chunks),
            )

        # No combining — use last partial as document summary
        return SummarizationResult(
            strategy=SummarizationStrategy.NAIVE,
            document_summary=partial_summaries[-1],
            summary_nodes=summary_nodes,
            num_layers=1,
            num_input_chunks=len(chunks),
        )
