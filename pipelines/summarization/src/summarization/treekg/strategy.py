"""TreeKG structure-aware hierarchical summarization strategy.

Leverages document section structure (section_path metadata) to build a
summary tree that mirrors the document heading hierarchy.  Leaf sections
are summarized first, then parent sections are summarized by combining
their children's summaries bottom-up until a single root summary remains.
"""

import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

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


class TreeKGStrategy(BaseSummarizationStrategy):
    """Structure-aware hierarchical summarization.

    Builds a summary tree that mirrors the document's heading hierarchy.
    Summarizes leaf sections first, then aggregates upward to produce a
    root-level document summary.

    Falls back to naive-like concatenate-and-summarize behaviour when
    chunks lack ``section_path`` metadata.
    """

    def __init__(self, config: SummarizationConfig, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self._llm_config = config.llm
        self._treekg_config = config.treekg

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _get_model(self, max_tokens: int):
        return AIFactory.create_language(
            self._llm_config.provider,
            model_name=self._llm_config.model_name,
            config={
                "base_url": self._llm_config.base_url,
                "temperature": self._llm_config.temperature,
                "max_tokens": max_tokens,
                "num_ctx": self._llm_config.num_ctx,
            },
        )

    async def _call_llm(self, prompt: str, max_tokens: int = 500) -> str:
        model = self._get_model(max_tokens)
        prompt = self.apply_output_instructions(prompt)
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        response = await model.achat_complete(messages)
        return self.normalize_response(response.content)

    # ------------------------------------------------------------------
    # Section tree construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_section_tree(
        chunks: List[ChunkInput],
    ) -> Dict[Tuple[str, ...], List[ChunkInput]]:
        """Group chunks by their section_path into a dict keyed by path tuple.

        Chunks without a ``section_path`` are placed under a ``("General",)``
        fallback key.
        """
        tree: Dict[Tuple[str, ...], List[ChunkInput]] = defaultdict(list)
        for chunk in chunks:
            if chunk.section_path:
                key = tuple(chunk.section_path)
            else:
                key = ("General",)
            tree[key].append(chunk)
        return dict(tree)

    @staticmethod
    def _get_sorted_paths(
        tree: Dict[Tuple[str, ...], List[ChunkInput]],
    ) -> List[Tuple[str, ...]]:
        """Return section paths sorted deepest-first (leaves before parents)."""
        return sorted(tree.keys(), key=lambda p: len(p), reverse=True)

    @staticmethod
    def _parent_path(path: Tuple[str, ...]) -> Optional[Tuple[str, ...]]:
        if len(path) <= 1:
            return None
        return path[:-1]

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    async def summarize(self, chunks: List[ChunkInput]) -> SummarizationResult:
        """Run TreeKG summarization on the provided chunks.

        Algorithm:
        1. Group chunks by ``section_path`` to build a tree structure.
        2. For leaf sections (deepest level): concatenate chunk texts and
           summarize if the content exceeds ``min_content_length``.
        3. For parent sections: concatenate children summaries and
           summarize upward.
        4. Root summary becomes the ``document_summary``.
        """
        if not chunks:
            return SummarizationResult(
                strategy=SummarizationStrategy.TREEKG,
                document_summary="",
                num_input_chunks=0,
            )

        # Single chunk shortcut
        if len(chunks) == 1:
            text = chunks[0].text.strip()
            node_id = str(uuid.uuid4())
            node = SummaryNode(
                text=text,
                layer=0,
                source_chunk_ids=[chunks[0].chunk_id or "0"],
                node_id=node_id,
                section_title=chunks[0].section_path[-1] if chunks[0].section_path else None,
            )
            return SummarizationResult(
                strategy=SummarizationStrategy.TREEKG,
                document_summary=text,
                summary_nodes=[node],
                num_layers=1,
                num_input_chunks=1,
            )

        # Check whether any chunk carries section_path
        has_structure = any(c.section_path for c in chunks)
        if not has_structure:
            logger.warning(
                "No section_path metadata found on any chunk; "
                "falling back to flat summarization."
            )
            return await self._flat_fallback(chunks)

        tree = self._build_section_tree(chunks)
        sorted_paths = self._get_sorted_paths(tree)

        logger.info(
            f"TreeKG strategy: {len(chunks)} chunks across "
            f"{len(tree)} sections (max depth "
            f"{max(len(p) for p in tree)})"
        )

        # section_path -> (summary_text, SummaryNode)
        section_summaries: Dict[Tuple[str, ...], Tuple[str, SummaryNode]] = {}
        all_nodes: List[SummaryNode] = []
        max_layer = 0

        for path in sorted_paths:
            section_chunks = tree.get(path, [])
            section_title = path[-1] if path else "Root"
            chunk_ids = [c.chunk_id or str(c.order) for c in section_chunks]

            # Collect children summaries for this section
            children_texts: List[str] = []
            children_node_ids: List[str] = []
            for other_path, (child_text, child_node) in section_summaries.items():
                if self._parent_path(other_path) == path:
                    children_texts.append(child_text)
                    children_node_ids.append(child_node.node_id or "")

            # Build content for this section
            own_text = "\n\n".join(c.text for c in section_chunks if c.text.strip())
            if children_texts:
                combined = own_text + "\n\n" + "\n\n".join(children_texts) if own_text else "\n\n".join(children_texts)
            else:
                combined = own_text

            # Determine layer based on depth (leaves = 0, root = deepest)
            depth = len(path)
            max_depth = max(len(p) for p in tree)
            layer = max_depth - depth

            if len(combined) < self._treekg_config.min_content_length:
                # Content too short to summarize; keep as-is
                summary_text = combined.strip() if combined.strip() else section_title
            else:
                # Truncate if needed
                content = combined[: self._treekg_config.max_content_length]
                prompt = (
                    f"Summarize the following content from the section "
                    f'"{section_title}". Preserve the key ideas and '
                    f"important details.\n\n"
                    f"CONTENT:\n{content}"
                )
                summary_text = await self._call_llm(
                    prompt,
                    max_tokens=self._treekg_config.summarization_max_tokens,
                )

            node_id = str(uuid.uuid4())
            node = SummaryNode(
                text=summary_text,
                layer=layer,
                source_chunk_ids=chunk_ids,
                children_ids=children_node_ids,
                node_id=node_id,
                section_title=section_title,
                metadata={"section_path": list(path)},
            )
            section_summaries[path] = (summary_text, node)
            all_nodes.append(node)
            max_layer = max(max_layer, layer)

        # Determine document summary: the root-level node (shortest path)
        root_paths = sorted(section_summaries.keys(), key=lambda p: len(p))
        if len(root_paths) == 1:
            document_summary = section_summaries[root_paths[0]][0]
        else:
            # Multiple top-level sections — combine them into a final summary
            top_summaries = [
                section_summaries[p][0]
                for p in root_paths
                if self._parent_path(p) is None
            ]
            if not top_summaries:
                top_summaries = [section_summaries[p][0] for p in root_paths[:3]]

            combined_top = "\n\n".join(
                f"[{root_paths[i][-1]}]: {s}"
                for i, s in enumerate(top_summaries)
            )
            prompt = (
                "Combine the following section summaries into a single "
                "coherent document summary:\n\n"
                f"{combined_top}"
            )
            document_summary = await self._call_llm(
                prompt,
                max_tokens=self._treekg_config.summarization_max_tokens,
            )
            max_layer += 1
            root_node = SummaryNode(
                text=document_summary,
                layer=max_layer,
                source_chunk_ids=[],
                children_ids=[
                    section_summaries[p][1].node_id or ""
                    for p in root_paths
                    if self._parent_path(p) is None
                ],
                node_id=str(uuid.uuid4()),
                section_title="Document Summary",
                metadata={"is_root": True},
            )
            all_nodes.append(root_node)

        return SummarizationResult(
            strategy=SummarizationStrategy.TREEKG,
            document_summary=document_summary,
            summary_nodes=all_nodes,
            num_layers=max_layer + 1,
            num_input_chunks=len(chunks),
            metadata={"num_sections": len(tree), "has_structure": True},
        )

    # ------------------------------------------------------------------
    # Fallback when no section_path metadata exists
    # ------------------------------------------------------------------

    async def _flat_fallback(self, chunks: List[ChunkInput]) -> SummarizationResult:
        """Naive-like flat summarization when section metadata is absent."""
        full_text = "\n\n".join(c.text for c in chunks if c.text.strip())
        chunk_ids = [c.chunk_id or str(c.order) for c in chunks]

        content = full_text[: self._treekg_config.max_content_length]
        prompt = (
            "Provide a comprehensive summary of the following text. "
            "Focus on the key ideas, arguments, and conclusions.\n\n"
            f"TEXT:\n{content}"
        )
        summary_text = await self._call_llm(
            prompt,
            max_tokens=self._treekg_config.summarization_max_tokens,
        )

        node = SummaryNode(
            text=summary_text,
            layer=0,
            source_chunk_ids=chunk_ids,
            node_id=str(uuid.uuid4()),
            section_title="General",
            metadata={"fallback": True},
        )

        return SummarizationResult(
            strategy=SummarizationStrategy.TREEKG,
            document_summary=summary_text,
            summary_nodes=[node],
            num_layers=1,
            num_input_chunks=len(chunks),
            metadata={"num_sections": 1, "has_structure": False},
        )
