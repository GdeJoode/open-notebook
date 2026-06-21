"""RAPTOR recursive clustering and summarization strategy.

Implements Recursive Abstractive Processing for Tree-Organized Retrieval.
Chunks are embedded, clustered via Gaussian Mixture Models with soft
assignment, summarized per cluster, and the process repeats on the
resulting summaries to build a hierarchical tree of increasingly abstract
representations.
"""

import uuid
from typing import Any, Callable, Coroutine, Dict, List, Optional

import numpy as np
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

EmbeddingFn = Callable[[List[str]], Coroutine[Any, Any, List[List[float]]]]


class RaptorStrategy(BaseSummarizationStrategy):
    """Recursive Abstractive Processing for Tree-Organized Retrieval.

    Clusters chunks by embedding similarity using GMM, summarizes each
    cluster, then repeats recursively to build a hierarchical summary
    tree.  Requires an async embedding function for clustering.

    The algorithm terminates when:
    - Only one summary node remains, or
    - Fewer than ``min_chunks_for_clustering`` nodes exist, or
    - ``max_layers`` is reached.
    """

    def __init__(
        self,
        config: SummarizationConfig,
        embedding_fn: EmbeddingFn,
        language_hint: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__(config, **kwargs)
        self._llm_config = config.llm
        self._raptor_config = config.raptor
        self._embedding_fn = embedding_fn
        self._language_hint = language_hint

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _get_model(self, max_tokens: int):
        if self.injected_model is not None:
            return self.injected_model
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

    async def _call_llm(self, prompt: str, max_tokens: int = 200) -> str:
        model = self._get_model(max_tokens)
        prompt = self.apply_output_instructions(prompt)
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        response = await model.achat_complete(messages)
        return self.normalize_response(response.content)

    async def _call_llm_minimal(self, prompt: str, max_tokens: int = 200) -> str:
        """LLM call without appending output_instructions.

        Used for intermediate cluster-summary calls: those should produce short
        prose, not the full meeting-note template that output_instructions
        describes (which only belongs at the final/root level). System prompt
        is still applied — it carries faithfulness constraints ("only report
        what was said", language matching) that we want on every call.
        """
        model = self._get_model(max_tokens)
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt},
        ]
        response = await model.achat_complete(messages)
        return self.normalize_response(response.content)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    async def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Retrieve embeddings for a list of texts via the injected function.

        Returns a 2-D numpy array of shape ``(len(texts), dim)``.
        """
        embeddings = await self._embedding_fn(texts)
        return np.array(embeddings, dtype=np.float64)

    # ------------------------------------------------------------------
    # Clustering
    # ------------------------------------------------------------------

    def _cluster_texts(
        self,
        embeddings: np.ndarray,
        threshold: float,
        reduction_dim: int,
    ) -> Dict[int, List[int]]:
        """Cluster embeddings using PCA + GMM with soft assignment.

        Returns a mapping of ``cluster_id -> [text_indices]``.  A single
        text may appear in multiple clusters when its soft-assignment
        probability exceeds ``threshold`` for more than one component.
        """
        from sklearn.decomposition import PCA
        from sklearn.mixture import GaussianMixture

        n_samples, n_features = embeddings.shape

        # PCA dimensionality reduction
        effective_dim = min(reduction_dim, n_samples, n_features)
        if effective_dim < n_features:
            pca = PCA(n_components=effective_dim)
            reduced = pca.fit_transform(embeddings)
        else:
            reduced = embeddings

        # Determine number of components (heuristic: sqrt of samples, min 2)
        n_components = max(2, min(int(np.sqrt(n_samples)), n_samples))

        gmm = GaussianMixture(
            n_components=n_components,
            covariance_type="full",
            random_state=42,
        )
        gmm.fit(reduced)
        probs = gmm.predict_proba(reduced)  # (n_samples, n_components)

        # Soft assignment: assign each text to all clusters above threshold
        clusters: Dict[int, List[int]] = {}
        for idx in range(n_samples):
            assigned = False
            for comp in range(n_components):
                if probs[idx, comp] >= threshold:
                    clusters.setdefault(comp, []).append(idx)
                    assigned = True
            # Fallback: assign to highest-probability cluster
            if not assigned:
                best = int(np.argmax(probs[idx]))
                clusters.setdefault(best, []).append(idx)

        # Remove empty clusters
        return {k: v for k, v in clusters.items() if v}

    # ------------------------------------------------------------------
    # Core algorithm
    # ------------------------------------------------------------------

    async def summarize(self, chunks: List[ChunkInput]) -> SummarizationResult:
        """Run RAPTOR summarization on the provided chunks.

        Algorithm:
        1. Start with input chunks as layer 0.
        2. Get embeddings for all chunk texts.
        3. Reduce dimensionality via PCA.
        4. Cluster using GMM with soft assignment.
        5. For each cluster: concatenate member texts (up to
           ``max_tokens_per_cluster`` chars), summarize to produce a
           layer N+1 node.
        6. Repeat from step 2 using the new summary nodes.
        7. Stop when termination criteria are met.
        8. Top-most summary becomes ``document_summary``.
        """
        if not chunks:
            return SummarizationResult(
                strategy=SummarizationStrategy.RAPTOR,
                document_summary="",
                num_input_chunks=0,
            )

        # Single chunk — no clustering needed, use full token budget
        if len(chunks) == 1:
            text = chunks[0].text.strip()
            prompt = (
                "Provide a comprehensive summary of the following text.\n\n"
                f"TEXT:\n{text}"
            )
            summary = await self._call_llm(
                prompt,
                max_tokens=self._llm_config.max_tokens,
            )
            node = SummaryNode(
                text=summary,
                layer=0,
                source_chunk_ids=[chunks[0].chunk_id or "0"],
                node_id=str(uuid.uuid4()),
            )
            return SummarizationResult(
                strategy=SummarizationStrategy.RAPTOR,
                document_summary=summary,
                summary_nodes=[node],
                num_layers=1,
                num_input_chunks=1,
            )

        logger.info(
            f"RAPTOR strategy: {len(chunks)} chunks, "
            f"max_layers={self._raptor_config.max_layers}"
        )

        all_nodes: List[SummaryNode] = []

        # Current layer texts and their IDs
        current_texts: List[str] = [c.text for c in chunks]
        current_ids: List[str] = [
            c.chunk_id or str(uuid.uuid4()) for c in chunks
        ]
        current_layer = 0

        for layer_num in range(1, self._raptor_config.max_layers + 1):
            n_texts = len(current_texts)

            # Termination: too few texts to cluster
            if n_texts < self._raptor_config.min_chunks_for_clustering:
                logger.info(
                    f"Layer {layer_num}: only {n_texts} texts, below "
                    f"min_chunks_for_clustering "
                    f"({self._raptor_config.min_chunks_for_clustering}). "
                    "Stopping."
                )
                break

            # Termination: single text remaining
            if n_texts == 1:
                logger.info("Layer {}: single text remaining. Stopping.", layer_num)
                break

            logger.info(
                f"RAPTOR layer {layer_num}: clustering {n_texts} texts"
            )

            # Step 2: get embeddings
            try:
                embeddings = await self._get_embeddings(current_texts)
            except Exception as e:
                logger.error(
                    f"Embedding failed at layer {layer_num}: {e}. "
                    "Falling back to concatenation."
                )
                break

            if embeddings.shape[0] != n_texts:
                logger.error(
                    f"Embedding count mismatch: got {embeddings.shape[0]}, "
                    f"expected {n_texts}. Stopping."
                )
                break

            # Steps 3-4: PCA + GMM clustering
            try:
                clusters = self._cluster_texts(
                    embeddings,
                    threshold=self._raptor_config.cluster_threshold,
                    reduction_dim=self._raptor_config.reduction_dimension,
                )
            except Exception as e:
                logger.error(
                    f"Clustering failed at layer {layer_num}: {e}. Stopping."
                )
                break

            if not clusters:
                logger.warning(f"No clusters produced at layer {layer_num}. Stopping.")
                break

            logger.info(
                f"Layer {layer_num}: {len(clusters)} clusters formed"
            )

            # Step 5: summarize each cluster
            next_texts: List[str] = []
            next_ids: List[str] = []

            for cluster_id, member_indices in sorted(clusters.items()):
                # Collect member texts up to max_tokens_per_cluster
                member_texts: List[str] = []
                member_chunk_ids: List[str] = []
                total_len = 0
                for idx in member_indices:
                    text = current_texts[idx]
                    if total_len + len(text) > self._raptor_config.max_tokens_per_cluster:
                        break
                    member_texts.append(text)
                    member_chunk_ids.append(current_ids[idx])
                    total_len += len(text)

                if not member_texts:
                    continue

                combined = "\n\n".join(member_texts)
                lang_directive = (
                    f"WRITE YOUR SUMMARY IN {self._language_hint.upper()}. "
                    f"This is non-negotiable: every word, including technical "
                    f"terms and proper nouns, must be in {self._language_hint}. "
                    f"Do not switch to another language even if the source "
                    f"contains foreign loanwords.\n\n"
                    if self._language_hint
                    else "Write your summary in the SAME language as the TEXTS below.\n\n"
                )
                prompt = (
                    f"{lang_directive}"
                    "Summarize the following related text fragments into a "
                    "concise, factual prose summary (3-6 short paragraphs). "
                    "Do NOT use section headers, bullet lists, or any "
                    "templated structure — only flowing paragraphs. Preserve "
                    "speaker attributions (SPEAKER_N or names) and concrete "
                    "content (names, numbers, decisions, quoted statements). "
                    "Do not interpret or add context that is not present in "
                    "the source.\n\n"
                    f"TEXTS:\n{combined}"
                )
                # Intermediate aggregate — bypass the meeting-note output_instructions.
                summary_text = await self._call_llm_minimal(
                    prompt,
                    max_tokens=self._raptor_config.summarization_max_tokens,
                )

                node_id = str(uuid.uuid4())
                node = SummaryNode(
                    text=summary_text,
                    layer=layer_num,
                    source_chunk_ids=member_chunk_ids,
                    children_ids=[],
                    node_id=node_id,
                    metadata={
                        "cluster_id": cluster_id,
                        "num_members": len(member_texts),
                    },
                )
                all_nodes.append(node)
                next_texts.append(summary_text)
                next_ids.append(node_id)

            if not next_texts:
                logger.warning(f"No summaries produced at layer {layer_num}. Stopping.")
                break

            current_texts = next_texts
            current_ids = next_ids
            current_layer = layer_num

            # Single summary remaining — we're done
            if len(current_texts) == 1:
                logger.info(f"Single summary at layer {layer_num}. Done.")
                break

        # Determine document summary
        if current_texts:
            if len(current_texts) == 1:
                document_summary = current_texts[0]
            else:
                # Multiple texts remain — combine into final summary
                combined_final = "\n\n".join(current_texts)
                prompt = (
                    "Combine the following summaries into a single coherent "
                    "document summary:\n\n"
                    f"{combined_final}"
                )
                document_summary = await self._call_llm(
                    prompt,
                    max_tokens=self._llm_config.max_tokens,
                )
                current_layer += 1
                root_node = SummaryNode(
                    text=document_summary,
                    layer=current_layer,
                    source_chunk_ids=current_ids,
                    children_ids=[],
                    node_id=str(uuid.uuid4()),
                    metadata={"is_root": True},
                )
                all_nodes.append(root_node)
        else:
            document_summary = ""

        return SummarizationResult(
            strategy=SummarizationStrategy.RAPTOR,
            document_summary=document_summary,
            summary_nodes=all_nodes,
            num_layers=current_layer + 1,
            num_input_chunks=len(chunks),
            metadata={
                "max_layers_configured": self._raptor_config.max_layers,
                "layers_built": current_layer,
                "total_nodes": len(all_nodes),
            },
        )
