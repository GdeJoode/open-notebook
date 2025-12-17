"""
RAPTOR Tree Builder

Builds hierarchical summary trees from document chunks.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from loguru import logger

from .config import RaptorConfig
from .clustering import cluster_embeddings
from .summarizer import summarize_cluster


@dataclass
class RaptorNode:
    """Represents a node in the RAPTOR tree."""

    text: str
    """Text content (original chunk or summary)."""

    layer: int
    """Abstraction layer: 0=leaf, 1+=summary."""

    embedding: List[float] = field(default_factory=list)
    """Embedding vector for this node."""

    children_ids: List[str] = field(default_factory=list)
    """IDs of child nodes (chunks or lower-level summaries)."""

    chunk_id: Optional[str] = None
    """Database ID after saving (set by processor)."""

    metadata: Dict[str, Any] = field(default_factory=dict)
    """Additional metadata."""


@dataclass
class RaptorTree:
    """Complete RAPTOR tree structure."""

    nodes: List[RaptorNode] = field(default_factory=list)
    """All summary nodes (layer >= 1). Layer 0 nodes are existing chunks."""

    num_layers: int = 0
    """Number of layers in the tree (excluding layer 0)."""

    source_id: Optional[str] = None
    """Source document ID."""

    def get_layer(self, layer: int) -> List[RaptorNode]:
        """Get all nodes at a specific layer."""
        return [n for n in self.nodes if n.layer == layer]

    def get_all_texts(self) -> List[str]:
        """Get all node texts."""
        return [n.text for n in self.nodes]

    @property
    def total_nodes(self) -> int:
        """Total number of summary nodes."""
        return len(self.nodes)


class RaptorTreeBuilder:
    """
    Builds a RAPTOR hierarchical summary tree from document chunks.

    The algorithm:
    1. Start with leaf nodes (existing chunks with embeddings)
    2. Cluster similar nodes using UMAP + GMM
    3. Summarize each cluster with LLM
    4. Create parent nodes from summaries
    5. Repeat until convergence or max_layers reached
    """

    def __init__(self, config: Optional[RaptorConfig] = None):
        self.config = config or RaptorConfig()
        self._embedding_service = None

    async def _get_embedding_service(self):
        """Lazy load embedding service."""
        if self._embedding_service is None:
            from open_notebook.processors.embeddings import EmbeddingService
            self._embedding_service = EmbeddingService()
        return self._embedding_service

    async def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        service = await self._get_embedding_service()
        return await service.embed_batch(texts)

    async def build_tree(
        self,
        chunks: List[Dict[str, Any]],
        existing_embeddings: Optional[List[List[float]]] = None
    ) -> RaptorTree:
        """
        Build RAPTOR tree from document chunks.

        Args:
            chunks: List of chunk dicts with 'text', 'id', and optionally 'embedding'
            existing_embeddings: Pre-computed embeddings (optional)

        Returns:
            RaptorTree with summary nodes (layer >= 1)
        """
        n_chunks = len(chunks)

        if n_chunks < self.config.min_chunks_for_clustering:
            logger.info(
                f"Too few chunks ({n_chunks}) for RAPTOR "
                f"(min: {self.config.min_chunks_for_clustering}), skipping"
            )
            return RaptorTree()

        logger.info(f"Building RAPTOR tree from {n_chunks} chunks")

        tree = RaptorTree()

        # Prepare leaf node data
        leaf_texts = [c['text'] for c in chunks]
        leaf_ids = [c.get('id', f'chunk_{i}') for i, c in enumerate(chunks)]

        # Get or generate embeddings for leaf nodes
        if existing_embeddings and len(existing_embeddings) == n_chunks:
            leaf_embeddings = existing_embeddings
        else:
            # Check if chunks have embeddings
            if all('embedding' in c and c['embedding'] for c in chunks):
                leaf_embeddings = [c['embedding'] for c in chunks]
            else:
                logger.info("Generating embeddings for leaf nodes...")
                leaf_embeddings = await self._get_embeddings(leaf_texts)

        # Filter out chunks without valid embeddings
        valid_indices = [
            i for i, emb in enumerate(leaf_embeddings)
            if emb and len(emb) > 0
        ]

        if len(valid_indices) < self.config.min_chunks_for_clustering:
            logger.warning(f"Not enough valid embeddings ({len(valid_indices)})")
            return RaptorTree()

        # Current layer state
        current_texts = [leaf_texts[i] for i in valid_indices]
        current_embeddings = np.array([leaf_embeddings[i] for i in valid_indices])
        current_ids = [leaf_ids[i] for i in valid_indices]

        # Build layers iteratively
        for layer in range(1, self.config.max_layers + 1):
            n_current = len(current_texts)

            if n_current < self.config.min_chunks_for_clustering:
                logger.info(
                    f"Layer {layer}: Too few nodes ({n_current}) to cluster, stopping"
                )
                break

            if self.config.verbose:
                logger.info(f"Layer {layer}: Clustering {n_current} nodes")

            # Cluster current layer
            clusters = cluster_embeddings(
                current_embeddings,
                method=self.config.cluster_method,
                threshold=self.config.cluster_threshold,
                reduction_dim=self.config.reduction_dimension,
            )

            # Filter empty clusters and single-node clusters
            clusters = [c for c in clusters if len(c) > 1]

            if len(clusters) <= 1:
                logger.info(f"Layer {layer}: Only {len(clusters)} cluster(s), stopping")
                break

            if self.config.verbose:
                logger.info(f"Layer {layer}: Created {len(clusters)} clusters")

            # Create summary nodes for each cluster
            new_texts = []
            new_ids = []

            for cluster_idx, cluster_indices in enumerate(clusters):
                # Get texts from cluster
                cluster_texts = [current_texts[i] for i in cluster_indices]
                cluster_child_ids = [current_ids[i] for i in cluster_indices]

                # Summarize cluster
                try:
                    summary = await summarize_cluster(
                        cluster_texts,
                        max_tokens=self.config.summarization_max_tokens,
                        model_id=self.config.summarization_model,
                        max_input_tokens=self.config.max_tokens_per_cluster,
                    )
                except Exception as e:
                    logger.error(f"Summarization failed for cluster {cluster_idx}: {e}")
                    # Use concatenation as fallback
                    summary = " [...] ".join(t[:100] for t in cluster_texts[:3])

                if self.config.verbose:
                    logger.info(
                        f"Layer {layer}, Cluster {cluster_idx}: "
                        f"{len(cluster_texts)} texts -> summary ({len(summary)} chars)"
                    )

                # Create summary node (embedding will be generated below)
                node = RaptorNode(
                    text=summary,
                    layer=layer,
                    children_ids=cluster_child_ids,
                    metadata={
                        "cluster_size": len(cluster_texts),
                        "cluster_index": cluster_idx,
                    }
                )
                tree.nodes.append(node)

                new_texts.append(summary)
                # Temporary ID until saved to DB
                new_ids.append(f"raptor_L{layer}_C{cluster_idx}")

            if not new_texts:
                logger.info(f"Layer {layer}: No summaries created, stopping")
                break

            # Generate embeddings for new summaries (for next layer clustering)
            logger.info(f"Layer {layer}: Generating embeddings for {len(new_texts)} summaries")
            new_embeddings = await self._get_embeddings(new_texts)

            # Update embeddings in nodes
            layer_nodes = [n for n in tree.nodes if n.layer == layer]
            for node, emb in zip(layer_nodes, new_embeddings):
                node.embedding = emb

            # Prepare for next iteration
            current_texts = new_texts
            current_embeddings = np.array([
                emb if emb else np.zeros(current_embeddings.shape[1])
                for emb in new_embeddings
            ])
            current_ids = new_ids
            tree.num_layers = layer

        logger.info(
            f"RAPTOR tree complete: {tree.total_nodes} summary nodes, "
            f"{tree.num_layers} layer(s)"
        )
        return tree
