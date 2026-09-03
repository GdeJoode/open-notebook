"""
Semantic blocking via UMAP dimensionality reduction + HDBSCAN clustering.

Groups entity embeddings into clusters so that pairwise comparison only
happens within each cluster, reducing the n² matching problem.

Requires optional dependencies: ``umap-learn``, ``hdbscan``, ``numpy``.
Falls back to single-block (all entities in one group) if unavailable.
"""

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from loguru import logger

try:
    import numpy as np

    _NUMPY = True
except ImportError:
    _NUMPY = False

try:
    import hdbscan
    import umap

    _HDBSCAN = True
except ImportError:
    _HDBSCAN = False


class SemanticBlocker:
    """Clusters entity embeddings to form candidate blocks for deduplication.

    Args:
        umap_n_components: Target dimensions for UMAP reduction (default 20).
        min_cluster_size: Minimum entities to form a cluster (default 2).
        min_samples: HDBSCAN density parameter (default 1 for dedup).
    """

    def __init__(
        self,
        umap_n_components: int = 20,
        min_cluster_size: int = 2,
        min_samples: int = 1,
    ) -> None:
        self._umap_dims = umap_n_components
        self._min_cluster_size = min_cluster_size
        self._min_samples = min_samples

    def block(
        self, entities: List[Dict[str, Any]]
    ) -> List[List[Dict[str, Any]]]:
        """Group entities into semantic blocks based on embedding similarity.

        Args:
            entities: List of entity dicts. Entities with
                ``properties["embedding"]`` are clustered; those without
                are placed in a single catch-all block.

        Returns:
            List of blocks, where each block is a list of entity dicts.
            Blocks with < 2 entities are excluded (singletons can't be
            matched).
        """
        if not _NUMPY or not _HDBSCAN:
            logger.warning(
                "UMAP/HDBSCAN not available — returning all entities "
                "in a single block (install umap-learn, hdbscan)"
            )
            return [entities] if len(entities) >= 2 else []

        # Split entities by embedding availability
        with_emb: List[Tuple[int, Dict[str, Any], List[float]]] = []
        without_emb: List[Dict[str, Any]] = []

        for i, entity in enumerate(entities):
            emb = entity.get("properties", {}).get("embedding")
            if emb and isinstance(emb, list) and len(emb) > 0:
                with_emb.append((i, entity, emb))
            else:
                without_emb.append(entity)

        if len(with_emb) < 2:
            # Not enough embedded entities to cluster
            all_entities = [e for _, e, _ in with_emb] + without_emb
            return [all_entities] if len(all_entities) >= 2 else []

        # Build embedding matrix
        embeddings = np.array([emb for _, _, emb in with_emb], dtype=np.float32)

        # UMAP dimensionality reduction
        n_components = min(self._umap_dims, embeddings.shape[1], len(with_emb) - 1)
        n_neighbors = min(15, len(with_emb) - 1)

        try:
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                metric="cosine",
                random_state=42,
            )
            reduced = reducer.fit_transform(embeddings)
        except Exception as e:
            logger.warning(f"UMAP reduction failed: {e} — using raw embeddings")
            reduced = embeddings

        # HDBSCAN clustering
        try:
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=self._min_cluster_size,
                min_samples=self._min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
            )
            labels = clusterer.fit_predict(reduced)
        except Exception as e:
            logger.warning(f"HDBSCAN clustering failed: {e} — single block")
            return [[e for _, e, _ in with_emb] + without_emb]

        # Group entities by cluster label
        clusters: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        noise_entities: List[Dict[str, Any]] = []

        for (_, entity, _), label in zip(with_emb, labels):
            if label == -1:
                noise_entities.append(entity)
            else:
                clusters[label].append(entity)

        # Build blocks (only those with 2+ entities)
        blocks: List[List[Dict[str, Any]]] = []
        for cluster_id, members in sorted(clusters.items()):
            if len(members) >= 2:
                blocks.append(members)
            else:
                noise_entities.extend(members)

        # Noise + non-embedded entities form a catch-all block
        catch_all = noise_entities + without_emb
        if len(catch_all) >= 2:
            blocks.append(catch_all)

        n_clustered = sum(len(b) for b in blocks)
        logger.info(
            f"SemanticBlocker: {len(entities)} entities → "
            f"{len(blocks)} blocks ({n_clustered} entities in blocks, "
            f"{len(set(labels)) - (1 if -1 in labels else 0)} clusters found)"
        )

        return blocks
