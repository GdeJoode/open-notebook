"""
Incremental entity resolution with cluster repair.

Compares new entity mentions against existing clusters (represented by
their centroid embeddings) and either assigns to an existing cluster or
creates a new one. Periodically repairs clusters by splitting low-coherence
groups and merging high-similarity neighbors.

Based on Saeedi et al. (2020) — incremental ER reduces dependency on
input ordering and avoids full re-computation as the graph grows.
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def _centroid(vectors: List[List[float]]) -> List[float]:
    """Compute the mean vector of a list of vectors."""
    if not vectors:
        return []
    dim = len(vectors[0])
    result = [0.0] * dim
    for v in vectors:
        for i in range(dim):
            result[i] += v[i]
    n = len(vectors)
    return [x / n for x in result]


class EntityCluster:
    """In-memory representation of an entity cluster."""

    def __init__(
        self,
        cluster_id: str,
        members: List[Dict[str, Any]],
        centroid: Optional[List[float]] = None,
        version: int = 1,
    ):
        self.cluster_id = cluster_id
        self.members = members  # list of entity dicts with "text", "embedding", etc.
        self.version = version
        self.centroid = centroid or self._compute_centroid()

    def _compute_centroid(self) -> List[float]:
        embeddings = [
            m.get("embedding", []) or m.get("properties", {}).get("embedding", [])
            for m in self.members
        ]
        embeddings = [e for e in embeddings if e]
        return _centroid(embeddings) if embeddings else []

    def add_member(self, entity: Dict[str, Any]) -> None:
        self.members.append(entity)
        self.centroid = self._compute_centroid()
        self.version += 1

    def internal_coherence(self) -> float:
        """Average pairwise similarity within the cluster. 1.0 = perfect."""
        if len(self.members) < 2 or not self.centroid:
            return 1.0
        embeddings = [
            m.get("embedding", []) or m.get("properties", {}).get("embedding", [])
            for m in self.members
        ]
        embeddings = [e for e in embeddings if e]
        if len(embeddings) < 2:
            return 1.0
        total, count = 0.0, 0
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                total += _cosine_similarity(embeddings[i], embeddings[j])
                count += 1
        return total / count if count > 0 else 1.0


class IncrementalResolver:
    """Assigns new entities to existing clusters or creates new ones.

    Args:
        similarity_threshold: Minimum cosine similarity to assign to a cluster.
        coherence_threshold: Minimum internal coherence before a cluster is
            flagged for splitting during repair.
        merge_threshold: Maximum inter-cluster centroid similarity before
            two clusters are flagged for merging during repair.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.85,
        coherence_threshold: float = 0.70,
        merge_threshold: float = 0.92,
    ):
        self._sim_threshold = similarity_threshold
        self._coherence_threshold = coherence_threshold
        self._merge_threshold = merge_threshold

    def resolve_incremental(
        self,
        new_entities: List[Dict[str, Any]],
        existing_clusters: List[EntityCluster],
    ) -> Tuple[List[EntityCluster], List[Dict[str, Any]]]:
        """Assign new entities to existing clusters or create new ones.

        Args:
            new_entities: Entities with embeddings to resolve.
            existing_clusters: Current clusters with centroids.

        Returns:
            (updated_clusters, assignments) where assignments is a list of
            dicts with entity_text, cluster_id, action (assigned|new), score.
        """
        assignments: List[Dict[str, Any]] = []

        for entity in new_entities:
            emb = entity.get("embedding") or entity.get("properties", {}).get("embedding")
            text = entity.get("text", "")

            if not emb:
                # No embedding — create a singleton cluster
                cluster_id = f"cluster_{text[:20]}_{len(existing_clusters)}"
                new_cluster = EntityCluster(
                    cluster_id=cluster_id,
                    members=[entity],
                )
                existing_clusters.append(new_cluster)
                assignments.append({
                    "entity_text": text,
                    "cluster_id": cluster_id,
                    "action": "new",
                    "score": 0.0,
                })
                continue

            # Find best matching cluster by centroid similarity
            best_score = -1.0
            best_cluster: Optional[EntityCluster] = None

            for cluster in existing_clusters:
                if not cluster.centroid:
                    continue
                sim = _cosine_similarity(emb, cluster.centroid)
                if sim > best_score:
                    best_score = sim
                    best_cluster = cluster

            if best_cluster and best_score >= self._sim_threshold:
                best_cluster.add_member(entity)
                assignments.append({
                    "entity_text": text,
                    "cluster_id": best_cluster.cluster_id,
                    "action": "assigned",
                    "score": round(best_score, 4),
                })
            else:
                cluster_id = f"cluster_{text[:20]}_{len(existing_clusters)}"
                new_cluster = EntityCluster(
                    cluster_id=cluster_id,
                    members=[entity],
                )
                existing_clusters.append(new_cluster)
                assignments.append({
                    "entity_text": text,
                    "cluster_id": cluster_id,
                    "action": "new",
                    "score": round(best_score, 4) if best_score >= 0 else 0.0,
                })

        logger.info(
            f"Incremental resolution: {len(new_entities)} entities → "
            f"{sum(1 for a in assignments if a['action'] == 'assigned')} assigned, "
            f"{sum(1 for a in assignments if a['action'] == 'new')} new clusters"
        )
        return existing_clusters, assignments

    def repair_clusters(
        self, clusters: List[EntityCluster]
    ) -> Tuple[List[EntityCluster], Dict[str, Any]]:
        """Check and repair cluster quality.

        1. Split clusters with low internal coherence.
        2. Merge clusters with high centroid similarity.

        Returns:
            (repaired_clusters, report) with repair statistics.
        """
        splits = 0
        merges = 0
        repaired: List[EntityCluster] = []

        # Phase 1: Split low-coherence clusters
        for cluster in clusters:
            coherence = cluster.internal_coherence()
            if coherence < self._coherence_threshold and len(cluster.members) >= 2:
                # Simple split: separate the most dissimilar member
                split_a, split_b = self._split_cluster(cluster)
                repaired.append(split_a)
                repaired.append(split_b)
                splits += 1
                logger.debug(
                    f"Split cluster {cluster.cluster_id} "
                    f"(coherence={coherence:.3f}): "
                    f"{len(split_a.members)} + {len(split_b.members)} members"
                )
            else:
                repaired.append(cluster)

        # Phase 2: Merge high-similarity clusters
        merged_indices: Set[int] = set()
        final: List[EntityCluster] = []

        for i in range(len(repaired)):
            if i in merged_indices:
                continue
            for j in range(i + 1, len(repaired)):
                if j in merged_indices:
                    continue
                if not repaired[i].centroid or not repaired[j].centroid:
                    continue
                sim = _cosine_similarity(repaired[i].centroid, repaired[j].centroid)
                if sim >= self._merge_threshold:
                    # Merge j into i
                    for member in repaired[j].members:
                        repaired[i].add_member(member)
                    merged_indices.add(j)
                    merges += 1
                    logger.debug(
                        f"Merged cluster {repaired[j].cluster_id} into "
                        f"{repaired[i].cluster_id} (similarity={sim:.3f})"
                    )
            final.append(repaired[i])

        report = {
            "clusters_before": len(clusters),
            "clusters_after": len(final),
            "splits": splits,
            "merges": merges,
        }
        logger.info(f"Cluster repair: {report}")
        return final, report

    def _split_cluster(
        self, cluster: EntityCluster
    ) -> Tuple[EntityCluster, EntityCluster]:
        """Split a cluster by finding the member most dissimilar to the centroid."""
        if not cluster.centroid or len(cluster.members) < 2:
            return cluster, EntityCluster(
                cluster_id=f"{cluster.cluster_id}_split",
                members=[],
                version=cluster.version + 1,
            )

        # Find the member furthest from centroid
        worst_idx = 0
        worst_sim = 1.0
        for i, m in enumerate(cluster.members):
            emb = m.get("embedding") or m.get("properties", {}).get("embedding")
            if not emb:
                continue
            sim = _cosine_similarity(emb, cluster.centroid)
            if sim < worst_sim:
                worst_sim = sim
                worst_idx = i

        outlier = cluster.members.pop(worst_idx)
        cluster.centroid = cluster._compute_centroid()
        cluster.version += 1

        new_cluster = EntityCluster(
            cluster_id=f"{cluster.cluster_id}_split_{cluster.version}",
            members=[outlier],
            version=1,
        )
        return cluster, new_cluster
