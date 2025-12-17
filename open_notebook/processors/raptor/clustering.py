"""
RAPTOR Clustering Module

Implements UMAP dimensionality reduction + GMM soft clustering.
Falls back to PCA + KMeans if UMAP is not available.

Based on: https://github.com/parthsarthi03/raptor
"""

from typing import List, Optional

import numpy as np
from loguru import logger
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans

# Optional UMAP import (heavy dependency)
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    logger.info("UMAP not installed. RAPTOR will use PCA fallback for clustering.")


RANDOM_SEED = 42


def reduce_embeddings(
    embeddings: np.ndarray,
    n_components: int = 10,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine"
) -> np.ndarray:
    """
    Reduce embedding dimensionality using UMAP or PCA.

    Args:
        embeddings: Array of shape (n_samples, n_features)
        n_components: Target dimensionality
        n_neighbors: UMAP neighbor parameter (auto-calculated if None)
        metric: Distance metric for UMAP

    Returns:
        Reduced embeddings of shape (n_samples, n_components)
    """
    n_samples = len(embeddings)

    # Ensure we don't request more components than samples
    n_components = min(n_components, n_samples - 1, embeddings.shape[1])

    if n_components < 2:
        # Can't reduce further, return as-is
        return embeddings

    if UMAP_AVAILABLE and n_samples > 10:
        # Use UMAP for better non-linear reduction
        if n_neighbors is None:
            n_neighbors = int((n_samples - 1) ** 0.5)
        n_neighbors = min(n_neighbors, n_samples - 1)

        try:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                metric=metric,
                random_state=RANDOM_SEED,
                min_dist=0.0,
            )
            return reducer.fit_transform(embeddings)
        except Exception as e:
            logger.warning(f"UMAP failed, falling back to PCA: {e}")

    # Fallback to PCA
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components, random_state=RANDOM_SEED).fit_transform(embeddings)


def get_optimal_clusters(
    embeddings: np.ndarray,
    max_clusters: int = 50,
    min_clusters: int = 2,
) -> int:
    """
    Find optimal number of clusters using BIC (Bayesian Information Criterion).

    Args:
        embeddings: Reduced embeddings
        max_clusters: Maximum clusters to try
        min_clusters: Minimum clusters

    Returns:
        Optimal number of clusters
    """
    n_samples = len(embeddings)
    max_clusters = min(max_clusters, n_samples - 1)

    if max_clusters < min_clusters:
        return min_clusters

    bics = []
    cluster_range = range(min_clusters, max_clusters + 1)

    for n in cluster_range:
        try:
            gm = GaussianMixture(
                n_components=n,
                random_state=RANDOM_SEED,
                max_iter=100,
                n_init=1
            )
            gm.fit(embeddings)
            bics.append(gm.bic(embeddings))
        except Exception:
            bics.append(float('inf'))

    if not bics or all(b == float('inf') for b in bics):
        return min_clusters

    optimal_idx = np.argmin(bics)
    return list(cluster_range)[optimal_idx]


def cluster_with_gmm(
    embeddings: np.ndarray,
    threshold: float = 0.1,
    reduction_dim: int = 10,
) -> List[List[int]]:
    """
    Cluster embeddings using GMM with soft assignment.

    Args:
        embeddings: Original embeddings
        threshold: Probability threshold for cluster assignment
        reduction_dim: Dimensionality for reduction

    Returns:
        List of clusters, where each cluster is a list of sample indices
    """
    if len(embeddings) <= 2:
        return [list(range(len(embeddings)))]

    # Reduce dimensionality
    reduced = reduce_embeddings(embeddings, n_components=reduction_dim)

    # Find optimal number of clusters
    n_clusters = get_optimal_clusters(reduced)

    # Fit GMM
    try:
        gm = GaussianMixture(
            n_components=n_clusters,
            random_state=RANDOM_SEED,
            max_iter=100,
        )
        gm.fit(reduced)
        probs = gm.predict_proba(reduced)
    except Exception as e:
        logger.warning(f"GMM clustering failed: {e}. Using hard clustering.")
        return cluster_with_kmeans(embeddings, reduction_dim=reduction_dim)

    # Soft assignment: each sample can belong to multiple clusters
    clusters_by_sample = []
    for prob in probs:
        assigned = np.where(prob > threshold)[0].tolist()
        if not assigned:
            # If no cluster meets threshold, assign to highest probability
            assigned = [np.argmax(prob)]
        clusters_by_sample.append(assigned)

    # Convert to list of clusters
    cluster_dict = {}
    for idx, cluster_ids in enumerate(clusters_by_sample):
        for cluster_id in cluster_ids:
            if cluster_id not in cluster_dict:
                cluster_dict[cluster_id] = []
            cluster_dict[cluster_id].append(idx)

    return list(cluster_dict.values())


def cluster_with_kmeans(
    embeddings: np.ndarray,
    n_clusters: Optional[int] = None,
    reduction_dim: int = 10,
) -> List[List[int]]:
    """
    Cluster embeddings using KMeans (fallback method).

    Args:
        embeddings: Original embeddings
        n_clusters: Number of clusters (auto if None)
        reduction_dim: Dimensionality for reduction

    Returns:
        List of clusters, where each cluster is a list of sample indices
    """
    if len(embeddings) <= 2:
        return [list(range(len(embeddings)))]

    # Reduce dimensionality
    reduced = reduce_embeddings(embeddings, n_components=reduction_dim)

    # Auto-determine clusters
    if n_clusters is None:
        n_clusters = min(max(2, len(embeddings) // 3), 10)

    n_clusters = min(n_clusters, len(embeddings) - 1)

    # Fit KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(reduced)

    # Convert to list of clusters
    cluster_dict = {}
    for idx, label in enumerate(labels):
        if label not in cluster_dict:
            cluster_dict[label] = []
        cluster_dict[label].append(idx)

    return list(cluster_dict.values())


def cluster_embeddings(
    embeddings: np.ndarray,
    method: str = "gmm",
    threshold: float = 0.1,
    reduction_dim: int = 10,
) -> List[List[int]]:
    """
    Main clustering function.

    Args:
        embeddings: Array of embeddings (n_samples, n_features)
        method: Clustering method ('gmm' or 'kmeans')
        threshold: Probability threshold for GMM soft clustering
        reduction_dim: Target dimensionality for reduction

    Returns:
        List of clusters, where each cluster is a list of sample indices
    """
    if len(embeddings) < 2:
        return [[0]] if len(embeddings) == 1 else []

    if method == "gmm":
        return cluster_with_gmm(embeddings, threshold, reduction_dim)
    else:
        return cluster_with_kmeans(embeddings, reduction_dim=reduction_dim)
