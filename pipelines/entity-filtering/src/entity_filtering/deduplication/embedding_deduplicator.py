"""
Embedding-based entity deduplicator.

Uses FAISS k-nearest-neighbor search over entity embeddings to find
semantically similar entities, then transitively merges them via
UnionFind.  Falls back to a no-op when FAISS or numpy are unavailable.

Embeddings are read from ``entity["properties"]["embedding"]``.
"""

from collections import Counter
from typing import Any, Dict, List, Set, Tuple

from loguru import logger

from entity_filtering.deduplication.union_find import UnionFind

# Optional heavy deps --------------------------------------------------
try:
    import numpy as np

    _NUMPY_AVAILABLE = True
except ImportError:
    np = None  # type: ignore[assignment]
    _NUMPY_AVAILABLE = False

try:
    import faiss

    _FAISS_AVAILABLE = True
except ImportError:
    faiss = None  # type: ignore[assignment]
    _FAISS_AVAILABLE = False


class EmbeddingDeduplicator:
    """Merge entities whose embeddings are within a cosine-similarity threshold.

    Args:
        similarity_threshold: Minimum cosine similarity (0-1) to consider
            two entities as duplicates.
        k_candidates: Number of nearest neighbours to retrieve per entity.
        use_faiss: When ``True`` (default), use FAISS for approximate
            nearest-neighbour search.  Falls back to brute-force numpy
            when FAISS is unavailable.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.90,
        k_candidates: int = 5,
        use_faiss: bool = True,
    ) -> None:
        self._threshold = similarity_threshold
        self._k = k_candidates
        self._use_faiss = use_faiss and _FAISS_AVAILABLE

        if use_faiss and not _FAISS_AVAILABLE:
            logger.warning(
                "FAISS not available, embedding dedup will use numpy fallback"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def deduplicate(
        self, entities: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[List[str]]]:
        """Merge entities with similar embeddings.

        Args:
            entities: List of entity dicts.  Entities **must** have
                ``properties.embedding`` (a ``List[float]``) to
                participate in embedding-based dedup; those without
                are passed through unchanged.

        Returns:
            Tuple of (merged entities, merge groups).
        """
        if not _NUMPY_AVAILABLE:
            logger.warning(
                "numpy not available, skipping embedding deduplication"
            )
            return list(entities), []

        if len(entities) <= 1:
            return list(entities), []

        # Split entities with/without embeddings
        with_emb: List[Tuple[int, Dict[str, Any]]] = []
        without_emb: List[Dict[str, Any]] = []
        for idx, entity in enumerate(entities):
            emb = entity.get("properties", {}).get("embedding")
            if emb is not None and len(emb) > 0:
                with_emb.append((idx, entity))
            else:
                without_emb.append(entity)

        if len(with_emb) < 2:
            return list(entities), []

        # Find candidate pairs via k-NN
        candidates = self._find_candidates(with_emb)

        if not candidates:
            return list(entities), []

        # Build Union-Find groups
        uf = UnionFind()
        texts = [e.get("text", "") for _, e in with_emb]
        for i, j, _sim in candidates:
            uf.union(texts[i], texts[j])

        groups = uf.get_groups(texts)

        # Merge groups
        result: List[Dict[str, Any]] = list(without_emb)
        merge_groups: List[List[str]] = []

        # Map text -> entity (with embedding index)
        text_to_entities: Dict[str, List[Dict[str, Any]]] = {}
        for _, entity in with_emb:
            t = entity.get("text", "")
            text_to_entities.setdefault(t, []).append(entity)

        for _root, group_texts in groups.items():
            group_entities: List[Dict[str, Any]] = []
            for t in group_texts:
                group_entities.extend(text_to_entities.get(t, []))

            if not group_entities:
                continue

            canonical = self._select_canonical(group_entities)
            result.append(canonical)

            unique_forms = sorted(set(group_texts))
            if len(unique_forms) > 1:
                merge_groups.append(unique_forms)

        if merge_groups:
            logger.debug(
                "EmbeddingDeduplicator merged {} groups ({} -> {} entities)",
                len(merge_groups),
                len(entities),
                len(result),
            )

        return result, merge_groups

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------

    def _find_candidates(
        self, with_emb: List[Tuple[int, Dict[str, Any]]]
    ) -> List[Tuple[int, int, float]]:
        """Find candidate pairs above the similarity threshold.

        Returns list of (local_idx_a, local_idx_b, similarity).
        """
        embeddings = np.array(
            [e.get("properties", {}).get("embedding") for _, e in with_emb],
            dtype="float32",
        )

        # L2-normalize for cosine similarity via inner product
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        embeddings = embeddings / norms

        if self._use_faiss:
            return self._faiss_search(embeddings)
        return self._brute_force_search(embeddings)

    def _faiss_search(
        self, embeddings: "np.ndarray"
    ) -> List[Tuple[int, int, float]]:
        """FAISS inner-product k-NN search."""
        n, dim = embeddings.shape
        k = min(self._k + 1, n)  # +1 because self is returned

        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        distances, indices = index.search(embeddings, k)

        candidates: List[Tuple[int, int, float]] = []
        seen: Set[Tuple[int, int]] = set()

        for i in range(n):
            for j_pos in range(k):
                j = int(indices[i][j_pos])
                if j <= i:
                    continue
                pair = (i, j)
                if pair in seen:
                    continue
                seen.add(pair)
                sim = float(distances[i][j_pos])
                if sim >= self._threshold:
                    candidates.append((i, j, sim))

        return candidates

    def _brute_force_search(
        self, embeddings: "np.ndarray"
    ) -> List[Tuple[int, int, float]]:
        """Brute-force cosine similarity search (numpy only)."""
        sim_matrix = embeddings @ embeddings.T
        n = sim_matrix.shape[0]

        candidates: List[Tuple[int, int, float]] = []
        for i in range(n):
            for j in range(i + 1, n):
                sim = float(sim_matrix[i, j])
                if sim >= self._threshold:
                    candidates.append((i, j, sim))

        return candidates

    # ------------------------------------------------------------------
    # Canonical selection
    # ------------------------------------------------------------------

    @staticmethod
    def _select_canonical(group: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Pick the best representative from a merge group.

        Strategy:
        - Most frequent surface form wins.
        - Highest confidence is kept.
        - Most common label is kept.
        - Properties are merged (embedding from canonical is kept).
        """
        text_counts: Counter[str] = Counter(
            e.get("text", "") for e in group
        )
        canonical_text = text_counts.most_common(1)[0][0]
        base = next(
            e for e in group if e.get("text") == canonical_text
        ).copy()

        merged_props: Dict[str, Any] = {}
        max_confidence = 0.0
        label_counts: Counter[str] = Counter()

        for entity in group:
            props = entity.get("properties", {})
            # Merge all properties except embedding (keep canonical's)
            for k, v in props.items():
                if k != "embedding":
                    merged_props[k] = v
            conf = entity.get("confidence", 0.0)
            if conf > max_confidence:
                max_confidence = conf
            label_counts[entity.get("label", "MISC")] += 1

        # Preserve canonical's embedding
        canonical_emb = base.get("properties", {}).get("embedding")
        merged_props["embedding"] = canonical_emb

        base["properties"] = merged_props
        base["confidence"] = max_confidence
        base["label"] = label_counts.most_common(1)[0][0]
        return base
