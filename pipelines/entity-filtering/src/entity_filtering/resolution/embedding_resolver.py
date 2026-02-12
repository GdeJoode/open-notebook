"""
Cosine-similarity-based semantic resolution.

Enriches entities by finding the most similar other entity within the
same batch using embedding vectors.  This is an *enrichment* step -- it
never removes entities; it only adds ``semantic_match_text`` and
``semantic_similarity`` to the properties of entities that have a close
neighbour above the configured threshold.

Embeddings are read from ``entity["properties"]["embedding"]``.
"""

from typing import Any, Dict, List

from loguru import logger


class EmbeddingResolver:
    """Find semantic matches within an entity batch using cosine similarity.

    For each entity that carries an embedding, the resolver scans the rest
    of the batch for the single most-similar neighbour.  When the similarity
    meets or exceeds *similarity_threshold*, two properties are added:

    * ``semantic_match_text`` -- the ``text`` of the matched entity.
    * ``semantic_similarity`` -- the cosine similarity score.

    The entity list is returned with the same length; only ``properties``
    dicts are mutated.

    Args:
        similarity_threshold: Minimum cosine similarity (0-1) to record
            a semantic match.
    """

    def __init__(self, similarity_threshold: float = 0.90) -> None:
        self._threshold = similarity_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def resolve(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Enrich entities with semantic-match metadata.

        Args:
            entities: List of entity dicts.  Entities with a non-empty
                ``properties.embedding`` list participate in matching.

        Returns:
            The same entity list (same length, same order) with added
            ``semantic_match_text`` and ``semantic_similarity`` properties
            on entities that have a close neighbour.
        """
        if len(entities) <= 1:
            return entities

        # Collect indices and embeddings for entities that have them
        indexed: List[tuple[int, List[float]]] = []
        for idx, entity in enumerate(entities):
            emb = entity.get("properties", {}).get("embedding")
            if emb is not None and len(emb) > 0:
                indexed.append((idx, emb))

        if len(indexed) < 2:
            logger.debug(
                "EmbeddingResolver: fewer than 2 entities with embeddings, "
                "skipping"
            )
            return entities

        matches_found = 0

        for pos_a, (idx_a, emb_a) in enumerate(indexed):
            best_sim = -1.0
            best_idx = -1

            for pos_b, (idx_b, emb_b) in enumerate(indexed):
                if pos_a == pos_b:
                    continue
                sim = self._cosine_similarity(emb_a, emb_b)
                if sim > best_sim:
                    best_sim = sim
                    best_idx = idx_b

            if best_sim >= self._threshold and best_idx >= 0:
                props = entities[idx_a].setdefault("properties", {})
                props["semantic_match_text"] = entities[best_idx].get(
                    "text", ""
                )
                props["semantic_similarity"] = round(best_sim, 6)
                matches_found += 1

        if matches_found:
            logger.debug(
                "EmbeddingResolver enriched {} of {} entities "
                "(threshold={:.2f})",
                matches_found,
                len(entities),
                self._threshold,
            )

        return entities

    # ------------------------------------------------------------------
    # Similarity computation
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(
        vec1: List[float], vec2: List[float]
    ) -> float:
        """Compute cosine similarity between two vectors (pure Python).

        Returns 0.0 for empty, mismatched, or zero-norm vectors.
        """
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        return dot_product / (norm1 * norm2)
