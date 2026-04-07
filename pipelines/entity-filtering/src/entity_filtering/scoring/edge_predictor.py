"""
Edge predictor — discovers implicit relations between entities.

Uses three complementary signals:
1. **Co-occurrence**: entities that appear in the same source chunk
2. **Embedding similarity**: cosine similarity between entity embeddings
3. **Graph structure**: Adamic-Adar index (shared neighbors in existing relations)

Signals are combined via weighted sum and thresholded.
"""

import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EdgePredictor:
    """Predicts new edges (relations) between entities.

    Combines co-occurrence, embedding similarity, and graph structure
    signals to discover plausible relations not explicitly extracted.

    Args:
        config: EdgePredictionConfig-like dict or object with weights and thresholds.
    """

    def __init__(self, config: Dict[str, Any] | None = None) -> None:
        self._config = config or {}
        self._cosine_weight = float(self._config.get("cosine_weight", 0.5))
        self._adamic_adar_weight = float(self._config.get("adamic_adar_weight", 0.2))
        self._common_ancestors_weight = float(self._config.get("common_ancestors_weight", 0.1))
        self._jaccard_weight = float(self._config.get("jaccard_weight", 0.2))
        self._similarity_threshold = float(self._config.get("similarity_threshold", 0.5))
        self._k_neighbors = int(self._config.get("k_neighbors", 10))

    def predict(
        self,
        entities: List[Dict[str, Any]],
        existing_relations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Predict new relations between entities.

        Args:
            entities: List of entity dicts (text, label, properties, source_chunk_id).
            existing_relations: Already-known relations (source_entity, target_entity).

        Returns:
            List of predicted relation dicts with confidence scores.
        """
        if len(entities) < 2:
            return []

        # Build lookup structures
        entity_texts = [e.get("text", "") for e in entities]
        entity_set = set(entity_texts)

        # Existing edge set for dedup
        existing_pairs: Set[Tuple[str, str]] = set()
        for rel in existing_relations:
            src = rel.get("source_entity", "")
            tgt = rel.get("target_entity", "")
            existing_pairs.add((src, tgt))
            existing_pairs.add((tgt, src))

        # 1. Co-occurrence signal
        cooccurrence = self._compute_cooccurrence(entities)

        # 2. Embedding similarity signal
        embedding_sim = self._compute_embedding_similarity(entities)

        # 3. Graph structure signal (Adamic-Adar)
        adamic_adar = self._compute_adamic_adar(entities, existing_relations)

        # 4. Jaccard neighbor overlap (stap 3)
        jaccard = self._compute_jaccard_overlap(entities, existing_relations)

        # Combine signals for all candidate pairs
        candidates: Dict[Tuple[str, str], float] = defaultdict(float)
        all_pairs: Set[Tuple[str, str]] = set()
        all_pairs.update(cooccurrence.keys())
        all_pairs.update(embedding_sim.keys())
        all_pairs.update(adamic_adar.keys())
        all_pairs.update(jaccard.keys())

        for pair in all_pairs:
            score = (
                self._cosine_weight * embedding_sim.get(pair, 0.0)
                + self._adamic_adar_weight * adamic_adar.get(pair, 0.0)
                + self._common_ancestors_weight * cooccurrence.get(pair, 0.0)
                + self._jaccard_weight * jaccard.get(pair, 0.0)
            )
            candidates[pair] = score

        # Filter and build results
        predicted: List[Dict[str, Any]] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        for (src, tgt), score in sorted(
            candidates.items(), key=lambda x: x[1], reverse=True
        ):
            if score < self._similarity_threshold:
                continue

            # Skip existing relations
            if (src, tgt) in existing_pairs:
                continue

            # Skip duplicates (undirected)
            canonical = (min(src, tgt), max(src, tgt))
            if canonical in seen_pairs:
                continue
            seen_pairs.add(canonical)

            # Skip self-loops
            if src == tgt:
                continue

            # Verify both entities exist
            if src not in entity_set or tgt not in entity_set:
                continue

            predicted.append({
                "source_entity": src,
                "target_entity": tgt,
                "relation_type": "RELATED",
                "confidence": round(min(score, 1.0), 4),
                "properties": {
                    "prediction_method": "edge_predictor",
                    "cosine_signal": round(embedding_sim.get((src, tgt), 0.0), 4),
                    "cooccurrence_signal": round(cooccurrence.get((src, tgt), 0.0), 4),
                    "adamic_adar_signal": round(adamic_adar.get((src, tgt), 0.0), 4),
                    "jaccard_signal": round(jaccard.get((src, tgt), 0.0), 4),
                },
            })

        logger.info(
            f"EdgePredictor: {len(entities)} entities, "
            f"{len(existing_relations)} existing relations → "
            f"{len(predicted)} predicted edges"
        )
        return predicted

    def _compute_cooccurrence(
        self, entities: List[Dict[str, Any]]
    ) -> Dict[Tuple[str, str], float]:
        """Compute co-occurrence scores for entity pairs in the same chunk."""
        chunk_entities: Dict[str, List[str]] = defaultdict(list)

        for e in entities:
            chunk_id = e.get("source_chunk_id") or e.get("properties", {}).get(
                "source_chunk_id"
            )
            if chunk_id:
                chunk_entities[chunk_id].append(e.get("text", ""))

        scores: Dict[Tuple[str, str], float] = defaultdict(float)
        for chunk_id, texts in chunk_entities.items():
            unique_texts = list(set(texts))
            for i in range(len(unique_texts)):
                for j in range(i + 1, len(unique_texts)):
                    pair = (unique_texts[i], unique_texts[j])
                    scores[pair] += 1.0
                    scores[(pair[1], pair[0])] += 1.0

        # Normalize to 0-1 range
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                for pair in scores:
                    scores[pair] /= max_score

        return dict(scores)

    def _compute_embedding_similarity(
        self, entities: List[Dict[str, Any]]
    ) -> Dict[Tuple[str, str], float]:
        """Compute pairwise cosine similarity for entities with embeddings."""
        entities_with_emb: List[Tuple[str, List[float]]] = []

        for e in entities:
            emb = e.get("properties", {}).get("embedding")
            if emb and isinstance(emb, list) and len(emb) > 0:
                entities_with_emb.append((e.get("text", ""), emb))

        if len(entities_with_emb) < 2:
            return {}

        scores: Dict[Tuple[str, str], float] = {}
        # Limit comparisons for performance
        limit = min(len(entities_with_emb), self._k_neighbors * 2)
        for i in range(limit):
            text_i, emb_i = entities_with_emb[i]
            for j in range(i + 1, limit):
                text_j, emb_j = entities_with_emb[j]
                sim = _cosine_similarity(emb_i, emb_j)
                if sim > 0:
                    scores[(text_i, text_j)] = sim
                    scores[(text_j, text_i)] = sim

        return scores

    def _compute_adamic_adar(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> Dict[Tuple[str, str], float]:
        """Compute Adamic-Adar index for entity pairs.

        AA(u,v) = sum(1/log(degree(w))) for common neighbors w of u and v.
        """
        if not relations:
            return {}

        # Build adjacency
        neighbors: Dict[str, Set[str]] = defaultdict(set)
        for rel in relations:
            src = rel.get("source_entity", "")
            tgt = rel.get("target_entity", "")
            if src and tgt:
                neighbors[src].add(tgt)
                neighbors[tgt].add(src)

        # Only consider entities that have at least one neighbor
        connected = [e.get("text", "") for e in entities if e.get("text", "") in neighbors]
        if len(connected) < 2:
            return {}

        scores: Dict[Tuple[str, str], float] = {}
        limit = min(len(connected), self._k_neighbors * 2)

        for i in range(limit):
            for j in range(i + 1, limit):
                u, v = connected[i], connected[j]
                common = neighbors.get(u, set()) & neighbors.get(v, set())
                if not common:
                    continue

                aa_score = 0.0
                for w in common:
                    degree = len(neighbors.get(w, set()))
                    if degree > 1:
                        aa_score += 1.0 / math.log(degree)

                if aa_score > 0:
                    scores[(u, v)] = aa_score
                    scores[(v, u)] = aa_score

        # Normalize
        if scores:
            max_score = max(scores.values())
            if max_score > 0:
                for pair in scores:
                    scores[pair] /= max_score

        return scores

    def _compute_jaccard_overlap(
        self,
        entities: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> Dict[Tuple[str, str], float]:
        """Compute Jaccard similarity of neighbor sets for entity pairs.

        Jaccard(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|

        Lightweight graph-structure feature that doesn't require GNN training.
        """
        if not relations:
            return {}

        neighbors: Dict[str, Set[str]] = defaultdict(set)
        for rel in relations:
            src = rel.get("source_entity", "")
            tgt = rel.get("target_entity", "")
            if src and tgt:
                neighbors[src].add(tgt)
                neighbors[tgt].add(src)

        connected = [e.get("text", "") for e in entities if e.get("text", "") in neighbors]
        if len(connected) < 2:
            return {}

        scores: Dict[Tuple[str, str], float] = {}
        limit = min(len(connected), self._k_neighbors * 2)

        for i in range(limit):
            for j in range(i + 1, limit):
                u, v = connected[i], connected[j]
                n_u = neighbors.get(u, set())
                n_v = neighbors.get(v, set())
                intersection = n_u & n_v
                union = n_u | n_v
                if not union:
                    continue
                jaccard = len(intersection) / len(union)
                if jaccard > 0:
                    scores[(u, v)] = jaccard
                    scores[(v, u)] = jaccard

        return scores
