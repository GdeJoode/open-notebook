"""
Contextual entity clusterer.

Groups entities by co-occurrence in shared source chunks using a
Union-Find (disjoint-set) data structure.  Two entities are placed in
the same cluster when they appear in the same ``source_chunk_id`` at
least *min_cooccurrence* times.

Because each entity dict carries at most one ``source_chunk_id``, the
co-occurrence count between two entities equals the number of
``source_chunk_id`` values they both appear with.  In the simplest
(and most common) case each entity has exactly one chunk, so sharing
that single chunk once satisfies ``min_cooccurrence=1``.

This is an *enrichment* step -- entities are never removed; only a
``cluster_id`` property is added.
"""

from collections import defaultdict
from typing import Any, Dict, List

from loguru import logger

from entity_filtering.deduplication.union_find import UnionFind


class ContextualClusterer:
    """Cluster entities by shared context (co-occurrence in same chunks).

    Args:
        min_cooccurrence: Minimum number of shared ``source_chunk_id``
            values for two entities to be considered co-occurring.
            Defaults to ``2``.
    """

    def __init__(self, min_cooccurrence: int = 2) -> None:
        self._min_cooccurrence = min_cooccurrence

    def cluster(
        self, entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Add ``cluster_id`` to entities based on co-occurrence patterns.

        Builds an adjacency graph where entities sharing
        ``source_chunk_id`` values (>= *min_cooccurrence* times) are
        connected.  Connected components are found via Union-Find and
        each component receives a unique integer ``cluster_id``.

        Entities without a ``source_chunk_id`` receive
        ``cluster_id = -1``.

        Args:
            entities: Entity dicts, each optionally carrying a
                ``source_chunk_id`` key.

        Returns:
            The same entity list with ``cluster_id`` added to each
            entity's ``properties``.
        """
        if not entities:
            return entities

        # Map: source_chunk_id -> list of entity indices
        chunk_to_indices: Dict[str, List[int]] = defaultdict(list)
        entity_keys: List[str] = []

        for idx, entity in enumerate(entities):
            chunk_id = entity.get("source_chunk_id")
            key = f"entity_{idx}"
            entity_keys.append(key)
            if chunk_id is not None and chunk_id != "":
                chunk_to_indices[str(chunk_id)].append(idx)

        # Build co-occurrence counts between entity pairs
        pair_counts: Dict[tuple[int, int], int] = defaultdict(int)
        for _chunk_id, indices in chunk_to_indices.items():
            for pos_a in range(len(indices)):
                for pos_b in range(pos_a + 1, len(indices)):
                    i, j = indices[pos_a], indices[pos_b]
                    pair_key = (min(i, j), max(i, j))
                    pair_counts[pair_key] += 1

        # Union entities that meet the co-occurrence threshold
        uf = UnionFind()
        edges_added = 0
        for (i, j), count in pair_counts.items():
            if count >= self._min_cooccurrence:
                uf.union(entity_keys[i], entity_keys[j])
                edges_added += 1

        # Assign cluster IDs via connected components
        # First, get the root for every entity that has a source_chunk_id
        root_to_cluster: Dict[str, int] = {}
        next_cluster_id = 0

        for idx, entity in enumerate(entities):
            chunk_id = entity.get("source_chunk_id")
            props = entity.setdefault("properties", {})

            if chunk_id is None or chunk_id == "":
                props["cluster_id"] = -1
                continue

            root = uf.find(entity_keys[idx])
            if root not in root_to_cluster:
                root_to_cluster[root] = next_cluster_id
                next_cluster_id += 1

            props["cluster_id"] = root_to_cluster[root]

        cluster_count = len(root_to_cluster)
        unclustered = sum(
            1
            for e in entities
            if e.get("properties", {}).get("cluster_id") == -1
        )

        logger.debug(
            "ContextualClusterer: {} clusters from {} entities "
            "({} unclustered, {} co-occurrence edges)",
            cluster_count,
            len(entities),
            unclustered,
            edges_added,
        )

        return entities
