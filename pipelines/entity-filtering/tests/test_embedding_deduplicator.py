"""Unit tests for EmbeddingDeduplicator.

Tests cover basic I/O, merging behaviour, canonical selection,
merge-group structure, and the numpy brute-force fallback path.

All embeddings use dim=4 to keep test data readable.
"""

import numpy as np
import pytest

from entity_filtering.deduplication.embedding_deduplicator import (
    EmbeddingDeduplicator,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entity(text, label="MISC", confidence=0.9, properties=None):
    return {
        "text": text,
        "label": label,
        "properties": properties or {},
        "confidence": confidence,
        "source_chunk_id": None,
    }


def _entity_with_embedding(
    text, embedding, label="MISC", confidence=0.9, extra_props=None
):
    props = {"embedding": embedding}
    if extra_props:
        props.update(extra_props)
    return _entity(text, label, confidence, props)


# Pre-computed reference vectors (dim=4)
# Cosine similarity between VEC_A and VEC_A_SIMILAR is ~0.9999
VEC_A = [1.0, 0.0, 0.0, 0.0]
VEC_A_SIMILAR = [0.99, 0.01, 0.0, 0.0]

# Cosine similarity between VEC_A and VEC_B_MODERATE is cos(45 deg) ~ 0.707
VEC_B_MODERATE = [0.7, 0.7, 0.0, 0.0]

# Cosine similarity between VEC_A and VEC_C_DIFFERENT is 0.0
VEC_C_DIFFERENT = [0.0, 0.0, 0.0, 1.0]

# A bridging vector that is similar to both VEC_A and a slightly rotated vector
VEC_BRIDGE = [0.95, 0.05, 0.0, 0.0]
VEC_BRIDGE_DEST = [0.90, 0.10, 0.0, 0.0]


class TestEmbeddingDedupBasicIO:
    """Passthrough and edge-case behaviour for trivial inputs."""

    def test_empty_input_returns_empty(self):
        dedup = EmbeddingDeduplicator(use_faiss=False)
        result, groups = dedup.deduplicate([])
        assert result == []
        assert groups == []

    def test_single_entity_passes_through(self):
        entity = _entity_with_embedding("Alpha", VEC_A)
        dedup = EmbeddingDeduplicator(use_faiss=False)
        result, groups = dedup.deduplicate([entity])
        assert len(result) == 1
        assert result[0]["text"] == "Alpha"
        assert groups == []

    def test_no_embeddings_present_passthrough(self):
        """Entities without embeddings are returned unchanged."""
        entities = [
            _entity("Alpha"),
            _entity("Beta"),
            _entity("Gamma"),
        ]
        dedup = EmbeddingDeduplicator(use_faiss=False)
        result, groups = dedup.deduplicate(entities)
        assert len(result) == 3
        assert groups == []
        result_texts = {e["text"] for e in result}
        assert result_texts == {"Alpha", "Beta", "Gamma"}

    def test_single_entity_with_embedding_passthrough(self):
        """Only one entity has an embedding -- not enough for comparison."""
        entities = [
            _entity_with_embedding("Alpha", VEC_A),
            _entity("Beta"),
        ]
        dedup = EmbeddingDeduplicator(use_faiss=False)
        result, groups = dedup.deduplicate(entities)
        assert len(result) == 2
        assert groups == []


class TestEmbeddingDedupMerging:
    """Core merging logic: similar embeddings merge, different do not."""

    def test_similar_embeddings_merge(self):
        """Entities with cosine similarity >= threshold merge."""
        entities = [
            _entity_with_embedding("Entity A", VEC_A, confidence=0.8),
            _entity_with_embedding("Entity B", VEC_A_SIMILAR, confidence=0.9),
        ]
        dedup = EmbeddingDeduplicator(
            similarity_threshold=0.90, use_faiss=False
        )
        result, groups = dedup.deduplicate(entities)
        assert len(result) == 1
        assert len(groups) == 1
        assert sorted(groups[0]) == ["Entity A", "Entity B"]

    def test_different_embeddings_do_not_merge(self):
        """Entities with low cosine similarity remain separate."""
        entities = [
            _entity_with_embedding("Alpha", VEC_A),
            _entity_with_embedding("Omega", VEC_C_DIFFERENT),
        ]
        dedup = EmbeddingDeduplicator(
            similarity_threshold=0.90, use_faiss=False
        )
        result, groups = dedup.deduplicate(entities)
        assert len(result) == 2
        assert groups == []

    def test_moderate_similarity_below_threshold(self):
        """Cosine ~0.707 should not merge at threshold=0.90."""
        entities = [
            _entity_with_embedding("Alpha", VEC_A),
            _entity_with_embedding("Beta", VEC_B_MODERATE),
        ]
        dedup = EmbeddingDeduplicator(
            similarity_threshold=0.90, use_faiss=False
        )
        result, groups = dedup.deduplicate(entities)
        assert len(result) == 2
        assert groups == []

    def test_moderate_similarity_above_lower_threshold(self):
        """Cosine ~0.707 should merge at threshold=0.70."""
        entities = [
            _entity_with_embedding("Alpha", VEC_A),
            _entity_with_embedding("Beta", VEC_B_MODERATE),
        ]
        dedup = EmbeddingDeduplicator(
            similarity_threshold=0.70, use_faiss=False
        )
        result, groups = dedup.deduplicate(entities)
        assert len(result) == 1
        assert len(groups) == 1

    def test_mixed_entities_with_and_without_embeddings(self):
        """Only entities that have embeddings participate in dedup."""
        entities = [
            _entity("No Embedding 1"),
            _entity_with_embedding("With Emb A", VEC_A),
            _entity("No Embedding 2"),
            _entity_with_embedding("With Emb B", VEC_A_SIMILAR),
        ]
        dedup = EmbeddingDeduplicator(
            similarity_threshold=0.90, use_faiss=False
        )
        result, groups = dedup.deduplicate(entities)
        # Two no-embedding pass through, two with-embedding merge to one
        assert len(result) == 3
        result_texts = {e["text"] for e in result}
        assert "No Embedding 1" in result_texts
        assert "No Embedding 2" in result_texts
        # One of the two with-embedding entities survives as canonical
        emb_survivors = result_texts - {"No Embedding 1", "No Embedding 2"}
        assert len(emb_survivors) == 1

    def test_transitive_merge(self):
        """A~B and B~C should transitively merge all three into one group."""
        # VEC_A ~ VEC_BRIDGE ~ VEC_BRIDGE_DEST (chain of similarity)
        entities = [
            _entity_with_embedding("A", VEC_A),
            _entity_with_embedding("B", VEC_BRIDGE),
            _entity_with_embedding("C", VEC_BRIDGE_DEST),
        ]
        dedup = EmbeddingDeduplicator(
            similarity_threshold=0.90, use_faiss=False
        )
        result, groups = dedup.deduplicate(entities)
        # All three should merge into one entity via transitive chain
        assert len(result) == 1
        assert len(groups) == 1
        assert sorted(groups[0]) == ["A", "B", "C"]

    def test_separate_clusters_remain_separate(self):
        """Two distinct clusters should not merge with each other."""
        cluster1_a = [1.0, 0.0, 0.0, 0.0]
        cluster1_b = [0.99, 0.01, 0.0, 0.0]
        cluster2_a = [0.0, 0.0, 1.0, 0.0]
        cluster2_b = [0.0, 0.0, 0.99, 0.01]

        entities = [
            _entity_with_embedding("C1A", cluster1_a),
            _entity_with_embedding("C1B", cluster1_b),
            _entity_with_embedding("C2A", cluster2_a),
            _entity_with_embedding("C2B", cluster2_b),
        ]
        dedup = EmbeddingDeduplicator(
            similarity_threshold=0.90, use_faiss=False
        )
        result, groups = dedup.deduplicate(entities)
        assert len(result) == 2
        assert len(groups) == 2


class TestEmbeddingDedupCanonical:
    """Canonical selection: most frequent text, max confidence, merged props."""

    def test_most_frequent_text_wins(self):
        """The surface form appearing most often becomes canonical."""
        entities = [
            _entity_with_embedding("Popular", VEC_A, confidence=0.7),
            _entity_with_embedding("Popular", VEC_A_SIMILAR, confidence=0.8),
            _entity_with_embedding("Rare", VEC_BRIDGE, confidence=0.9),
        ]
        dedup = EmbeddingDeduplicator(
            similarity_threshold=0.90, use_faiss=False
        )
        result, groups = dedup.deduplicate(entities)
        assert len(result) == 1
        assert result[0]["text"] == "Popular"

    def test_max_confidence_preserved(self):
        """The highest confidence across the merge group is kept."""
        entities = [
            _entity_with_embedding("Low", VEC_A, confidence=0.3),
            _entity_with_embedding("High", VEC_A_SIMILAR, confidence=0.99),
        ]
        dedup = EmbeddingDeduplicator(
            similarity_threshold=0.90, use_faiss=False
        )
        result, _ = dedup.deduplicate(entities)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.99

    def test_non_embedding_properties_merged(self):
        """Extra properties from all group members are merged."""
        entities = [
            _entity_with_embedding(
                "A", VEC_A, extra_props={"country": "NL"}
            ),
            _entity_with_embedding(
                "B", VEC_A_SIMILAR, extra_props={"population": 17_000_000}
            ),
        ]
        dedup = EmbeddingDeduplicator(
            similarity_threshold=0.90, use_faiss=False
        )
        result, _ = dedup.deduplicate(entities)
        assert len(result) == 1
        props = result[0]["properties"]
        assert props.get("country") == "NL"
        assert props.get("population") == 17_000_000

    def test_embedding_from_canonical_preserved(self):
        """The embedding on the canonical entity is kept in the result."""
        entities = [
            _entity_with_embedding("Winner", VEC_A, confidence=0.9),
            _entity_with_embedding("Winner", VEC_A_SIMILAR, confidence=0.8),
            _entity_with_embedding("Loser", VEC_BRIDGE, confidence=0.7),
        ]
        dedup = EmbeddingDeduplicator(
            similarity_threshold=0.90, use_faiss=False
        )
        result, _ = dedup.deduplicate(entities)
        assert len(result) == 1
        emb = result[0]["properties"]["embedding"]
        assert emb == VEC_A  # from the canonical "Winner" entity

    def test_most_common_label_preserved(self):
        """The most frequent label in the group is selected."""
        entities = [
            _entity_with_embedding("X", VEC_A, label="ORG", confidence=0.8),
            _entity_with_embedding(
                "Y", VEC_A_SIMILAR, label="ORG", confidence=0.8
            ),
            _entity_with_embedding(
                "Z", VEC_BRIDGE, label="PERSON", confidence=0.8
            ),
        ]
        dedup = EmbeddingDeduplicator(
            similarity_threshold=0.90, use_faiss=False
        )
        result, _ = dedup.deduplicate(entities)
        assert len(result) == 1
        assert result[0]["label"] == "ORG"


class TestEmbeddingDedupMergeGroups:
    """Merge group structure: only groups with >1 unique text form appear."""

    def test_multi_text_group_recorded(self):
        """Groups with multiple distinct surface forms are in merge_groups."""
        entities = [
            _entity_with_embedding("Form A", VEC_A),
            _entity_with_embedding("Form B", VEC_A_SIMILAR),
        ]
        dedup = EmbeddingDeduplicator(
            similarity_threshold=0.90, use_faiss=False
        )
        _, groups = dedup.deduplicate(entities)
        assert len(groups) == 1
        assert sorted(groups[0]) == ["Form A", "Form B"]

    def test_single_text_group_not_in_merge_groups(self):
        """When all entities in a group share the same text, no merge group."""
        entities = [
            _entity_with_embedding("Same", VEC_A),
            _entity_with_embedding("Same", VEC_A_SIMILAR),
        ]
        dedup = EmbeddingDeduplicator(
            similarity_threshold=0.90, use_faiss=False
        )
        result, groups = dedup.deduplicate(entities)
        assert len(result) == 1
        assert groups == []

    def test_no_merge_groups_when_nothing_merges(self):
        """Completely dissimilar entities produce no merge groups."""
        entities = [
            _entity_with_embedding("Alpha", VEC_A),
            _entity_with_embedding("Omega", VEC_C_DIFFERENT),
        ]
        dedup = EmbeddingDeduplicator(
            similarity_threshold=0.90, use_faiss=False
        )
        _, groups = dedup.deduplicate(entities)
        assert groups == []


class TestEmbeddingDedupFallback:
    """Numpy brute-force path and passthrough for entities without embeddings."""

    def test_use_faiss_false_forces_numpy_brute_force(self):
        """Explicit use_faiss=False produces the same results as default."""
        entities = [
            _entity_with_embedding("A", VEC_A),
            _entity_with_embedding("B", VEC_A_SIMILAR),
            _entity_with_embedding("C", VEC_C_DIFFERENT),
        ]
        dedup_np = EmbeddingDeduplicator(
            similarity_threshold=0.90, use_faiss=False
        )
        result_np, groups_np = dedup_np.deduplicate(entities)
        # A and B should merge, C stays separate
        assert len(result_np) == 2
        assert len(groups_np) == 1
        assert sorted(groups_np[0]) == ["A", "B"]

    def test_brute_force_matches_faiss_when_available(self):
        """If FAISS is available, both paths should produce identical groups."""
        faiss = pytest.importorskip("faiss")

        entities = [
            _entity_with_embedding("X", VEC_A),
            _entity_with_embedding("Y", VEC_A_SIMILAR),
            _entity_with_embedding("Z", VEC_C_DIFFERENT),
        ]

        dedup_faiss = EmbeddingDeduplicator(
            similarity_threshold=0.90, use_faiss=True
        )
        dedup_np = EmbeddingDeduplicator(
            similarity_threshold=0.90, use_faiss=False
        )

        result_faiss, groups_faiss = dedup_faiss.deduplicate(entities)
        result_np, groups_np = dedup_np.deduplicate(entities)

        assert len(result_faiss) == len(result_np)
        # Both should produce the same merge group content
        sorted_faiss = [sorted(g) for g in groups_faiss]
        sorted_np = [sorted(g) for g in groups_np]
        assert sorted(sorted_faiss) == sorted(sorted_np)

    def test_entities_without_embeddings_pass_through_unchanged(self):
        """Entities lacking embeddings are in the output with original data."""
        original = _entity("NoEmb", label="ORG", confidence=0.75)
        original["properties"]["wiki_id"] = "Q12345"

        entities = [
            original,
            _entity_with_embedding("A", VEC_A),
            _entity_with_embedding("B", VEC_A_SIMILAR),
        ]
        dedup = EmbeddingDeduplicator(
            similarity_threshold=0.90, use_faiss=False
        )
        result, _ = dedup.deduplicate(entities)

        no_emb = [e for e in result if e["text"] == "NoEmb"]
        assert len(no_emb) == 1
        assert no_emb[0]["label"] == "ORG"
        assert no_emb[0]["confidence"] == 0.75
        assert no_emb[0]["properties"]["wiki_id"] == "Q12345"

    def test_empty_embedding_list_treated_as_no_embedding(self):
        """An entity with an empty embedding list does not participate."""
        entities = [
            _entity("Empty Emb", properties={"embedding": []}),
            _entity_with_embedding("A", VEC_A),
            _entity_with_embedding("B", VEC_A_SIMILAR),
        ]
        dedup = EmbeddingDeduplicator(
            similarity_threshold=0.90, use_faiss=False
        )
        result, groups = dedup.deduplicate(entities)
        # A and B merge; "Empty Emb" passes through
        assert len(result) == 2
        result_texts = {e["text"] for e in result}
        assert "Empty Emb" in result_texts
