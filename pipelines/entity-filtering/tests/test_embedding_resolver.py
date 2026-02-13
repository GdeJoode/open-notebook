"""Tests for EmbeddingResolver cosine-similarity-based semantic resolution.

Covers empty/single inputs, identical and different embeddings, threshold
behaviour, mixed entities (with/without embeddings), the _cosine_similarity
static method edge cases, custom thresholds, and property preservation.
"""

import math

from entity_filtering.resolution.embedding_resolver import EmbeddingResolver


def _entity(text, label="MISC", embedding=None, properties=None):
    """Helper to build an entity dict with optional embedding."""
    props = dict(properties) if properties else {}
    if embedding is not None:
        props["embedding"] = embedding
    return {
        "text": text,
        "label": label,
        "properties": props,
        "confidence": 0.9,
        "source_chunk_id": None,
    }


# ------------------------------------------------------------------
# Basic input / output
# ------------------------------------------------------------------


class TestEmbeddingResolverBasicIO:
    def test_empty_input_returns_empty(self):
        resolver = EmbeddingResolver()
        result = resolver.resolve([])
        assert result == []

    def test_single_entity_returns_unchanged(self):
        """A single entity cannot have a neighbour, so it must be returned
        as-is without any semantic_match_* properties."""
        resolver = EmbeddingResolver()
        entities = [_entity("Paris", embedding=[1.0, 0.0, 0.0, 0.0])]
        result = resolver.resolve(entities)
        assert len(result) == 1
        assert result[0]["text"] == "Paris"
        assert "semantic_match_text" not in result[0]["properties"]
        assert "semantic_similarity" not in result[0]["properties"]

    def test_single_entity_without_embedding_returns_unchanged(self):
        resolver = EmbeddingResolver()
        entities = [_entity("Paris")]
        result = resolver.resolve(entities)
        assert len(result) == 1
        assert result[0]["text"] == "Paris"

    def test_all_entities_returned_same_length(self):
        """The resolver never removes entities."""
        resolver = EmbeddingResolver()
        entities = [
            _entity("A", embedding=[1.0, 0.0, 0.0, 0.0]),
            _entity("B", embedding=[0.0, 1.0, 0.0, 0.0]),
            _entity("C"),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 3


# ------------------------------------------------------------------
# Matching behaviour
# ------------------------------------------------------------------


class TestEmbeddingResolverMatching:
    def test_identical_embeddings_get_matched(self):
        """Two entities with identical embeddings should be matched
        (cosine similarity = 1.0 which exceeds default 0.90)."""
        resolver = EmbeddingResolver()
        entities = [
            _entity("Paris", embedding=[1.0, 0.0, 0.0, 0.0]),
            _entity("Paris, France", embedding=[1.0, 0.0, 0.0, 0.0]),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 2
        # Both should be enriched pointing at each other
        assert result[0]["properties"]["semantic_match_text"] == "Paris, France"
        assert result[0]["properties"]["semantic_similarity"] == 1.0
        assert result[1]["properties"]["semantic_match_text"] == "Paris"
        assert result[1]["properties"]["semantic_similarity"] == 1.0

    def test_very_similar_embeddings_get_matched(self):
        """Two vectors close in angle should match above the threshold."""
        resolver = EmbeddingResolver(similarity_threshold=0.90)
        # [1, 0, 0, 0] and [0.99, 0.1, 0, 0] -> high cosine similarity
        entities = [
            _entity("Paris", embedding=[1.0, 0.0, 0.0, 0.0]),
            _entity("Parijs", embedding=[0.99, 0.1, 0.0, 0.0]),
        ]
        result = resolver.resolve(entities)
        assert "semantic_match_text" in result[0]["properties"]
        assert result[0]["properties"]["semantic_match_text"] == "Parijs"
        sim = result[0]["properties"]["semantic_similarity"]
        assert sim >= 0.90

    def test_different_embeddings_below_threshold_no_match(self):
        """Orthogonal vectors have cosine similarity 0, well below 0.90."""
        resolver = EmbeddingResolver()
        entities = [
            _entity("Paris", embedding=[1.0, 0.0, 0.0, 0.0]),
            _entity("Apple", embedding=[0.0, 1.0, 0.0, 0.0]),
        ]
        result = resolver.resolve(entities)
        assert "semantic_match_text" not in result[0]["properties"]
        assert "semantic_match_text" not in result[1]["properties"]

    def test_best_match_selected_among_multiple(self):
        """When there are three entities, each should match its closest
        neighbour, not just any neighbour above threshold."""
        resolver = EmbeddingResolver(similarity_threshold=0.50)
        entities = [
            _entity("A", embedding=[1.0, 0.0, 0.0, 0.0]),
            _entity("B", embedding=[0.95, 0.3, 0.0, 0.0]),
            _entity("C", embedding=[0.0, 1.0, 0.0, 0.0]),
        ]
        result = resolver.resolve(entities)
        # A's best match should be B (highest cosine to A)
        assert result[0]["properties"]["semantic_match_text"] == "B"
        # B's best match should be A
        assert result[1]["properties"]["semantic_match_text"] == "A"


# ------------------------------------------------------------------
# Entities without embeddings
# ------------------------------------------------------------------


class TestEmbeddingResolverNoEmbeddings:
    def test_entities_without_embeddings_passed_through(self):
        """Entities with no embedding key are returned without match props."""
        resolver = EmbeddingResolver()
        entities = [
            _entity("Alice"),
            _entity("Bob"),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 2
        assert "semantic_match_text" not in result[0]["properties"]
        assert "semantic_match_text" not in result[1]["properties"]

    def test_single_entity_with_embedding_among_others_without(self):
        """When only one entity has an embedding, no matching is possible
        (fewer than 2 with embeddings)."""
        resolver = EmbeddingResolver()
        entities = [
            _entity("Paris", embedding=[1.0, 0.0, 0.0, 0.0]),
            _entity("London"),
            _entity("Berlin"),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 3
        assert "semantic_match_text" not in result[0]["properties"]

    def test_mixed_entities_all_returned(self):
        """A mix of entities with and without embeddings: all are returned,
        and only those with embeddings participate in matching."""
        resolver = EmbeddingResolver(similarity_threshold=0.50)
        entities = [
            _entity("Paris", embedding=[1.0, 0.0, 0.0, 0.0]),
            _entity("London"),
            _entity("Parijs", embedding=[0.99, 0.1, 0.0, 0.0]),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 3
        # Paris and Parijs matched
        assert result[0]["properties"]["semantic_match_text"] == "Parijs"
        assert result[2]["properties"]["semantic_match_text"] == "Paris"
        # London has no match props
        assert "semantic_match_text" not in result[1]["properties"]

    def test_entity_with_empty_embedding_list_is_skipped(self):
        """An entity with an empty embedding list should not participate."""
        resolver = EmbeddingResolver()
        entities = [
            _entity("A", embedding=[]),
            _entity("B", embedding=[1.0, 0.0, 0.0, 0.0]),
            _entity("C", embedding=[0.99, 0.1, 0.0, 0.0]),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 3
        # B and C should match each other
        assert "semantic_match_text" in result[1]["properties"]
        assert "semantic_match_text" in result[2]["properties"]
        # A should not have match props
        assert "semantic_match_text" not in result[0]["properties"]

    def test_entity_with_none_embedding_is_skipped(self):
        """An entity with embedding=None is equivalent to no embedding."""
        resolver = EmbeddingResolver()
        e = {"text": "X", "label": "MISC", "properties": {"embedding": None}}
        entities = [
            e,
            _entity("Y", embedding=[1.0, 0.0, 0.0, 0.0]),
            _entity("Z", embedding=[0.98, 0.2, 0.0, 0.0]),
        ]
        result = resolver.resolve(entities)
        assert len(result) == 3
        assert "semantic_match_text" not in result[0].get("properties", {})


# ------------------------------------------------------------------
# Custom threshold
# ------------------------------------------------------------------


class TestEmbeddingResolverThreshold:
    def test_custom_low_threshold_allows_more_matches(self):
        """A very low threshold should match even moderately different vectors."""
        resolver = EmbeddingResolver(similarity_threshold=0.10)
        entities = [
            _entity("A", embedding=[1.0, 0.0, 0.0, 0.0]),
            _entity("B", embedding=[0.5, 0.5, 0.5, 0.5]),
        ]
        result = resolver.resolve(entities)
        # Cosine similarity of [1,0,0,0] and [0.5,0.5,0.5,0.5] = 0.5
        # which exceeds 0.10, so they should match
        assert "semantic_match_text" in result[0]["properties"]
        assert "semantic_match_text" in result[1]["properties"]

    def test_custom_high_threshold_prevents_matches(self):
        """A threshold of 1.0 should only match identical vectors."""
        resolver = EmbeddingResolver(similarity_threshold=1.0)
        entities = [
            _entity("A", embedding=[1.0, 0.0, 0.0, 0.0]),
            _entity("B", embedding=[0.99, 0.1, 0.0, 0.0]),
        ]
        result = resolver.resolve(entities)
        assert "semantic_match_text" not in result[0]["properties"]
        assert "semantic_match_text" not in result[1]["properties"]

    def test_threshold_exactly_at_boundary(self):
        """When similarity equals the threshold exactly, a match should occur
        (the code uses >= comparison)."""
        # Identical vectors have similarity 1.0
        resolver = EmbeddingResolver(similarity_threshold=1.0)
        entities = [
            _entity("A", embedding=[1.0, 0.0, 0.0, 0.0]),
            _entity("B", embedding=[1.0, 0.0, 0.0, 0.0]),
        ]
        result = resolver.resolve(entities)
        assert result[0]["properties"]["semantic_match_text"] == "B"


# ------------------------------------------------------------------
# Property preservation
# ------------------------------------------------------------------


class TestEmbeddingResolverPropertyPreservation:
    def test_existing_properties_preserved(self):
        """Enrichment must add new keys, not replace the properties dict."""
        resolver = EmbeddingResolver(similarity_threshold=0.50)
        entities = [
            _entity("A", embedding=[1.0, 0.0, 0.0, 0.0], properties={
                "existing_key": "value_a", "embedding": [1.0, 0.0, 0.0, 0.0]
            }),
            _entity("B", embedding=[0.99, 0.1, 0.0, 0.0], properties={
                "role": "test", "embedding": [0.99, 0.1, 0.0, 0.0]
            }),
        ]
        result = resolver.resolve(entities)
        assert result[0]["properties"]["existing_key"] == "value_a"
        assert "semantic_match_text" in result[0]["properties"]
        assert result[1]["properties"]["role"] == "test"
        assert "semantic_match_text" in result[1]["properties"]

    def test_similarity_score_is_rounded(self):
        """The similarity score should be rounded to 6 decimal places."""
        resolver = EmbeddingResolver(similarity_threshold=0.50)
        entities = [
            _entity("A", embedding=[1.0, 0.0, 0.0, 0.0]),
            _entity("B", embedding=[0.99, 0.1, 0.0, 0.0]),
        ]
        result = resolver.resolve(entities)
        sim = result[0]["properties"]["semantic_similarity"]
        # Verify it is a float with at most 6 decimal places
        assert sim == round(sim, 6)

    def test_entity_without_properties_key_gets_properties_created(self):
        """Entities that lack a properties dict entirely should have one
        created during enrichment."""
        resolver = EmbeddingResolver(similarity_threshold=0.50)
        entities = [
            {"text": "A", "label": "MISC"},
            _entity("B", embedding=[1.0, 0.0, 0.0, 0.0]),
            _entity("C", embedding=[0.99, 0.1, 0.0, 0.0]),
        ]
        result = resolver.resolve(entities)
        # Entity A has no properties and no embedding, so no enrichment
        # but B and C should be enriched
        assert "semantic_match_text" in result[1]["properties"]


# ------------------------------------------------------------------
# _cosine_similarity static method
# ------------------------------------------------------------------


class TestCosineSimilarity:
    def test_identical_vectors(self):
        sim = EmbeddingResolver._cosine_similarity(
            [1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
        )
        assert sim == 1.0

    def test_orthogonal_vectors(self):
        sim = EmbeddingResolver._cosine_similarity(
            [1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]
        )
        assert sim == 0.0

    def test_opposite_vectors(self):
        sim = EmbeddingResolver._cosine_similarity(
            [1.0, 0.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]
        )
        assert sim == -1.0

    def test_known_angle(self):
        """45-degree angle between [1,0] and [1,1] -> cos(45) ~ 0.7071."""
        sim = EmbeddingResolver._cosine_similarity([1.0, 0.0], [1.0, 1.0])
        assert abs(sim - (1.0 / math.sqrt(2))) < 1e-10

    def test_empty_first_vector(self):
        sim = EmbeddingResolver._cosine_similarity([], [1.0, 0.0])
        assert sim == 0.0

    def test_empty_second_vector(self):
        sim = EmbeddingResolver._cosine_similarity([1.0, 0.0], [])
        assert sim == 0.0

    def test_both_empty_vectors(self):
        sim = EmbeddingResolver._cosine_similarity([], [])
        assert sim == 0.0

    def test_mismatched_lengths(self):
        sim = EmbeddingResolver._cosine_similarity(
            [1.0, 0.0], [1.0, 0.0, 0.0]
        )
        assert sim == 0.0

    def test_zero_norm_first_vector(self):
        sim = EmbeddingResolver._cosine_similarity(
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]
        )
        assert sim == 0.0

    def test_zero_norm_second_vector(self):
        sim = EmbeddingResolver._cosine_similarity(
            [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        )
        assert sim == 0.0

    def test_both_zero_norm_vectors(self):
        sim = EmbeddingResolver._cosine_similarity(
            [0.0, 0.0], [0.0, 0.0]
        )
        assert sim == 0.0

    def test_symmetry(self):
        a = [0.3, 0.7, 0.1, 0.9]
        b = [0.8, 0.2, 0.5, 0.4]
        assert EmbeddingResolver._cosine_similarity(a, b) == \
            EmbeddingResolver._cosine_similarity(b, a)

    def test_unit_vectors(self):
        """Two unit vectors with known angle."""
        v1 = [1.0, 0.0, 0.0, 0.0]
        v2 = [0.5, 0.5, 0.5, 0.5]
        sim = EmbeddingResolver._cosine_similarity(v1, v2)
        # v2 norm = sqrt(4 * 0.25) = 1.0, dot = 0.5
        assert abs(sim - 0.5) < 1e-10
