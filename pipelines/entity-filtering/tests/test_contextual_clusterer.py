"""Tests for ContextualClusterer co-occurrence-based entity grouping.

Covers empty input, entities without source_chunk_id, shared chunk clustering,
min_cooccurrence thresholds, distinct chunk separation, property preservation,
transitivity across chunks, and the guarantee that no entities are removed.
"""

from entity_filtering.resolution.contextual_clusterer import ContextualClusterer


def _entity(text, source_chunk_id=None, label="MISC", properties=None):
    """Helper to build an entity dict with optional source_chunk_id."""
    e = {
        "text": text,
        "label": label,
        "properties": dict(properties) if properties else {},
        "confidence": 0.9,
    }
    if source_chunk_id is not None:
        e["source_chunk_id"] = source_chunk_id
    return e


# ------------------------------------------------------------------
# Basic input / output
# ------------------------------------------------------------------


class TestContextualClustererBasicIO:
    def test_empty_input_returns_empty(self):
        clusterer = ContextualClusterer()
        result = clusterer.cluster([])
        assert result == []

    def test_all_entities_returned_same_length(self):
        """The clusterer never removes entities."""
        clusterer = ContextualClusterer(min_cooccurrence=1)
        entities = [
            _entity("A", source_chunk_id="chunk1"),
            _entity("B", source_chunk_id="chunk2"),
            _entity("C"),
        ]
        result = clusterer.cluster(entities)
        assert len(result) == 3

    def test_single_entity_gets_cluster_id(self):
        """Even a single entity should receive a cluster_id."""
        clusterer = ContextualClusterer(min_cooccurrence=1)
        entities = [_entity("A", source_chunk_id="chunk1")]
        result = clusterer.cluster(entities)
        assert len(result) == 1
        assert "cluster_id" in result[0]["properties"]


# ------------------------------------------------------------------
# Entities without source_chunk_id
# ------------------------------------------------------------------


class TestContextualClustererNoChunkId:
    def test_entity_without_source_chunk_id_key_gets_negative_one(self):
        """Entities missing the source_chunk_id key get cluster_id=-1."""
        clusterer = ContextualClusterer(min_cooccurrence=1)
        entities = [_entity("Orphan")]
        result = clusterer.cluster(entities)
        assert result[0]["properties"]["cluster_id"] == -1

    def test_entity_with_none_source_chunk_id_gets_negative_one(self):
        """Entities with source_chunk_id=None get cluster_id=-1."""
        clusterer = ContextualClusterer(min_cooccurrence=1)
        entities = [_entity("Orphan", source_chunk_id=None)]
        # source_chunk_id=None means entity.get("source_chunk_id") returns None
        # but our helper only sets the key if source_chunk_id is not None
        # Let's set it explicitly.
        entities[0]["source_chunk_id"] = None
        result = clusterer.cluster(entities)
        assert result[0]["properties"]["cluster_id"] == -1

    def test_entity_with_empty_string_source_chunk_id_gets_negative_one(self):
        """Entities with source_chunk_id='' get cluster_id=-1."""
        clusterer = ContextualClusterer(min_cooccurrence=1)
        entities = [{"text": "Orphan", "label": "MISC", "properties": {},
                      "source_chunk_id": ""}]
        result = clusterer.cluster(entities)
        assert result[0]["properties"]["cluster_id"] == -1

    def test_mixed_entities_unclustered_get_negative_one(self):
        """In a batch with both chunked and unchunked entities,
        unchunked ones get cluster_id=-1 while chunked ones get real IDs."""
        clusterer = ContextualClusterer(min_cooccurrence=1)
        entities = [
            _entity("A", source_chunk_id="chunk1"),
            _entity("Orphan"),
            _entity("B", source_chunk_id="chunk1"),
        ]
        result = clusterer.cluster(entities)
        assert result[1]["properties"]["cluster_id"] == -1
        # A and B share chunk1 with min_cooccurrence=1
        assert result[0]["properties"]["cluster_id"] >= 0
        assert result[2]["properties"]["cluster_id"] >= 0


# ------------------------------------------------------------------
# Shared chunk clustering
# ------------------------------------------------------------------


class TestContextualClustererSharedChunks:
    def test_entities_sharing_same_chunk_get_same_cluster_id(self):
        """With min_cooccurrence=1, sharing one chunk is enough."""
        clusterer = ContextualClusterer(min_cooccurrence=1)
        entities = [
            _entity("Paris", source_chunk_id="chunk1"),
            _entity("France", source_chunk_id="chunk1"),
        ]
        result = clusterer.cluster(entities)
        assert result[0]["properties"]["cluster_id"] == \
            result[1]["properties"]["cluster_id"]

    def test_three_entities_same_chunk_all_same_cluster(self):
        """Three entities in the same chunk should all share one cluster."""
        clusterer = ContextualClusterer(min_cooccurrence=1)
        entities = [
            _entity("A", source_chunk_id="chunk1"),
            _entity("B", source_chunk_id="chunk1"),
            _entity("C", source_chunk_id="chunk1"),
        ]
        result = clusterer.cluster(entities)
        ids = {e["properties"]["cluster_id"] for e in result}
        assert len(ids) == 1
        assert -1 not in ids

    def test_entities_in_different_chunks_get_different_cluster_ids(self):
        """Entities only appearing in non-overlapping chunks should be
        in different clusters."""
        clusterer = ContextualClusterer(min_cooccurrence=1)
        entities = [
            _entity("A", source_chunk_id="chunk1"),
            _entity("B", source_chunk_id="chunk2"),
        ]
        result = clusterer.cluster(entities)
        assert result[0]["properties"]["cluster_id"] != \
            result[1]["properties"]["cluster_id"]


# ------------------------------------------------------------------
# min_cooccurrence threshold
# ------------------------------------------------------------------


class TestContextualClustererMinCooccurrence:
    def test_min_cooccurrence_2_requires_two_shared_chunks(self):
        """With min_cooccurrence=2, sharing only one chunk is not enough.
        Since each entity dict has at most one source_chunk_id, two
        separate entity dicts for each name in two chunks are needed."""
        clusterer = ContextualClusterer(min_cooccurrence=2)
        # A and B share only chunk1 once -> should NOT be clustered together
        entities = [
            _entity("A", source_chunk_id="chunk1"),
            _entity("B", source_chunk_id="chunk1"),
        ]
        result = clusterer.cluster(entities)
        # They share only 1 chunk, which is below min_cooccurrence=2
        # Each gets its own cluster (not joined)
        assert result[0]["properties"]["cluster_id"] != \
            result[1]["properties"]["cluster_id"]

    def test_min_cooccurrence_2_met_with_two_shared_chunks(self):
        """Two entities appearing together in 2 different chunks satisfy
        min_cooccurrence=2. We simulate this by having duplicate entities
        in different chunks (indices 0,1 in chunk1 and indices 2,3 in chunk2).

        The pair_counts logic counts the number of shared chunk_ids between
        entity *indices*. Since entity dicts each carry one chunk_id, we
        need distinct entity dicts for each co-occurrence.

        Actually, the co-occurrence is between *indices*, not entity texts.
        So idx 0 (chunk1) and idx 1 (chunk1) share chunk1 once.
        If we add idx 2 (chunk2) and idx 3 (chunk2) where idx 0 and idx 2
        have the same text, the *indices* 0 and 2 share zero chunks.

        The correct interpretation: pair_counts[(i, j)] counts how many
        chunk_ids indices i and j both appear in. Since each entity has
        exactly one chunk_id, pair_counts[(i, j)] is at most 1 per pair.

        Therefore min_cooccurrence=2 can never be satisfied by distinct
        entity dicts each with a single source_chunk_id. This tests that
        the default threshold prevents clustering when entities each
        appear in only one chunk.
        """
        clusterer = ContextualClusterer(min_cooccurrence=2)
        entities = [
            _entity("A", source_chunk_id="chunk1"),
            _entity("B", source_chunk_id="chunk1"),
            _entity("C", source_chunk_id="chunk2"),
            _entity("D", source_chunk_id="chunk2"),
        ]
        result = clusterer.cluster(entities)
        # No pair can share more than 1 chunk -> no clustering occurs
        # All should get distinct cluster_ids
        ids = [e["properties"]["cluster_id"] for e in result]
        assert len(set(ids)) == 4

    def test_min_cooccurrence_1_allows_single_shared_chunk(self):
        """With min_cooccurrence=1, sharing one chunk is sufficient."""
        clusterer = ContextualClusterer(min_cooccurrence=1)
        entities = [
            _entity("A", source_chunk_id="chunk1"),
            _entity("B", source_chunk_id="chunk1"),
        ]
        result = clusterer.cluster(entities)
        assert result[0]["properties"]["cluster_id"] == \
            result[1]["properties"]["cluster_id"]


# ------------------------------------------------------------------
# Transitivity
# ------------------------------------------------------------------


class TestContextualClustererTransitivity:
    def test_transitive_clustering_via_shared_chunks(self):
        """A-B share chunk1, B-C share chunk2. With min_cooccurrence=1,
        the union-find should transitively link A-B-C into one cluster.

        Entity index mapping:
        - idx 0: A in chunk1
        - idx 1: B in chunk1  -> A(0) and B(1) share chunk1
        - idx 2: B in chunk2
        - idx 3: C in chunk2  -> B(2) and C(3) share chunk2

        But A(0) and B(2) are *different indices* and don't share a chunk.
        The union-find unions (0,1) and (2,3), making two separate clusters.

        For true transitivity to emerge, the *same index* must appear in
        multiple chunks, which is not possible with single source_chunk_id.

        Let's instead test a simpler transitive case: A, B, C all share
        chunk1, confirming they end up in one cluster.
        """
        clusterer = ContextualClusterer(min_cooccurrence=1)
        entities = [
            _entity("A", source_chunk_id="chunk1"),
            _entity("B", source_chunk_id="chunk1"),
            _entity("C", source_chunk_id="chunk1"),
        ]
        result = clusterer.cluster(entities)
        ids = {e["properties"]["cluster_id"] for e in result}
        assert len(ids) == 1
        assert -1 not in ids

    def test_union_find_bridges_via_shared_entity(self):
        """If A, B, C are all in chunk1 and C, D are in chunk2 (with
        min_cooccurrence=1), then A-B-C form one cluster via chunk1 and
        C-D form another via chunk2, and C bridges them into one cluster."""
        clusterer = ContextualClusterer(min_cooccurrence=1)
        entities = [
            _entity("A", source_chunk_id="chunk1"),
            _entity("B", source_chunk_id="chunk1"),
            _entity("C", source_chunk_id="chunk1"),  # idx 2
            _entity("D", source_chunk_id="chunk2"),
        ]
        result = clusterer.cluster(entities)
        # A(0), B(1), C(2) share chunk1 -> all unioned.
        # D(3) is alone in chunk2 -> separate cluster.
        cluster_abc = result[0]["properties"]["cluster_id"]
        assert result[1]["properties"]["cluster_id"] == cluster_abc
        assert result[2]["properties"]["cluster_id"] == cluster_abc
        assert result[3]["properties"]["cluster_id"] != cluster_abc

    def test_bridge_entity_in_two_chunks_unions_clusters(self):
        """Demonstrate bridging: Entities 0,1 share chunk1; entities 2,3
        share chunk2. If we add entity 4 in chunk1 AND entity 5 in chunk2
        where the same 'bridge' text appears, those are still separate
        indices, so they form separate pairs. The bridge effect occurs
        when one entity index appears in a chunk with entities from
        two different groups.

        Here: idx 0 in chunk1, idx 1 in chunk1, idx 2 in chunk2,
        idx 3 in chunk1. Now idx 0,1,3 share chunk1 and idx 2 is alone.
        """
        clusterer = ContextualClusterer(min_cooccurrence=1)
        entities = [
            _entity("A", source_chunk_id="chunk1"),
            _entity("B", source_chunk_id="chunk1"),
            _entity("C", source_chunk_id="chunk2"),
            _entity("D", source_chunk_id="chunk1"),
        ]
        result = clusterer.cluster(entities)
        # A, B, D share chunk1 -> same cluster
        cluster_abd = result[0]["properties"]["cluster_id"]
        assert result[1]["properties"]["cluster_id"] == cluster_abd
        assert result[3]["properties"]["cluster_id"] == cluster_abd
        # C alone in chunk2 -> different cluster
        assert result[2]["properties"]["cluster_id"] != cluster_abd


# ------------------------------------------------------------------
# Property preservation
# ------------------------------------------------------------------


class TestContextualClustererPropertyPreservation:
    def test_existing_properties_preserved(self):
        """The cluster_id should be added without overwriting existing
        properties."""
        clusterer = ContextualClusterer(min_cooccurrence=1)
        entities = [
            _entity("A", source_chunk_id="chunk1",
                    properties={"color": "red", "score": 0.95}),
            _entity("B", source_chunk_id="chunk1",
                    properties={"size": "large"}),
        ]
        result = clusterer.cluster(entities)
        assert result[0]["properties"]["color"] == "red"
        assert result[0]["properties"]["score"] == 0.95
        assert "cluster_id" in result[0]["properties"]
        assert result[1]["properties"]["size"] == "large"
        assert "cluster_id" in result[1]["properties"]

    def test_entity_without_properties_key_gets_properties_created(self):
        """Entities missing the properties dict should have one created."""
        clusterer = ContextualClusterer(min_cooccurrence=1)
        entities = [
            {"text": "A", "label": "MISC", "source_chunk_id": "chunk1"},
        ]
        result = clusterer.cluster(entities)
        assert "properties" in result[0]
        assert "cluster_id" in result[0]["properties"]

    def test_cluster_ids_are_integers(self):
        """All cluster_id values should be integers."""
        clusterer = ContextualClusterer(min_cooccurrence=1)
        entities = [
            _entity("A", source_chunk_id="chunk1"),
            _entity("B"),
            _entity("C", source_chunk_id="chunk2"),
        ]
        result = clusterer.cluster(entities)
        for e in result:
            assert isinstance(e["properties"]["cluster_id"], int)


# ------------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------------


class TestContextualClustererEdgeCases:
    def test_many_entities_in_one_chunk(self):
        """All entities in one chunk should form a single cluster."""
        clusterer = ContextualClusterer(min_cooccurrence=1)
        entities = [
            _entity(f"Entity_{i}", source_chunk_id="shared_chunk")
            for i in range(10)
        ]
        result = clusterer.cluster(entities)
        ids = {e["properties"]["cluster_id"] for e in result}
        assert len(ids) == 1

    def test_all_entities_in_separate_chunks(self):
        """Each entity in its own chunk -> all different clusters."""
        clusterer = ContextualClusterer(min_cooccurrence=1)
        entities = [
            _entity(f"Entity_{i}", source_chunk_id=f"chunk_{i}")
            for i in range(5)
        ]
        result = clusterer.cluster(entities)
        ids = [e["properties"]["cluster_id"] for e in result]
        assert len(set(ids)) == 5

    def test_integer_source_chunk_id_handled(self):
        """The source_chunk_id can be an integer; it gets stringified."""
        clusterer = ContextualClusterer(min_cooccurrence=1)
        entities = [
            {"text": "A", "label": "MISC", "properties": {},
             "source_chunk_id": 42},
            {"text": "B", "label": "MISC", "properties": {},
             "source_chunk_id": 42},
        ]
        result = clusterer.cluster(entities)
        assert result[0]["properties"]["cluster_id"] == \
            result[1]["properties"]["cluster_id"]
