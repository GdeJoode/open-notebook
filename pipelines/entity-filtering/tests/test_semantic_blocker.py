"""Tests for SemanticBlocker."""

import pytest

from entity_filtering.deduplication.semantic_blocker import SemanticBlocker


def _make_entity(text: str, label: str = "ORG", embedding=None):
    e = {"text": text, "label": label, "confidence": 0.9, "properties": {}}
    if embedding is not None:
        e["properties"]["embedding"] = embedding
    return e


class TestSemanticBlockerNoEmbeddings:
    def test_returns_single_block_without_embeddings(self):
        blocker = SemanticBlocker()
        entities = [_make_entity("A"), _make_entity("B"), _make_entity("C")]
        blocks = blocker.block(entities)
        # Without embeddings, all go into one catch-all block
        assert len(blocks) == 1
        assert len(blocks[0]) == 3

    def test_returns_empty_for_single_entity(self):
        blocker = SemanticBlocker()
        blocks = blocker.block([_make_entity("A")])
        # Can't form pairs from a single entity
        assert len(blocks) == 0

    def test_returns_empty_for_no_entities(self):
        blocker = SemanticBlocker()
        blocks = blocker.block([])
        assert len(blocks) == 0


class TestSemanticBlockerWithEmbeddings:
    def test_groups_similar_embeddings(self):
        blocker = SemanticBlocker(min_cluster_size=2)

        # Two clusters: [A, B] close together, [C, D] close together
        entities = [
            _make_entity("A", embedding=[1.0, 0.0, 0.0]),
            _make_entity("B", embedding=[0.95, 0.05, 0.0]),
            _make_entity("C", embedding=[0.0, 1.0, 0.0]),
            _make_entity("D", embedding=[0.0, 0.95, 0.05]),
        ]

        blocks = blocker.block(entities)
        # Should have at least 1 block (exact count depends on HDBSCAN behavior)
        assert len(blocks) >= 1
        total_entities = sum(len(b) for b in blocks)
        assert total_entities >= 2  # At least some entities are blocked

    def test_mixed_embedded_and_non_embedded(self):
        blocker = SemanticBlocker(min_cluster_size=2)

        entities = [
            _make_entity("A", embedding=[1.0, 0.0]),
            _make_entity("B", embedding=[0.9, 0.1]),
            _make_entity("C"),  # No embedding
            _make_entity("D"),  # No embedding
        ]

        blocks = blocker.block(entities)
        # Non-embedded entities go to catch-all block
        assert len(blocks) >= 1
        all_texts = set()
        for block in blocks:
            for e in block:
                all_texts.add(e["text"])
        # All entities should appear somewhere
        assert "C" in all_texts
        assert "D" in all_texts


class TestSemanticBlockerFallback:
    def test_fallback_without_hdbscan(self):
        """If HDBSCAN not available, returns single block."""
        import entity_filtering.deduplication.semantic_blocker as mod

        original = mod._HDBSCAN
        try:
            mod._HDBSCAN = False
            blocker = SemanticBlocker()
            entities = [
                _make_entity("A", embedding=[1.0]),
                _make_entity("B", embedding=[0.9]),
            ]
            blocks = blocker.block(entities)
            assert len(blocks) == 1
            assert len(blocks[0]) == 2
        finally:
            mod._HDBSCAN = original
