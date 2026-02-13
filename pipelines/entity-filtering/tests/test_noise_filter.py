"""Tests for NoiseFilter entity and relation filtering."""

import pytest

from entity_filtering.filters.noise_filter import NoiseFilter


def _entity(text, label="MISC", confidence=0.9):
    """Helper to build an entity dict."""
    return {
        "text": text,
        "label": label,
        "properties": {},
        "confidence": confidence,
        "source_chunk_id": None,
    }


def _relation(source, target, rel_type="RELATED_TO", confidence=0.8):
    """Helper to build a relation dict."""
    return {
        "source_entity": source,
        "target_entity": target,
        "relation_type": rel_type,
        "properties": {},
        "confidence": confidence,
        "source_chunk_id": None,
    }


class TestNoiseFilterEntities:
    def test_filters_below_min_entity_length(self):
        nf = NoiseFilter(min_entity_length=2)
        entities = [_entity("A"), _entity("John Doe")]
        result = nf.filter_entities(entities)
        assert len(result) == 1
        assert result[0]["text"] == "John Doe"

    def test_filters_single_character(self):
        nf = NoiseFilter(min_entity_length=2)
        entities = [_entity("X")]
        result = nf.filter_entities(entities)
        assert len(result) == 0

    def test_filters_citation_et_al(self):
        nf = NoiseFilter()
        entities = [_entity("et al."), _entity("et al")]
        result = nf.filter_entities(entities)
        assert len(result) == 0

    def test_filters_citation_ibid(self):
        nf = NoiseFilter()
        entities = [_entity("ibid."), _entity("ibid")]
        result = nf.filter_entities(entities)
        assert len(result) == 0

    def test_filters_citation_op_cit(self):
        nf = NoiseFilter()
        entities = [_entity("op. cit."), _entity("op cit")]
        result = nf.filter_entities(entities)
        assert len(result) == 0

    def test_filters_pure_numbers(self):
        nf = NoiseFilter()
        entities = [_entity("123"), _entity("45.6%"), _entity("1,000")]
        result = nf.filter_entities(entities)
        assert len(result) == 0

    def test_filters_urls(self):
        nf = NoiseFilter()
        entities = [_entity("https://example.com"), _entity("http://test.org/path")]
        result = nf.filter_entities(entities)
        assert len(result) == 0

    def test_filters_email_addresses(self):
        nf = NoiseFilter()
        entities = [_entity("user@example.com")]
        result = nf.filter_entities(entities)
        assert len(result) == 0

    def test_filters_isolated_punctuation(self):
        nf = NoiseFilter()
        entities = [_entity("---"), _entity("***"), _entity("...")]
        result = nf.filter_entities(entities)
        assert len(result) == 0

    def test_keeps_valid_entities(self):
        nf = NoiseFilter()
        entities = [
            _entity("John Doe", "PERSON"),
            _entity("Microsoft", "ORG"),
            _entity("Climate Change", "CONCEPT"),
        ]
        result = nf.filter_entities(entities)
        assert len(result) == 3
        texts = {e["text"] for e in result}
        assert texts == {"John Doe", "Microsoft", "Climate Change"}

    def test_custom_noise_patterns(self):
        nf = NoiseFilter(custom_patterns=[r"^REDACTED$"])
        entities = [_entity("REDACTED"), _entity("Valid Entity")]
        result = nf.filter_entities(entities)
        assert len(result) == 1
        assert result[0]["text"] == "Valid Entity"

    def test_empty_input(self):
        nf = NoiseFilter()
        result = nf.filter_entities([])
        assert result == []

    def test_mixed_valid_and_noise(self):
        nf = NoiseFilter()
        entities = [
            _entity("John Doe", "PERSON"),
            _entity("123"),
            _entity("Microsoft", "ORG"),
            _entity("---"),
            _entity("et al."),
        ]
        result = nf.filter_entities(entities)
        assert len(result) == 2
        texts = {e["text"] for e in result}
        assert texts == {"John Doe", "Microsoft"}


class TestNoiseFilterRelations:
    def test_removes_relations_with_removed_entities(self):
        nf = NoiseFilter()
        valid_entities = [
            _entity("John Doe", "PERSON"),
            _entity("Microsoft", "ORG"),
        ]
        relations = [
            _relation("John Doe", "Microsoft", "WORKS_AT"),
            _relation("123", "Microsoft", "RELATED_TO"),
            _relation("John Doe", "et al.", "CITED_BY"),
        ]
        result = nf.filter_relations(relations, valid_entities)
        assert len(result) == 1
        assert result[0]["source_entity"] == "John Doe"
        assert result[0]["target_entity"] == "Microsoft"

    def test_keeps_relations_with_valid_entities(self):
        nf = NoiseFilter()
        valid_entities = [
            _entity("Alice", "PERSON"),
            _entity("Bob", "PERSON"),
            _entity("Acme Corp", "ORG"),
        ]
        relations = [
            _relation("Alice", "Bob", "KNOWS"),
            _relation("Alice", "Acme Corp", "WORKS_AT"),
        ]
        result = nf.filter_relations(relations, valid_entities)
        assert len(result) == 2

    def test_empty_relations(self):
        nf = NoiseFilter()
        result = nf.filter_relations([], [_entity("Alice")])
        assert result == []

    def test_empty_valid_entities_removes_all_relations(self):
        nf = NoiseFilter()
        relations = [_relation("Alice", "Bob", "KNOWS")]
        result = nf.filter_relations(relations, [])
        assert result == []
