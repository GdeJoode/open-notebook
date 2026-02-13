"""Tests for EntityDeduplicator string-based deduplication."""

from entity_filtering.deduplication.entity_deduplicator import EntityDeduplicator


def _entity(text, label="MISC", confidence=0.9, properties=None):
    """Helper to build an entity dict."""
    return {
        "text": text,
        "label": label,
        "properties": properties or {},
        "confidence": confidence,
        "source_chunk_id": None,
    }


class TestExactDuplicates:
    def test_exact_duplicates_merged(self):
        dedup = EntityDeduplicator()
        entities = [
            _entity("John Doe", "PERSON", 0.9),
            _entity("John Doe", "PERSON", 0.85),
        ]
        result, merge_groups = dedup.deduplicate(entities)
        assert len(result) == 1
        assert result[0]["text"] == "John Doe"
        # Exact same surface form -- no multi-form merge group produced
        assert len(merge_groups) == 0

    def test_case_insensitive_deduplication(self):
        dedup = EntityDeduplicator()
        entities = [
            _entity("john doe", "PERSON", 0.8),
            _entity("John Doe", "PERSON", 0.9),
        ]
        result, merge_groups = dedup.deduplicate(entities)
        assert len(result) == 1
        # Merge group should record both surface forms
        assert len(merge_groups) == 1
        assert sorted(merge_groups[0]) == ["John Doe", "john doe"]


class TestMergeGroups:
    def test_merge_groups_track_surface_forms(self):
        dedup = EntityDeduplicator()
        entities = [
            _entity("new york", "LOC"),
            _entity("New York", "LOC"),
            _entity("NEW YORK", "LOC"),
        ]
        result, merge_groups = dedup.deduplicate(entities)
        assert len(result) == 1
        assert len(merge_groups) == 1
        assert len(merge_groups[0]) == 3


class TestCanonicalSelection:
    def test_most_frequent_surface_form_wins(self):
        dedup = EntityDeduplicator()
        entities = [
            _entity("Climate Change"),
            _entity("climate change"),
            _entity("Climate Change"),
        ]
        result, _ = dedup.deduplicate(entities)
        assert len(result) == 1
        assert result[0]["text"] == "Climate Change"

    def test_max_confidence_preserved(self):
        dedup = EntityDeduplicator()
        entities = [
            _entity("John Doe", "PERSON", 0.7),
            _entity("john doe", "PERSON", 0.95),
        ]
        result, _ = dedup.deduplicate(entities)
        assert result[0]["confidence"] == 0.95

    def test_properties_merged(self):
        dedup = EntityDeduplicator()
        entities = [
            _entity("Paris", "LOC", properties={"country": "France"}),
            _entity("paris", "LOC", properties={"population": 2_000_000}),
        ]
        result, _ = dedup.deduplicate(entities)
        assert len(result) == 1
        props = result[0]["properties"]
        assert "country" in props
        assert "population" in props

    def test_most_common_label_preserved(self):
        dedup = EntityDeduplicator()
        entities = [
            _entity("Apple", "ORG"),
            _entity("apple", "PRODUCT"),
            _entity("Apple", "ORG"),
        ]
        result, _ = dedup.deduplicate(entities)
        assert result[0]["label"] == "ORG"


class TestPassThrough:
    def test_unique_entities_unchanged(self):
        dedup = EntityDeduplicator()
        entities = [
            _entity("Alice", "PERSON"),
            _entity("Bob", "PERSON"),
            _entity("Acme Corp", "ORG"),
        ]
        result, merge_groups = dedup.deduplicate(entities)
        assert len(result) == 3
        assert len(merge_groups) == 0

    def test_empty_input(self):
        dedup = EntityDeduplicator()
        result, merge_groups = dedup.deduplicate([])
        assert result == []
        assert merge_groups == []

    def test_single_entity(self):
        dedup = EntityDeduplicator()
        entities = [_entity("Singleton")]
        result, merge_groups = dedup.deduplicate(entities)
        assert len(result) == 1
        assert result[0]["text"] == "Singleton"
        assert merge_groups == []

    def test_empty_text_entity_skipped(self):
        dedup = EntityDeduplicator()
        entities = [_entity(""), _entity("Valid")]
        result, _ = dedup.deduplicate(entities)
        assert len(result) == 1
        assert result[0]["text"] == "Valid"
