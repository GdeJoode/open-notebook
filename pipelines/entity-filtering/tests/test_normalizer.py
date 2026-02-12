"""Tests for EntityNormalizer text normalization and entity merging."""

from entity_filtering.filters.normalizer import EntityNormalizer


def _entity(text, label="MISC", confidence=0.9, properties=None):
    """Helper to build an entity dict."""
    return {
        "text": text,
        "label": label,
        "properties": properties or {},
        "confidence": confidence,
        "source_chunk_id": None,
    }


class TestNormalizerTextTransforms:
    def test_strips_leading_article_the(self):
        normalizer = EntityNormalizer()
        entities = [_entity("The United Nations")]
        result = normalizer.normalize(entities)
        assert len(result) == 1
        # The entity text itself is not mutated; merging happens by key.
        # "The United Nations" and "United Nations" would merge if both present.
        # With a single entity the original text is preserved.
        assert result[0]["text"] == "The United Nations"

    def test_merges_with_and_without_article(self):
        normalizer = EntityNormalizer()
        entities = [
            _entity("The United Nations"),
            _entity("United Nations"),
            _entity("United Nations"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1
        # Most frequent surface form wins: "United Nations" appears twice.
        assert result[0]["text"] == "United Nations"

    def test_whitespace_collapsing(self):
        normalizer = EntityNormalizer()
        entities = [
            _entity("John   Doe"),
            _entity("John Doe"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1

    def test_unicode_normalization(self):
        normalizer = EntityNormalizer()
        # NFKC normalizes the full-width 'A' (\uff21) to regular 'A'
        entities = [
            _entity("\uff21pple"),  # full-width A
            _entity("Apple"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1

    def test_no_strip_articles_when_disabled(self):
        normalizer = EntityNormalizer(strip_articles=False)
        entities = [
            _entity("The United Nations"),
            _entity("United Nations"),
        ]
        result = normalizer.normalize(entities)
        # Without article stripping these are different keys
        assert len(result) == 2


class TestNormalizerMerging:
    def test_canonical_selection_most_frequent_wins(self):
        normalizer = EntityNormalizer()
        # Use article variants (normalizer strips "The ") so they share a key
        entities = [
            _entity("The Climate Change"),
            _entity("Climate Change"),
            _entity("Climate Change"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1
        # "Climate Change" appears twice vs "The Climate Change" once
        assert result[0]["text"] == "Climate Change"

    def test_max_confidence_preserved(self):
        normalizer = EntityNormalizer()
        # Use whitespace variant so they share a normalized key
        entities = [
            _entity("Climate  Change", confidence=0.7),
            _entity("Climate Change", confidence=0.95),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1
        assert result[0]["confidence"] == 0.95

    def test_properties_merged(self):
        normalizer = EntityNormalizer()
        # Use article variant so they share a normalized key
        entities = [
            _entity("The New York", properties={"country": "US"}),
            _entity("New York", properties={"population": 8_000_000}),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1
        props = result[0]["properties"]
        assert "country" in props
        assert "population" in props

    def test_single_entity_passes_through(self):
        normalizer = EntityNormalizer()
        entities = [_entity("Unique Entity")]
        result = normalizer.normalize(entities)
        assert len(result) == 1
        assert result[0]["text"] == "Unique Entity"

    def test_empty_input(self):
        normalizer = EntityNormalizer()
        result = normalizer.normalize([])
        assert result == []

    def test_custom_articles(self):
        normalizer = EntityNormalizer(custom_articles=["De ", "Het "])
        entities = [
            _entity("De Volkskrant"),
            _entity("Volkskrant"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1

    def test_empty_text_entity_skipped(self):
        normalizer = EntityNormalizer()
        entities = [_entity(""), _entity("Valid")]
        result = normalizer.normalize(entities)
        assert len(result) == 1
        assert result[0]["text"] == "Valid"
