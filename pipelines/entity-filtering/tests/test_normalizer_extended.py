"""Tests for the NEW features added to EntityNormalizer in Phase 1.

Covers diacritics removal, HTML tag stripping, OCR artifact cleanup,
page-number filtering, combined features, and backward compatibility
(all features disabled by default).

The existing tests in test_normalizer.py are left untouched.
"""

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


# ------------------------------------------------------------------
# Diacritics removal
# ------------------------------------------------------------------


class TestNormalizerDiacritics:
    def test_diacritics_removal_merges_accented_and_plain(self):
        """With remove_diacritics=True, 'cafe' and 'cafe' (from 'cafe')
        should normalize to the same key and merge."""
        normalizer = EntityNormalizer(remove_diacritics=True)
        entities = [
            _entity("caf\u00e9"),  # cafe with accent
            _entity("cafe"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1

    def test_diacritics_disabled_by_default(self):
        """Without remove_diacritics, 'cafe' and 'cafe' are different keys."""
        normalizer = EntityNormalizer()
        entities = [
            _entity("caf\u00e9"),
            _entity("cafe"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 2

    def test_diacritics_removal_preserves_base_characters(self):
        """Removing diacritics from 'uber' (u with diaeresis) should yield 'uber'."""
        normalizer = EntityNormalizer(remove_diacritics=True)
        entities = [
            _entity("\u00fcber"),  # u-umlaut-ber
            _entity("uber"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1

    def test_diacritics_multiple_accents(self):
        """An entity with multiple accented characters normalizes correctly."""
        normalizer = EntityNormalizer(remove_diacritics=True)
        entities = [
            _entity("r\u00e9sum\u00e9"),  # resume with accents
            _entity("resume"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1


# ------------------------------------------------------------------
# HTML stripping
# ------------------------------------------------------------------


class TestNormalizerHTMLStripping:
    def test_html_tags_stripped_when_enabled(self):
        normalizer = EntityNormalizer(html_strip_enabled=True)
        entities = [
            _entity("<b>Entity</b>"),
            _entity("Entity"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1

    def test_html_stripping_disabled_by_default(self):
        normalizer = EntityNormalizer()
        entities = [
            _entity("<b>Entity</b>"),
            _entity("Entity"),
        ]
        result = normalizer.normalize(entities)
        # Tags are part of the text, so keys differ
        assert len(result) == 2

    def test_nested_html_tags_stripped(self):
        normalizer = EntityNormalizer(html_strip_enabled=True)
        entities = [
            _entity("<div><span>Deep</span></div>"),
            _entity("Deep"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1

    def test_html_with_attributes_stripped(self):
        normalizer = EntityNormalizer(html_strip_enabled=True)
        entities = [
            _entity('<a href="http://example.com">Link Text</a>'),
            _entity("Link Text"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1


# ------------------------------------------------------------------
# OCR cleanup
# ------------------------------------------------------------------


class TestNormalizerOCRCleanup:
    def test_ocr_cleanup_with_custom_patterns(self):
        """Remove form-feed characters via custom OCR pattern."""
        normalizer = EntityNormalizer(
            ocr_cleanup_enabled=True,
            ocr_artifact_patterns=[r"\f"],
        )
        entities = [
            _entity("Hello\fWorld"),
            _entity("HelloWorld"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1

    def test_ocr_cleanup_removes_soft_hyphens(self):
        """Remove soft-hyphen (U+00AD) via OCR pattern."""
        normalizer = EntityNormalizer(
            ocr_cleanup_enabled=True,
            ocr_artifact_patterns=[r"\u00ad"],
        )
        entities = [
            _entity("exam\u00adple"),  # soft hyphen inside word
            _entity("example"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1

    def test_ocr_cleanup_disabled_by_default(self):
        normalizer = EntityNormalizer()
        entities = [
            _entity("Hello\fWorld"),
            _entity("HelloWorld"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 2

    def test_multiple_ocr_patterns_applied(self):
        """Multiple patterns are all applied in sequence."""
        normalizer = EntityNormalizer(
            ocr_cleanup_enabled=True,
            ocr_artifact_patterns=[r"\f", r"\u00ad"],
        )
        entities = [
            _entity("He\fllo\u00adWorld"),
            _entity("HelloWorld"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1


# ------------------------------------------------------------------
# Page number filtering
# ------------------------------------------------------------------


class TestNormalizerPageNumberFilter:
    def test_page_number_dropped_when_enabled(self):
        normalizer = EntityNormalizer(page_number_filter=True)
        entities = [
            _entity("Page 1"),
            _entity("Real Entity"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1
        assert result[0]["text"] == "Real Entity"

    def test_page_number_variants_dropped(self):
        """Various page-number forms: 'p. 42', 'pp.3', 'page 100'."""
        normalizer = EntityNormalizer(page_number_filter=True)
        entities = [
            _entity("Page 1"),
            _entity("p. 42"),
            _entity("pp.3"),
            _entity("page 100"),
            _entity("Actual Data"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1
        assert result[0]["text"] == "Actual Data"

    def test_page_number_filter_disabled_by_default(self):
        normalizer = EntityNormalizer()
        entities = [_entity("Page 1")]
        result = normalizer.normalize(entities)
        assert len(result) == 1
        assert result[0]["text"] == "Page 1"

    def test_page_number_case_insensitive(self):
        normalizer = EntityNormalizer(page_number_filter=True)
        entities = [
            _entity("PAGE 5"),
            _entity("page 5"),
            _entity("Page 5"),
            _entity("Keep Me"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1
        assert result[0]["text"] == "Keep Me"


# ------------------------------------------------------------------
# Combined features
# ------------------------------------------------------------------


class TestNormalizerCombinedFeatures:
    def test_diacritics_and_html_together(self):
        normalizer = EntityNormalizer(
            remove_diacritics=True, html_strip_enabled=True
        )
        entities = [
            _entity("<b>caf\u00e9</b>"),
            _entity("cafe"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1

    def test_all_features_together(self):
        """Diacritics + HTML + page numbers all active at once."""
        normalizer = EntityNormalizer(
            remove_diacritics=True,
            html_strip_enabled=True,
            page_number_filter=True,
        )
        entities = [
            _entity("Page 1"),  # dropped by page filter
            _entity("<i>caf\u00e9</i>"),  # HTML stripped, diacritics removed
            _entity("cafe"),  # matches the above after normalization
            _entity("Unique"),
        ]
        result = normalizer.normalize(entities)
        # Page 1 dropped, cafe+cafe merged, Unique kept -> 2 entities
        assert len(result) == 2
        result_texts = {e["text"] for e in result}
        assert "Unique" in result_texts
        # Page 1 should not be present
        assert "Page 1" not in result_texts


# ------------------------------------------------------------------
# Backward compatibility
# ------------------------------------------------------------------


class TestNormalizerBackwardCompatibility:
    def test_all_new_features_disabled_by_default(self):
        """Default EntityNormalizer() has all Phase 1 features off."""
        normalizer = EntityNormalizer()
        assert normalizer._remove_diacritics is False
        assert normalizer._ocr_cleanup is False
        assert normalizer._ocr_patterns == []
        assert normalizer._html_strip is False
        assert normalizer._page_number_filter is False

    def test_existing_normalization_unchanged(self):
        """The original article-stripping and whitespace behavior still works.

        Both forms normalize to the same key ('United Nations') so they
        merge.  With equal frequency the first entity seen becomes canonical.
        Adding a second 'United Nations' makes it the most-frequent form.
        """
        normalizer = EntityNormalizer()
        entities = [
            _entity("The  United  Nations"),
            _entity("United Nations"),
            _entity("United Nations"),
        ]
        result = normalizer.normalize(entities)
        assert len(result) == 1
        assert result[0]["text"] == "United Nations"
