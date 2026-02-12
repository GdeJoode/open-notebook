"""Tests for EntityReclassifier rule matching and label updates."""

from entity_filtering.filters.reclassifier import EntityReclassifier


def _entity(text, label="MISC", confidence=0.9, properties=None):
    """Helper to build an entity dict."""
    return {
        "text": text,
        "label": label,
        "properties": properties or {},
        "confidence": confidence,
        "source_chunk_id": None,
    }


class TestGenericRules:
    def test_hyphenated_name_reclassified_to_person(self):
        rc = EntityReclassifier()
        entities = [_entity("Jean-Pierre", label="MISC")]
        result = rc.reclassify(entities)
        assert result[0]["label"] == "PERSON"

    def test_hyphenated_name_preserves_original_label(self):
        rc = EntityReclassifier()
        entities = [_entity("Mary-Ann", label="ORG")]
        result = rc.reclassify(entities)
        assert result[0]["label"] == "PERSON"
        assert result[0]["properties"]["original_label"] == "ORG"

    def test_short_allcaps_reclassified_to_abbreviation(self):
        rc = EntityReclassifier()
        entities = [_entity("NATO", label="ORG")]
        result = rc.reclassify(entities)
        assert result[0]["label"] == "ABBREVIATION"
        assert result[0]["properties"]["original_label"] == "ORG"

    def test_short_allcaps_boundary_two_chars(self):
        rc = EntityReclassifier()
        entities = [_entity("UN", label="ORG")]
        result = rc.reclassify(entities)
        assert result[0]["label"] == "ABBREVIATION"

    def test_short_allcaps_boundary_six_chars(self):
        rc = EntityReclassifier()
        entities = [_entity("UNESCO", label="ORG")]
        result = rc.reclassify(entities)
        assert result[0]["label"] == "ABBREVIATION"

    def test_long_allcaps_not_reclassified(self):
        rc = EntityReclassifier()
        # 7 characters -- exceeds 2-6 range
        entities = [_entity("UNICEFF", label="ORG")]
        result = rc.reclassify(entities)
        assert result[0]["label"] == "ORG"


class TestCustomRules:
    def test_custom_rule_overrides_generic(self):
        # NATO would normally match ABBREVIATION via generic rule.
        # A custom rule mapping it to ORGANIZATION should take precedence.
        custom = {r"^NATO$": "ORGANIZATION"}
        rc = EntityReclassifier(custom_rules=custom)
        entities = [_entity("NATO", label="MISC")]
        result = rc.reclassify(entities)
        assert result[0]["label"] == "ORGANIZATION"

    def test_custom_rule_applied(self):
        custom = {r"^Prof\.\s": "PERSON"}
        rc = EntityReclassifier(custom_rules=custom)
        entities = [_entity("Prof. Smith", label="MISC")]
        result = rc.reclassify(entities)
        assert result[0]["label"] == "PERSON"

    def test_custom_rule_case_insensitive(self):
        custom = {r"^prof\.\s": "PERSON"}
        rc = EntityReclassifier(custom_rules=custom)
        entities = [_entity("Prof. Smith", label="MISC")]
        result = rc.reclassify(entities)
        assert result[0]["label"] == "PERSON"


class TestNoReclassification:
    def test_no_rule_matches_keeps_original_label(self):
        rc = EntityReclassifier()
        entities = [_entity("Climate Change", label="CONCEPT")]
        result = rc.reclassify(entities)
        assert result[0]["label"] == "CONCEPT"
        assert "original_label" not in result[0]["properties"]

    def test_matching_rule_same_label_no_change(self):
        rc = EntityReclassifier()
        # "Jean-Pierre" matches PERSON rule, but label is already PERSON
        entities = [_entity("Jean-Pierre", label="PERSON")]
        result = rc.reclassify(entities)
        assert result[0]["label"] == "PERSON"
        assert "original_label" not in result[0]["properties"]

    def test_empty_input(self):
        rc = EntityReclassifier()
        result = rc.reclassify([])
        assert result == []

    def test_entity_dict_is_copied_not_mutated(self):
        rc = EntityReclassifier()
        original = _entity("Jean-Pierre", label="MISC")
        result = rc.reclassify([original])
        # The result should be a copy, not the same dict
        assert result[0] is not original
        # Original should be unchanged
        assert original["label"] == "MISC"
