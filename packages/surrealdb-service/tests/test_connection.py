"""Tests for connection module."""

import pytest
from surrealdb import RecordID

from surrealdb_service.connection import ensure_record_id, parse_record_ids


class TestParseRecordIds:
    """Tests for parse_record_ids function."""

    def test_parse_dict_with_record_id(self):
        """Test parsing dictionary containing RecordID."""
        record_id = RecordID.parse("user:123")
        data = {"id": record_id, "name": "test"}
        result = parse_record_ids(data)
        assert result["id"] == "user:123"
        assert result["name"] == "test"

    def test_parse_nested_dict(self):
        """Test parsing nested dictionary."""
        record_id = RecordID.parse("source:456")
        data = {
            "outer": {
                "id": record_id,
                "value": "nested",
            }
        }
        result = parse_record_ids(data)
        assert result["outer"]["id"] == "source:456"

    def test_parse_list_with_record_ids(self):
        """Test parsing list containing RecordIDs."""
        record_ids = [RecordID.parse("item:1"), RecordID.parse("item:2")]
        result = parse_record_ids(record_ids)
        assert result == ["item:1", "item:2"]

    def test_parse_mixed_data(self):
        """Test parsing mixed data types."""
        record_id = RecordID.parse("test:789")
        data = {
            "id": record_id,
            "items": [record_id, "string", 123],
            "nested": {"ref": record_id},
            "plain": "text",
        }
        result = parse_record_ids(data)
        assert result["id"] == "test:789"
        assert result["items"][0] == "test:789"
        assert result["items"][1] == "string"
        assert result["items"][2] == 123
        assert result["nested"]["ref"] == "test:789"
        assert result["plain"] == "text"

    def test_parse_non_record_values(self):
        """Test that non-RecordID values pass through unchanged."""
        data = {
            "string": "test",
            "number": 42,
            "float": 3.14,
            "bool": True,
            "none": None,
        }
        result = parse_record_ids(data)
        assert result == data


class TestEnsureRecordId:
    """Tests for ensure_record_id function."""

    def test_string_input(self):
        """Test converting string to RecordID."""
        result = ensure_record_id("user:123")
        assert isinstance(result, RecordID)
        assert str(result) == "user:123"

    def test_record_id_input(self):
        """Test passing through existing RecordID."""
        original = RecordID.parse("source:456")
        result = ensure_record_id(original)
        assert result is original

    def test_complex_id(self):
        """Test complex ID with special characters."""
        result = ensure_record_id("table:complex-id_123")
        assert isinstance(result, RecordID)
