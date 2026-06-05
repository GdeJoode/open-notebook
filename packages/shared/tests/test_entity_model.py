"""Pydantic round-trip tests for Entity and Relation models (Phase B.1a).

These are pure-Pydantic tests — no SurrealDB involvement. The DB-backed
round-trip lives in
``packages/surrealdb-service/tests/test_entity_repository_roundtrip.py``.
"""

from __future__ import annotations

import pytest

from shared.models import Entity, Relation


class TestEntityModel:
    def test_minimal_construction(self):
        """canonical_name + entity_type are required; everything else has a default."""
        ent = Entity(canonical_name="BZK", entity_type="Org", embedding=[])
        assert ent.canonical_name == "BZK"
        assert ent.entity_type == "Org"
        assert ent.confidence == 1.0
        assert ent.status == "active"
        assert ent.extraction_method == "llm"
        assert ent.source_documents == []
        assert ent.type_tags == []
        assert ent.primary_type is None
        assert ent.embedding == []
        assert ent.properties == {}

    def test_canonical_name_required(self):
        with pytest.raises(Exception):  # noqa: B017 — Pydantic ValidationError
            Entity(entity_type="Org", embedding=[])  # type: ignore[call-arg]

    def test_table_name(self):
        assert Entity.table_name == "entity"

    def test_model_dump_roundtrip_with_multi_type_tagging(self):
        """The full B.1a field set survives model_dump → re-parse."""
        original = Entity(
            canonical_name="Foo",
            entity_type="Person",
            type_tags=["Person", "Researcher"],
            primary_type="Researcher",
            confidence=0.87,
            source_documents=["source:abc", "source:def"],
            properties={"role": "scientist", "merged_from": ["Foo", "F. Bar"]},
            embedding=[0.1, 0.2, 0.3],
        )
        dumped = original.model_dump()
        reparsed = Entity(**dumped)

        assert reparsed.canonical_name == original.canonical_name
        assert reparsed.entity_type == original.entity_type
        assert reparsed.type_tags == ["Person", "Researcher"]
        assert reparsed.primary_type == "Researcher"
        assert reparsed.confidence == pytest.approx(0.87)
        assert reparsed.source_documents == ["source:abc", "source:def"]
        assert reparsed.properties == original.properties
        assert reparsed.embedding == [0.1, 0.2, 0.3]

    def test_confidence_clamped(self):
        """confidence must be in [0, 1] — Pydantic enforces the bounds."""
        with pytest.raises(Exception):  # ValidationError
            Entity(canonical_name="X", entity_type="Org", confidence=1.5, embedding=[])
        with pytest.raises(Exception):
            Entity(canonical_name="X", entity_type="Org", confidence=-0.1, embedding=[])

    def test_legacy_none_fields_coerced(self):
        """type_tags / source_documents / properties tolerate None on parse."""
        # Simulate a row from before migration 44 where the DB returns NONE.
        ent = Entity(
            canonical_name="LegacyRow",
            entity_type="Org",
            embedding=[],
            type_tags=None,  # type: ignore[arg-type]
            source_documents=None,  # type: ignore[arg-type]
            properties=None,  # type: ignore[arg-type]
            provenance_chain=None,  # type: ignore[arg-type]
        )
        assert ent.type_tags == []
        assert ent.source_documents == []
        assert ent.properties == {}
        assert ent.provenance_chain == []

    def test_primary_type_can_be_unset(self):
        """primary_type stays None when not provided (matches DB ``option<string>``)."""
        ent = Entity(canonical_name="X", entity_type="Org", embedding=[])
        assert ent.primary_type is None

    def test_to_dict_excludes_none_by_default(self):
        """Inherited ``to_dict`` strips None fields for cleaner persistence."""
        ent = Entity(canonical_name="X", entity_type="Org", embedding=[])
        data = ent.to_dict()
        assert "primary_type" not in data  # None excluded
        assert data["canonical_name"] == "X"


class TestRelationModel:
    def test_minimal_construction(self):
        rel = Relation(relation_type="WORKS_AT")
        assert rel.relation_type == "WORKS_AT"
        assert rel.confidence == 1.0
        assert rel.status == "active"
        assert rel.source_documents == []
        assert rel.properties == {}

    def test_table_name(self):
        assert Relation.table_name == "relation"

    def test_model_dump_roundtrip(self):
        original = Relation(
            in_entity="entity:abc",
            out_entity="entity:def",
            relation_type="MEMBER_OF",
            confidence=0.71,
            source_documents=["source:doc1"],
            properties={"role": "co-author"},
        )
        reparsed = Relation(**original.model_dump())
        assert reparsed.relation_type == "MEMBER_OF"
        assert reparsed.confidence == pytest.approx(0.71)
        assert reparsed.in_entity == "entity:abc"
        assert reparsed.out_entity == "entity:def"
        assert reparsed.properties == {"role": "co-author"}

    def test_constructs_from_db_row_with_in_out_keys(self):
        """A raw SurrealDB RELATE row uses ``in`` / ``out`` (Python keywords).

        The ``Field(alias="in")`` / ``Field(alias="out")`` wiring lets us
        construct directly from such a dict — important so the repository
        can pass ``Relation(**row)`` without manually re-keying.
        """
        db_row = {
            "id": "relation:xyz",
            "in": "entity:abc",
            "out": "entity:def",
            "relation_type": "PART_OF",
            "confidence": 0.9,
            "source_documents": ["source:doc1"],
            "properties": {"since": "2024"},
        }
        rel = Relation(**db_row)
        assert rel.in_entity == "entity:abc"
        assert rel.out_entity == "entity:def"
        assert rel.relation_type == "PART_OF"
        assert rel.confidence == pytest.approx(0.9)
        assert rel.source_documents == ["source:doc1"]
        assert rel.properties == {"since": "2024"}
