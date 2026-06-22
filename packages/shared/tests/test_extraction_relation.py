"""Tests for the ExtractedRelation endpoint-type fields (K.7a).

``source_type`` / ``target_type`` are additive, default ``None`` for
back-compat. These tests pin both the default (existing callers unaffected)
and the populated form (the K.7a disambiguation seam), and that the fields
survive ``model_dump`` (the persist path round-trips relations as dicts).
"""

from shared.models.extraction import ExtractedRelation


def test_relation_endpoint_types_default_none():
    """An ExtractedRelation built the old way carries no endpoint types."""
    rel = ExtractedRelation(
        source_entity="BZK",
        target_entity="Klimaatwet",
        relation_type="SIGNS",
    )
    assert rel.source_type is None
    assert rel.target_type is None


def test_relation_endpoint_types_can_be_set():
    """The K.7a seam: endpoint types disambiguate cross-type homographs."""
    rel = ExtractedRelation(
        source_entity="BZK",
        target_entity="Klimaatwet",
        relation_type="SIGNS",
        source_type="organization",
        target_type="legislation",
    )
    assert rel.source_type == "organization"
    assert rel.target_type == "legislation"


def test_relation_endpoint_types_survive_model_dump():
    """The persist path round-trips relations via model_dump → dict; the new
    fields must be present so the persist seam can read them back."""
    rel = ExtractedRelation(
        source_entity="BZK",
        target_entity="Klimaatwet",
        relation_type="SIGNS",
        source_type="organization",
    )
    dumped = rel.model_dump()
    assert dumped["source_type"] == "organization"
    assert dumped["target_type"] is None
