"""Tests for the ontology service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app_main.services.ontology_service import OntologyService


def _make_ontology_mock():
    """Create a mock Ontology with realistic attributes."""
    ontology = MagicMock()
    metadata = MagicMock()
    metadata.name = "general"
    metadata.version = "1.0.0"
    metadata.description = "General purpose ontology"
    metadata.authors = ["test"]
    metadata.extends = None
    ontology.metadata = metadata

    # Entity types
    prop = MagicMock()
    prop.name = "name"
    prop.description = "Full name"
    prop.data_type = MagicMock(value="string")
    prop.validation = MagicMock(required=True)
    prop.examples = None

    entity_type = MagicMock()
    entity_type.name = "Person"
    entity_type.description = "A person"
    entity_type.extraction_hints = ["names", "people"]
    entity_type.examples = ["John Doe"]
    entity_type.aliases = None
    entity_type.parent_type = None
    entity_type.properties = [prop]

    ontology.entity_types = {"Person": entity_type}

    # Relationship types
    rel = MagicMock()
    rel.name = "works_for"
    rel.description = "Employment"
    rel.domain = ["Person"]
    rel.range = ["Organization"]
    rel.cardinality = MagicMock(value="many-to-one")
    rel.symmetric = False
    rel.transitive = False
    rel.inverse = "employs"
    rel.extraction_hints = None

    ontology.relationship_types = {"works_for": rel}

    # Concepts
    concept = MagicMock()
    concept.name = "Leadership"
    concept.description = "Leadership concept"
    concept.related_concepts = None
    concept.examples = None

    ontology.concepts = {"Leadership": concept}

    return ontology


class TestListOntologies:

    @pytest.mark.asyncio
    async def test_list_returns_summaries(self):
        manager = AsyncMock()
        manager.list_ontologies.return_value = ["general"]
        manager.get_ontology.return_value = _make_ontology_mock()
        svc = OntologyService(manager=manager)

        result = await svc.list_ontologies()

        assert len(result) == 1
        assert result[0]["name"] == "general"
        assert result[0]["entity_type_count"] == 1

    @pytest.mark.asyncio
    async def test_list_skips_none(self):
        manager = AsyncMock()
        manager.list_ontologies.return_value = ["general", "missing"]
        manager.get_ontology.side_effect = [_make_ontology_mock(), None]
        svc = OntologyService(manager=manager)

        result = await svc.list_ontologies()

        assert len(result) == 1


class TestGetOntology:

    @pytest.mark.asyncio
    async def test_get_returns_detail(self):
        manager = AsyncMock()
        manager.get_ontology.return_value = _make_ontology_mock()
        svc = OntologyService(manager=manager)

        result = await svc.get_ontology("general")

        assert result is not None
        assert result["name"] == "general"
        assert len(result["entity_types"]) == 1
        assert len(result["relationship_types"]) == 1
        assert len(result["concepts"]) == 1

    @pytest.mark.asyncio
    async def test_get_returns_none(self):
        manager = AsyncMock()
        manager.get_ontology.return_value = None
        svc = OntologyService(manager=manager)

        result = await svc.get_ontology("missing")

        assert result is None


class TestGetEntityTypes:

    @pytest.mark.asyncio
    async def test_returns_types(self):
        manager = AsyncMock()
        manager.get_ontology.return_value = _make_ontology_mock()
        svc = OntologyService(manager=manager)

        result = await svc.get_entity_types("general")

        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "Person"
        assert result[0]["properties"][0]["name"] == "name"

    @pytest.mark.asyncio
    async def test_returns_none_for_missing(self):
        manager = AsyncMock()
        manager.get_ontology.return_value = None
        svc = OntologyService(manager=manager)

        result = await svc.get_entity_types("missing")

        assert result is None


class TestGetRelationshipTypes:

    @pytest.mark.asyncio
    async def test_returns_types(self):
        manager = AsyncMock()
        manager.get_ontology.return_value = _make_ontology_mock()
        svc = OntologyService(manager=manager)

        result = await svc.get_relationship_types("general")

        assert result is not None
        assert len(result) == 1
        assert result[0]["name"] == "works_for"
        assert result[0]["domain"] == ["Person"]
