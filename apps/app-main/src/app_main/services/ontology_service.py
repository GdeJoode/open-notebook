"""
Ontology service - business logic for ontology browsing.

Thin wrapper around the OntologyManager singleton that serializes
Ontology models into API-friendly dictionaries.
"""

from typing import Any, Dict, List, Optional

from loguru import logger
from ontology_manager.manager import OntologyManager, get_ontology_manager
from ontology_manager.schema import Ontology


class OntologyService:
    """Service for browsing ontology definitions."""

    def __init__(self, manager: Optional[OntologyManager] = None):
        self._manager = manager or get_ontology_manager()

    async def list_ontologies(self) -> List[Dict[str, Any]]:
        """List all ontologies with summary info."""
        names = await self._manager.list_ontologies()
        summaries = []
        for name in names:
            ontology = await self._manager.get_ontology(name)
            if ontology:
                summaries.append(_ontology_summary(ontology))
        return summaries

    async def get_ontology(self, name: str) -> Optional[Dict[str, Any]]:
        """Get full ontology detail."""
        ontology = await self._manager.get_ontology(name)
        if not ontology:
            return None
        return _ontology_detail(ontology)

    async def get_entity_types(self, name: str) -> Optional[List[Dict[str, Any]]]:
        """Get entity types for an ontology."""
        ontology = await self._manager.get_ontology(name)
        if not ontology:
            return None
        return [_entity_type_dict(et) for et in ontology.entity_types.values()]

    async def get_relationship_types(self, name: str) -> Optional[List[Dict[str, Any]]]:
        """Get relationship types for an ontology."""
        ontology = await self._manager.get_ontology(name)
        if not ontology:
            return None
        return [_relationship_type_dict(rt) for rt in ontology.relationship_types.values()]


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------

def _ontology_summary(o: Ontology) -> Dict[str, Any]:
    return {
        "name": o.metadata.name,
        "version": o.metadata.version,
        "description": o.metadata.description,
        "entity_type_count": len(o.entity_types),
        "relationship_type_count": len(o.relationship_types),
        "concept_count": len(o.concepts),
    }


def _ontology_detail(o: Ontology) -> Dict[str, Any]:
    return {
        **_ontology_summary(o),
        "authors": o.metadata.authors,
        "extends": o.metadata.extends,
        "entity_types": [_entity_type_dict(et) for et in o.entity_types.values()],
        "relationship_types": [
            _relationship_type_dict(rt) for rt in o.relationship_types.values()
        ],
        "concepts": [
            {
                "name": c.name,
                "description": c.description,
                "related_concepts": c.related_concepts,
                "examples": c.examples,
            }
            for c in o.concepts.values()
        ],
    }


def _entity_type_dict(et) -> Dict[str, Any]:
    return {
        "name": et.name,
        "description": et.description,
        "extraction_hints": et.extraction_hints,
        "examples": et.examples,
        "aliases": et.aliases,
        "parent_type": et.parent_type,
        "properties": [
            {
                "name": p.name,
                "description": p.description,
                "data_type": p.data_type.value,
                "required": p.validation.required if p.validation else False,
                "examples": p.examples,
            }
            for p in et.properties
        ],
    }


def _relationship_type_dict(rt) -> Dict[str, Any]:
    return {
        "name": rt.name,
        "description": rt.description,
        "domain": rt.domain,
        "range": rt.range,
        "cardinality": rt.cardinality.value,
        "symmetric": rt.symmetric,
        "transitive": rt.transitive,
        "inverse": rt.inverse,
        "extraction_hints": rt.extraction_hints,
    }
