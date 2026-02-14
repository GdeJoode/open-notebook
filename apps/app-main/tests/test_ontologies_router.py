"""Tests for the ontologies router."""

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.ontologies import router
from app_main.dependencies import get_ontology_service
from app_main.services.ontology_service import OntologyService


def _make_app(svc):
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_ontology_service] = lambda: svc
    return TestClient(app)


_SUMMARY = {
    "name": "general",
    "version": "1.0.0",
    "description": "General ontology",
    "entity_type_count": 5,
    "relationship_type_count": 3,
    "concept_count": 2,
}

_DETAIL = {
    **_SUMMARY,
    "authors": ["test"],
    "extends": None,
    "entity_types": [
        {
            "name": "Person",
            "description": "A person",
            "extraction_hints": ["names"],
            "examples": ["John Doe"],
            "aliases": None,
            "parent_type": None,
            "properties": [
                {
                    "name": "name",
                    "description": "Full name",
                    "data_type": "string",
                    "required": True,
                    "examples": None,
                }
            ],
        }
    ],
    "relationship_types": [
        {
            "name": "works_for",
            "description": "Employment relation",
            "domain": ["Person"],
            "range": ["Organization"],
            "cardinality": "many-to-one",
            "symmetric": False,
            "transitive": False,
            "inverse": "employs",
            "extraction_hints": None,
        }
    ],
    "concepts": [
        {
            "name": "Leadership",
            "description": "Leadership concept",
            "related_concepts": None,
            "examples": None,
        }
    ],
}


class TestListOntologies:

    def test_list_returns_summaries(self):
        svc = AsyncMock(spec=OntologyService)
        svc.list_ontologies.return_value = [_SUMMARY]
        client = _make_app(svc)

        resp = client.get("/api/ontologies")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "general"

    def test_list_empty(self):
        svc = AsyncMock(spec=OntologyService)
        svc.list_ontologies.return_value = []
        client = _make_app(svc)

        resp = client.get("/api/ontologies")

        assert resp.status_code == 200
        assert resp.json() == []


class TestGetOntology:

    def test_get_found(self):
        svc = AsyncMock(spec=OntologyService)
        svc.get_ontology.return_value = _DETAIL
        client = _make_app(svc)

        resp = client.get("/api/ontologies/general")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "general"
        assert len(data["entity_types"]) == 1
        assert len(data["relationship_types"]) == 1

    def test_get_not_found(self):
        svc = AsyncMock(spec=OntologyService)
        svc.get_ontology.return_value = None
        client = _make_app(svc)

        resp = client.get("/api/ontologies/missing")

        assert resp.status_code == 404


class TestGetEntityTypes:

    def test_entity_types(self):
        svc = AsyncMock(spec=OntologyService)
        svc.get_entity_types.return_value = _DETAIL["entity_types"]
        client = _make_app(svc)

        resp = client.get("/api/ontologies/general/entity-types")

        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["name"] == "Person"

    def test_entity_types_not_found(self):
        svc = AsyncMock(spec=OntologyService)
        svc.get_entity_types.return_value = None
        client = _make_app(svc)

        resp = client.get("/api/ontologies/missing/entity-types")

        assert resp.status_code == 404


class TestGetRelationshipTypes:

    def test_relationship_types(self):
        svc = AsyncMock(spec=OntologyService)
        svc.get_relationship_types.return_value = _DETAIL["relationship_types"]
        client = _make_app(svc)

        resp = client.get("/api/ontologies/general/relationship-types")

        assert resp.status_code == 200
        assert len(resp.json()) == 1
        assert resp.json()[0]["name"] == "works_for"

    def test_relationship_types_not_found(self):
        svc = AsyncMock(spec=OntologyService)
        svc.get_relationship_types.return_value = None
        client = _make_app(svc)

        resp = client.get("/api/ontologies/missing/relationship-types")

        assert resp.status_code == 404
