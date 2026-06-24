"""Tests for the knowledge graph router."""

from unittest.mock import AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app_main.api.routers.knowledge_graph import router
from app_main.dependencies import get_knowledge_graph_service
from app_main.services.knowledge_graph_service import KnowledgeGraphService


def _make_app(svc):
    app = FastAPI()
    app.include_router(router, prefix="/api")
    app.dependency_overrides[get_knowledge_graph_service] = lambda: svc
    return TestClient(app)


_ENTITY = {
    "id": "entity:abc",
    "name": "John Doe",
    "entity_type": "Person",
    "weight": 5,
}

_ENTITY_DETAIL = {
    **_ENTITY,
    "relations": [
        {
            "id": "relation:1",
            "source": "entity:abc",
            "target": "entity:xyz",
            "relation_type": "works_for",
        }
    ],
}


class TestListEntities:

    def test_list_entities(self):
        svc = AsyncMock(spec=KnowledgeGraphService)
        svc.list_entities.return_value = [_ENTITY]
        svc.count_entities.return_value = 1
        client = _make_app(svc)

        resp = client.get("/api/knowledge-graph/entities")

        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert len(data["items"]) == 1

    def test_list_with_type_filter(self):
        svc = AsyncMock(spec=KnowledgeGraphService)
        svc.list_entities.return_value = [_ENTITY]
        svc.count_entities.return_value = 1
        client = _make_app(svc)

        resp = client.get("/api/knowledge-graph/entities?entity_type=Person")

        assert resp.status_code == 200
        svc.list_entities.assert_called_once_with(
            limit=50, offset=0, entity_type="Person", status=None
        )

    def test_list_with_pagination(self):
        svc = AsyncMock(spec=KnowledgeGraphService)
        svc.list_entities.return_value = []
        svc.count_entities.return_value = 100
        client = _make_app(svc)

        resp = client.get("/api/knowledge-graph/entities?limit=10&offset=20")

        assert resp.status_code == 200
        svc.list_entities.assert_called_once_with(
            limit=10, offset=20, entity_type=None, status=None
        )

    def test_list_with_status_filter(self):
        """Q.5: the status query param threads through to the service."""
        svc = AsyncMock(spec=KnowledgeGraphService)
        svc.list_entities.return_value = []
        svc.count_entities.return_value = 0
        client = _make_app(svc)

        resp = client.get("/api/knowledge-graph/entities?status=reference")

        assert resp.status_code == 200
        svc.list_entities.assert_called_once_with(
            limit=50, offset=0, entity_type=None, status="reference"
        )
        svc.count_entities.assert_called_once_with(
            entity_type=None, status="reference"
        )


class TestGetEntity:

    def test_get_found(self):
        svc = AsyncMock(spec=KnowledgeGraphService)
        svc.get_entity.return_value = _ENTITY_DETAIL
        client = _make_app(svc)

        resp = client.get("/api/knowledge-graph/entities/entity:abc")

        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "John Doe"
        assert len(data["relations"]) == 1

    def test_get_not_found(self):
        svc = AsyncMock(spec=KnowledgeGraphService)
        svc.get_entity.return_value = None
        client = _make_app(svc)

        resp = client.get("/api/knowledge-graph/entities/entity:missing")

        assert resp.status_code == 404


class TestGetEntityTypes:

    def test_entity_types_summary(self):
        svc = AsyncMock(spec=KnowledgeGraphService)
        svc.get_entity_types_summary.return_value = [
            {"entity_type": "Person", "count": 42},
            {"entity_type": "Organization", "count": 15},
        ]
        client = _make_app(svc)

        resp = client.get("/api/knowledge-graph/entity-types")

        assert resp.status_code == 200
        assert len(resp.json()) == 2


class TestGetGraph:

    def test_get_graph_data(self):
        svc = AsyncMock(spec=KnowledgeGraphService)
        svc.get_graph_data.return_value = {
            "nodes": [{"id": "e:1", "label": "John", "entity_type": "Person", "weight": 1}],
            "edges": [{"source": "e:1", "target": "e:2", "label": "works_for"}],
        }
        client = _make_app(svc)

        resp = client.get("/api/knowledge-graph/graph")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["nodes"]) == 1
        assert len(data["edges"]) == 1


class TestSearchEntities:

    def test_search(self):
        svc = AsyncMock(spec=KnowledgeGraphService)
        svc.search_entities.return_value = [_ENTITY]
        client = _make_app(svc)

        resp = client.get("/api/knowledge-graph/search?q=John")

        assert resp.status_code == 200
        assert len(resp.json()) == 1
        svc.search_entities.assert_called_once_with(query="John", limit=20)

    def test_search_requires_query(self):
        svc = AsyncMock(spec=KnowledgeGraphService)
        client = _make_app(svc)

        resp = client.get("/api/knowledge-graph/search")

        assert resp.status_code == 422  # Missing required query param
