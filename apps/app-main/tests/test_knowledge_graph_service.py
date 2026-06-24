"""Tests for the knowledge graph service."""

from unittest.mock import AsyncMock

import pytest

from app_main.services.knowledge_graph_service import KnowledgeGraphService


@pytest.fixture
def entity_repo():
    repo = AsyncMock()
    repo.list_entities = AsyncMock(return_value=[])
    repo.count_entities = AsyncMock(return_value=0)
    repo.get_entity_detail = AsyncMock(return_value=None)
    repo.get_entity_types_summary = AsyncMock(return_value=[])
    repo.search_entities = AsyncMock(return_value=[])
    repo.get_all_entities_and_relations = AsyncMock(
        return_value={"nodes": [], "edges": []}
    )
    # Q.5 signals join: the service annotates list/detail rows with effective
    # degree via this batched relation helper. Default to "no edges" so the
    # join contributes degree 0 unless a test overrides it.
    repo.effective_degree_for_entities = AsyncMock(return_value={})
    return repo


class TestListEntities:

    @pytest.mark.asyncio
    async def test_delegates_to_repo(self, entity_repo):
        entity_repo.list_entities.return_value = [
            {"id": "entity:1", "name": "Test", "entity_type": "Person", "weight": 1}
        ]
        svc = KnowledgeGraphService(entity_repo=entity_repo)

        result = await svc.list_entities(limit=10, offset=5, entity_type="Person")

        entity_repo.list_entities.assert_called_once_with(
            limit=10, offset=5, entity_type="Person", status=None
        )
        assert len(result) == 1

    @pytest.mark.asyncio
    async def test_threads_status_filter(self, entity_repo):
        """Q.5: the optional status filter reaches the repository."""
        svc = KnowledgeGraphService(entity_repo=entity_repo)

        await svc.list_entities(limit=10, offset=0, status="reference")

        entity_repo.list_entities.assert_called_once_with(
            limit=10, offset=0, entity_type=None, status="reference"
        )

    @pytest.mark.asyncio
    async def test_attaches_degree_and_doc_count(self, entity_repo):
        """Q.5 AC2: page rows carry structural_degree + doc_count."""
        entity_repo.list_entities.return_value = [
            {
                "id": "entity:1",
                "name": "Test",
                "entity_type": "Person",
                "weight": 1,
                "source_documents": ["source:a", "source:b", "source:a"],
            }
        ]
        entity_repo.effective_degree_for_entities.return_value = {"entity:1": 4}
        svc = KnowledgeGraphService(entity_repo=entity_repo)

        result = await svc.list_entities(limit=10, offset=0)

        assert result[0]["structural_degree"] == 4
        # Distinct docs: source:a + source:b == 2 (the duplicate collapses).
        assert result[0]["doc_count"] == 2


class TestCountEntities:

    @pytest.mark.asyncio
    async def test_count(self, entity_repo):
        entity_repo.count_entities.return_value = 42
        svc = KnowledgeGraphService(entity_repo=entity_repo)

        result = await svc.count_entities(entity_type="Person")

        assert result == 42
        entity_repo.count_entities.assert_called_once_with(
            entity_type="Person", status=None
        )

    @pytest.mark.asyncio
    async def test_count_threads_status(self, entity_repo):
        entity_repo.count_entities.return_value = 7
        svc = KnowledgeGraphService(entity_repo=entity_repo)

        result = await svc.count_entities(status="reference")

        assert result == 7
        entity_repo.count_entities.assert_called_once_with(
            entity_type=None, status="reference"
        )


class TestGetEntity:

    @pytest.mark.asyncio
    async def test_get_found(self, entity_repo):
        entity_repo.get_entity_detail.return_value = {
            "id": "entity:1",
            "name": "Test",
            "entity_type": "Person",
            "relations": [],
        }
        svc = KnowledgeGraphService(entity_repo=entity_repo)

        result = await svc.get_entity("entity:1")

        assert result is not None
        assert result["name"] == "Test"

    @pytest.mark.asyncio
    async def test_get_not_found(self, entity_repo):
        entity_repo.get_entity_detail.return_value = None
        svc = KnowledgeGraphService(entity_repo=entity_repo)

        result = await svc.get_entity("entity:missing")

        assert result is None


class TestGetEntityTypesSummary:

    @pytest.mark.asyncio
    async def test_summary(self, entity_repo):
        entity_repo.get_entity_types_summary.return_value = [
            {"entity_type": "Person", "count": 10}
        ]
        svc = KnowledgeGraphService(entity_repo=entity_repo)

        result = await svc.get_entity_types_summary()

        assert len(result) == 1
        assert result[0]["count"] == 10


class TestSearchEntities:

    @pytest.mark.asyncio
    async def test_search(self, entity_repo):
        entity_repo.search_entities.return_value = [
            {"id": "entity:1", "name": "John"}
        ]
        svc = KnowledgeGraphService(entity_repo=entity_repo)

        result = await svc.search_entities("John", limit=5)

        entity_repo.search_entities.assert_called_once_with(query="John", limit=5)
        assert len(result) == 1


class TestGetGraphData:

    @pytest.mark.asyncio
    async def test_graph_data(self, entity_repo):
        entity_repo.get_all_entities_and_relations.return_value = {
            "nodes": [
                {"id": "e:1", "name": "John", "entity_type": "Person", "weight": 3},
                {"id": "e:2", "name": "Acme", "entity_type": "Org", "weight": 1},
            ],
            "edges": [
                {"source": "e:1", "target": "e:2", "relation_type": "works_for"}
            ],
        }
        svc = KnowledgeGraphService(entity_repo=entity_repo)

        result = await svc.get_graph_data(limit=100)

        assert len(result["nodes"]) == 2
        assert len(result["edges"]) == 1
        # Verify node format
        assert result["nodes"][0]["label"] == "John"
        # Verify edge format
        assert result["edges"][0]["source"] == "e:1"

    @pytest.mark.asyncio
    async def test_graph_data_empty(self, entity_repo):
        svc = KnowledgeGraphService(entity_repo=entity_repo)

        result = await svc.get_graph_data()

        assert result["nodes"] == []
        assert result["edges"] == []
