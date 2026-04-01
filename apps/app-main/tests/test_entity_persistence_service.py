"""Tests for EntityPersistenceService."""

from unittest.mock import AsyncMock, patch

import pytest

from app_main.services.entity_persistence_service import EntityPersistenceService


class TestPersistFilteredResult:
    @pytest.mark.asyncio
    async def test_upserts_entities(self):
        svc = EntityPersistenceService()

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_query:
            result = await svc.persist_filtered_result(
                source_id="source:1",
                entities=[
                    {"text": "BZK", "label": "ORG", "confidence": 0.9, "properties": {}},
                    {"text": "EZK", "label": "ORG", "confidence": 0.8, "properties": {}},
                ],
                relations=[],
            )

        assert result["entities_upserted"] == 2
        assert result["relations_created"] == 0
        # Each entity triggers one upsert query
        assert mock_query.call_count == 2

    @pytest.mark.asyncio
    async def test_creates_relations(self):
        svc = EntityPersistenceService()

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_query:
            result = await svc.persist_filtered_result(
                source_id="source:1",
                entities=[],
                relations=[
                    {
                        "source_entity": "BZK",
                        "target_entity": "EZK",
                        "relation_type": "RELATED",
                        "confidence": 0.7,
                        "properties": {},
                    },
                ],
            )

        assert result["relations_created"] == 1

    @pytest.mark.asyncio
    async def test_skips_empty_entity_text(self):
        svc = EntityPersistenceService()

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_query:
            result = await svc.persist_filtered_result(
                source_id="source:1",
                entities=[
                    {"text": "", "label": "ORG", "confidence": 0.9, "properties": {}},
                    {"text": "  ", "label": "ORG", "confidence": 0.8, "properties": {}},
                ],
                relations=[],
            )

        assert result["entities_upserted"] == 0

    @pytest.mark.asyncio
    async def test_skips_empty_relation_entities(self):
        svc = EntityPersistenceService()

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await svc.persist_filtered_result(
                source_id="source:1",
                entities=[],
                relations=[
                    {"source_entity": "", "target_entity": "B", "relation_type": "X",
                     "confidence": 0.5, "properties": {}},
                ],
            )

        assert result["relations_created"] == 0

    @pytest.mark.asyncio
    async def test_merge_groups_stored_in_properties(self):
        svc = EntityPersistenceService()

        queries_called = []

        async def capture_query(query, params=None, config=None):
            queries_called.append((query, params))
            return []

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            side_effect=capture_query,
        ):
            await svc.persist_filtered_result(
                source_id="source:1",
                entities=[
                    {"text": "BZK", "label": "ORG", "confidence": 0.9, "properties": {}},
                ],
                relations=[],
                merge_groups=[["BZK", "Binnenlandse Zaken"]],
            )

        # Check that merged_from was included in the properties
        upsert_params = queries_called[0][1]
        assert "merged_from" in upsert_params["properties"]
        assert upsert_params["properties"]["merged_from"] == ["BZK", "Binnenlandse Zaken"]

    @pytest.mark.asyncio
    async def test_excludes_embedding_from_stored_properties(self):
        svc = EntityPersistenceService()

        queries_called = []

        async def capture_query(query, params=None, config=None):
            queries_called.append((query, params))
            return []

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            side_effect=capture_query,
        ):
            await svc.persist_filtered_result(
                source_id="source:1",
                entities=[
                    {
                        "text": "BZK",
                        "label": "ORG",
                        "confidence": 0.9,
                        "properties": {"embedding": [0.1, 0.2], "custom_key": "value"},
                    },
                ],
                relations=[],
            )

        upsert_params = queries_called[0][1]
        assert "embedding" not in upsert_params["properties"]
        assert upsert_params["properties"]["custom_key"] == "value"

    @pytest.mark.asyncio
    async def test_handles_query_failure_gracefully(self):
        svc = EntityPersistenceService()

        async def failing_query(query, params=None, config=None):
            raise RuntimeError("DB connection failed")

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            side_effect=failing_query,
        ):
            # Should not raise — logs warning and continues
            result = await svc.persist_filtered_result(
                source_id="source:1",
                entities=[
                    {"text": "BZK", "label": "ORG", "confidence": 0.9, "properties": {}},
                ],
                relations=[
                    {"source_entity": "A", "target_entity": "B",
                     "relation_type": "X", "confidence": 0.5, "properties": {}},
                ],
            )

        # Failures are logged, counts stay at 0
        assert result["entities_upserted"] == 0
        assert result["relations_created"] == 0
