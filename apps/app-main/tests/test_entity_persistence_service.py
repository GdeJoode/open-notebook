"""Tests for EntityPersistenceService.

Phase B.1a routed entity upserts through ``EntityRepository.upsert_entity``
(see ``apps/app-main/src/app_main/services/entity_persistence_service.py``
docstring). The tests here mock the injected repository so we still assert
behavior at the service boundary without booting SurrealDB.

Relation creation and match-candidate persistence still call
``execute_query`` directly at the service-module level, so those tests
keep mocking ``app_main.services.entity_persistence_service.execute_query``.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app_main.services.entity_persistence_service import EntityPersistenceService


def _make_service_with_mock_repo() -> tuple[EntityPersistenceService, AsyncMock]:
    """Return (service, mock_upsert_entity) — entity writes are intercepted at the repo."""
    mock_repo = MagicMock()
    mock_repo.upsert_entity = AsyncMock(return_value="entity:fake-id")
    svc = EntityPersistenceService(entity_repository=mock_repo)
    return svc, mock_repo.upsert_entity


class TestPersistFilteredResult:
    @pytest.mark.asyncio
    async def test_upserts_entities(self):
        svc, mock_upsert = _make_service_with_mock_repo()

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            new_callable=AsyncMock,
            return_value=[],
        ):
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
        # Each entity triggers one upsert_entity call
        assert mock_upsert.call_count == 2

    @pytest.mark.asyncio
    async def test_creates_relations(self):
        svc, _mock_upsert = _make_service_with_mock_repo()

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
        # Relation path still hits execute_query (one RELATE statement)
        assert mock_query.call_count == 1

    @pytest.mark.asyncio
    async def test_skips_empty_entity_text(self):
        svc, mock_upsert = _make_service_with_mock_repo()

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await svc.persist_filtered_result(
                source_id="source:1",
                entities=[
                    {"text": "", "label": "ORG", "confidence": 0.9, "properties": {}},
                    {"text": "  ", "label": "ORG", "confidence": 0.8, "properties": {}},
                ],
                relations=[],
            )

        assert result["entities_upserted"] == 0
        assert mock_upsert.call_count == 0

    @pytest.mark.asyncio
    async def test_skips_empty_relation_entities(self):
        svc, _mock_upsert = _make_service_with_mock_repo()

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
        """Merged-from group is recorded inside the Entity's properties bag."""
        svc, mock_upsert = _make_service_with_mock_repo()

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await svc.persist_filtered_result(
                source_id="source:1",
                entities=[
                    {"text": "BZK", "label": "ORG", "confidence": 0.9, "properties": {}},
                ],
                relations=[],
                merge_groups=[["BZK", "Binnenlandse Zaken"]],
            )

        # First positional arg to upsert_entity is the Entity model
        called_entity = mock_upsert.call_args.args[0]
        assert "merged_from" in called_entity.properties
        assert called_entity.properties["merged_from"] == ["BZK", "Binnenlandse Zaken"]

    @pytest.mark.asyncio
    async def test_excludes_embedding_from_stored_properties(self):
        """Embedding in input ``properties`` is stripped before persisting."""
        svc, mock_upsert = _make_service_with_mock_repo()

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            new_callable=AsyncMock,
            return_value=[],
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

        called_entity = mock_upsert.call_args.args[0]
        assert "embedding" not in called_entity.properties
        assert called_entity.properties["custom_key"] == "value"
        # And the Entity.embedding field is an explicit empty list (the SCHEMAFULL
        # `entity.embedding` column has no DB default — see migration 39 line 30).
        assert called_entity.embedding == []

    @pytest.mark.asyncio
    async def test_uses_canonical_schema_field_names(self):
        """B.1a alignment guard: Entity passed to upsert uses migration-39 field names."""
        svc, mock_upsert = _make_service_with_mock_repo()

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await svc.persist_filtered_result(
                source_id="source:doc1",
                entities=[
                    {"text": "BZK", "label": "ORG", "confidence": 0.9, "properties": {}},
                ],
                relations=[],
            )

        ent = mock_upsert.call_args.args[0]
        # canonical_name + source_documents, NOT legacy name + source_ids
        assert ent.canonical_name == "BZK"
        assert ent.entity_type == "ORG"
        assert ent.source_documents == ["source:doc1"]
        assert ent.confidence == 0.9

    @pytest.mark.asyncio
    async def test_handles_query_failure_gracefully(self):
        """Repository raising on upsert AND relation execute_query failing — both swallowed."""

        async def failing_upsert(_entity):
            raise RuntimeError("DB connection failed")

        mock_repo = MagicMock()
        mock_repo.upsert_entity = AsyncMock(side_effect=failing_upsert)
        svc = EntityPersistenceService(entity_repository=mock_repo)

        async def failing_query(query, params=None, config=None):
            raise RuntimeError("DB connection failed")

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            side_effect=failing_query,
        ):
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
