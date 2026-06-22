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
    async def test_relation_endpoint_type_from_batch_entities(self):
        """K.7a: when both endpoints are in the batch's entities, the RELATE
        SELECT is type-filtered (the entity_type from the batch) so a
        cross-type homograph resolves to the type-correct entity."""
        svc, _mock_upsert = _make_service_with_mock_repo()

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_query:
            await svc.persist_filtered_result(
                source_id="source:1",
                entities=[
                    {"text": "BZK", "label": "ORG", "confidence": 0.9, "properties": {}},
                    {"text": "Klimaatwet", "label": "law", "confidence": 0.8, "properties": {}},
                ],
                relations=[
                    {
                        "source_entity": "BZK",
                        "target_entity": "Klimaatwet",
                        "relation_type": "SIGNS",
                        "confidence": 0.7,
                        "properties": {},
                    },
                ],
            )

        # The relation RELATE is the call carrying src_name.
        relate_call = next(
            c for c in mock_query.call_args_list
            if (c.args[1] if len(c.args) > 1 else {}).get("src_name") == "BZK"
        )
        sql, params = relate_call.args[0], relate_call.args[1]
        # Type filter present for BOTH endpoints (both came from the batch).
        assert "entity_type = $src_type" in sql
        assert "entity_type = $tgt_type" in sql
        # Normalized types: ORG → organization, law → legislation.
        assert params["src_type"] == "organization"
        assert params["tgt_type"] == "legislation"

    @pytest.mark.asyncio
    async def test_relation_prefers_relation_carried_type(self):
        """K.7a: a type the relation already carries (source_type/target_type)
        wins over the same-batch entity map."""
        svc, _mock_upsert = _make_service_with_mock_repo()

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_query:
            await svc.persist_filtered_result(
                source_id="source:1",
                entities=[],
                relations=[
                    {
                        "source_entity": "BZK",
                        "target_entity": "Klimaatwet",
                        "relation_type": "SIGNS",
                        "source_type": "organization",
                        "target_type": "legislation",
                        "confidence": 0.7,
                        "properties": {},
                    },
                ],
            )

        relate_call = next(
            c for c in mock_query.call_args_list
            if (c.args[1] if len(c.args) > 1 else {}).get("src_name") == "BZK"
        )
        params = relate_call.args[1]
        assert params["src_type"] == "organization"
        assert params["tgt_type"] == "legislation"

    @pytest.mark.asyncio
    async def test_relation_unknown_endpoint_falls_back_to_name_only(self):
        """K.7a regression guard: a cross-batch endpoint (not in this batch's
        entities, no carried type) resolves by NAME ONLY — the original
        ``LIMIT 1`` behaviour, zero regression for today's working cases."""
        svc, _mock_upsert = _make_service_with_mock_repo()

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            new_callable=AsyncMock,
            return_value=[],
        ) as mock_query:
            result = await svc.persist_filtered_result(
                source_id="source:1",
                entities=[],  # no entities in batch → no type map
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
        relate_call = next(
            c for c in mock_query.call_args_list
            if (c.args[1] if len(c.args) > 1 else {}).get("src_name") == "BZK"
        )
        sql, params = relate_call.args[0], relate_call.args[1]
        # No type filter on either endpoint — pure name-only resolution.
        assert "entity_type = $src_type" not in sql
        assert "entity_type = $tgt_type" not in sql
        assert params["src_type"] is None
        assert params["tgt_type"] is None

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
        # entity_type is normalized onto the canonical enum: "ORG" -> "organization"
        assert ent.entity_type == "organization"
        assert ent.source_documents == ["source:doc1"]
        assert ent.confidence == 0.9

    @pytest.mark.asyncio
    async def test_fully_failed_entity_batch_raises(self):
        """B.8a: when EVERY entity upsert fails, persist must RAISE — not report
        a silent success of 0 written. Previously this was swallowed (the bug
        that hid extraction failures)."""

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
            with pytest.raises(RuntimeError, match="wrote 0 of 1 entities"):
                await svc.persist_filtered_result(
                    source_id="source:1",
                    entities=[
                        {"text": "BZK", "label": "ORG", "confidence": 0.9, "properties": {}},
                    ],
                    relations=[],
                )

    @pytest.mark.asyncio
    async def test_partial_entity_failure_is_counted_not_raised(self):
        """B.8a: a partial failure (some succeed) is reported via entities_failed,
        not raised — only a fully-empty write is fatal."""
        call_count = {"n": 0}

        async def flaky_upsert(_entity):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("transient")
            return "entity:ok"

        mock_repo = MagicMock()
        mock_repo.upsert_entity = AsyncMock(side_effect=flaky_upsert)
        svc = EntityPersistenceService(entity_repository=mock_repo)

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

        assert result["entities_upserted"] == 1
        assert result["entities_failed"] == 1

    @pytest.mark.asyncio
    async def test_threads_extraction_method_and_model(self):
        """B.8a: extraction_method/model provenance must reach the persisted
        Entity instead of silently defaulting to 'llm'."""
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
                extraction_method="langextract",
                extraction_model="qwen2.5:14b-instruct-q5_K_M",
            )

        ent = mock_upsert.call_args.args[0]
        assert ent.extraction_method == "langextract"
        assert ent.properties["extraction_model"] == "qwen2.5:14b-instruct-q5_K_M"


# --- L.1: ontology->canonical bridge at the persist boundary -------------

from app_main.services.entity_persistence_service import (  # noqa: E402
    _ALLOWED_ENTITY_TYPES,
    _resolve_entity_type,
)
from ontology_manager.schema import (  # noqa: E402
    EntityTypeDefinition,
    Ontology,
    OntologyMetadata,
)


def _gov_schema() -> Ontology:
    """government ontology fragment with the live parent_type chains.

    AdministrativeArea / GovernmentOrganization are intentionally NOT loaded
    here (they live in policy.yaml in prod) — the bridge terminates on the
    base NAME.
    """
    return Ontology(
        metadata=OntologyMetadata(name="government", version="1.0"),
        entity_types={
            "Gemeente": EntityTypeDefinition(
                name="Gemeente", parent_type="AdministrativeArea"
            ),
            "Ministerie": EntityTypeDefinition(
                name="Ministerie", parent_type="GovernmentOrganization"
            ),
        },
    )


def _deals_schema() -> Ontology:
    return Ontology(
        metadata=OntologyMetadata(name="deals", version="1.0"),
        entity_types={
            "Deal": EntityTypeDefinition(name="Deal", parent_type="GovernmentService"),
            "RegioDeal": EntityTypeDefinition(name="RegioDeal", parent_type="Deal"),
        },
    )


def _instruments_schema() -> Ontology:
    return Ontology(
        metadata=OntologyMetadata(name="instruments", version="1.0"),
        entity_types={"Wet": EntityTypeDefinition(name="Wet", parent_type="Legislation")},
    )


def _policy_themes_schema() -> Ontology:
    return Ontology(
        metadata=OntologyMetadata(name="policy_themes", version="1.0"),
        entity_types={
            "BeleidsThema": EntityTypeDefinition(
                name="BeleidsThema", parent_type="Concept"
            )
        },
    )


class TestResolveEntityType:
    """Unit tests on the bridge-aware resolver used at the persist boundary."""

    def test_gemeente_bridges_and_preserves(self):
        res = _resolve_entity_type("Gemeente", [_gov_schema()])
        assert res.entity_type == "administrative_area"
        assert res.primary_type == "Gemeente"
        assert "Gemeente" in res.type_tags

    def test_ministerie_bridges(self):
        res = _resolve_entity_type("Ministerie", [_gov_schema()])
        assert res.entity_type == "government_organization"
        assert res.primary_type == "Ministerie"

    def test_regiodeal_includes_deal_parent(self):
        res = _resolve_entity_type("RegioDeal", [_deals_schema()])
        # interim canonical until L.3 introduces ``programme``
        assert res.entity_type == "creative_work"
        assert res.primary_type == "RegioDeal"
        assert "Deal" in res.type_tags

    def test_wet_bridges_to_legislation(self):
        res = _resolve_entity_type("Wet", [_instruments_schema()])
        assert res.entity_type == "legislation"
        assert res.primary_type == "Wet"

    def test_beleidsthema_bridges_to_topic(self):
        res = _resolve_entity_type("BeleidsThema", [_policy_themes_schema()])
        assert res.entity_type == "topic"
        assert res.primary_type == "BeleidsThema"

    def test_unknown_label_falls_through_to_alias_path(self):
        # Not in the applied ontology -> None from bridge -> alias/enum path.
        res = _resolve_entity_type("Quux", [_gov_schema()])
        assert res.entity_type == "other"
        assert res.primary_type is None
        assert res.type_tags == []

    def test_no_schemas_uses_alias_path(self):
        # Graceful degradation (AC7): no schemas -> alias/enum path, no crash.
        res = _resolve_entity_type("ORG", None)
        assert res.entity_type == "organization"
        assert res.primary_type is None

    def test_canonical_always_in_allowed_enum(self):
        # B.8 contract: every bridged canonical is a valid enum member.
        for label, schema in [
            ("Gemeente", _gov_schema()),
            ("Ministerie", _gov_schema()),
            ("RegioDeal", _deals_schema()),
            ("Wet", _instruments_schema()),
            ("BeleidsThema", _policy_themes_schema()),
        ]:
            res = _resolve_entity_type(label, [schema])
            assert res.entity_type in _ALLOWED_ENTITY_TYPES


class TestPersistStampsRichType:
    """The persist path stamps primary_type/type_tags + keeps raw_entity_type."""

    @pytest.mark.asyncio
    async def test_bridge_stamps_primary_type_and_type_tags(self):
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
                        "text": "Gemeente Groningen",
                        "label": "Gemeente",
                        "confidence": 0.9,
                        "properties": {},
                    }
                ],
                relations=[],
                applicable_schemas=[_gov_schema()],
            )

        ent = mock_upsert.call_args.args[0]
        assert ent.entity_type == "administrative_area"
        assert ent.primary_type == "Gemeente"
        assert "Gemeente" in ent.type_tags
        # raw label preserved for provenance / L.5 retro re-typing
        assert ent.properties["raw_entity_type"] == "Gemeente"

    @pytest.mark.asyncio
    async def test_no_schemas_leaves_rich_type_empty(self):
        # Without schemas the alias/enum path runs; rich-type slots stay empty.
        svc, mock_upsert = _make_service_with_mock_repo()

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            new_callable=AsyncMock,
            return_value=[],
        ):
            await svc.persist_filtered_result(
                source_id="source:1",
                entities=[
                    {"text": "BZK", "label": "ORG", "confidence": 0.9, "properties": {}}
                ],
                relations=[],
            )

        ent = mock_upsert.call_args.args[0]
        assert ent.entity_type == "organization"
        assert ent.primary_type is None
        assert ent.type_tags == []

    @pytest.mark.asyncio
    async def test_bridge_failure_degrades_not_crashes(self):
        # AC7: an unknown label with schemas present still persists (falls to
        # alias path), never raising.
        svc, mock_upsert = _make_service_with_mock_repo()

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            new_callable=AsyncMock,
            return_value=[],
        ):
            result = await svc.persist_filtered_result(
                source_id="source:1",
                entities=[
                    {"text": "Frobnicator", "label": "Quux", "confidence": 0.5, "properties": {}}
                ],
                relations=[],
                applicable_schemas=[_gov_schema()],
            )

        assert result["entities_upserted"] == 1
        ent = mock_upsert.call_args.args[0]
        assert ent.entity_type == "other"
