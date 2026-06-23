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

        # O.1: the relation path now resolves each endpoint id via
        # ``_resolve_endpoint_id`` (a SELECT) then RELATEs. A non-empty SELECT
        # return means both endpoints resolve so the edge is created.
        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            new_callable=AsyncMock,
            return_value=["entity:fake"],
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
        # Two endpoint lookups (src + tgt) + one RELATE.
        assert mock_query.call_count == 3

    @pytest.mark.asyncio
    async def test_relation_endpoint_type_from_batch_entities(self):
        """K.7a: when both endpoints are in the batch's entities, each endpoint
        lookup is type-filtered (the bridge-resolved entity_type from the batch)
        so a cross-type homograph resolves to the type-correct entity.

        O.1: the lookup is now a per-endpoint SELECT (``_resolve_endpoint_id``).
        We assert the FIRST (typed) SELECT for each endpoint carries the right
        ``etype``.
        """
        svc, _mock_upsert = _make_service_with_mock_repo()

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            new_callable=AsyncMock,
            return_value=["entity:fake"],  # both typed lookups hit
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

        # The typed endpoint SELECTs carry name + etype. ORG → organization,
        # law → legislation (via the bridge/alias path).
        typed_lookups = [
            c.args[1] for c in mock_query.call_args_list
            if "etype" in (c.args[1] if len(c.args) > 1 else {})
        ]
        by_name = {p["name"]: p["etype"] for p in typed_lookups}
        assert by_name.get("BZK") == "organization"
        assert by_name.get("Klimaatwet") == "legislation"

    @pytest.mark.asyncio
    async def test_relation_bridge_only_type_endpoint_resolves(self):
        """O.1: an endpoint whose entity type is a BRIDGE-only canonical (e.g.
        a label that bridges to ``topic``) must be typed via the bridge so the
        (name, type) lookup matches the persisted row.

        Pre-fix the relation side used ``_normalize_entity_type`` (alias-only),
        which for a bridge-only label resolves to ``other`` and diverged from
        the entity's bridge type → the typed lookup missed and the edge was
        skipped. Here the carried ``source_type`` bridges to ``topic``.
        """
        svc, _mock_upsert = _make_service_with_mock_repo()

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            new_callable=AsyncMock,
            return_value=["entity:fake"],
        ) as mock_query:
            await svc.persist_filtered_result(
                source_id="source:1",
                entities=[],
                relations=[
                    {
                        "source_entity": "Indicator X",
                        "target_entity": "Klimaatwet",
                        "relation_type": "MEASURES",
                        # raw ontology label that the alias map coarses to a NL
                        # theme → bridge-resolved canonical ``topic``.
                        "source_type": "BeleidsThema",
                        "target_type": "legislation",
                        "confidence": 0.7,
                        "properties": {},
                    },
                ],
                applicable_schemas=[_policy_themes_schema()],
            )

        typed_lookups = [
            c.args[1] for c in mock_query.call_args_list
            if "etype" in (c.args[1] if len(c.args) > 1 else {})
        ]
        by_name = {p["name"]: p["etype"] for p in typed_lookups}
        # The endpoint type is the BRIDGE canonical, not the alias-only ``other``.
        assert by_name.get("Indicator X") == "topic"
        assert by_name.get("Klimaatwet") == "legislation"

    @pytest.mark.asyncio
    async def test_relation_carried_type_wins_for_homograph_disambiguation(self):
        """K.7a: a relation-CARRIED source_type wins over the same-batch entity
        map, because it is the per-edge disambiguator for an in-batch homograph.

        ``type_by_name`` can only hold ONE type per name, so when a batch has a
        ``person`` BZK AND an ``organization`` BZK, only the carried type can
        tell the two edges apart. O.1: the carried type is now bridge-resolved
        (same as the entity), so the typed lookup still matches the persisted
        row.
        """
        svc, _mock_upsert = _make_service_with_mock_repo()

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            new_callable=AsyncMock,
            return_value=["entity:fake"],
        ) as mock_query:
            await svc.persist_filtered_result(
                source_id="source:1",
                entities=[
                    # batch map for BZK resolves to organization (last write wins)
                    {"text": "BZK", "label": "person", "confidence": 0.9, "properties": {}},
                    {"text": "BZK", "label": "ORG", "confidence": 0.9, "properties": {}},
                ],
                relations=[
                    {
                        "source_entity": "BZK",
                        "target_entity": "Klimaatwet",
                        "relation_type": "LEADS",
                        # carried type disambiguates THIS edge to the person BZK
                        "source_type": "person",
                        "target_type": "legislation",
                        "confidence": 0.7,
                        "properties": {},
                    },
                ],
            )

        typed_lookups = [
            c.args[1] for c in mock_query.call_args_list
            if "etype" in (c.args[1] if len(c.args) > 1 else {})
        ]
        by_name = {p["name"]: p["etype"] for p in typed_lookups}
        # Carried type wins: person, NOT the batch map's organization.
        assert by_name.get("BZK") == "person"

    @pytest.mark.asyncio
    async def test_relation_name_only_fallback_when_typed_miss(self):
        """O.1: when the typed lookup misses (entity exists under a different
        type) the endpoint resolves by NAME ONLY so the edge is not dropped.

        The mock returns [] for the typed SELECT (carrying ``etype``) and a fake
        id for the name-only SELECT — proving the fallback fires and the RELATE
        runs."""
        svc, _mock_upsert = _make_service_with_mock_repo()

        async def fake_query(query, params=None, config=None):
            params = params or {}
            if "etype" in params:
                return []  # typed lookup misses
            if "name" in params:
                return ["entity:fallback"]  # name-only hit
            return []  # RELATE

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            side_effect=fake_query,
        ) as mock_query:
            result = await svc.persist_filtered_result(
                source_id="source:1",
                entities=[
                    {"text": "BZK", "label": "ORG", "confidence": 0.9, "properties": {}},
                    {"text": "EZK", "label": "ORG", "confidence": 0.9, "properties": {}},
                ],
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
        # Both a typed and a name-only SELECT ran per endpoint, then a RELATE.
        had_typed = any(
            "etype" in (c.args[1] or {}) for c in mock_query.call_args_list
        )
        had_name_only = any(
            (c.args[1] or {}).get("name") and "etype" not in (c.args[1] or {})
            for c in mock_query.call_args_list
        )
        assert had_typed and had_name_only

    @pytest.mark.asyncio
    async def test_relation_unknown_endpoint_falls_back_to_name_only(self):
        """K.7a regression guard: a cross-batch endpoint (not in this batch's
        entities, no carried type) resolves by NAME ONLY — the original
        ``LIMIT 1`` behaviour, zero regression for today's working cases."""
        svc, _mock_upsert = _make_service_with_mock_repo()

        with patch(
            "app_main.services.entity_persistence_service.execute_query",
            new_callable=AsyncMock,
            return_value=["entity:fake"],
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
        # No type known for either endpoint → no typed SELECT ran (only
        # name-only lookups + the RELATE).
        assert not any(
            "etype" in (c.args[1] if len(c.args) > 1 else {})
            for c in mock_query.call_args_list
        )

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
        """P.1: the embedding is lifted OUT of ``properties`` onto the first-class
        ``Entity.embedding`` field (not dropped, not double-stored)."""
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
        # Not double-stored in the properties bag (saves the ~3KB vector there).
        assert "embedding" not in called_entity.properties
        assert called_entity.properties["custom_key"] == "value"

    @pytest.mark.asyncio
    async def test_embedding_persisted_on_first_class_field(self):
        """P.1 forward fix: the computed embedding reaches ``Entity.embedding``
        (not the historical ``[]``) so K.5's semantic dedup band has a vector.
        """
        svc, mock_upsert = _make_service_with_mock_repo()

        vector = [0.1, 0.2, 0.3]
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
                        "properties": {"embedding": vector},
                    },
                ],
                relations=[],
            )

        called_entity = mock_upsert.call_args.args[0]
        assert called_entity.embedding == vector
        assert "embedding" not in called_entity.properties

    @pytest.mark.asyncio
    async def test_missing_embedding_defaults_to_empty_list(self):
        """P.1: an entity without a vector persists ``embedding=[]`` gracefully —
        the dedup band degrades to fuzzy-only for it (no crash, no fabrication).
        """
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
            )

        called_entity = mock_upsert.call_args.args[0]
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
        # L.3: Deal/GovernmentService -> programme (was interim creative_work).
        assert res.entity_type == "programme"
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

    def test_unknown_label_falls_through_to_residual_fallback(self):
        # Not in the applied ontology -> None from bridge -> L.2 residual. L.2
        # changed the historical silent ``other``: a genuine-unknown label now
        # coarses to a default BUT preserves the rich label (anti-flattening).
        res = _resolve_entity_type("Quux", [_gov_schema()])
        assert res.entity_type == "concept"
        assert res.primary_type == "Quux"
        assert res.type_tags == ["Quux"]

    def test_no_schemas_uses_alias_path(self):
        # Graceful degradation (AC7): no schemas -> L.2 residual alias path, no
        # crash. L.2: an alias hit now PRESERVES the raw label in primary_type
        # (was empty under L.1-only); the coarse projection is still the enum
        # value.
        res = _resolve_entity_type("ORG", None)
        assert res.entity_type == "organization"
        assert res.primary_type == "ORG"

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


class TestL3ProgrammeTechnologyActivation:
    """L.3: programme/technology are valid enum members — no more ``other`` re-pin.

    Pre-L.3 the runtime enum-guard re-pinned ``technology`` / ``programme`` to
    ``other`` because they were not in ``_ALLOWED_ENTITY_TYPES``. L.3 adds them,
    so the L.1 bridge (Deal/GovernmentService -> programme) and the L.2 residual
    alias (technologie -> technology) now land on the real canonical.
    """

    def test_enum_contains_programme_and_technology(self):
        # AC1/AC2: the new canonical types are members of the code-side enum.
        assert "programme" in _ALLOWED_ENTITY_TYPES
        assert "technology" in _ALLOWED_ENTITY_TYPES

    def test_regiodeal_bridge_lands_on_programme_not_other(self):
        # AC1: {label: "RegioDeal"} (deals applied) -> programme via the
        # Deal->GovernmentService chain; NOT other, NOT creative_work.
        res = _resolve_entity_type("RegioDeal", [_deals_schema()])
        assert res.entity_type == "programme"
        assert res.entity_type != "other"
        assert res.entity_type != "creative_work"
        assert res.primary_type == "RegioDeal"

    def test_technologie_residual_alias_lands_on_technology(self):
        # AC2: {label: "Technologie"} with no schema applied -> the L.2 residual
        # alias maps to ``technology`` and the guard no longer re-pins to other.
        res = _resolve_entity_type("Technologie", None)
        assert res.entity_type == "technology"
        assert res.entity_type != "other"
        assert res.primary_type == "Technologie"

    @pytest.mark.parametrize(
        ("label", "schema", "expected"),
        [
            ("RegioDeal", "deals", "programme"),
            ("Programma", None, "programme"),
            ("Project", None, "programme"),
            ("Technologie", None, "technology"),
            ("Technology", None, "technology"),
        ],
    )
    def test_deal_programme_technology_labels_no_longer_other(
        self, label, schema, expected
    ):
        schemas = [_deals_schema()] if schema == "deals" else None
        res = _resolve_entity_type(label, schemas)
        assert res.entity_type == expected
        assert res.entity_type in _ALLOWED_ENTITY_TYPES


class TestCanonicalPassthrough:
    """L.2-fix: a label that is ALREADY a valid canonical enum member passes
    through as that type instead of falling to the ``concept`` default.

    Regression: L.2's ``resolve_residual_type`` only checks the alias map then
    falls to ``concept``; it never checked whether the label is itself a
    canonical enum member. A model (llama3.1:8b) that emitted bare English
    canonical names (``Organization`` / ``Person`` / ``Event`` ...) dumped them
    all into ``concept``. The fix restores the old ``_normalize_entity_type``
    canonical-passthrough in ``_resolve_entity_type``.
    """

    def test_organization_passes_through_not_concept(self):
        # The proven regression case: was ``concept`` live, must be organization.
        res = _resolve_entity_type("Organization")
        assert res.entity_type == "organization"
        assert res.entity_type != "concept"
        assert res.primary_type == "Organization"
        assert res.type_tags == ["Organization"]

    @pytest.mark.parametrize(
        ("label", "expected"),
        [
            ("Person", "person"),
            ("Event", "event"),
            ("Product", "product"),
            ("Location", "location"),
            ("Topic", "topic"),
            ("Concept", "concept"),
        ],
    )
    def test_bare_canonical_names_pass_through(self, label, expected):
        res = _resolve_entity_type(label)
        assert res.entity_type == expected
        # raw label preserved (anti-flattening guarantee)
        assert res.primary_type == label
        assert res.type_tags == [label]

    @pytest.mark.parametrize(
        "label",
        ["Government Organization", "government_organization", "Government-Organization"],
    )
    def test_multiword_canonical_passes_through(self, label):
        # Multi-word / underscore / dash canonical must normalize consistently.
        res = _resolve_entity_type(label)
        assert res.entity_type == "government_organization"
        assert res.primary_type == label

    def test_creative_work_multiword_canonical(self):
        res = _resolve_entity_type("Creative Work")
        assert res.entity_type == "creative_work"
        assert res.primary_type == "Creative Work"

    def test_genuine_unknown_still_falls_back_to_concept(self):
        # The L.2 non-silent fallback is unchanged: a genuine-unknown label still
        # coarses to ``concept`` with the raw label preserved.
        res = _resolve_entity_type("Frobnicator")
        assert res.entity_type == "concept"
        assert res.primary_type == "Frobnicator"
        assert res.type_tags == ["Frobnicator"]

    def test_dutch_alias_still_resolves(self):
        # No regression to the L.2 alias map: a Dutch alias still maps.
        res = _resolve_entity_type("Persoon")
        assert res.entity_type == "person"
        assert res.primary_type == "Persoon"

    def test_l1_bridge_still_wins_over_passthrough(self):
        # The bridge runs first: an ontology type resolves via L.1, not the
        # passthrough (Gemeente is not a bare canonical anyway, but assert the
        # ordering holds for a bridged label).
        res = _resolve_entity_type("Gemeente", [_gov_schema()])
        assert res.entity_type == "administrative_area"
        assert res.primary_type == "Gemeente"

    def test_passthrough_results_in_allowed_enum(self):
        # B.8 contract: every passthrough result is a valid enum member.
        for label in [
            "Organization", "Person", "Event", "Product", "Location",
            "Topic", "Concept", "Government Organization", "Creative Work",
        ]:
            res = _resolve_entity_type(label)
            assert res.entity_type in _ALLOWED_ENTITY_TYPES

    @pytest.mark.asyncio
    async def test_persist_path_stamps_canonical_passthrough(self):
        # End-to-end at the persist boundary: a bare ``Organization`` label
        # persists as entity_type organization (not concept) with the raw label
        # preserved.
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
                        "text": "Ministerie van BZK",
                        "label": "Organization",
                        "confidence": 0.9,
                        "properties": {},
                    }
                ],
                relations=[],
            )

        ent = mock_upsert.call_args.args[0]
        assert ent.entity_type == "organization"
        assert ent.entity_type != "concept"
        assert ent.primary_type == "Organization"


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
    async def test_no_schemas_alias_path_preserves_raw_label(self):
        # Without schemas the L.2 residual path runs. An alias hit (ORG ->
        # organization) now preserves the raw label in primary_type / type_tags
        # (the L.2 anti-flattening change; L.1-only left these empty).
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
        assert ent.primary_type == "ORG"
        assert ent.type_tags == ["ORG"]

    @pytest.mark.asyncio
    async def test_bridge_failure_degrades_not_crashes(self):
        # AC7: an unknown label with schemas present still persists (falls to
        # the L.2 residual fallback), never raising. L.2: the rich label is
        # preserved (coarse default ``concept``, primary_type retained) — no
        # longer the silent ``other`` of L.1.
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
        assert ent.entity_type == "concept"
        assert ent.primary_type == "Quux"

    @pytest.mark.asyncio
    async def test_programme_entity_persists_without_repin(self):
        # AC3: a RegioDeal (deals applied) persists with entity_type=="programme"
        # — the runtime guard accepts it; the upserted entity carries the real
        # canonical, not ``other``.
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
                        "text": "Regio Deal Groningen",
                        "label": "RegioDeal",
                        "confidence": 0.9,
                        "properties": {},
                    }
                ],
                relations=[],
                applicable_schemas=[_deals_schema()],
            )

        ent = mock_upsert.call_args.args[0]
        assert ent.entity_type == "programme"
        assert ent.primary_type == "RegioDeal"

    @pytest.mark.asyncio
    async def test_technology_entity_persists_without_repin(self):
        # AC3: a Technologie label (residual alias, no schema) persists with
        # entity_type=="technology" — accepted by the runtime guard.
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
                        "text": "GPT-4",
                        "label": "Technologie",
                        "confidence": 0.9,
                        "properties": {},
                    }
                ],
                relations=[],
            )

        ent = mock_upsert.call_args.args[0]
        assert ent.entity_type == "technology"
        assert ent.primary_type == "Technologie"
