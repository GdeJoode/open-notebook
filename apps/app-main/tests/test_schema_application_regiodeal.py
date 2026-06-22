"""Track L.4 — schema application over a Regio-Deal document.

The 44% generic ``concept``/``topic`` bucket is a schema-APPLICATION gap: the
``policy_themes`` schema (``BeleidsThema`` / ``BeleidsPijler`` / ``Indicator`` /
``BredeWelvaart`` / ``Leefbaarheid`` …) exists for exactly the Regio-Deal
themes, but its abstract type NAMES keyword-match poorly so it is crowded out of
``detect_applicable_schemas``' top-K. L.4 closes the gap two ways:

* **schema-affinity co-selection** — when the keyword-rich gov stack
  (``deals`` / ``government``) is selected, ``policy_themes`` is co-selected
  (gated + additive). AC1 / AC4 / AC5.
* **per-notebook default** — a configured ``notebook_schema`` forces the base +
  its affinity bundle regardless of keyword score (config beats auto-detection).
  AC3.

With ``policy_themes`` applied, the L.1 canonical bridge maps a ``BeleidsThema``
to ``entity_type == "topic"`` with the rich ``primary_type == "BeleidsThema"``
(not generic ``concept``). AC2.

These tests run against the REAL ontologies (loaded from YAML by the registry —
no DB, no LLM, no network) so they assert the production vocabulary, not a
synthetic fixture.
"""

from __future__ import annotations

import pytest
from app_main.services.entity_extraction_service import EntityExtractionService
from app_main.services.entity_persistence_service import _resolve_entity_type
from ontology_extraction.multi_schema_orchestrator import (
    SCHEMA_AFFINITY,
    detect_applicable_schemas,
)
from ontology_manager import get_ontology_manager
from shared.models.notebook_schema import NotebookSchema

# A representative Regio-Deal document body. Mentions the concrete gov/deals
# vocabulary (``Gemeente`` / ``Provincie`` / ``Ministerie`` / ``RegioDeal``) that
# keyword-fires, while the abstract policy-theme NAMES (``BeleidsThema`` …) do NOT
# appear verbatim — which is exactly why policy_themes is crowded out without
# affinity co-selection.
REGIODEAL_TEXT = (
    "Regio Deal Groningen Noord. De Gemeente Groningen, de Provincie Groningen "
    "en het Ministerie van Binnenlandse Zaken sluiten deze Regio Deal. De Deal "
    "versterkt de brede welvaart en leefbaarheid in de regio. Wethouder Jansen "
    "ondertekent het convenant. De RegioDeal investeert in werkgelegenheid, "
    "gezondheid en bereikbaarheid."
)


async def _load_real_ontologies():
    manager = get_ontology_manager()
    names = await manager.list_ontologies()
    ontologies = []
    for name in names:
        ont = await manager.get_ontology(name)
        if ont is not None:
            ontologies.append(ont)
    return ontologies


def _deals_mapper(_document_type):
    """Stub mapper: a Regio-Deal document maps to the ``deals`` ontology.

    In production the document-type signal is how the keyword-rich gov stack is
    selected for these documents (the keyword-overlap score is diluted by the
    real ontologies' large type-count). We inject the mapping here so the test
    exercises the AFFINITY mechanism — the unit under test for L.4 — against the
    real ontologies rather than re-testing the keyword scorer.
    """
    return "deals"


class TestRegioDealSchemaDetection:
    @pytest.mark.asyncio
    async def test_before_policy_themes_not_selected(self):
        # AC1 (BEFORE): with affinity disabled, policy_themes is NOT among the
        # applied schemas even though deals fires — it loses on keywords.
        ontologies = await _load_real_ontologies()
        ranked = await detect_applicable_schemas(
            document_type="regio_deal",
            document_text=REGIODEAL_TEXT,
            ontologies=ontologies,
            top_k=3,
            mapper=_deals_mapper,
            affinity={},
        )
        names = [ont.metadata.name for ont, _conf in ranked]
        assert "deals" in names, f"deals should fire via the mapper; got {names}"
        assert "policy_themes" not in names

    @pytest.mark.asyncio
    async def test_after_policy_themes_co_selected(self):
        # AC1 (AFTER): with the default affinity, policy_themes is co-selected
        # because deals fired.
        ontologies = await _load_real_ontologies()
        ranked = await detect_applicable_schemas(
            document_type="regio_deal",
            document_text=REGIODEAL_TEXT,
            ontologies=ontologies,
            top_k=3,
            mapper=_deals_mapper,
        )
        names = [ont.metadata.name for ont, _conf in ranked]
        assert "policy_themes" in names, (
            f"policy_themes should be co-selected via affinity; got {names}"
        )
        assert "deals" in names  # conservative — trigger retained

    @pytest.mark.asyncio
    async def test_gated_non_policy_document_excludes_policy_themes(self):
        # AC5: a non-policy document (no gov/deals trigger) does NOT pull in
        # policy_themes. Use the real mapper so no trigger is forced.
        ontologies = await _load_real_ontologies()
        ranked = await detect_applicable_schemas(
            document_type="academic_paper",
            document_text="A short paper on graph theory by a researcher.",
            ontologies=ontologies,
            top_k=3,
        )
        names = [ont.metadata.name for ont, _conf in ranked]
        assert "policy_themes" not in names


class TestRegioDealRichTyping:
    @pytest.mark.asyncio
    async def test_beleidsthema_resolves_to_topic_with_rich_primary_type(self):
        # AC2: with policy_themes applied, a BeleidsThema entity persists as
        # entity_type == "topic" (canonical) with primary_type == "BeleidsThema"
        # (rich) — NOT generic concept. This is the L.1 bridge over a
        # policy_themes type, gated by L.4 having applied the schema.
        manager = get_ontology_manager()
        policy_themes = await manager.get_ontology("policy_themes")
        assert policy_themes is not None

        resolved = _resolve_entity_type("BeleidsThema", [policy_themes])
        assert resolved.entity_type == "topic"
        assert resolved.primary_type == "BeleidsThema"
        assert "BeleidsThema" in resolved.type_tags

    @pytest.mark.asyncio
    async def test_beleidspijler_also_gets_rich_type(self):
        # AC2 (sibling): BeleidsPijler → topic + rich primary_type.
        manager = get_ontology_manager()
        policy_themes = await manager.get_ontology("policy_themes")
        resolved = _resolve_entity_type("BeleidsPijler", [policy_themes])
        assert resolved.entity_type == "topic"
        assert resolved.primary_type == "BeleidsPijler"


class TestPerNotebookDefault:
    """AC3: a configured notebook_schema forces deals + policy_themes."""

    def _service(self) -> EntityExtractionService:
        # The override helper is pure (no I/O); construct the service with the
        # minimum the constructor needs. We only call the pure helper.
        return EntityExtractionService.__new__(EntityExtractionService)

    @pytest.mark.asyncio
    async def test_config_beats_auto_detection(self):
        # A notebook configured with base_ontology="deals". Auto-detection (with
        # a deliberately empty applicable set, simulating a doc with no theme
        # keywords) must be overridden so deals + policy_themes are applied.
        ontologies = await _load_real_ontologies()
        notebook_schema = NotebookSchema(
            notebook="notebook:regiodeal",
            base_ontology="deals",
        )
        service = self._service()
        forced = service._apply_notebook_schema_default(
            applicable_schemas=[],  # auto-detection found nothing
            candidate_ontologies=ontologies,
            notebook_schema=notebook_schema,
        )
        names = [ont.metadata.name for ont, _conf in forced]
        assert "deals" in names, f"config base must be forced; got {names}"
        assert "policy_themes" in names, (
            f"base affinity bundle must be forced; got {names}"
        )

    @pytest.mark.asyncio
    async def test_config_is_additive_not_destructive(self):
        # The forced base is added ON TOP of whatever auto-detection found; an
        # auto-detected schema is never dropped by the override.
        ontologies = await _load_real_ontologies()
        scholarly = next(
            o for o in ontologies if o.metadata.name == "scholarly"
        )
        notebook_schema = NotebookSchema(
            notebook="notebook:regiodeal",
            base_ontology="deals",
        )
        service = self._service()
        forced = service._apply_notebook_schema_default(
            applicable_schemas=[(scholarly, 0.92)],
            candidate_ontologies=ontologies,
            notebook_schema=notebook_schema,
        )
        names = [ont.metadata.name for ont, _conf in forced]
        assert "scholarly" in names  # auto-detected schema retained
        assert "deals" in names
        assert "policy_themes" in names

    @pytest.mark.asyncio
    async def test_misconfigured_base_does_not_crash(self):
        # A base_ontology that doesn't exist in the candidate set is skipped,
        # never crashes extraction.
        ontologies = await _load_real_ontologies()
        notebook_schema = NotebookSchema(
            notebook="notebook:x",
            base_ontology="does_not_exist",
        )
        service = self._service()
        forced = service._apply_notebook_schema_default(
            applicable_schemas=[],
            candidate_ontologies=ontologies,
            notebook_schema=notebook_schema,
        )
        assert forced == []


def test_affinity_contract_is_data_driven():
    # The affinity is declared as DATA so it is extensible without touching the
    # selection logic.
    assert SCHEMA_AFFINITY["deals"] == ["policy_themes"]
    assert SCHEMA_AFFINITY["government"] == ["policy_themes"]
