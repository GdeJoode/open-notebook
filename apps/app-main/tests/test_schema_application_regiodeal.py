"""Track L.4 rev2 — content-driven schema application over policy-theme docs.

The 44% generic ``concept``/``topic`` bucket is a schema-APPLICATION gap: the
``policy_themes`` schema (``BeleidsThema`` / ``BeleidsPijler`` / ``Indicator`` /
``BredeWelvaart`` / ``Leefbaarheid`` …) exists for exactly the brede-welvaart
themes, but its abstract type NAMES match poorly so it was crowded out of
``detect_applicable_schemas``' top-K.

rev1 tried to fix this by gating ``policy_themes`` co-selection on the gov/deals
stack firing. That was wrong: policy themes are a property of CONTENT, not
provenance (a scientific paper about brede welvaart is about policy themes
without being a gov document), AND the gate was inert in production — the real
``document_mapper`` maps Regio-Deal docs to ``policy`` / ``general``, never
``deals`` / ``government``, so the trigger never fired on real data. rev1's test
only passed by injecting a fake ``_deals_mapper`` the production path lacks.

rev2 makes ``policy_themes`` earn selection on its OWN content merit: the
applicability scorer now scores the schema's ``extraction_hints`` /
``description`` text / ``concepts`` (the real theme vocabulary — ``brede
welvaart``, ``leefbaarheid``, ``vergrijzing`` …), not just abstract type names.
So ANY document discussing the themes selects ``policy_themes`` on content:

* a Regio-Deal document (AC1),
* a NON-government scientific paper about the themes (AC2),

while an unrelated document does not (AC3, bounded by
``MIN_APPLICABLE_CONFIDENCE``). The per-notebook default stays as the explicit
operator override (AC5).

With ``policy_themes`` applied, the L.1 canonical bridge maps a ``BeleidsThema``
to ``entity_type == "topic"`` with the rich ``primary_type == "BeleidsThema"``
(not generic ``concept``). AC4.

These tests run against the REAL ontologies and the REAL document_mapper (loaded
from YAML by the registry — no DB, no LLM, no network, NO injected mapper) so
they prove the production path, not a synthetic fixture.
"""

from __future__ import annotations

import pytest
from app_main.services.entity_extraction_service import (
    NOTEBOOK_DEFAULT_BUNDLE,
    EntityExtractionService,
)
from app_main.services.entity_persistence_service import _resolve_entity_type
from ontology_extraction.multi_schema_orchestrator import (
    detect_applicable_schemas,
)
from ontology_manager import get_ontology_manager
from shared.models.notebook_schema import NotebookSchema

# A representative Regio-Deal document body. Mentions both the concrete gov/deals
# vocabulary (``Gemeente`` / ``Provincie`` / ``Ministerie`` / ``RegioDeal``) AND
# the brede-welvaart theme vocabulary (``brede welvaart`` / ``leefbaarheid`` /
# ``gezondheid`` …) that the rev2 content scorer matches on. ``document_type`` is
# ``policy_document`` — what the REAL mapper assigns to these docs (it maps to
# ``policy``, never ``deals``), so the test exercises the production path.
REGIODEAL_TEXT = (
    "Regio Deal Groningen Noord. De Gemeente Groningen, de Provincie Groningen "
    "en het Ministerie van Binnenlandse Zaken sluiten deze Regio Deal. De Deal "
    "versterkt de brede welvaart en leefbaarheid in de regio. Wethouder Jansen "
    "ondertekent het convenant. De RegioDeal investeert in werkgelegenheid, "
    "gezondheid en bereikbaarheid."
)

# A NON-government scientific paper discussing the same themes WITHOUT any
# gov/deals provenance. policy_themes must still be selected — purely on content.
SCIPAPER_TEXT = (
    "Abstract. This paper studies brede welvaart and leefbaarheid in shrinking "
    "Dutch regions. Drawing on CBS Monitor data we analyse vergrijzing and "
    "bevolkingsdaling and their effect on voorzieningenniveau, gezondheid and "
    "wonen. We argue that materiele welvaart alone understates regional "
    "leefbaarheid and propose a composite welfare index."
)

# An unrelated technical paper with no policy-theme vocabulary at all.
MLPAPER_TEXT = (
    "We present a transformer architecture for image classification. Our model "
    "achieves state-of-the-art accuracy on ImageNet using a novel attention "
    "mechanism and gradient checkpointing to reduce GPU memory consumption."
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


class TestRegioDealSchemaDetection:
    @pytest.mark.asyncio
    async def test_regiodeal_selects_policy_themes_real_path(self):
        # AC1: a Regio-Deal document selects policy_themes via the REAL
        # production path — real document_mapper (policy_document -> policy),
        # real ontologies, NO injected mapper — on CONTENT overlap.
        ontologies = await _load_real_ontologies()
        ranked = await detect_applicable_schemas(
            document_type="policy_document",
            document_text=REGIODEAL_TEXT,
            ontologies=ontologies,
            top_k=3,
        )
        names = [ont.metadata.name for ont, _conf in ranked]
        assert "policy_themes" in names, (
            f"policy_themes must be selected on content via the real path; "
            f"got {names}"
        )

    @pytest.mark.asyncio
    async def test_non_gov_theme_paper_selects_policy_themes(self):
        # AC2: a NON-government scientific paper about the themes ALSO selects
        # policy_themes — proving content-driven, not provenance-gated. This is
        # the user's requirement: themes are a property of content. The mapper
        # routes academic_paper -> scholarly (no gov/deals anywhere).
        ontologies = await _load_real_ontologies()
        ranked = await detect_applicable_schemas(
            document_type="academic_paper",
            document_text=SCIPAPER_TEXT,
            ontologies=ontologies,
            top_k=3,
        )
        names = [ont.metadata.name for ont, _conf in ranked]
        assert "policy_themes" in names, (
            f"policy_themes must be selected on content alone for a non-gov "
            f"theme paper; got {names}"
        )
        # No provenance dependency: the gov stack does not fire on this paper.
        assert "deals" not in names
        assert "government" not in names

    @pytest.mark.asyncio
    async def test_unrelated_document_excludes_policy_themes(self):
        # AC3: an unrelated technical paper (no theme vocabulary) does NOT
        # select policy_themes — over-application bounded by the applicability
        # floor. Real mapper, real ontologies.
        ontologies = await _load_real_ontologies()
        ranked = await detect_applicable_schemas(
            document_type="academic_paper",
            document_text=MLPAPER_TEXT,
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


def test_notebook_default_bundle_is_data_driven():
    # The explicit per-notebook override bundle is declared as DATA so it is
    # extensible without touching the override logic. This is the EXPLICIT
    # operator-config path — distinct from auto-detection, which selects
    # policy_themes on content with no bundle dependency.
    assert NOTEBOOK_DEFAULT_BUNDLE["deals"] == ["policy_themes"]
    assert NOTEBOOK_DEFAULT_BUNDLE["government"] == ["policy_themes"]
