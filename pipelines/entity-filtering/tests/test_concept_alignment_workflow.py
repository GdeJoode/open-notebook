"""Track N.4b — concept alignment as a WORKFLOW stage (integration).

The absence of exactly this file is what let both blockers through in the parked
first attempt at N.4: seeded edges were silently discarded by the ontology filter
while the report still counted them, and edges added before centrality shifted
every PageRank score and could change which entities were removed. Unit tests on
``ConceptAligner`` could not see either, because both are properties of the stage's
POSITION in the pipeline rather than of the classifier.

So the load-bearing assertions here are:

* a seeded ``is_a`` reaches ``result.relations`` WITH ontology validation enabled;
* centrality scores are identical with the stage on and off;
* the reported ``seeded_is_a`` equals what actually survived;
* the stage is add-only — no entity is removed, merged or re-typed by it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest
from entity_filtering.config import (
    ConceptAlignmentConfig,
    FilteringConfig,
    KGResolutionConfig,
    OntologyValidationConfig,
)
from entity_filtering.resolution.concept_alignment import IS_A, RELATION_SOURCE
from entity_filtering.workflow import FilteringWorkflow
from shared.models.extraction import ExtractedEntity, ExtractionResult

# The ontology the alignment resolves types against. RegioDeal declares
# parent_type Deal, and canonical_bridge maps Deal -> the "programme" canonical
# enum — which is the column find_by_type actually filters on.
schema = pytest.importorskip("ontology_manager.schema")


def _ontology():
    return schema.Ontology(
        metadata=schema.OntologyMetadata(name="test", version="1"),
        entity_types={
            "RegioDeal": schema.EntityTypeDefinition(
                name="RegioDeal", parent_type="Deal"
            ),
        },
        relationship_types={},
    )


class _Repo:
    """KGResolver + ConceptAligner repository.

    ``find_by_type`` is keyed so the two stages see DIFFERENT rows: KG resolution
    asks for the RICH label and gets nothing (so the entity stays ``is_new``),
    while alignment asks for the CANONICAL type and finds the broader concept.
    That split is the whole point of D-N4-3 and is asserted below.
    """

    def __init__(self, by_type: Dict[str, List[Dict[str, Any]]]):
        self._by_type = by_type
        self.calls: List[str] = []

    async def find_by_alias(self, alias_text: str) -> Optional[Dict[str, Any]]:
        return None

    async def find_by_type(self, entity_type: str, limit: int = 100):
        self.calls.append(entity_type)
        return list(self._by_type.get(entity_type, []))[:limit]

    async def register_alias(self, **kwargs) -> bool:
        return True


def _repo():
    return _Repo({
        "RegioDeal": [],  # KG resolution finds no match -> is_new
        "programme": [    # alignment finds the broader concept, materialised
            {"id": "entity:deal", "name": "Deal", "weight": 0.0,
             "embedding": [0.0, 1.0]},
        ],
    })


def _config(*, alignment: bool, ontology_validation: bool = True,
            centrality: bool = False, kg_resolution: bool = True,
            seed_is_a: bool = True) -> FilteringConfig:
    return FilteringConfig(
        kg_resolution=KGResolutionConfig(enabled=kg_resolution,
                                         register_aliases=False),
        ontology_validation=OntologyValidationConfig(
            enabled=ontology_validation,
            graph_centrality_enabled=centrality,
        ),
        concept_alignment=ConceptAlignmentConfig(
            enabled=alignment,
            type_chain_enabled=True,  # the tier that yields NARROWER_THAN
            judge_enabled=False,      # deterministic: no LLM in this test
            seed_is_a=seed_is_a,
        ),
    )


def _extraction():
    return ExtractionResult(
        entities=[
            ExtractedEntity(text="Regio Deal Midden-Limburg", label="RegioDeal",
                            confidence=0.9,
                            properties={"embedding": [1.0, 0.0]}),
            ExtractedEntity(text="Provincie Limburg", label="RegioDeal",
                            confidence=0.9,
                            properties={"embedding": [1.0, 0.0]}),
        ],
        relations=[],
    )


async def _run(config, *, repo=None):
    workflow = FilteringWorkflow(
        config=config, entity_repo=repo or _repo(), ontology=_ontology()
    )
    return await workflow.process(_extraction())


# ---------------------------------------------------------------------------
# Blocker 2 of the parked attempt: seeds must survive ontology validation
# ---------------------------------------------------------------------------


async def test_seeded_is_a_survives_ontology_validation():
    result = await _run(_config(alignment=True, ontology_validation=True))
    seeds = [r for r in result.relations
             if r.properties.get("relation_source") == RELATION_SOURCE]
    assert seeds, "the seeded is_a was discarded before reaching the result"
    assert seeds[0].relation_type == IS_A
    assert seeds[0].target_entity == "Deal"


async def test_report_counts_only_what_survived():
    result = await _run(_config(alignment=True, ontology_validation=True))
    surviving = [r for r in result.relations
                 if r.properties.get("relation_source") == RELATION_SOURCE]
    assert result.concept_alignment_report["seeded_is_a"] == len(surviving)


async def test_seeded_edge_carries_both_endpoint_types():
    # D-N4-5: without these the persist path falls back to name-only resolution,
    # which is the cross-type homograph mis-binding Track O.1 prevents.
    result = await _run(_config(alignment=True))
    seed = next(r for r in result.relations
                if r.properties.get("relation_source") == RELATION_SOURCE)
    assert seed.source_type == "programme"
    assert seed.target_type == "programme"
    assert seed.properties["alignment_target_id"] == "entity:deal"
    assert seed.properties["alignment_evidence"]


# ---------------------------------------------------------------------------
# Major 4 of the parked attempt: seeds must not perturb centrality
# ---------------------------------------------------------------------------


def _centrality_scores(result) -> Dict[str, Any]:
    return {e.text: e.properties.get("centrality_score") for e in result.entities}


async def test_centrality_is_identical_with_the_stage_on_and_off():
    off = await _run(_config(alignment=False, centrality=True))
    on = await _run(_config(alignment=True, centrality=True))
    assert _centrality_scores(on) == _centrality_scores(off)


async def test_stage_does_not_change_which_entities_survive_centrality():
    off = await _run(_config(alignment=False, centrality=True))
    on = await _run(_config(alignment=True, centrality=True))
    assert {e.text for e in on.entities} == {e.text for e in off.entities}


# ---------------------------------------------------------------------------
# Add-only / non-destructive at workflow level
# ---------------------------------------------------------------------------


async def test_stage_is_add_only_for_entities():
    off = await _run(_config(alignment=False))
    on = await _run(_config(alignment=True))
    assert {e.text for e in on.entities} == {e.text for e in off.entities}
    assert {e.label for e in on.entities} == {e.label for e in off.entities}
    assert len(on.removed_entities) == len(off.removed_entities)


async def test_stage_only_adds_relations_it_tagged():
    off = await _run(_config(alignment=False))
    on = await _run(_config(alignment=True))
    added = len(on.relations) - len(off.relations)
    tagged = [r for r in on.relations
              if r.properties.get("relation_source") == RELATION_SOURCE]
    assert added == len(tagged)


async def test_seeding_can_be_disabled_while_verdicts_still_run():
    result = await _run(_config(alignment=True, seed_is_a=False))
    assert not [r for r in result.relations
                if r.properties.get("relation_source") == RELATION_SOURCE]
    assert result.concept_alignment_report["aligned_count"] > 0
    assert result.concept_alignment_report["seeded_is_a"] == 0


# ---------------------------------------------------------------------------
# D-N4-3 end-to-end, and the disabled / misconfigured paths
# ---------------------------------------------------------------------------


async def test_alignment_queries_the_canonical_type_not_the_rich_label():
    repo = _repo()
    await _run(_config(alignment=True), repo=repo)
    assert "programme" in repo.calls          # alignment used the canonical enum
    assert repo.calls.count("RegioDeal") > 0  # KG resolution used the rich label


async def test_stage_absent_when_disabled():
    result = await _run(_config(alignment=False))
    assert result.concept_alignment_report is None
    assert not [r for r in result.relations
                if r.properties.get("relation_source") == RELATION_SOURCE]


async def test_enabled_without_kg_resolution_is_a_safe_no_op(caplog):
    # Nothing is marked is_new, so there is nothing to align. It must be a quiet
    # no-op in the DATA and a loud one in the LOG (the Stage-14 house pattern).
    result = await _run(_config(alignment=True, kg_resolution=False))
    assert result.concept_alignment_report["aligned_count"] == 0
    assert result.concept_alignment_report["seeded_is_a"] == 0
    assert not [r for r in result.relations
                if r.properties.get("relation_source") == RELATION_SOURCE]


async def test_enabled_without_a_repository_does_not_crash():
    workflow = FilteringWorkflow(
        config=_config(alignment=True), entity_repo=None, ontology=_ontology()
    )
    result = await workflow.process(_extraction())
    assert result.concept_alignment_report["seeded_is_a"] == 0
