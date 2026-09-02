"""Track N.4b — concept alignment as a WORKFLOW stage (integration).

The absence of exactly this file is what let both blockers through in the parked
first attempt at N.4: seeded edges were silently discarded by the ontology filter
while the report still counted them, and edges added before centrality shifted
every PageRank score and could change which entities were removed. Unit tests on
``ConceptAligner`` could not see either, because both are properties of the stage's
POSITION in the pipeline rather than of the classifier.

So the load-bearing assertions here are:

* the stage emits NO relations (the subsumption tier was retired in N.4d.0), and
  ``result.relations`` is identical with the stage on and off;
* centrality scores are identical with the stage on and off;
* the stage is non-destructive — no entity is removed, merged or re-typed by it.

The placement assertions are kept even though nothing is emitted today, but NOT
as a guard: the N.4d.0 review showed by mutation that a producer emitting the
shape that actually mattered — an edge into an existing, OFF-BATCH node — passes
all of them. They assert the stage is inert, nothing more. Whoever reintroduces a
producer must add a test with an off-batch endpoint.
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
            centrality_min_score: float = 0.01) -> FilteringConfig:
    return FilteringConfig(
        kg_resolution=KGResolutionConfig(enabled=kg_resolution,
                                         register_aliases=False),
        ontology_validation=OntologyValidationConfig(
            enabled=ontology_validation,
            graph_centrality_enabled=centrality,
            centrality_min_score=centrality_min_score,
        ),
        concept_alignment=ConceptAlignmentConfig(
            enabled=alignment,
            judge_enabled=False,      # deterministic: no LLM in this test
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


async def _run(config, *, repo=None, source_id=None, **workflow_kwargs):
    workflow = FilteringWorkflow(
        config=config,
        entity_repo=repo or _repo(),
        ontology=_ontology(),
        **workflow_kwargs,
    )
    return await workflow.process(_extraction(), source_id=source_id)


# ---------------------------------------------------------------------------
# D-N4-12 / N.4d.0 — the stage emits no relations at all
# ---------------------------------------------------------------------------


async def test_stage_emits_no_relations():
    """The subsumption tier that seeded ``is_a`` was retired in N.4d.0.

    Asserted at workflow level rather than only on the module, because the seed
    used to be added here — this is the test that would notice a reintroduction.
    """
    off = await _run(_config(alignment=False))
    on = await _run(_config(alignment=True))
    assert [r.model_dump() for r in on.relations] == [
        r.model_dump() for r in off.relations
    ]
    assert not any(
        r.properties.get("relation_source") == "concept_alignment"
        for r in on.relations
    )


async def test_report_has_no_seeding_counter():
    result = await _run(_config(alignment=True))
    assert "seeded_is_a" not in result.concept_alignment_report
    assert result.concept_alignment_report["aligned_count"] > 0


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
    """The stage must not change the surviving set.

    The floor sits at 0.4 while both entities score 0.5, which was chosen in N.4b
    to be discriminating against a seed's phantom node. With no producer left
    there is no phantom node, so this now asserts only that the stage is inert
    here — it does NOT prove a misplacement would be caught (the N.4d.0 review
    disproved that by mutation).
    """
    cfg = dict(centrality=True, centrality_min_score=0.4)
    off = await _run(_config(alignment=False, **cfg))
    on = await _run(_config(alignment=True, **cfg))
    assert {e.text for e in off.entities} == {"Regio Deal Midden-Limburg",
                                              "Provincie Limburg"}
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


async def test_enabled_without_kg_resolution_is_a_safe_no_op(caplog):
    # Nothing is marked is_new, so there is nothing to align. It must be a quiet
    # no-op in the DATA and a loud one in the LOG (the Stage-14 house pattern).
    result = await _run(_config(alignment=True, kg_resolution=False))
    assert result.concept_alignment_report["aligned_count"] == 0


async def test_enabled_without_a_repository_does_not_crash():
    workflow = FilteringWorkflow(
        config=_config(alignment=True), entity_repo=None, ontology=_ontology()
    )
    result = await workflow.process(_extraction())
    assert result.concept_alignment_report["aligned_count"] == 2


# ---------------------------------------------------------------------------
# The misconfiguration WARNING (review M2/M3)
# ---------------------------------------------------------------------------
# loguru does not propagate into pytest's caplog by itself, so a bare `caplog`
# argument silently asserts nothing. This suite already solved that once for the
# orphan-connector (test_workflow.py) — same bridge here, otherwise the warning
# is untested while looking tested.


def _loguru_to_caplog():
    import logging

    from loguru import logger as loguru_logger

    class _PropagateHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            logging.getLogger(record.name).handle(record)

    return loguru_logger.add(_PropagateHandler(), level="WARNING", format="{message}")


async def test_warns_that_nothing_is_classified_when_kg_resolution_is_off(caplog):
    import logging

    from loguru import logger as loguru_logger

    sink = _loguru_to_caplog()
    try:
        with caplog.at_level(logging.WARNING):
            await _run(_config(alignment=True, kg_resolution=False))
    finally:
        loguru_logger.remove(sink)
    messages = " | ".join(r.message for r in caplog.records)
    assert "nothing will be classified" in messages
    assert "kg_resolution is disabled" in messages


async def test_warns_that_the_stage_is_degraded_not_silent_without_a_repo(caplog):
    # M3: with no repo the stage still RUNS and records NOVEL verdicts, so the
    # log must not claim it classifies nothing.
    import logging

    from loguru import logger as loguru_logger

    sink = _loguru_to_caplog()
    try:
        with caplog.at_level(logging.WARNING):
            workflow = FilteringWorkflow(
                config=_config(alignment=True), entity_repo=None,
                ontology=_ontology(),
            )
            result = await workflow.process(_extraction())
    finally:
        loguru_logger.remove(sink)
    messages = " | ".join(r.message for r in caplog.records)
    assert "DEGRADED" in messages and "entity_repo" in messages
    assert "nothing will be classified" not in messages
    # and the claim is true: verdicts WERE recorded
    assert result.concept_alignment_report["aligned_count"] == 2
    assert result.concept_alignment_report["reason_counts"] == {"no_repo": 2}


async def test_warns_when_the_judge_is_enabled_but_has_no_caller(caplog):
    import logging

    from loguru import logger as loguru_logger

    sink = _loguru_to_caplog()
    try:
        cfg = _config(alignment=True)
        cfg.concept_alignment.judge_enabled = True
        with caplog.at_level(logging.WARNING):
            await _run(cfg)
    finally:
        loguru_logger.remove(sink)
    assert "alignment_llm_caller" in " | ".join(r.message for r in caplog.records)


async def test_degraded_warning_is_silent_when_nothing_is_classified(caplog):
    # Residual from the N.4b re-review: with kg_resolution OFF and a missing DI
    # input, the accurate "nothing will be classified" line must not be
    # contradicted by a DEGRADED line claiming verdicts are being recorded.
    import logging

    from loguru import logger as loguru_logger

    sink = _loguru_to_caplog()
    try:
        with caplog.at_level(logging.WARNING):
            workflow = FilteringWorkflow(
                config=_config(alignment=True, kg_resolution=False),
                entity_repo=None, ontology=_ontology(),
            )
            result = await workflow.process(_extraction())
    finally:
        loguru_logger.remove(sink)
    messages = " | ".join(r.message for r in caplog.records)
    assert "nothing will be classified" in messages
    assert "DEGRADED" not in messages
    assert result.concept_alignment_report["aligned_count"] == 0


# ---------------------------------------------------------------------------
# N.4d.4 — the gap loop reaches the aligner from HERE
# ---------------------------------------------------------------------------
#
# The unit tests in test_concept_alignment.py construct a ConceptAligner
# directly, so they cannot see whether the workflow hands it anything. A review
# measured that deleting `gap_recorder=` and `source_id=` from this stage left
# all 597 entity-filtering tests green — the whole feature became a no-op in
# production with every suite passing. These are the guards for that seam.


class _Recorder:
    """Stands in for `OntologyEvolutionAgent`, matching its real return shape."""

    def __init__(self):
        self.calls = []

    async def record_gap(self, **kwargs):
        self.calls.append(kwargs)
        return type("_Gap", (), {"id": "ontology_gap:1"})()


# The file's own fixtures already produce the verdict this loop needs: the
# entities embed at [1.0, 0.0] and the only "programme" candidate at [0.0, 1.0],
# so cosine is 0, below the related floor, and the verdict is NOVEL with
# `EV_NONE_CLOSE` — one of the two codes that license a gap.


async def test_the_recorder_reaches_the_aligner():
    recorder = _Recorder()
    result = await _run(_config(alignment=True), gap_recorder=recorder)
    report = result.concept_alignment_report
    assert report["gap_recorder_wired"] is True
    assert report["gap_eligible"] >= 1
    assert report["gaps_recorded"] == report["gap_eligible"]
    assert recorder.calls, "the workflow never handed the recorder to the aligner"


async def test_the_source_id_reaches_the_aligner():
    """Provenance for the gap row. Without it a concept recurring across
    documents is indistinguishable from one seen once — which is the whole
    reason the accumulation exists.
    """
    recorder = _Recorder()
    await _run(
        _config(alignment=True),
        gap_recorder=recorder,
        source_id="source:abc",
    )
    assert recorder.calls[0]["source_id"] == "source:abc"


async def test_the_gap_ontology_name_reaches_the_aligner():
    """B1: gaps are filed under the notebook's DECLARED vocabulary, not a member
    of the per-document applied set.
    """
    recorder = _Recorder()
    await _run(
        _config(alignment=True),
        gap_recorder=recorder,
        gap_ontology_name="deals",
    )
    assert recorder.calls[0]["ontology_name"] == "deals"


async def test_without_a_declared_name_the_applied_schema_names_the_gap():
    """The fallback, and the vacuity guard for the test above: the name really is
    read from the caller rather than always being the schema's.
    """
    recorder = _Recorder()
    await _run(_config(alignment=True), gap_recorder=recorder)
    assert recorder.calls[0]["ontology_name"] == _ontology().metadata.name


async def test_all_applied_schemas_reach_the_aligner():
    """`detect_applicable_schemas(top_k=3)` returns up to three. Passing one
    makes every type declared in the other two fail to resolve, which yields a
    reason code that licenses no gap — the loop silently under-fires.
    """
    unrelated = schema.Ontology(
        metadata=schema.OntologyMetadata(name="unrelated", version="1"),
        entity_types={
            "Tranche": schema.EntityTypeDefinition(name="Tranche", parent_type="Deal")
        },
        relationship_types={},
    )
    # `RegioDeal` — the entities' label — is declared ONLY in the second element.
    result = await _run(
        _config(alignment=True),
        gap_recorder=_Recorder(),
        alignment_schemas=[unrelated, _ontology()],
    )
    assert result.concept_alignment_report["gap_eligible"] >= 1, (
        "the label resolved in no schema — only the first was searched"
    )

    # Vacuity guard: with ONLY the unrelated schema the label does not resolve,
    # so the assertion above is about the search and not about a label that
    # would have resolved regardless.
    without = await _run(
        _config(alignment=True),
        gap_recorder=_Recorder(),
        alignment_schemas=[unrelated],
    )
    assert without.concept_alignment_report["gap_eligible"] == 0


async def test_warns_when_no_gap_recorder_is_wired(caplog):
    """D-N4-8's honest DEGRADED warning, for the tier this phase added."""
    import logging

    from loguru import logger as loguru_logger

    sink = _loguru_to_caplog()
    try:
        with caplog.at_level(logging.WARNING):
            await _run(_config(alignment=True))
    finally:
        loguru_logger.remove(sink)
    messages = " | ".join(r.message for r in caplog.records)
    assert "gap_recorder" in messages


async def test_no_degraded_warning_about_the_recorder_when_it_is_wired(caplog):
    """Vacuity guard: the warning is about the recorder's absence, not a line
    that always fires.
    """
    import logging

    from loguru import logger as loguru_logger

    sink = _loguru_to_caplog()
    try:
        with caplog.at_level(logging.WARNING):
            await _run(
                _config(alignment=True), gap_recorder=_Recorder()
            )
    finally:
        loguru_logger.remove(sink)
    messages = " | ".join(r.message for r in caplog.records)
    assert "gap_recorder" not in messages
