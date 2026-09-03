"""Tests for ``ontology_extraction.multi_schema_orchestrator``.

Covers Phase B.1e acceptance criteria:

1. Merge correctness for an entity present in ≥ 2 passes (type_tags
   length ≥ 2, primary_type from highest-confidence pass).
2. Token budget per Pass-2 call ≤ 3000 tokens (perf guard).
3. ``pass1_results`` row is written per schema attempted (fake repo
   captures every call).
4. Single-schema input is bit-identical to the B.1d ``run_pass2``
   path (snapshot regression).
5. Soft-nudge thresholds match the shared-config source of truth and
   round-trip through the shared import.
6. Coverage ≥ 85% on the new orchestrator module.

All tests inject fake LLM callers and fake repos — no DB, no network.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from unittest.mock import patch

import pytest
from ontology_extraction.multi_schema_orchestrator import (
    MIN_APPLICABLE_CONFIDENCE,
    SOFT_NUDGE_COVERAGE_HIGH,
    SOFT_NUDGE_COVERAGE_LOW,
    MergedEntity,
    SoftNudgeDecision,
    _content_signal_phrases,
    _decide_soft_nudge,
    _dedupe_extensions,
    _merge_results,
    _sample_text_for_pass1,
    _score_content_overlap,
    detect_applicable_schemas,
    run_multi_schema,
)
from ontology_extraction.pass2_typed_extraction import (
    TOKEN_BUDGET_TARGET,
    _estimate_tokens,
)
from ontology_extraction.prompts.pass2 import build_pass2_prompt
from ontology_manager.schema import (
    EntityTypeDefinition,
    Ontology,
    OntologyMetadata,
    RelationshipTypeDefinition,
)
from shared.config import (
    MIN_APPLICABLE_CONFIDENCE as SHARED_MIN_APPLICABLE_CONFIDENCE,
)
from shared.config import (
    SOFT_NUDGE_COVERAGE_HIGH as SHARED_SOFT_NUDGE_COVERAGE_HIGH,
)
from shared.config import (
    SOFT_NUDGE_COVERAGE_LOW as SHARED_SOFT_NUDGE_COVERAGE_LOW,
)
from shared.models.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from shared.models.notebook_schema import Pass1Result

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _scholarly_ontology() -> Ontology:
    """A 3-type scholarly ontology."""
    return Ontology(
        metadata=OntologyMetadata(name="scholarly", version="1.0"),
        entity_types={
            "Researcher": EntityTypeDefinition(
                name="Researcher",
                description="A person who conducts research.",
            ),
            "Paper": EntityTypeDefinition(
                name="Paper",
                description="A peer-reviewed publication.",
            ),
            "Institution": EntityTypeDefinition(
                name="Institution",
                description="A research-performing organisation.",
            ),
        },
        relationship_types={
            "AUTHORED": RelationshipTypeDefinition(
                name="AUTHORED",
                description="A researcher authored a paper.",
                domain=["Researcher"],
                range=["Paper"],
            ),
        },
    )


def _policy_ontology() -> Ontology:
    """A 3-type policy ontology (overlaps with scholarly on Author)."""
    return Ontology(
        metadata=OntologyMetadata(name="policy", version="1.0"),
        entity_types={
            "Policymaker": EntityTypeDefinition(
                name="Policymaker",
                description="A person who shapes policy.",
            ),
            "Legislation": EntityTypeDefinition(
                name="Legislation",
                description="A legal instrument.",
            ),
            "Agency": EntityTypeDefinition(
                name="Agency",
                description="A government body.",
            ),
        },
    )


def _base_ontology() -> Ontology:
    """A minimal base ontology with one generic Person type."""
    return Ontology(
        metadata=OntologyMetadata(name="general", version="1.0"),
        entity_types={
            "Person": EntityTypeDefinition(
                name="Person", description="Any human."
            ),
            "Organization": EntityTypeDefinition(
                name="Organization", description="Any group of people."
            ),
        },
    )


# ---------------------------------------------------------------------------
# L.4 affinity fixtures: the keyword-rich gov stack (deals / government) vs the
# keyword-poor theme schema (policy_themes). These mirror the real ontologies'
# defining property: gov/deals type NAMES read like Regio-Deal vocabulary and
# appear in document bodies, while policy_themes type NAMES are abstract and
# rarely appear verbatim — exactly why policy_themes is crowded out of top-K.
# ---------------------------------------------------------------------------


def _government_ontology() -> Ontology:
    return Ontology(
        metadata=OntologyMetadata(name="government", version="1.0"),
        entity_types={
            "Gemeente": EntityTypeDefinition(
                name="Gemeente", description="A municipality."
            ),
            "Provincie": EntityTypeDefinition(
                name="Provincie", description="A province."
            ),
            "Ministerie": EntityTypeDefinition(
                name="Ministerie", description="A ministry."
            ),
        },
    )


def _deals_ontology() -> Ontology:
    return Ontology(
        metadata=OntologyMetadata(name="deals", version="1.0"),
        entity_types={
            "Deal": EntityTypeDefinition(name="Deal", description="A deal."),
            "RegioDeal": EntityTypeDefinition(
                name="RegioDeal", description="A regional deal."
            ),
        },
    )


def _policy_themes_ontology() -> Ontology:
    """Theme schema whose abstract type NAMES keyword-match poorly but whose
    ``extraction_hints`` carry the real theme vocabulary the L.4 rev2 content
    scorer matches on (mirrors the production ``policy_themes.yaml``)."""
    return Ontology(
        metadata=OntologyMetadata(name="policy_themes", version="1.0"),
        entity_types={
            "BeleidsThema": EntityTypeDefinition(
                name="BeleidsThema",
                description="An overarching policy theme.",
                parent_type="Concept",
                extraction_hints=[
                    "brede welvaart",
                    "leefbaarheid",
                    "vergrijzing",
                    "bevolkingsdaling",
                    "gezondheid",
                    "werkgelegenheid",
                    "voorzieningenniveau",
                    "materiele welvaart",
                ],
            ),
            "BeleidsPijler": EntityTypeDefinition(
                name="BeleidsPijler",
                description="A pillar within a programme or deal.",
                parent_type="Concept",
            ),
            "Indicator": EntityTypeDefinition(
                name="Indicator",
                description="A measurable indicator.",
                parent_type="Concept",
            ),
        },
    )


def _three_chunks() -> List[Dict[str, Any]]:
    return [
        {"text": "Alice is a researcher at MIT.", "id": "c-1"},
        {"text": "Bob authored 'Quantum 101'.", "id": "c-2"},
        {"text": "Carol works on graph theory.", "id": "c-3"},
    ]


def _make_pass1_response(
    detected: str,
    coverage: float,
    confidence: float = 0.8,
    extensions: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build a canned Pass-1 LLM response in the canonical shape."""
    return json.dumps(
        {
            "detected_schema": detected,
            "confidence_in_choice": confidence,
            "coverage_pct": coverage,
            "uncovered_concepts": [],
            "proposed_extensions": extensions or [],
            "alternative_schemas": [],
        }
    )


def _make_pass2_response(
    entities: List[Dict[str, Any]],
    relations: Optional[List[Dict[str, Any]]] = None,
) -> str:
    """Build a canned Pass-2 LLM response."""
    return json.dumps({"entities": entities, "relations": relations or []})


class _FakePass1Repo:
    """Test fake mirroring the ``Pass1ResultRepository.record`` contract.

    Captures every ``Pass1Result`` written so tests can assert on the
    per-schema persistence guarantee (AC #3).
    """

    def __init__(self) -> None:
        self.records: List[Pass1Result] = []

    async def record(self, result: Pass1Result) -> str:
        # Defensive copy so a downstream mutation doesn't taint the
        # captured ledger. ``model_copy`` is cheap for our tiny test
        # payloads.
        self.records.append(result.model_copy(deep=True))
        return f"pass1_results:fake-{len(self.records)}"


# ---------------------------------------------------------------------------
# Threshold round-trip tests (AC #5)
# ---------------------------------------------------------------------------


class TestSharedThresholdsRoundTrip:
    """Constants in the orchestrator come from ``shared.config``.

    Ensures B.3c UI can import the same values without drift (RETRO
    lesson #2 — single source of truth).
    """

    def test_soft_nudge_high_matches_shared(self):
        assert SOFT_NUDGE_COVERAGE_HIGH == SHARED_SOFT_NUDGE_COVERAGE_HIGH
        assert SOFT_NUDGE_COVERAGE_HIGH == 0.95

    def test_soft_nudge_low_matches_shared(self):
        assert SOFT_NUDGE_COVERAGE_LOW == SHARED_SOFT_NUDGE_COVERAGE_LOW
        assert SOFT_NUDGE_COVERAGE_LOW == 0.80

    def test_min_applicable_matches_shared(self):
        assert MIN_APPLICABLE_CONFIDENCE == SHARED_MIN_APPLICABLE_CONFIDENCE
        assert MIN_APPLICABLE_CONFIDENCE == 0.3


# ---------------------------------------------------------------------------
# Soft-nudge decision tests
# ---------------------------------------------------------------------------


class TestSoftNudgeDecision:
    """Coverage threshold → :class:`SoftNudgeDecision` mapping."""

    def test_high_coverage_returns_none(self):
        assert _decide_soft_nudge(0.97) == SoftNudgeDecision.NONE

    def test_top_of_band_returns_extension_suggested(self):
        # Exactly HIGH falls into EXTENSION_SUGGESTED (the >HIGH check
        # is strict so 0.95 itself still wants extensions).
        assert (
            _decide_soft_nudge(0.95) == SoftNudgeDecision.EXTENSION_SUGGESTED
        )

    def test_mid_band_returns_extension_suggested(self):
        assert (
            _decide_soft_nudge(0.85) == SoftNudgeDecision.EXTENSION_SUGGESTED
        )

    def test_low_boundary_returns_extension_suggested(self):
        # Inclusive at LOW so 0.80 itself stays in suggestion bucket.
        assert (
            _decide_soft_nudge(0.80) == SoftNudgeDecision.EXTENSION_SUGGESTED
        )

    def test_below_low_returns_schema_mismatch(self):
        assert _decide_soft_nudge(0.75) == SoftNudgeDecision.SCHEMA_MISMATCH

    def test_zero_coverage_returns_schema_mismatch(self):
        assert _decide_soft_nudge(0.0) == SoftNudgeDecision.SCHEMA_MISMATCH


# ---------------------------------------------------------------------------
# detect_applicable_schemas tests
# ---------------------------------------------------------------------------


class TestDetectApplicableSchemas:
    @pytest.mark.asyncio
    async def test_document_type_match_wins(self):
        # When document_type maps to scholarly via the real mapper,
        # the scholarly ontology should score top.
        ontologies = [_scholarly_ontology(), _policy_ontology(), _base_ontology()]
        ranked = await detect_applicable_schemas(
            document_type="academic_paper",
            document_text="A short paper.",
            ontologies=ontologies,
        )
        assert ranked, "Expected at least one applicable schema"
        assert ranked[0][0].metadata.name == "scholarly"
        assert ranked[0][1] >= 0.92

    @pytest.mark.asyncio
    async def test_keyword_overlap_only(self):
        # No document_type → keyword overlap is the only signal.
        ontologies = [_scholarly_ontology(), _policy_ontology()]
        # Text references Researcher + Paper from scholarly (2/3 types)
        text = "The researcher published a paper about graph theory."
        ranked = await detect_applicable_schemas(
            document_type=None,
            document_text=text,
            ontologies=ontologies,
        )
        # Scholarly should rank first due to keyword hits.
        names = [ontology.metadata.name for ontology, _conf in ranked]
        assert "scholarly" in names

    @pytest.mark.asyncio
    async def test_top_k_caps_result_size(self):
        ontologies = [_scholarly_ontology(), _policy_ontology(), _base_ontology()]
        ranked = await detect_applicable_schemas(
            document_type="academic_paper",
            document_text="Researcher Paper Institution Policymaker Agency Person",
            ontologies=ontologies,
            top_k=2,
        )
        assert len(ranked) <= 2

    @pytest.mark.asyncio
    async def test_min_applicable_floor_filters_zero_match(self):
        # An ontology with zero keyword matches and no document_type
        # signal should fall below MIN_APPLICABLE_CONFIDENCE.
        ontologies = [_base_ontology()]
        ranked = await detect_applicable_schemas(
            document_type=None,
            document_text="completely unrelated content xyz",
            ontologies=ontologies,
        )
        # base has 0 keyword matches against this text — should drop.
        assert ranked == []

    @pytest.mark.asyncio
    async def test_empty_ontology_list_returns_empty(self):
        ranked = await detect_applicable_schemas(
            document_type="academic_paper",
            document_text="hello",
            ontologies=[],
        )
        assert ranked == []

    @pytest.mark.asyncio
    async def test_custom_mapper_injected(self):
        # Inject a stub mapper so we can drive the document-type
        # signal without coupling to the real document_mapper.
        def stub_mapper(dt: Optional[str]) -> str:
            return "policy"

        ranked = await detect_applicable_schemas(
            document_type="anything",
            document_text="",
            ontologies=[_scholarly_ontology(), _policy_ontology()],
            mapper=stub_mapper,
        )
        assert ranked[0][0].metadata.name == "policy"


# ---------------------------------------------------------------------------
# L.4 rev2: content-driven applicability of policy_themes
# ---------------------------------------------------------------------------


class TestContentDrivenPolicyThemes:
    """``policy_themes`` is selected on CONTENT overlap, not on provenance (L.4
    rev2).

    rev1 gated ``policy_themes`` on the gov/deals stack firing — wrong because
    policy themes are a property of content, and the gate was inert in
    production (the real document-type mapper never selects deals/government for
    these docs). rev2 scores against the schema's content-signal vocabulary
    (extraction_hints / descriptions / concepts), so ANY document discussing the
    themes selects ``policy_themes`` — a Regio-Deal doc and a non-gov scientific
    paper alike — and an unrelated document does not.
    """

    # A Regio-Deal-style body: gov vocabulary AND theme vocabulary present.
    _REGIODEAL_TEXT = (
        "De Gemeente Groningen en de Provincie sluiten een RegioDeal met het "
        "Ministerie. De Deal investeert in brede welvaart en leefbaarheid, in "
        "gezondheid en werkgelegenheid."
    )
    # A NON-government scientific-paper body: theme vocabulary WITHOUT any
    # gov/deals provenance. This is the user's requirement — content, not
    # provenance, drives selection.
    _SCIPAPER_TEXT = (
        "This paper studies brede welvaart and leefbaarheid in shrinking "
        "regions. We analyse vergrijzing and bevolkingsdaling and their effect "
        "on voorzieningenniveau and gezondheid. We argue materiele welvaart "
        "alone understates regional leefbaarheid."
    )
    # An unrelated ML paper: no policy-theme vocabulary at all.
    _MLPAPER_TEXT = (
        "We present a transformer architecture for image classification. Our "
        "model achieves state-of-the-art accuracy on ImageNet using a novel "
        "attention mechanism and gradient checkpointing."
    )

    def _all_ontologies(self) -> List[Ontology]:
        return [
            _government_ontology(),
            _deals_ontology(),
            _policy_themes_ontology(),
            _scholarly_ontology(),
        ]

    @pytest.mark.asyncio
    async def test_regiodeal_doc_selects_policy_themes_on_content(self):
        # AC1: a Regio-Deal doc selects policy_themes via content overlap — no
        # mapper, no affinity, no injected trigger.
        ranked = await detect_applicable_schemas(
            document_type=None,
            document_text=self._REGIODEAL_TEXT,
            ontologies=self._all_ontologies(),
            top_k=3,
        )
        names = [ont.metadata.name for ont, _conf in ranked]
        assert "policy_themes" in names

    @pytest.mark.asyncio
    async def test_non_gov_theme_paper_also_selects_policy_themes(self):
        # AC2: a non-government scientific paper about the themes ALSO selects
        # policy_themes — proving content-driven, not provenance-gated. The
        # gov/deals schemas do NOT fire here.
        ranked = await detect_applicable_schemas(
            document_type=None,
            document_text=self._SCIPAPER_TEXT,
            ontologies=self._all_ontologies(),
            top_k=3,
        )
        names = [ont.metadata.name for ont, _conf in ranked]
        assert "policy_themes" in names
        assert "deals" not in names
        assert "government" not in names

    @pytest.mark.asyncio
    async def test_unrelated_doc_does_not_select_policy_themes(self):
        # AC3: an unrelated ML paper (no theme vocabulary) does NOT select
        # policy_themes — over-application bounded by MIN_APPLICABLE_CONFIDENCE.
        ranked = await detect_applicable_schemas(
            document_type=None,
            document_text=self._MLPAPER_TEXT,
            ontologies=self._all_ontologies(),
            top_k=3,
        )
        names = [ont.metadata.name for ont, _conf in ranked]
        assert "policy_themes" not in names

    @pytest.mark.asyncio
    async def test_content_score_clears_floor_for_theme_doc(self):
        # The policy_themes content score is comfortably above the applicability
        # floor for a theme doc — the selection is not a knife-edge.
        ranked = await detect_applicable_schemas(
            document_type=None,
            document_text=self._SCIPAPER_TEXT,
            ontologies=self._all_ontologies(),
            top_k=5,
        )
        by_name = {ont.metadata.name: conf for ont, conf in ranked}
        assert by_name["policy_themes"] >= MIN_APPLICABLE_CONFIDENCE

    def test_content_signal_phrases_include_hints_and_descriptions(self):
        # The signal set is the union of names + hints + descriptions (+ concept
        # vocabulary). Pin that hints are included — that is the rev2 fix.
        phrases = _content_signal_phrases(_policy_themes_ontology())
        lowered = {p.lower() for p in phrases}
        assert "beleidsthema" in lowered  # type name
        assert "brede welvaart" in lowered  # extraction_hint
        assert "leefbaarheid" in lowered  # extraction_hint

    def test_score_content_overlap_zero_for_unrelated_text(self):
        score = _score_content_overlap(
            _policy_themes_ontology(),
            "a paper on transformer attention and gpu memory",
        )
        assert score == 0.0

    def test_score_content_overlap_nonzero_for_theme_text(self):
        score = _score_content_overlap(
            _policy_themes_ontology(),
            "a study of brede welvaart and leefbaarheid in the region",
        )
        assert score > 0.0

    def test_score_content_overlap_capped_below_doctype_signal(self):
        # The content score is capped at 0.9 so a document-type match (0.92)
        # always dominates when both fire.
        score = _score_content_overlap(
            _policy_themes_ontology(),
            " ".join(
                [
                    "brede welvaart",
                    "leefbaarheid",
                    "vergrijzing",
                    "bevolkingsdaling",
                    "gezondheid",
                    "werkgelegenheid",
                    "voorzieningenniveau",
                    "materiele welvaart",
                ]
            ),
        )
        assert score <= 0.9


# ---------------------------------------------------------------------------
# Sample / dedup helper tests
# ---------------------------------------------------------------------------


class TestSampleForPass1:
    def test_concatenates_chunks_up_to_budget(self):
        chunks = [
            {"text": "alpha", "id": "1"},
            {"text": "beta", "id": "2"},
        ]
        sample = _sample_text_for_pass1(chunks)
        assert "alpha" in sample
        assert "beta" in sample

    def test_truncates_oversized(self):
        long_chunk = "x" * 100_000
        chunks = [{"text": long_chunk, "id": "1"}]
        sample = _sample_text_for_pass1(chunks)
        # Budget is 6000 chars; allow a small overhead for the join
        # separator (\n\n adds 2 chars).
        assert len(sample) <= 6000

    def test_skips_empty_chunks(self):
        chunks = [
            {"text": "", "id": "1"},
            {"text": "   ", "id": "2"},
            {"text": "real text", "id": "3"},
        ]
        sample = _sample_text_for_pass1(chunks)
        assert sample == "real text"


class TestDedupeExtensions:
    def test_dedup_by_type_name(self):
        extensions = [
            {"type_name": "Foo", "rationale": "first"},
            {"type_name": "Bar", "rationale": "another"},
            {"type_name": "Foo", "rationale": "duplicate"},
        ]
        result = _dedupe_extensions(extensions)
        assert len(result) == 2
        # First-seen wins.
        foo = [e for e in result if e["type_name"] == "Foo"][0]
        assert foo["rationale"] == "first"

    def test_drops_missing_type_name(self):
        extensions = [{"rationale": "no name"}, {"type_name": "Bar"}]
        result = _dedupe_extensions(extensions)
        assert len(result) == 1
        assert result[0]["type_name"] == "Bar"


# ---------------------------------------------------------------------------
# Merge-step tests (AC #1)
# ---------------------------------------------------------------------------


def _pass_result(
    schema, *, kept=0, extracted=0, abstained=0, removed=0, judged=0, chunks=10
):
    """A `run_pass2` result with the counter metadata it really emits."""
    return (
        schema,
        ExtractionResult(
            entities=[
                ExtractedEntity(text=f"{schema}-e{i}", label="L")
                for i in range(kept)
            ],
            metadata={
                "ontology_name": schema,
                "chunk_count": chunks,
                "extensions_count": 0,
                "total_entities": kept,
                "total_relations": 0,
                "parse_failures": 0,
                "entities_extracted": extracted,
                "entities_kept": kept,
                "not_a_concept_removed": removed,
                "not_a_concept_judged": judged,
                "abstained_chunks": abstained,
            },
        ),
    )


class TestRelationProvenanceSurvivesCollapse:
    """Track N.5c / R2 — the reversibility claim, made true.

    The relation merge collapses duplicates on
    `(source, target, relation_type)` and keeps the highest confidence. The
    loser's `relation_source` went with it, so two passes that both found
    `is_a(dorpshuizen, ontmoetingspunten)` — the LLM at 0.9 in one, the Hearst
    seeder at 0.5 in the other — produced a surviving edge with no trace of the
    seeder. That falsifies what the provenance is for: that one
    `WHERE relation_source = ...` drops everything a pass contributed.

    Latent today, since the Hearst miner ships off (N.5b) and is the only `is_a`
    producer left. Not `is_a`-specific though: it applies to any relation
    carrying provenance, and it returns the moment that flag does.
    """

    @staticmethod
    def _rel(conf, source=None, target="ontmoetingspunten"):
        return ExtractedRelation(
            source_entity="dorpshuizen",
            target_entity=target,
            relation_type="is_a",
            confidence=conf,
            properties={"relation_source": source} if source else {},
        )

    def test_the_loser_s_provenance_is_not_dropped(self):
        merged = _merge_results(
            [
                ("deals", ExtractionResult(relations=[self._rel(0.9)])),
                ("policy", ExtractionResult(relations=[self._rel(0.5, "hearst")])),
            ]
        )
        assert len(merged.relations) == 1
        assert merged.relations[0].confidence == 0.9  # max-confidence still wins
        assert merged.relations[0].properties["relation_sources"] == ["hearst"]

    def test_every_contributor_is_named_not_just_the_loser(self):
        merged = _merge_results(
            [
                ("deals", ExtractionResult(relations=[self._rel(0.9, "llm")])),
                ("policy", ExtractionResult(relations=[self._rel(0.5, "hearst")])),
            ]
        )
        assert merged.relations[0].properties["relation_sources"] == ["hearst", "llm"]

    def test_a_single_contributor_adds_no_list(self):
        """A relation nothing collapsed into keeps the shape it already had —
        the fix must not put a redundant key on every edge in the graph.
        """
        merged = _merge_results(
            [
                ("deals", ExtractionResult(relations=[self._rel(0.5, "hearst")])),
                ("policy", ExtractionResult(relations=[self._rel(0.4, "hearst")])),
            ]
        )
        props = merged.relations[0].properties
        assert props == {"relation_source": "hearst"}
        assert "relation_sources" not in props

    def test_relations_with_no_provenance_are_untouched(self):
        merged = _merge_results(
            [
                ("deals", ExtractionResult(relations=[self._rel(0.9)])),
                ("policy", ExtractionResult(relations=[self._rel(0.5)])),
            ]
        )
        assert merged.relations[0].properties == {}

    def test_distinct_relations_do_not_share_provenance(self):
        """A vacuity guard: the union is per KEY. Pooling it across the merge
        would tag every edge with every source and still satisfy the tests above.
        """
        merged = _merge_results(
            [
                (
                    "deals",
                    ExtractionResult(
                        relations=[
                            self._rel(0.9),
                            self._rel(0.9, target="voorzieningen"),
                        ]
                    ),
                ),
                ("policy", ExtractionResult(relations=[self._rel(0.5, "hearst")])),
            ]
        )
        by_target = {r.target_entity: r.properties for r in merged.relations}
        assert by_target["ontmoetingspunten"]["relation_sources"] == ["hearst"]
        assert "relation_sources" not in by_target["voorzieningen"]


class TestPassCountersSurviveTheMerge:
    """Track N.5a — the observability N.3 measured, on the path production takes.

    `_merge_results` built a fresh four-key metadata dict, so every counter
    `run_pass2` emitted was discarded whenever more than one schema applied. The
    cost is not that the numbers went missing — it is that their absence reads as
    a CLEAN RUN: measured on the fixture below, the merged result reported
    `over_generation_rate` 0.00 where the passes had culled 14 entities to 5, and
    `abstain_rate` 0.00 where 9 of 20 chunk-passes yielded nothing.

    Every test here asserts on a MERGED result — two or more passes. The
    single-schema path returns its input untouched, so the counters were always
    intact there and a test on that path could not fail.
    """

    def test_the_counters_reach_the_merged_record(self):
        merged = _merge_results(
            [
                _pass_result("deals", kept=3, extracted=9, abstained=2, removed=6, judged=4),
                _pass_result("policy", kept=2, extracted=5, abstained=7, removed=3, judged=3),
            ]
        )
        assert merged.metadata["entities_extracted"] == 14
        assert merged.metadata["not_a_concept_removed"] == 9
        assert merged.metadata["not_a_concept_judged"] == 7
        assert merged.metadata["abstained_chunks"] == 9
        assert merged.metadata["chunk_count"] == 20

    def test_a_document_that_yields_nothing_explains_itself(self):
        """The measured case: `Bennett_test.pdf` produced ten chunks and zero
        entities, and its stored record could not say whether the model found
        nothing or the not-a-concept gate removed everything. Those are opposite
        diagnoses — one is a document with no concepts, the other is a gate that
        is too aggressive — and the summed totals alone still cannot tell them
        apart when the passes disagree. `per_schema` can.
        """
        merged = _merge_results(
            [
                # The model found nothing at all in this pass.
                _pass_result("deals", kept=0, extracted=0, abstained=10),
                # The model found seven and the gate removed all seven.
                _pass_result("policy", kept=0, extracted=7, abstained=3, removed=7, judged=7),
            ]
        )
        assert merged.entities == []

        per_schema = merged.metadata["per_schema"]
        assert per_schema["deals"]["entities_extracted"] == 0
        assert per_schema["deals"]["not_a_concept_removed"] == 0
        assert per_schema["policy"]["entities_extracted"] == 7
        assert per_schema["policy"]["not_a_concept_removed"] == 7

    def test_entities_kept_counts_gate_survivors_not_distinct_entities(self):
        """Two passes finding the SAME entity is duplication, not
        over-generation, and folding one into the other would make the merge
        itself look like a gate rejection.
        """
        same = ExtractionResult(
            entities=[ExtractedEntity(text="Alice", label="Person")],
            metadata={"entities_extracted": 4, "entities_kept": 1, "chunk_count": 5},
        )
        merged = _merge_results([("deals", same), ("policy", same)])

        assert len(merged.entities) == 1  # one distinct entity...
        assert merged.metadata["entities_kept"] == 2  # ...two gate survivors
        assert merged.metadata["total_entities"] == 1
        # And the difference is published rather than left to be inferred.
        assert merged.metadata["merged_duplicates_collapsed"] == 1

    def test_the_merged_record_no_longer_reads_as_a_clean_run(self):
        """The property the defect actually broke, stated as a ratio rather than
        as a key list: a run that culled most of what the model produced must not
        measure as one that culled nothing.
        """
        merged = _merge_results(
            [
                _pass_result("deals", kept=3, extracted=9, abstained=2, removed=6, judged=4),
                _pass_result("policy", kept=2, extracted=5, abstained=7, removed=3, judged=3),
            ]
        )
        extracted = merged.metadata["entities_extracted"]
        survivors = merged.metadata["entities_kept"]
        assert extracted > 0 and survivors < extracted
        assert (extracted - survivors) / extracted > 0.5

        abstained = merged.metadata["abstained_chunks"]
        chunks = merged.metadata["chunk_count"]
        assert chunks > 0 and 0.0 < abstained / chunks < 1.0

    def test_a_pass_from_before_these_keys_contributes_zero(self):
        """A result with no counter metadata degrades to a low count, not an
        error — the keys are additive and a missing pass is worth 0.
        """
        merged = _merge_results(
            [
                _pass_result("deals", kept=1, extracted=4, abstained=1, removed=3, judged=2),
                ("legacy", ExtractionResult(entities=[], metadata={})),
            ]
        )
        assert merged.metadata["entities_extracted"] == 4
        assert merged.metadata["per_schema"]["legacy"]["entities_extracted"] == 0

    def test_a_non_numeric_counter_does_not_break_the_merge(self):
        """Metadata is a free-form bag; a bad value must not cost the extraction
        that already succeeded.
        """
        merged = _merge_results(
            [
                _pass_result("deals", kept=1, extracted=4),
                (
                    "broken",
                    ExtractionResult(
                        entities=[],
                        metadata={"entities_extracted": "not a number"},
                    ),
                ),
            ]
        )
        assert merged.metadata["entities_extracted"] == 4

    def test_the_existing_merge_output_is_not_disturbed(self):
        """A vacuity guard for the whole class: the keys the merge already
        published still say what they said, so "the counters arrived" cannot be
        satisfied by a merge that broke everything else.
        """
        merged = _merge_results(
            [
                _pass_result("deals", kept=2, extracted=5),
                _pass_result("policy", kept=1, extracted=3),
            ]
        )
        assert merged.metadata["merged_from_schemas"] == ["deals", "policy"]
        assert merged.metadata["schema_count"] == 2
        assert merged.metadata["total_entities"] == len(merged.entities) == 3


class TestMergeResults:
    """Direct tests of ``_merge_results`` so the merge correctness
    contract is testable without spinning up the full pipeline."""

    def test_entity_in_two_passes_gets_type_tags(self):
        result_a = ExtractionResult(
            entities=[
                ExtractedEntity(
                    text="Alice", label="Researcher", confidence=0.9
                )
            ]
        )
        result_b = ExtractionResult(
            entities=[
                ExtractedEntity(text="Alice", label="Person", confidence=0.7)
            ]
        )
        merged = _merge_results([("scholarly", result_a), ("general", result_b)])
        assert len(merged.entities) == 1
        e = merged.entities[0]
        assert len(e.type_tags) == 2
        assert "Researcher" in e.type_tags
        assert "Person" in e.type_tags
        # Higher confidence wins for primary_type and label.
        assert e.primary_type == "Researcher"
        assert e.label == "Researcher"
        assert e.confidence == 0.9

    def test_entity_in_single_pass_has_one_tag(self):
        result_a = ExtractionResult(
            entities=[
                ExtractedEntity(
                    text="Alice", label="Researcher", confidence=0.9
                )
            ]
        )
        result_b = ExtractionResult(
            entities=[
                ExtractedEntity(text="Bob", label="Person", confidence=0.7)
            ]
        )
        merged = _merge_results([("scholarly", result_a), ("general", result_b)])
        alice = [e for e in merged.entities if e.text == "Alice"][0]
        bob = [e for e in merged.entities if e.text == "Bob"][0]
        assert len(alice.type_tags) == 1
        assert len(bob.type_tags) == 1

    def test_name_normalisation_merges_variants(self):
        # "  Foo Bar.  " and "foo bar" should merge into one entity.
        result_a = ExtractionResult(
            entities=[
                ExtractedEntity(
                    text="  Foo Bar.  ", label="LabelA", confidence=0.6
                )
            ]
        )
        result_b = ExtractionResult(
            entities=[
                ExtractedEntity(text="foo bar", label="LabelB", confidence=0.85)
            ]
        )
        merged = _merge_results([("a", result_a), ("b", result_b)])
        assert len(merged.entities) == 1
        merged_entity = merged.entities[0]
        # Highest-confidence pass's text wins.
        assert merged_entity.text == "foo bar"
        assert merged_entity.primary_type == "LabelB"
        assert set(merged_entity.type_tags) == {"LabelA", "LabelB"}

    def test_relation_dedup_max_confidence_wins(self):
        result_a = ExtractionResult(
            relations=[
                ExtractedRelation(
                    source_entity="Alice",
                    target_entity="MIT",
                    relation_type="AFFILIATED_WITH",
                    confidence=0.6,
                )
            ]
        )
        result_b = ExtractionResult(
            relations=[
                ExtractedRelation(
                    source_entity="alice",
                    target_entity="mit",
                    relation_type="AFFILIATED_WITH",
                    confidence=0.85,
                )
            ]
        )
        merged = _merge_results([("a", result_a), ("b", result_b)])
        assert len(merged.relations) == 1
        assert merged.relations[0].confidence == 0.85

    def test_relation_different_types_not_deduped(self):
        result_a = ExtractionResult(
            relations=[
                ExtractedRelation(
                    source_entity="A",
                    target_entity="B",
                    relation_type="X",
                    confidence=0.6,
                )
            ]
        )
        result_b = ExtractionResult(
            relations=[
                ExtractedRelation(
                    source_entity="A",
                    target_entity="B",
                    relation_type="Y",
                    confidence=0.8,
                )
            ]
        )
        merged = _merge_results([("a", result_a), ("b", result_b)])
        assert len(merged.relations) == 2

    def test_single_schema_input_is_passthrough(self):
        # AC #4: with one schema, output is bit-identical to input.
        result_a = ExtractionResult(
            entities=[
                ExtractedEntity(text="Alice", label="X", confidence=0.9)
            ],
            relations=[
                ExtractedRelation(
                    source_entity="Alice",
                    target_entity="MIT",
                    relation_type="AT",
                    confidence=0.8,
                )
            ],
            metadata={"ontology_name": "scholarly"},
        )
        merged = _merge_results([("scholarly", result_a)])
        # No type_tags / primary_type added in the single-schema path.
        assert merged.entities[0].type_tags == []
        assert merged.entities[0].primary_type is None
        assert merged.entities[0].label == "X"
        assert merged.entities[0].confidence == 0.9
        # Metadata preserved.
        assert merged.metadata == result_a.metadata

    def test_empty_input_returns_empty_result(self):
        merged = _merge_results([])
        assert merged.entities == []
        assert merged.relations == []

    def test_entity_with_empty_text_dropped(self):
        result = ExtractionResult(
            entities=[
                ExtractedEntity(text="   ", label="X", confidence=0.5),
                ExtractedEntity(text="Real", label="Y", confidence=0.5),
            ]
        )
        # With two schemas (so merge actually runs).
        merged = _merge_results([("a", result), ("b", ExtractionResult())])
        # Empty-text entity dropped.
        assert len(merged.entities) == 1
        assert merged.entities[0].text == "Real"

    def test_relation_endpoints_relinked_to_canonical_entity_text(self):
        """B.4 fix: when entity merge picks one pass-winner's surface form
        and relation merge picks another's, the relation's endpoint must
        be rewritten to the canonical entity ``text``.

        Scenario (mirrors the B.1e review's concrete bug report):

        - Pass-A: ``Alice@0.9`` + ``Alice→MIT@0.6``
        - Pass-B: ``alice@0.6`` + ``alice→MIT@0.9``

        Entity merge: Alice@0.9 wins → ``text="Alice"``.
        Relation merge: alice→MIT@0.9 wins (Pass-B has higher conf).

        Without the B.4 fix the relation's ``source_entity`` stays
        ``"alice"`` (Pass-B's raw form), leaving the relation pointing
        at a name no entity owns. The fix rewrites it to ``"Alice"``.
        """
        result_a = ExtractionResult(
            entities=[
                ExtractedEntity(text="Alice", label="Researcher", confidence=0.9),
                ExtractedEntity(text="MIT", label="Institution", confidence=0.9),
            ],
            relations=[
                ExtractedRelation(
                    source_entity="Alice",
                    target_entity="MIT",
                    relation_type="AFFILIATED_WITH",
                    confidence=0.6,
                ),
            ],
        )
        result_b = ExtractionResult(
            entities=[
                ExtractedEntity(text="alice", label="Person", confidence=0.6),
                ExtractedEntity(text="mit", label="Org", confidence=0.6),
            ],
            relations=[
                ExtractedRelation(
                    source_entity="alice",
                    target_entity="mit",
                    relation_type="AFFILIATED_WITH",
                    confidence=0.9,
                ),
            ],
        )

        merged = _merge_results(
            [("scholarly", result_a), ("general", result_b)]
        )

        # Entity merge: canonical text is the highest-confidence form.
        alice_entity = [
            e for e in merged.entities
            if e.text.lower().strip() == "alice"
        ][0]
        mit_entity = [
            e for e in merged.entities
            if e.text.lower().strip() == "mit"
        ][0]
        assert alice_entity.text == "Alice", (
            f"Entity merge should keep 'Alice' (conf 0.9 won over 0.6); "
            f"got {alice_entity.text!r}"
        )
        assert mit_entity.text == "MIT"

        # Relation merge picked Pass-B (conf 0.9), but the B.4 fix
        # rewrites endpoints to the canonical entity surface form.
        assert len(merged.relations) == 1
        rel = merged.relations[0]
        assert rel.confidence == 0.9, "Higher-confidence relation must win"
        assert rel.source_entity == "Alice", (
            f"B.4 re-link: source_entity should match canonical entity "
            f"text 'Alice'; got {rel.source_entity!r}"
        )
        assert rel.target_entity == "MIT", (
            f"B.4 re-link: target_entity should match canonical entity "
            f"text 'MIT'; got {rel.target_entity!r}"
        )

    def test_relation_endpoint_unchanged_when_already_canonical(self):
        """B.4 fix: the rewrite is a no-op when the surviving relation
        already references the canonical entity surface form.

        Pins that the new pass doesn't break the common case where the
        entity merge and the relation merge both pick the same pass.
        """
        result_a = ExtractionResult(
            entities=[
                ExtractedEntity(text="Alice", label="Researcher", confidence=0.9),
                ExtractedEntity(text="MIT", label="Institution", confidence=0.9),
            ],
            relations=[
                ExtractedRelation(
                    source_entity="Alice",
                    target_entity="MIT",
                    relation_type="AFFILIATED_WITH",
                    confidence=0.95,
                ),
            ],
        )
        result_b = ExtractionResult(
            entities=[
                ExtractedEntity(text="alice", label="Person", confidence=0.5),
            ],
        )
        merged = _merge_results([("a", result_a), ("b", result_b)])
        assert len(merged.relations) == 1
        assert merged.relations[0].source_entity == "Alice"
        assert merged.relations[0].target_entity == "MIT"

    def test_relation_with_orphan_endpoint_passes_through_unchanged(self):
        """B.4 re-link contract: when a relation references an entity
        that no schema yielded, the relation's endpoint stays as the
        LLM-raw text (NOT rewritten, NOT dropped).

        Rationale: downstream KG persistence
        (``entity_persistence_service``) drops relations whose endpoints
        don't match any entity text, so an orphan relation will be
        silently filtered. We deliberately *do not* synthesise a
        placeholder entity here — the orchestrator stays pure.

        Pins minor #1 from the B.1f attempt-1 review.
        """
        # Pass A: knows Alice.
        result_a = ExtractionResult(
            entities=[
                ExtractedEntity(text="Alice", label="Researcher", confidence=0.9),
            ],
            relations=[
                ExtractedRelation(
                    source_entity="Alice",
                    target_entity="MIT",  # MIT is not in any entity list!
                    relation_type="AFFILIATED_WITH",
                    confidence=0.8,
                ),
            ],
        )
        # Pass B is empty (forces the merge path to actually run; the
        # single-schema short-circuit would bypass the re-link).
        result_b = ExtractionResult(entities=[])

        merged = _merge_results([("a", result_a), ("b", result_b)])

        assert len(merged.relations) == 1
        rel = merged.relations[0]
        # Source endpoint IS in the entity list → re-linked to canon.
        assert rel.source_entity == "Alice"
        # Target endpoint is an orphan → left as the LLM-raw text so
        # downstream filtering decides what to do.
        assert rel.target_entity == "MIT"


class TestMergedEntityHelper:
    """Direct tests for the MergedEntity bookkeeper.

    Mostly defensive — ensures the type-tag dedup invariant inside
    ``add()`` doesn't accidentally explode tag lists.
    """

    def test_duplicate_label_not_duplicated(self):
        e1 = ExtractedEntity(text="Alice", label="Researcher", confidence=0.6)
        e2 = ExtractedEntity(text="Alice", label="Researcher", confidence=0.9)
        m = MergedEntity(e1)
        m.add(e2)
        assert m.type_tags == ["Researcher"]
        assert m.best_confidence == 0.9
        assert m.best_label == "Researcher"


# ---------------------------------------------------------------------------
# Token-budget guard (AC #2)
# ---------------------------------------------------------------------------


class TestTokenBudgetGuard:
    """Pass-2 prompts emitted by the orchestrator stay under budget.

    Cross-check on the orchestrator's contract: handing ``run_pass2``
    a realistic chunk + ontology must produce prompts ≤ 3000 tokens
    (we enforce the 2400-token internal cap from B.1d, which has the
    20% safety margin baked in).
    """

    def test_each_pass2_prompt_under_3000_tokens(self):
        ontology = _scholarly_ontology()
        chunks = _three_chunks()
        for chunk in chunks:
            prompt = build_pass2_prompt(ontology, chunk["text"], [])
            assert _estimate_tokens(prompt) <= 3000

    def test_internal_budget_target_holds(self):
        # The B.1d-internal legacy default cap. Raised 2400 → 2800 in N.1
        # to reserve room for the exhaustive-extraction prompt overhead
        # (~+290 tokens). If this assertion drifts, B.1e's perf guarantee
        # drifts too.
        assert TOKEN_BUDGET_TARGET == 2800


# ---------------------------------------------------------------------------
# run_multi_schema end-to-end tests (AC #1, #3, #4)
# ---------------------------------------------------------------------------


def _make_dual_pass_callers(
    pass1_responses: Dict[str, str],
    pass2_responses: Dict[str, str],
):
    """Build a fake LLM caller that branches on prompt content.

    The orchestrator calls the same caller for Pass-1 and Pass-2;
    we distinguish by looking for unique markers in each prompt.
    Returns a sync callable matching the ``LLMCaller`` signature.
    """

    def fake_llm(system_prompt: str, user_prompt: str, model: str) -> str:
        # Pass-1 user prompts begin with the schema-validation header
        # built by ``build_pass1_prompt``; Pass-2 begins with the
        # typed-extraction header from ``build_pass2_prompt``.
        is_pass1 = "# Pass-1 Schema Validation" in user_prompt
        responses = pass1_responses if is_pass1 else pass2_responses
        # Schema name appears as ``## Schema: <name>`` in Pass-1 and
        # ``**<name>**`` in Pass-2 — match whichever pattern fits.
        for schema_name, payload in responses.items():
            if is_pass1 and f"## Schema: {schema_name}" in user_prompt:
                return payload
            if not is_pass1 and f"**{schema_name}**" in user_prompt:
                return payload
        # Default: empty result.
        return json.dumps({}) if is_pass1 else json.dumps({"entities": []})

    return fake_llm


class TestRunMultiSchema:
    @pytest.mark.asyncio
    async def test_two_schemas_persist_pass1_records(self):
        # AC #3: pass1_results row written for each schema attempted.
        repo = _FakePass1Repo()
        applicable = [
            (_scholarly_ontology(), 0.92),
            (_policy_ontology(), 0.55),
        ]
        fake_llm = _make_dual_pass_callers(
            pass1_responses={
                "scholarly": _make_pass1_response("scholarly", 0.9),
                "policy": _make_pass1_response("policy", 0.5),
            },
            pass2_responses={
                "scholarly": _make_pass2_response(
                    [{"text": "Alice", "label": "Researcher", "confidence": 0.9}]
                ),
                "policy": _make_pass2_response([]),
            },
        )
        result, decision = await run_multi_schema(
            source_id="source:abc",
            notebook_id="notebook:xyz",
            chunks=_three_chunks(),
            applicable_schemas=applicable,
            llm_caller=fake_llm,
            pass1_repo=repo,
        )
        assert len(repo.records) == 2
        schemas = {r.schema_attempted for r in repo.records}
        assert schemas == {"scholarly", "policy"}
        # Per AC #3: source and notebook propagate.
        assert all(r.source == "source:abc" for r in repo.records)
        assert all(r.notebook == "notebook:xyz" for r in repo.records)

    @pytest.mark.asyncio
    async def test_one_run_stamps_one_run_id_on_all_its_rows(self):
        """Track PC.1. `pass1_results` is append-only and one run writes one row
        per applied schema, so without a shared id the rows of this run cannot be
        told apart from the rows of the run before it.

        The notebook's rolling coverage depends on exactly that: after a curator
        edits the schema set and re-extracts, the abandoned schema's row must stop
        counting. A review measured that grouping on `(source, schema_attempted)`
        instead — the only alternative available without this id — still reported
        0.9 while the current run scored 0.30, so coverage could never fall.

        Losing the stamp is silent: `_rolling_coverage` treats an unstamped row as
        legacy and keeps averaging, so nothing downstream raises. Hence a test.
        """
        applicable = [
            (_scholarly_ontology(), 0.92),
            (_policy_ontology(), 0.55),
        ]
        fake_llm = _make_dual_pass_callers(
            pass1_responses={
                "scholarly": _make_pass1_response("scholarly", 0.9),
                "policy": _make_pass1_response("policy", 0.5),
            },
            pass2_responses={
                "scholarly": _make_pass2_response([]),
                "policy": _make_pass2_response([]),
            },
        )

        async def _one_run(repo):
            await run_multi_schema(
                source_id="source:abc",
                notebook_id="notebook:xyz",
                chunks=_three_chunks(),
                applicable_schemas=applicable,
                llm_caller=fake_llm,
                pass1_repo=repo,
            )
            return [r.pass1_metadata.get("run_id") for r in repo.records]

        first = await _one_run(_FakePass1Repo())
        assert len(first) == 2
        assert all(first), "every row of a run must carry a run_id"
        assert len(set(first)) == 1, "one run, one id across all its schemas"

        # ...and a SECOND extraction of the same source is a different run, or
        # the previous run's rows would never be superseded.
        second = await _one_run(_FakePass1Repo())
        assert set(second).isdisjoint(set(first))

    @pytest.mark.asyncio
    async def test_multi_tagged_entity_from_three_passes(self):
        # AC #1: entity in ≥ 2 passes has type_tags length ≥ 2 and
        # primary_type from highest-confidence pass.
        applicable = [
            (_scholarly_ontology(), 0.92),
            (_policy_ontology(), 0.6),
            (_base_ontology(), 0.4),
        ]
        fake_llm = _make_dual_pass_callers(
            pass1_responses={
                "scholarly": _make_pass1_response("scholarly", 0.85),
                "policy": _make_pass1_response("policy", 0.7),
                "general": _make_pass1_response("general", 0.5),
            },
            pass2_responses={
                "scholarly": _make_pass2_response(
                    [{"text": "Alice", "label": "Researcher", "confidence": 0.95}]
                ),
                "policy": _make_pass2_response(
                    [{"text": "Alice", "label": "Policymaker", "confidence": 0.6}]
                ),
                "general": _make_pass2_response(
                    [{"text": "Alice", "label": "Person", "confidence": 0.8}]
                ),
            },
        )
        result, decision = await run_multi_schema(
            source_id="source:abc",
            notebook_id="notebook:xyz",
            chunks=_three_chunks(),
            applicable_schemas=applicable,
            llm_caller=fake_llm,
            pass1_repo=_FakePass1Repo(),
        )
        assert len(result.entities) == 1
        alice = result.entities[0]
        assert len(alice.type_tags) == 3
        # primary_type from highest-confidence pass.
        assert alice.primary_type == "Researcher"
        assert alice.confidence == 0.95

    @pytest.mark.asyncio
    async def test_soft_nudge_extension_suggested_in_band(self):
        applicable = [(_scholarly_ontology(), 0.92)]
        fake_llm = _make_dual_pass_callers(
            pass1_responses={
                "scholarly": _make_pass1_response("scholarly", 0.85)
            },
            pass2_responses={"scholarly": _make_pass2_response([])},
        )
        _result, decision = await run_multi_schema(
            source_id="source:1",
            notebook_id="notebook:1",
            chunks=_three_chunks(),
            applicable_schemas=applicable,
            llm_caller=fake_llm,
            pass1_repo=_FakePass1Repo(),
        )
        assert decision == SoftNudgeDecision.EXTENSION_SUGGESTED

    @pytest.mark.asyncio
    async def test_soft_nudge_none_above_threshold(self):
        applicable = [(_scholarly_ontology(), 0.92)]
        fake_llm = _make_dual_pass_callers(
            pass1_responses={
                "scholarly": _make_pass1_response("scholarly", 0.97)
            },
            pass2_responses={"scholarly": _make_pass2_response([])},
        )
        _result, decision = await run_multi_schema(
            source_id="source:1",
            notebook_id="notebook:1",
            chunks=_three_chunks(),
            applicable_schemas=applicable,
            llm_caller=fake_llm,
            pass1_repo=_FakePass1Repo(),
        )
        assert decision == SoftNudgeDecision.NONE

    @pytest.mark.asyncio
    async def test_soft_nudge_schema_mismatch_below_low(self):
        applicable = [(_scholarly_ontology(), 0.92)]
        fake_llm = _make_dual_pass_callers(
            pass1_responses={
                "scholarly": _make_pass1_response("scholarly", 0.75)
            },
            pass2_responses={"scholarly": _make_pass2_response([])},
        )
        _result, decision = await run_multi_schema(
            source_id="source:1",
            notebook_id="notebook:1",
            chunks=_three_chunks(),
            applicable_schemas=applicable,
            llm_caller=fake_llm,
            pass1_repo=_FakePass1Repo(),
        )
        assert decision == SoftNudgeDecision.SCHEMA_MISMATCH

    @pytest.mark.asyncio
    async def test_empty_applicable_schemas_short_circuits(self):
        result, decision = await run_multi_schema(
            source_id="source:1",
            notebook_id="notebook:1",
            chunks=_three_chunks(),
            applicable_schemas=[],
            llm_caller=lambda s, u, m: "{}",
            pass1_repo=_FakePass1Repo(),
        )
        assert result.entities == []
        assert result.relations == []
        assert decision == SoftNudgeDecision.NONE

    @pytest.mark.asyncio
    async def test_extensions_deduped_across_passes(self):
        applicable = [
            (_scholarly_ontology(), 0.92),
            (_policy_ontology(), 0.55),
        ]
        fake_llm = _make_dual_pass_callers(
            pass1_responses={
                "scholarly": _make_pass1_response(
                    "scholarly",
                    0.85,
                    extensions=[
                        {"type_name": "Postdoc", "parent_type": "Researcher"}
                    ],
                ),
                "policy": _make_pass1_response(
                    "policy",
                    0.6,
                    extensions=[
                        # Same type_name → should dedup.
                        {"type_name": "Postdoc", "parent_type": "Person"},
                        {"type_name": "Lobbyist", "parent_type": "Policymaker"},
                    ],
                ),
            },
            pass2_responses={
                "scholarly": _make_pass2_response([]),
                "policy": _make_pass2_response([]),
            },
        )
        result, _decision = await run_multi_schema(
            source_id="source:1",
            notebook_id="notebook:1",
            chunks=_three_chunks(),
            applicable_schemas=applicable,
            llm_caller=fake_llm,
            pass1_repo=_FakePass1Repo(),
        )
        proposed = result.metadata.get("proposed_extensions", [])
        names = {e["type_name"] for e in proposed}
        assert names == {"Postdoc", "Lobbyist"}

    @pytest.mark.asyncio
    async def test_single_schema_input_bitidentical_to_pass2(self):
        # AC #4: single-schema input is bit-identical to direct run_pass2.
        # We compare entity text, label, confidence — the contract surface.
        from ontology_extraction.pass2_typed_extraction import run_pass2

        ontology = _scholarly_ontology()
        chunks = _three_chunks()

        def pass2_only_llm(s: str, u: str, m: str) -> str:
            return _make_pass2_response(
                [{"text": "Alice", "label": "Researcher", "confidence": 0.95}]
            )

        # Direct B.1d path.
        direct = await run_pass2(
            chunks=chunks,
            ontology=ontology,
            accepted_extensions=[],
            llm_caller=pass2_only_llm,
        )

        # Multi-schema path with a single applicable schema.
        fake_llm = _make_dual_pass_callers(
            pass1_responses={
                "scholarly": _make_pass1_response("scholarly", 0.85)
            },
            pass2_responses={
                "scholarly": _make_pass2_response(
                    [{"text": "Alice", "label": "Researcher", "confidence": 0.95}]
                )
            },
        )
        multi, _decision = await run_multi_schema(
            source_id="source:1",
            notebook_id="notebook:1",
            chunks=chunks,
            applicable_schemas=[(ontology, 0.92)],
            llm_caller=fake_llm,
            pass1_repo=_FakePass1Repo(),
        )
        # Compare the load-bearing fields. Per spec single-schema
        # output is byte-identical: no type_tags, no primary_type, same
        # confidence and label.
        assert len(multi.entities) == len(direct.entities) == 3
        for d_ent, m_ent in zip(direct.entities, multi.entities):
            assert d_ent.text == m_ent.text
            assert d_ent.label == m_ent.label
            assert d_ent.confidence == m_ent.confidence
            assert m_ent.type_tags == []
            assert m_ent.primary_type is None

    @pytest.mark.asyncio
    async def test_pass1_failure_skips_schema_does_not_abort_run(self):
        # If Pass-1 raises, the orchestrator should log + skip the
        # schema, continue with remaining schemas, and still produce
        # a usable result.
        from ontology_extraction.pass1_schema_validation import Pass1ParseError

        applicable = [
            (_scholarly_ontology(), 0.92),
            (_policy_ontology(), 0.55),
        ]

        def failing_then_ok(s: str, u: str, m: str) -> str:
            # Scholarly pass-1 will fail (we raise from the caller).
            # Pass-1 for scholarly recognised by the schema-name in
            # the prompt. Policy and pass-2 responses succeed.
            if "# Pass-1 Schema Validation" in u and "## Schema: scholarly" in u:
                raise RuntimeError("boom")
            if "# Pass-1 Schema Validation" in u:
                return _make_pass1_response("policy", 0.6)
            # Pass-2: empty
            return _make_pass2_response([])

        repo = _FakePass1Repo()
        result, _decision = await run_multi_schema(
            source_id="source:1",
            notebook_id="notebook:1",
            chunks=_three_chunks(),
            applicable_schemas=applicable,
            llm_caller=failing_then_ok,
            pass1_repo=repo,
        )
        # Only the policy pass-1 row was persisted.
        assert len(repo.records) == 1
        assert repo.records[0].schema_attempted == "policy"

    @pytest.mark.asyncio
    async def test_pass1_repo_none_skips_persistence(self):
        applicable = [(_scholarly_ontology(), 0.92)]
        fake_llm = _make_dual_pass_callers(
            pass1_responses={
                "scholarly": _make_pass1_response("scholarly", 0.85)
            },
            pass2_responses={"scholarly": _make_pass2_response([])},
        )
        # Should not raise even though pass1_repo is None.
        result, _decision = await run_multi_schema(
            source_id="source:1",
            notebook_id="notebook:1",
            chunks=_three_chunks(),
            applicable_schemas=applicable,
            llm_caller=fake_llm,
            pass1_repo=None,
        )
        assert result is not None

    @pytest.mark.asyncio
    async def test_pass1_repo_record_failure_does_not_abort(self):
        class _FlakyRepo:
            def __init__(self) -> None:
                self.calls = 0

            async def record(self, result: Pass1Result) -> str:
                self.calls += 1
                raise RuntimeError("disk full")

        repo = _FlakyRepo()
        applicable = [(_scholarly_ontology(), 0.92)]
        fake_llm = _make_dual_pass_callers(
            pass1_responses={
                "scholarly": _make_pass1_response("scholarly", 0.85)
            },
            pass2_responses={"scholarly": _make_pass2_response([])},
        )
        # Should not propagate the repo failure.
        result, _decision = await run_multi_schema(
            source_id="source:1",
            notebook_id="notebook:1",
            chunks=_three_chunks(),
            applicable_schemas=applicable,
            llm_caller=fake_llm,
            pass1_repo=repo,
        )
        assert repo.calls == 1
        assert result is not None

    @pytest.mark.asyncio
    async def test_per_schema_accepted_extensions_passed_through(self):
        applicable = [
            (_scholarly_ontology(), 0.92),
            (_policy_ontology(), 0.55),
        ]
        # Capture the accepted-extensions argument seen by each Pass-2
        # call so we can assert the per-schema routing.
        seen_for: Dict[str, List[Dict[str, Any]]] = {}

        async def fake_run_pass2(
            chunks: List[Dict[str, Any]],
            ontology: Ontology,
            accepted_extensions=None,
            llm_caller=None,
            model: str = "default",
            token_budget=None,
        ) -> ExtractionResult:
            seen_for[ontology.metadata.name] = accepted_extensions or []
            return ExtractionResult()

        fake_llm = _make_dual_pass_callers(
            pass1_responses={
                "scholarly": _make_pass1_response("scholarly", 0.85),
                "policy": _make_pass1_response("policy", 0.6),
            },
            pass2_responses={
                "scholarly": _make_pass2_response([]),
                "policy": _make_pass2_response([]),
            },
        )

        with patch(
            "ontology_extraction.multi_schema_orchestrator.run_pass2",
            side_effect=fake_run_pass2,
        ):
            await run_multi_schema(
                source_id="source:1",
                notebook_id="notebook:1",
                chunks=_three_chunks(),
                applicable_schemas=applicable,
                llm_caller=fake_llm,
                pass1_repo=_FakePass1Repo(),
                accepted_extensions_by_schema={
                    "scholarly": [{"type_name": "Postdoc"}],
                    "policy": [{"type_name": "Lobbyist"}],
                },
            )

        assert [e["type_name"] for e in seen_for["scholarly"]] == ["Postdoc"]
        assert [e["type_name"] for e in seen_for["policy"]] == ["Lobbyist"]

    @pytest.mark.asyncio
    async def test_metadata_carries_schemas_attempted(self):
        applicable = [
            (_scholarly_ontology(), 0.92),
            (_policy_ontology(), 0.55),
        ]
        fake_llm = _make_dual_pass_callers(
            pass1_responses={
                "scholarly": _make_pass1_response("scholarly", 0.85),
                "policy": _make_pass1_response("policy", 0.5),
            },
            pass2_responses={
                "scholarly": _make_pass2_response([]),
                "policy": _make_pass2_response([]),
            },
        )
        result, _decision = await run_multi_schema(
            source_id="source:1",
            notebook_id="notebook:1",
            chunks=_three_chunks(),
            applicable_schemas=applicable,
            llm_caller=fake_llm,
            pass1_repo=_FakePass1Repo(),
        )
        assert result.metadata["schemas_attempted"] == ["scholarly", "policy"]
        assert result.metadata["best_coverage"] == 0.85
        assert result.metadata["soft_nudge"] == "extension_suggested"

    @pytest.mark.asyncio
    async def test_pass2_failure_skips_schema(self):
        # If run_pass2 raises for one schema, the orchestrator should
        # continue and produce a result from the surviving schemas.
        applicable = [
            (_scholarly_ontology(), 0.92),
            (_policy_ontology(), 0.55),
        ]

        async def fake_run_pass2(
            chunks: List[Dict[str, Any]],
            ontology: Ontology,
            accepted_extensions=None,
            llm_caller=None,
            model: str = "default",
            token_budget=None,
        ) -> ExtractionResult:
            if ontology.metadata.name == "scholarly":
                raise RuntimeError("pass2 boom")
            return ExtractionResult(
                entities=[
                    ExtractedEntity(text="X", label="Y", confidence=0.5)
                ]
            )

        fake_llm = _make_dual_pass_callers(
            pass1_responses={
                "scholarly": _make_pass1_response("scholarly", 0.85),
                "policy": _make_pass1_response("policy", 0.5),
            },
            pass2_responses={
                "scholarly": _make_pass2_response([]),
                "policy": _make_pass2_response([]),
            },
        )
        with patch(
            "ontology_extraction.multi_schema_orchestrator.run_pass2",
            side_effect=fake_run_pass2,
        ):
            result, _decision = await run_multi_schema(
                source_id="source:1",
                notebook_id="notebook:1",
                chunks=_three_chunks(),
                applicable_schemas=applicable,
                llm_caller=fake_llm,
                pass1_repo=_FakePass1Repo(),
            )
        # Policy result survived; scholarly was skipped.
        assert len(result.entities) == 1
        assert result.entities[0].text == "X"


# ---------------------------------------------------------------------------
# Workflow integration smoke test (mode="multi")
# ---------------------------------------------------------------------------


class TestWorkflowMultiMode:
    @pytest.mark.asyncio
    async def test_workflow_multi_mode_dispatches_to_orchestrator(self):
        # Smoke test: ExtractionWorkflow.extract(mode="multi") should
        # invoke run_multi_schema and return the merged result.
        from ontology_extraction.workflow import ExtractionWorkflow

        wf = ExtractionWorkflow()
        applicable = [(_scholarly_ontology(), 0.92)]
        fake_llm = _make_dual_pass_callers(
            pass1_responses={
                "scholarly": _make_pass1_response("scholarly", 0.85)
            },
            pass2_responses={
                "scholarly": _make_pass2_response(
                    [{"text": "Alice", "label": "Researcher", "confidence": 0.9}]
                )
            },
        )
        called = {"yes": False}

        async def fake_run_multi(*args, **kwargs):
            called["yes"] = True
            return ExtractionResult(metadata={"soft_nudge": "none"}), SoftNudgeDecision.NONE

        with patch(
            "ontology_extraction.workflow.run_multi_schema",
            side_effect=fake_run_multi,
        ):
            result = await wf.extract(
                chunks=_three_chunks(),
                mode="multi",
                applicable_schemas=applicable,
                source_id="source:1",
                notebook_id="notebook:1",
            )
        assert called["yes"] is True
        assert result.metadata["soft_nudge"] == "none"

    @pytest.mark.asyncio
    async def test_workflow_multi_mode_empty_schemas_short_circuits(self):
        # mode="multi" + empty applicable_schemas → empty result,
        # NONE decision, no orchestrator call needed.
        from ontology_extraction.workflow import ExtractionWorkflow

        wf = ExtractionWorkflow()
        with patch(
            "ontology_extraction.workflow.run_multi_schema"
        ) as mocked:
            result = await wf.extract(
                chunks=_three_chunks(),
                mode="multi",
                applicable_schemas=[],
            )
            mocked.assert_not_called()
        assert result.metadata["soft_nudge"] == SoftNudgeDecision.NONE.value
