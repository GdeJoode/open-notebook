"""Tests for FilteringWorkflow end-to-end pipeline orchestration."""

import json
from typing import Any, Dict, List

import pytest

from shared.models.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
    FilteredResult,
)

from entity_filtering.config import FilteringConfig, OrphanConnectorConfig
from entity_filtering.resolution import orphan_connector as _orphan_connector
from entity_filtering.workflow import FilteringWorkflow


def _extraction_result(entities=None, relations=None, metadata=None):
    """Helper to build an ExtractionResult."""
    return ExtractionResult(
        entities=[ExtractedEntity(**e) for e in (entities or [])],
        relations=[ExtractedRelation(**r) for r in (relations or [])],
        metadata=metadata or {},
    )


def _entity(text, label="MISC", confidence=0.9):
    return {
        "text": text,
        "label": label,
        "properties": {},
        "confidence": confidence,
        "source_chunk_id": None,
    }


def _relation(source, target, rel_type="RELATED_TO", confidence=0.8):
    return {
        "source_entity": source,
        "target_entity": target,
        "relation_type": rel_type,
        "properties": {},
        "confidence": confidence,
        "source_chunk_id": None,
    }


class TestWorkflowFullPipeline:
    async def test_full_pipeline_with_extraction_result(self):
        entities = [
            _entity("John Doe", "PERSON", 0.9),
            _entity("Microsoft", "ORG", 0.95),
            _entity("123"),
        ]
        relations = [
            _relation("John Doe", "Microsoft", "WORKS_AT"),
            _relation("123", "Microsoft", "RELATED_TO"),
        ]
        extraction = _extraction_result(entities, relations)
        workflow = FilteringWorkflow()
        result = await workflow.process(extraction)

        assert isinstance(result, FilteredResult)
        # "123" should be removed (pure number noise)
        result_texts = {e.text for e in result.entities}
        assert "John Doe" in result_texts
        assert "Microsoft" in result_texts
        assert "123" not in result_texts
        # Relation referencing "123" should be removed
        assert len(result.relations) == 1
        assert result.relations[0].source_entity == "John Doe"

    async def test_empty_input_returns_empty_filtered_result(self):
        extraction = _extraction_result()
        workflow = FilteringWorkflow()
        result = await workflow.process(extraction)

        assert isinstance(result, FilteredResult)
        assert len(result.entities) == 0
        assert len(result.relations) == 0
        assert len(result.removed_entities) == 0
        assert len(result.merged_entity_groups) == 0
        assert len(result.predicted_edges) == 0


class TestWorkflowNoiseRemoval:
    async def test_noise_entities_are_removed(self):
        entities = [
            _entity("Valid Entity", "MISC"),
            _entity("---"),
            _entity("et al."),
            _entity("https://example.com"),
        ]
        extraction = _extraction_result(entities)
        workflow = FilteringWorkflow()
        result = await workflow.process(extraction)

        result_texts = {e.text for e in result.entities}
        assert "Valid Entity" in result_texts
        assert "---" not in result_texts
        assert "et al." not in result_texts
        assert "https://example.com" not in result_texts

    async def test_removed_entities_tracked(self):
        entities = [
            _entity("Good", "MISC"),
            _entity("123"),
        ]
        extraction = _extraction_result(entities)
        workflow = FilteringWorkflow()
        result = await workflow.process(extraction)

        removed_texts = {e.text for e in result.removed_entities}
        assert "123" in removed_texts


class TestWorkflowNormalization:
    async def test_normalization_merges_article_variants(self):
        entities = [
            _entity("The United Nations", "ORG", 0.8),
            _entity("United Nations", "ORG", 0.9),
            _entity("United Nations", "ORG", 0.85),
        ]
        extraction = _extraction_result(entities)
        workflow = FilteringWorkflow()
        result = await workflow.process(extraction)

        # All three should normalize to the same key and merge
        un_entities = [e for e in result.entities if "United Nations" in e.text]
        assert len(un_entities) == 1


class TestWorkflowDeduplication:
    async def test_deduplication_merges_case_variants(self):
        entities = [
            _entity("john doe", "PERSON", 0.8),
            _entity("John Doe", "PERSON", 0.9),
        ]
        extraction = _extraction_result(entities)
        workflow = FilteringWorkflow()
        result = await workflow.process(extraction)

        person_entities = [e for e in result.entities if "doe" in e.text.lower()]
        assert len(person_entities) == 1

    async def test_deduplication_disabled(self):
        config = FilteringConfig(dedup_enabled=False)
        entities = [
            _entity("john doe", "PERSON"),
            _entity("John Doe", "PERSON"),
        ]
        extraction = _extraction_result(entities)
        workflow = FilteringWorkflow(config=config)
        result = await workflow.process(extraction)

        # Without dedup, normalizer may still merge if they share a key,
        # but deduplicator step is skipped.
        assert len(result.merged_entity_groups) == 0


class TestWorkflowMetadata:
    async def test_metadata_includes_filtering_statistics(self):
        entities = [
            _entity("Alice", "PERSON"),
            _entity("Bob", "PERSON"),
            _entity("---"),
        ]
        relations = [
            _relation("Alice", "Bob", "KNOWS"),
        ]
        extraction = _extraction_result(
            entities, relations, metadata={"source": "test"}
        )
        workflow = FilteringWorkflow()
        result = await workflow.process(extraction)

        assert "filtering" in result.metadata
        stats = result.metadata["filtering"]
        assert stats["input_entities"] == 3
        assert stats["input_relations"] == 1
        assert "output_entities" in stats
        assert "output_relations" in stats
        assert "removed_count" in stats
        assert "merge_groups" in stats
        assert "predicted_edges" in stats

    async def test_original_metadata_preserved(self):
        extraction = _extraction_result(
            metadata={"source": "test_doc", "version": "1.0"}
        )
        workflow = FilteringWorkflow()
        result = await workflow.process(extraction)

        assert result.metadata["source"] == "test_doc"
        assert result.metadata["version"] == "1.0"
        assert "filtering" in result.metadata


# ===========================================================================
# Stage 14: Orphan-connector integration (B.5a review attempt 2)
# ===========================================================================


class _MockOrphanRepo:
    """In-memory implementation of OrphanEntityRepoProtocol."""

    def __init__(self, orphans: List[Dict[str, Any]]):
        self._orphans = list(orphans)
        self.call_count = 0

    async def list_orphans_for_source(
        self, source_id: str
    ) -> List[Dict[str, Any]]:
        self.call_count += 1
        return list(self._orphans)


class _ScriptedLLM:
    """Async LLM stub: returns scripted responses, records every call."""

    def __init__(self, responses):
        if isinstance(responses, str):
            self._responses = [responses]
            self._sticky = True
        else:
            self._responses = list(responses)
            self._sticky = False
        self.calls: List[Dict[str, str]] = []

    async def __call__(
        self, system_prompt: str, user_prompt: str, model: str
    ) -> str:
        self.calls.append(
            {"system": system_prompt, "user": user_prompt, "model": model}
        )
        if self._sticky:
            return self._responses[0]
        if not self._responses:
            return "{}"
        return self._responses.pop(0)


def _orphan_entity_row(name: str, entity_id: str = "entity:o1") -> Dict[str, Any]:
    """Repository-shaped orphan entity row."""
    return {
        "id": entity_id,
        "canonical_name": name,
        "entity_type": "PERSON",
        "source_documents": ["source:demo"],
        "properties": {},
    }


def _confirm_response(rel_type: str = "KNOWS", confidence: float = 0.85) -> str:
    return json.dumps({"relation_type": rel_type, "confidence": confidence})


class TestStage14OrphanConnector:
    """M2 + M3: Stage 14 wiring inside ``FilteringWorkflow.process``."""

    async def test_stage14_disabled_skips_orphan_connect(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M2 #1: ``OrphanConnectorConfig(enabled=False)`` skips the stage.

        With the stage disabled, the orphan-connector module's ``run``
        function MUST NOT be invoked even when all DI inputs are
        present. We assert this with a monkeypatched sentinel that
        records call attempts.
        """
        call_log: List[Dict[str, Any]] = []

        async def _record_run(**kwargs: Any) -> List[ExtractedRelation]:
            call_log.append(kwargs)
            return []

        monkeypatch.setattr(_orphan_connector, "run", _record_run)

        config = FilteringConfig(
            orphan_connector=OrphanConnectorConfig(enabled=False)
        )
        workflow = FilteringWorkflow(config=config)
        extraction = _extraction_result(
            entities=[_entity("Alice", "PERSON")],
            relations=[],
        )

        result = await workflow.process(
            extraction,
            source_id="source:demo",
            chunks=[
                {
                    "id": "c-1",
                    "text": "Alice met Bob.",
                    "entities": ["Alice", "Bob"],
                }
            ],
            orphan_entity_repo=_MockOrphanRepo([_orphan_entity_row("Alice")]),
            orphan_llm_caller=_ScriptedLLM(_confirm_response()),
        )

        assert isinstance(result, FilteredResult)
        assert call_log == [], "orphan-connector must NOT be invoked when disabled"

    async def test_stage14_enabled_happy_path(self) -> None:
        """M2 #2: enabled + full DI -> confirmed relations appear in result.

        The orphan-connector confirms one orphan-partner pair; we assert
        the resulting :class:`ExtractedRelation` appears in
        ``result.relations`` with the orphan as the source and the
        partner as the target. ``properties.extraction_method`` is set
        to ``"orphan_connector"`` by the stage so downstream telemetry
        can attribute the edge.
        """
        config = FilteringConfig(
            orphan_connector=OrphanConnectorConfig(
                enabled=True, max_proposals_per_orphan=2, min_confidence=0.6
            )
        )
        workflow = FilteringWorkflow(config=config)
        # Two entities — Alice is the orphan (per repo); Bob is the
        # non-orphan partner reachable in the chunk.
        extraction = _extraction_result(
            entities=[
                _entity("Alice", "PERSON"),
                _entity("Bob", "PERSON"),
            ],
            relations=[],
        )

        result = await workflow.process(
            extraction,
            source_id="source:demo",
            chunks=[
                {
                    "id": "c-1",
                    "text": "Alice met Bob at the cafe.",
                    "entities": ["Alice", "Bob"],
                }
            ],
            orphan_entity_repo=_MockOrphanRepo(
                [_orphan_entity_row("Alice", "entity:alice")]
            ),
            orphan_llm_caller=_ScriptedLLM(_confirm_response("KNOWS", 0.85)),
        )

        orphan_relations = [
            r
            for r in result.relations
            if r.properties.get("extraction_method") == "orphan_connector"
        ]
        assert len(orphan_relations) == 1
        assert orphan_relations[0].source_entity == "Alice"
        assert orphan_relations[0].target_entity == "Bob"
        assert orphan_relations[0].relation_type == "KNOWS"
        assert orphan_relations[0].confidence == pytest.approx(0.85)

    async def test_stage14_token_budget_exceeded_recovers(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """M2 #3: ``OrphanTokenBudgetExceeded`` mid-call -> graceful skip.

        The stage catches the budget breach, logs a WARNING, and
        continues — the workflow returns a normal :class:`FilteredResult`
        with no orphan-confirmed relations attached.
        """

        async def _raise_budget(**kwargs: Any):
            raise _orphan_connector.OrphanTokenBudgetExceeded(
                estimated=2400, budget=1500
            )

        monkeypatch.setattr(_orphan_connector, "run", _raise_budget)

        config = FilteringConfig(
            orphan_connector=OrphanConnectorConfig(enabled=True)
        )
        workflow = FilteringWorkflow(config=config)
        extraction = _extraction_result(
            entities=[_entity("Alice", "PERSON")],
            relations=[],
        )

        # Must not raise; the orphan-connector path catches and logs.
        result = await workflow.process(
            extraction,
            source_id="source:demo",
            chunks=[
                {
                    "id": "c-1",
                    "text": "Alice " * 5000,
                    "entities": ["Alice", "Bob"],
                }
            ],
            orphan_entity_repo=_MockOrphanRepo([_orphan_entity_row("Alice")]),
            orphan_llm_caller=_ScriptedLLM(_confirm_response()),
        )

        assert isinstance(result, FilteredResult)
        orphan_relations = [
            r
            for r in result.relations
            if r.properties.get("extraction_method") == "orphan_connector"
        ]
        assert orphan_relations == []

    async def test_stage14_enabled_missing_di_logs_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Minor-1 (review attempt 1): missing DI inputs while enabled
        emits a WARNING listing the missing fields.

        Until the production caller threads all four inputs through, a
        silent skip would hide the misconfiguration; the WARNING makes
        it diagnosable from the logs.

        loguru routes through the standard logging stack here via
        ``loguru.logger`` -> root logger when ``caplog`` is in use; we
        check both ``caplog.text`` and ``caplog.messages`` for the
        substring to keep the assert robust to logger configuration.
        """
        from loguru import logger as loguru_logger

        # Bridge loguru's WARNING to the standard logging stack so
        # caplog can capture it.
        import logging

        class _PropagateHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                logging.getLogger(record.name).handle(record)

        sink_id = loguru_logger.add(
            _PropagateHandler(),
            level="WARNING",
            format="{message}",
        )
        try:
            config = FilteringConfig(
                orphan_connector=OrphanConnectorConfig(enabled=True)
            )
            workflow = FilteringWorkflow(config=config)
            extraction = _extraction_result(
                entities=[_entity("Alice", "PERSON")],
                relations=[],
            )

            with caplog.at_level(logging.WARNING):
                # Pass NONE of the DI inputs.
                await workflow.process(extraction)
        finally:
            loguru_logger.remove(sink_id)

        joined = "\n".join(caplog.messages) + "\n" + caplog.text
        assert "Orphan-connector enabled but skipped" in joined
        # All four fields should be flagged as missing.
        for field in (
            "source_id",
            "chunks",
            "orphan_entity_repo",
            "orphan_llm_caller",
        ):
            assert field in joined

    async def test_orphan_relation_type_bypasses_ontology(self) -> None:
        """M3: orphan-confirmed relations bypass Stage 11 (ontology filter).

        Behaviour-pin (B.5a attempt 2): the ontology allows only
        ``"KNOWS"`` and ``"WORKS_WITH"``. The mock LLM returns the
        OUT-OF-ONTOLOGY relation type ``"ATE_LUNCH_WITH"``. The bypass
        is by design — see ``orphan_connector.run`` and the Stage 14
        docstring block in ``workflow.py`` for rationale. This test
        guards against accidental tightening; a future refactor that
        decides to validate the orphan output through the ontology
        filter MUST consciously delete this test.
        """
        from entity_filtering.config import OntologyValidationConfig
        from ontology_manager.schema import (
            EntityTypeDefinition,
            Ontology,
            OntologyMetadata,
            RelationshipTypeDefinition,
        )

        # Build a constrained ontology: PERSON entities only, two
        # allowed relation types — neither is "ATE_LUNCH_WITH".
        ontology = Ontology(
            metadata=OntologyMetadata(name="bypass-test", version="1.0"),
            entity_types={
                "PERSON": EntityTypeDefinition(
                    name="PERSON", description="A person"
                ),
            },
            relationship_types={
                "KNOWS": RelationshipTypeDefinition(
                    name="KNOWS",
                    description="acquaintance",
                    domain=["PERSON"],
                    range=["PERSON"],
                ),
                "WORKS_WITH": RelationshipTypeDefinition(
                    name="WORKS_WITH",
                    description="colleague",
                    domain=["PERSON"],
                    range=["PERSON"],
                ),
            },
        )

        config = FilteringConfig(
            ontology_validation=OntologyValidationConfig(
                enabled=True,
                filter_invalid_relations=True,
            ),
            orphan_connector=OrphanConnectorConfig(
                enabled=True,
                min_confidence=0.6,
            ),
        )
        workflow = FilteringWorkflow(config=config, ontology=ontology)
        extraction = _extraction_result(
            entities=[
                _entity("Alice", "PERSON"),
                _entity("Bob", "PERSON"),
            ],
            relations=[],
        )

        result = await workflow.process(
            extraction,
            source_id="source:demo",
            chunks=[
                {
                    "id": "c-1",
                    "text": "Alice and Bob ate lunch together.",
                    "entities": ["Alice", "Bob"],
                }
            ],
            orphan_entity_repo=_MockOrphanRepo(
                [_orphan_entity_row("Alice", "entity:alice")]
            ),
            # LLM returns an OUT-OF-ONTOLOGY relation type — the bypass
            # contract says this still ends up in result.relations.
            orphan_llm_caller=_ScriptedLLM(
                _confirm_response("ATE_LUNCH_WITH", 0.9)
            ),
        )

        orphan_relations = [
            r
            for r in result.relations
            if r.properties.get("extraction_method") == "orphan_connector"
        ]
        assert len(orphan_relations) == 1, (
            "orphan-confirmed relation must bypass Stage 11 (ontology) "
            "even when relation_type is not in the ontology"
        )
        assert orphan_relations[0].relation_type == "ATE_LUNCH_WITH"
