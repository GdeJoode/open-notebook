"""
Service bridging the app layer to the ontology-extraction and entity-filtering pipelines.

Fetches source chunks, runs ExtractionWorkflow, optionally runs
FilteringWorkflow for deduplication, and persists results to SurrealDB.

Phase B.1f wires the multi-schema orchestrator (B.1e) into
``run_extraction``. The default behaviour now is "multi-schema when a
notebook_id is supplied"; legacy CLI callers omit ``notebook_id`` and
fall through to the single-schema path so no existing test relies on
B.1e being green.

A back-compat kill-switch (``multi_schema_enabled=False``) forces the
single-schema path regardless of ``notebook_id``. Ops runbook: pass
``False`` via the handler payload to roll back if the orchestrator
misbehaves in production.
"""

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Tuple

from entity_filtering.config import FilteringConfig
from entity_filtering.workflow import FilteringWorkflow
from job_queue import JobPausedForReviewError
from llm_manager.gate import classificatie_van
from loguru import logger
from ontology_extraction.config import ExtractionConfig
from ontology_extraction.multi_schema_orchestrator import (
    detect_applicable_schemas,
)
from ontology_extraction.workflow import ExtractionWorkflow
from ontology_manager.canonical_bridge import _find_definition
from shared.config_coherence import ConfigurationError
from shared.services.metrics import record_metric
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories import (
    NotebookSchemaRepository,
    Pass1ResultRepository,
    SourceRepository,
)

from app_main.config import get_concept_alignment_enabled

if TYPE_CHECKING:
    from ontology_manager.schema import Ontology
    from shared.models import NotebookSchema

from app_main.services.entity_persistence_service import (
    EntityPersistenceService,
    load_typing_ontologies,
    typing_vocabulary,
)

# Explicit per-notebook override bundle (Track L.4). When an operator configures
# a ``base_ontology`` for a notebook, that is a declaration that the notebook is
# *about* a domain — so the related theme schema is bundled in alongside the
# base, regardless of any per-document content score. This is an EXPLICIT
# override (config beats auto-detection), NOT the auto-detection path: ordinary
# auto-detection selects ``policy_themes`` purely on content overlap via
# ``detect_applicable_schemas`` (no provenance gating). Keeping the bundle as
# data here means the override is extensible without touching the override logic.
# Applicability-sample shape (PC.1). `_score_content_overlap` counts distinct
# vocabulary phrases appearing anywhere in the sample, so what it needs is
# COVERAGE of the document, not contiguity: windows spread across the whole
# document beat one long prefix of it. The budget is generous because the scorer
# is substring matching over a lowercased string — measured in milliseconds for
# every ontology in the registry at this size.
# What `ensure_row` writes into `base_ontology`. Empty means "nobody has
# declared one", which is the honest state for a notebook whose first document
# has just arrived: the read paths resolve `base_ontology or
# DEFAULT_BASE_ONTOLOGY`, and `_apply_notebook_schema_default` skips a falsy
# name, so nothing is forced onto any extraction until a curator chooses.
NO_DECLARED_BASE = ""

_SAMPLE_BUDGET_CHARS = 40000
# 40 windows sharing that budget: 1000 characters each. Both numbers are
# measured on the project's 14 sources, sweeping each against how many documents
# detect a schema at all:
#
#   budget 20000   w=25 10/14   w=40 12/14   w=60 13/14   w=120 13/14
#   budget 40000   w=25 11/14   w=40 13/14   w=60 13/14   w=120 13/14
#   budget 60000   w=25 11/14   w=40 13/14   w=60 13/14   w=120 13/14
#
# 13/14 is the ceiling. Two settings reach it equally cheaply and this is the one
# with the WIDER windows (1000 characters against 333), because the scorer
# matches multi-word phrases as substrings and every window boundary is a place
# a phrase can be cut in half. Cost at this setting: 4 ms per document for all
# eleven ontologies. The curve is not monotonic — 50 windows scores 12/14 —
# which says the scorer is borderline for one document rather than that the
# sample is short.
#
# The remaining document is a journal article ("J of Common Market Studies"),
# and it is what the notebook-history fallback exists for.
_SAMPLE_MAX_WINDOWS = 40


def _applicability_sample(chunks: List[Dict[str, Any]]) -> Optional[str]:
    """Build the text `detect_applicable_schemas` scores a document by.

    Takes up to ``_SAMPLE_MAX_WINDOWS`` evenly spaced windows so a long document
    is represented end to end rather than by its cover page. Returns ``None``
    when there is no text at all, which is what the scorer expects for "skip the
    content path".
    """
    texts = [
        (chunk.get("text") or "") for chunk in chunks if (chunk.get("text") or "").strip()
    ]
    if not texts:
        return None

    # Indices spread across the WHOLE range, last one pinned to the final chunk.
    # A plain `range(0, n, step)` stops short of the end — for a 500-chunk
    # document it samples up to chunk 480 and never sees the conclusions or the
    # annexes, which is where a policy document names what it is about.
    n = len(texts)
    if n <= _SAMPLE_MAX_WINDOWS:
        indices = list(range(n))
    else:
        last = n - 1
        indices = sorted(
            {
                round(i * last / (_SAMPLE_MAX_WINDOWS - 1))
                for i in range(_SAMPLE_MAX_WINDOWS)
            }
        )

    # Each window gets an equal share of the budget. The previous version capped
    # every window at a fixed 1200 characters and stopped once the running total
    # reached the budget — and that break fires in ascending index order, so it
    # re-introduced the exact head bias the spread exists to remove. A review
    # measured it: at 1500-character chunks only 17 of 40 windows survived and
    # the document was scored on its first 41%, an order of magnitude worse than
    # the cover-page problem this started as. Sharing the budget means all 40
    # windows are always taken, whatever the chunk size.
    per_window = max(1, _SAMPLE_BUDGET_CHARS // _SAMPLE_MAX_WINDOWS)
    return " ".join(texts[i][:per_window] for i in indices)[:_SAMPLE_BUDGET_CHARS]


# The counters N.3 measures and N.5a carries through the multi-schema merge.
# They live in `ExtractionResult.metadata`; this is the list that crosses out of
# the service, so a consumer does not have to know the metadata bag's shape.
_OBSERVABILITY_COUNTERS = (
    "chunk_count",
    "entities_extracted",
    "entities_kept",
    "not_a_concept_removed",
    "not_a_concept_judged",
    "abstained_chunks",
    "parse_failures",
)


def _persisted_counts(persist_result: Any) -> Dict[str, int]:
    """What `persist_filtered_result` actually wrote, for the caller.

    Track PC.1b / W3. Returns ``{}`` when persistence did not run — filtering was
    skipped, or the extraction found nothing — so the caller omits the key rather
    than reporting five fabricated zeros. "Not measured" and "measured zero" stay
    distinguishable, the rule N.5d's gate rests on and the same reason
    `_observability_counters` below returns an empty dict.
    """
    if not isinstance(persist_result, dict):
        return {}
    return {
        key: int(persist_result[key] or 0)
        for key in (
            "entities_upserted",
            "entities_failed",
            "relations_created",
            "relations_merged",
            "candidates_stored",
        )
        if key in persist_result
    }


def _observability_counters(metadata: Any) -> Dict[str, int]:
    """Lift the N.3/N.5a counters out of a result's metadata for the caller.

    Track N.5d closed a seam here. `run_extraction` returned only entity and
    relation counts plus the filtering stats, so the counters N.5a had just
    fixed the merge to carry never reached anything that reads a run — including
    the regression gate, whose two cost dimensions read a key NOTHING in the
    repository wrote. A review caught that: half the gate was structurally unable
    to fail while three documents claimed re-measuring would populate it. Exactly
    the anti-pattern this track's retrospective says it learned to detect,
    reached from the other side.

    Returns ``{}`` when the metadata carries none of them, so the caller can omit
    the key entirely. "Not measured" and "measured zero" stay distinguishable,
    which is the distinction the gate's SKIPPED-versus-PASSED rule rests on.
    """
    if not isinstance(metadata, dict):
        return {}
    counters: Dict[str, int] = {}
    for key in _OBSERVABILITY_COUNTERS:
        if key not in metadata:
            continue
        try:
            counters[key] = int(metadata[key] or 0)
        except (TypeError, ValueError):
            continue
    return counters


NOTEBOOK_DEFAULT_BUNDLE: Dict[str, List[str]] = {
    "deals": ["policy_themes"],
    "government": ["policy_themes"],
}


# N.4d.3: the discriminator on an ``accepted_extensions`` entry that records a
# re-parent instead of a new type. Imported nowhere else in this module's import
# graph, so it is spelled here rather than pulling in a service-layer symbol.
_REPARENT_OP = "reparent"


def _counts_toward_review(extension: Dict[str, Any]) -> bool:
    """Whether an ``accepted_extensions`` entry satisfies the B.1f review gate.

    The gate is ``review_required AND accepted_extensions is empty``, and it
    means "the curator has reviewed the schema this notebook proposes". B.3c
    appends a sentinel row precisely to trip it, so the predicate is known and
    load-bearing.

    N.4d.3 — a re-parent does NOT count. Moving a type that already exists says
    nothing about the pending extensions awaiting review, and letting it count
    would resume a paused extraction for a notebook with forty unreviewed
    proposals because the curator adjusted one parent. The sentinel is excluded
    for the mirror-image reason: it carries no ontology content, and it exists to
    trip the gate DELIBERATELY, so it is handled at its own call site rather than
    here.
    """
    return extension.get("op") != _REPARENT_OP


def _avg_entity_confidence(entities: Iterable[Any]) -> float:
    """Mean ``confidence`` across ``entities`` — 0.0 for an empty iterable.

    Defined at module scope so unit tests can exercise it without
    booting the service. Skips entities whose ``confidence`` attribute
    is missing or non-numeric — the metric column is a single float and
    we'd rather report a slightly-low average than crash the telemetry
    write.
    """
    scores: List[float] = []
    for entity in entities:
        raw = getattr(entity, "confidence", None)
        try:
            scores.append(float(raw))
        except (TypeError, ValueError):
            continue
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


# Het B.8-veld ``default_field`` bepaalt de taak in het register.
_DEFAULT_FIELD_TO_TASK = {
    "default_extraction_model": "ENTITY_EXTRACTION",
    "default_transformation_model": "SUMMARIZATION",
    "default_chat_model": "CHAT",
}


class RoutedLLMCaller:
    """Een async LLM-aanroeper die via ``ModelGate`` naar llama-swap gaat (F2-08).

    Behoudt de B.8-vorm ``(system, user, model) -> str``, zodat elke aanroeper
    van de extractie blijft werken. Het model komt uit het register bij de taak;
    er is geen route, geen failover en geen terugval (ADR 0006). De poort
    schrijft per aanroep één regel in ``model_call``.

    Na elke aanroep geven :attr:`served_provider` (altijd ``llama-swap``) en
    :attr:`served_model_id` (het registermodel) de herkomst voor de KG.
    """

    def __init__(
        self,
        *,
        task: "Any",
        classificatie: Optional[str],
        run_id: Optional[str] = None,
        json_mode: bool = False,
    ) -> None:
        from app_main.services.model_routing.llm_call import taak_van

        self._taak = taak_van(task)
        self._classificatie = classificatie
        self._run_id = run_id
        self._json_mode = json_mode
        self.served_provider: Optional[str] = None
        self.served_model_id: Optional[str] = None

    async def __call__(
        self, system_prompt: str, user_prompt: str, _model: str = "default"
    ) -> str:
        # ``_model`` is de oude keuze uit de modelrijen; het register beslist.
        from app_main.services.model_routing.llm_call import roep_taalmodel_aan

        antwoord = await roep_taalmodel_aan(
            system_prompt,
            user_prompt,
            taak=self._taak,
            classificatie=self._classificatie,
            run_id=self._run_id,
            json_mode=self._json_mode,
        )
        self.served_provider = "llama-swap"
        self.served_model_id = antwoord.model
        return antwoord.tekst


async def make_default_llm_caller(
    model_id: Optional[str] = None,
    *,
    classificatie: Optional[str],
    default_field: str = "default_chat_model",
    run_id: Optional[str] = None,
    json_mode: Optional[bool] = None,
):
    """Een async ``LLMCaller`` via de modelpoort (F2-08, ADR 0006).

    ``default_field`` bepaalt de taak (extractie, samenvatten of chat); het
    model komt uit het register bij die taak. ``classificatie`` is die van de
    invoer (verplicht; ``onbekend`` telt als intern). ``model_id`` telt niet:
    het register beslist.

    ``json_mode``: ``None`` zet JSON-uitvoer aan voor de extractie en uit voor
    chat en samenvatten.
    """
    task_name = _DEFAULT_FIELD_TO_TASK.get(default_field, "CHAT")
    resolved_json_mode = (
        json_mode if json_mode is not None else task_name == "ENTITY_EXTRACTION"
    )
    return RoutedLLMCaller(
        task=task_name,
        classificatie=classificatie,
        run_id=run_id,
        json_mode=resolved_json_mode,
    )


async def extract_from_text(text: str, ontology_name: Optional[str] = None):
    """Extract typed entities from RAW TEXT with NO DB writes (Track G.1, G-D5).

    Runs the existing single-ontology :class:`ExtractionWorkflow` over one
    synthetic in-memory chunk and returns its ``ExtractionResult.entities``
    (``shared.models.extraction.ExtractedEntity`` — name / entity_type /
    confidence). ``mode="single"`` needs no ``notebook_id`` / ``pass1_repo``, so
    nothing is persisted — this is the stateless capability behind the agent API's
    ``POST /extract-entities``. Empty/whitespace text yields ``[]``.
    """
    from ontology_extraction.config import ExtractionConfig
    from ontology_extraction.workflow import ExtractionWorkflow

    if not (text or "").strip():
        return []

    config = ExtractionConfig(ontology_name=ontology_name or "general")
    workflow = ExtractionWorkflow(config=config)
    # Losse tekst van de agent-API: de classificatie is onbekend, dus intern.
    llm_caller = await make_default_llm_caller(classificatie="onbekend")
    result = await workflow.extract(
        [{"text": text, "id": "agent:0"}], mode="single", llm_caller=llm_caller
    )
    return list(result.entities)


class SchemaReviewPendingError(JobPausedForReviewError):
    """Raised when extraction blocks on user review of pending schema extensions.

    Phase B.1f introduces a per-notebook ``review_required`` toggle. When
    set, the multi-schema orchestrator should not run until the user has
    explicitly approved (or rejected) the pending extension proposals.
    The handler catches this exception and parks the background job in
    ``PAUSED_FOR_REVIEW``; the API translates it into 409 Conflict.

    Subclassing :class:`JobPausedForReviewError` lets the generic
    :class:`job_queue.worker.JobWorker` route this to
    ``PAUSED_FOR_REVIEW`` without knowing about extraction internals.
    """

    def __init__(self, notebook_id: str, source_id: str, pending_count: int = 0):
        self.notebook_id = notebook_id
        self.source_id = source_id
        self.pending_count = pending_count
        # F2-15: het notebook waarvan het schema op review wacht.
        self.review_item = notebook_id
        super().__init__(
            f"Notebook {notebook_id} requires review of pending schema "
            f"extensions before source {source_id} can be extracted "
            f"({pending_count} pending)."
        )


class EntityExtractionService:
    """Runs ontology-guided entity extraction on a source's chunks."""

    def __init__(
        self,
        source_repo: SourceRepository,
        notebook_schema_repo: Optional[NotebookSchemaRepository] = None,
        pass1_repo: Optional[Pass1ResultRepository] = None,
        notebook_event_repo: Optional[Any] = None,
    ):
        self._source_repo = source_repo
        # B.1f wiring: the multi-schema orchestrator needs both repos.
        # The legacy single-schema path doesn't, so they default to
        # ``None`` and are lazily constructed when needed. Tests can
        # inject fakes to assert on the multi-schema branch.
        self._notebook_schema_repo = notebook_schema_repo
        self._pass1_repo = pass1_repo
        # PC.1b: injected only by tests. Production lazily constructs it, like
        # the two repos above — the soft-nudge write is best-effort and must not
        # make the service harder to build.
        self._notebook_event_repo = notebook_event_repo
        self._persistence = EntityPersistenceService()
        # Q.4: the after-extraction triage pipeline (match + merge + signals +
        # status + queue + backbone). Constructed lazily on first persist so a
        # service instance that never extracts pays no triage import cost, and so
        # tests can inject a fake via ``self._triage`` before running. ``None``
        # until first use; ``_run_triage`` builds it from the DI factory.
        self._triage: Any = None
        # F2-08: de classificatie van de fragmenten van deze run, voor de poort.
        # ``None`` (geen run) telt als intern.
        self._classificatie: Optional[str] = None
        # F2-15: de run van de stap, voor ``model_call`` en ``processing_run.models``.
        self._run_id: Optional[str] = None
        # Run-scoped applied schemas (Track L.1). Set in ``_run_multi_schema``
        # after ``detect_applicable_schemas`` and consumed at the persist site
        # so the canonical bridge has them WITHOUT re-detecting. Reset at the
        # top of each ``run_extraction`` so a single-schema / no-schema run does
        # not inherit a prior run's schemas.
        self._applicable_schemas: Optional[List[Any]] = None
        # PC.4: what the persist step TYPES against — the applied set followed by
        # every other ontology (`_typing_schemas_for`). Reset with
        # `_applicable_schemas`.
        self._typing_schemas: Optional[List[Any]] = None
        # N.4d.4: the notebook's declared vocabulary name, for gap rows. See the
        # stash site in `_run_multi_schema` for why it is not read off the
        # applied set.
        self._gap_ontology_name: Optional[str] = None
        # The most-recently-built routed caller for this run. After extraction
        # its ``served_provider`` / ``served_model_id`` carry the provider that
        # actually answered (J-Q7) — read at provenance-stamp time so the KG
        # records the SERVED model, not just the resolved head.
        self._last_routed_caller: Any = None

    async def _llm_chat(self, messages: List[Dict[str, Any]]):
        """De chatfunctie voor de LLM-matcher van de filtering (F2-08).

        Via de modelpoort, taak ``entity_dedup``, JSON-uitvoer, met de
        classificatie van de fragmenten van deze run.
        """
        from app_main.services.model_routing.llm_call import get_aanroeper

        return await get_aanroeper().chat(
            taak="entity_dedup",
            classificatie=self._classificatie,
            run_id=self._run_id,
            messages=messages,
            json_mode=True,
        )

    async def _make_routed_caller(self, *, model_id: Optional[str] = None):
        """De LLM-aanroeper van deze run, via de modelpoort.

        Eén bouwer voor elke extractieaanroep (single-schema, Pass-1/Pass-2,
        B.5b-herhaling), zodat ze allemaal de classificatie van de run meegeven.
        """
        caller = await make_default_llm_caller(
            model_id=model_id,
            classificatie=self._classificatie,
            default_field="default_extraction_model",
            run_id=self._run_id,
        )
        self._last_routed_caller = caller
        return caller

    async def _run_triage(
        self, source_id: str, persist_result: Any
    ) -> None:
        """Fire the Q.4 after-extraction triage pipeline. Non-blocking.

        This is the R1 hook: it runs RIGHT AFTER a ``persist_filtered_result`` and
        is wrapped in its OWN log-and-continue guard (mirroring the filtering-
        failure isolation at the main-path persist site). A triage failure MUST
        NOT fail the extraction — the entities are already persisted; triage only
        re-derives status / fills the review queue. So any exception here is logged
        and swallowed and the extraction reports success.

        ``persist_result`` is the dict returned by ``persist_filtered_result``; its
        ``persisted_entity_ids`` are the batch the pipeline re-triages. A stable
        ``batch_id`` (source + persist signature) ties the run's status-log + queue
        rows together; re-running the same source produces the same logical batch,
        and the pipeline's idempotency keeps the re-run a no-op.
        """
        try:
            entity_ids = (
                persist_result.get("persisted_entity_ids")
                if isinstance(persist_result, dict)
                else None
            )
            if not entity_ids:
                return
            if self._triage is None:
                # Local import keeps the triage graph out of the module import
                # path for non-extracting callers (and avoids a circular import
                # via dependencies -> services -> this module).
                from app_main.dependencies import get_triage_pipeline

                self._triage = get_triage_pipeline()
            batch_id = f"triage:{source_id}:{len(entity_ids)}"
            await self._triage.run(
                source_id=source_id,
                persisted_entity_ids=list(entity_ids),
                batch_id=batch_id,
            )
        except Exception as e:
            # Non-blocking: triage never fails extraction (mirrors the filtering
            # isolation). The persisted KG is intact; only triage's status/queue
            # side-effects are skipped for this run.
            logger.error(f"Triage failed for source {source_id} (non-blocking): {e}")

    async def _served_extraction_model(self, config) -> Optional[str]:
        """The model id to stamp as extraction provenance (J-Q7).

        Prefers the provider that ACTUALLY served the run (the routed caller's
        ``served_model_id`` after extraction); falls back to the register model
        for ``entity_extraction`` when nothing was served — e.g. a
        zero-entity run where the caller was never invoked, or a caller-build
        failure.
        """
        served = getattr(self._last_routed_caller, "served_model_id", None)
        if served:
            return served
        from app_main.services.model_routing.llm_call import registermodel

        return registermodel("entity_extraction")

    async def _embed_entities(
        self, result: "ExtractionResult"
    ) -> None:
        """Embed entity texts and store vectors in entity properties.

        This enables embedding-based deduplication in the FilteringWorkflow.
        Runs in-place — modifies entity properties directly.
        """
        texts = [e.text for e in result.entities if e.text.strip()]
        if not texts:
            return

        try:
            from app_main.dependencies import get_embedding_service

            embed_svc = await get_embedding_service()
            # Batch embed all entity texts
            vectors = await embed_svc.embedding_model.aembed(
                texts, classificatie=self._classificatie, run_id=self._run_id
            )

            # Map vectors back to entities
            text_to_vec = dict(zip(texts, vectors))
            embedded = 0
            for entity in result.entities:
                vec = text_to_vec.get(entity.text)
                if vec:
                    entity.properties["embedding"] = vec
                    embedded += 1

            logger.info(f"Embedded {embedded}/{len(result.entities)} entities")
        except Exception as e:
            logger.warning(
                f"Entity embedding failed (dedup will use string-only): {e}"
            )

    async def _build_single_schema_workflow(
        self, config: "ExtractionConfig"
    ) -> "ExtractionWorkflow":
        """Build a single-schema ``ExtractionWorkflow`` whose ``LLMExtractor``
        is wired with the extraction-model caller.

        Single-mode ``ExtractionWorkflow.extract`` ignores any ``llm_caller``
        kwarg and uses ``self._get_extractor()``, which (when the workflow was
        built without an extractor) lazily constructs a CALLER-LESS
        ``LLMExtractor`` that silently returns 0 entities (the B.1f canary). So
        every single-schema run — including the multi-schema
        no-applicable-schema fallback — MUST go through this builder, not a bare
        ``ExtractionWorkflow(config)``.
        """
        extractor = None
        if config.extractor_type == "llm":
            try:
                llm_caller = await self._make_routed_caller(
                    model_id=config.llm_model
                    if config.llm_model != "default"
                    else None,
                )
                from ontology_extraction.extractors.llm_extractor import (
                    LLMExtractor,
                )

                extractor = LLMExtractor(
                    llm_model=config.llm_model,
                    confidence_threshold=config.confidence_threshold,
                    llm_caller=llm_caller,
                )
            except Exception as e:
                logger.warning(
                    f"single-schema: failed to wire LLM caller ({e}); "
                    "extractor will fall back to lazy-default empty path."
                )
        return ExtractionWorkflow(config, extractor=extractor)

    def _apply_notebook_schema_default(
        self,
        applicable_schemas: List[Tuple["Ontology", float]],
        candidate_ontologies: List["Ontology"],
        notebook_schema: "NotebookSchema",
    ) -> List[Tuple["Ontology", float]]:
        """Force a per-notebook ``notebook_schema`` default over auto-detection.

        Config beats auto-detection (L.4 AC3): when an operator has configured a
        ``base_ontology`` for a notebook, that base — plus its explicit override
        bundle (``NOTEBOOK_DEFAULT_BUNDLE``, e.g. ``deals`` → ``policy_themes``)
        — is applied regardless of the content score. A document with no theme
        keywords still gets ``policy_themes`` because the notebook is *declared*
        to be about policy.

        This changes the SELECTION only, never the type a label gets: typing
        runs against every ontology with the selection first
        (``typing_vocabulary``), and the type is the same under every selection
        (PC.4). OE.5 dropped the schemas named on accepted extensions from the
        forced set along with the rest of the per-notebook layer.

        This is the EXPLICIT-override path. The ordinary auto-detection path
        (``detect_applicable_schemas``) already selects ``policy_themes`` on
        content overlap for any document discussing the themes, with no
        dependency on this bundle — see the orchestrator's content-signal scorer.

        Additive: the auto-detected ``applicable_schemas`` are preserved; the
        forced schemas are merged in (de-duplicated by name) at a fixed
        config-confidence so they sort ahead of weakly-keyword-matched ones but
        the auto-detected set is never dropped. Forced schemas that don't exist
        in ``candidate_ontologies`` are skipped (a misconfigured base never
        crashes extraction).

        Pure (no I/O) so the override rule is unit-testable in isolation.
        """
        # Config-confidence: above MIN_APPLICABLE_CONFIDENCE and above a weak
        # keyword score so the operator's choice leads, but it does not need to
        # beat a 0.92 document-type match — both stay in the applied set.
        config_conf = 0.85

        by_name: Dict[str, "Ontology"] = {
            (ont.metadata.name if ont.metadata else ""): ont
            for ont in candidate_ontologies
        }

        forced_names: List[str] = [notebook_schema.base_ontology]
        forced_names.extend(
            NOTEBOOK_DEFAULT_BUNDLE.get(notebook_schema.base_ontology, [])
        )

        existing_names = {
            (ont.metadata.name if ont.metadata else "")
            for ont, _conf in applicable_schemas
        }

        result = list(applicable_schemas)
        appended: set[str] = set()
        for name in forced_names:
            if (
                not name
                or name in existing_names
                or name in appended
                or name not in by_name
            ):
                continue
            result.append((by_name[name], config_conf))
            appended.add(name)
            logger.info(
                "notebook_schema default forced '{name}' for notebook={nb} "
                "(base={base})",
                name=name,
                nb=notebook_schema.notebook,
                base=notebook_schema.base_ontology,
            )

        return result

    async def _record_pass1_outcome(
        self,
        result: Any,
        notebook_id: str,
        notebook_schema_repo: "NotebookSchemaRepository",
        pass1_repo: "Pass1ResultRepository",
        source_id: str = "",
        shared_ontologies: Optional[List["Ontology"]] = None,
    ) -> None:
        """Move Pass 1's findings to where a curator can act on them (PC.1).

        OE.5 (review M3): a proposal whose name (or an alias of it) one of the
        ``shared_ontologies`` already defines, matched as typing matches, is not
        queued: typing already resolves it, and accepting it would publish a
        second definition. Pass 1 proposes against the SELECTED ontologies only,
        so this is common.

        Pass 1 already worked and was already persisted to ``pass1_results``. What
        did not happen is the step that makes it ACTIONABLE:
        ``notebook_schema.pending_extensions`` had no writer anywhere in
        production — `add_pending_extension`'s only callers were in its own
        roundtrip test — and `coverage_pct`, documented as driving the B.3c
        soft-nudge, was never written either.

        Everything downstream reads those two fields: accept, reject, the
        `PendingExtensionsPanel`, and through the accept step the whole of
        N.4d.1-N.4d.3. A pipeline review measured the consequence: after eight
        documents and fourteen `pass1_results` rows, `pending_extensions` was
        empty and `coverage_pct` was 0.0.

        Best-effort by design: extraction has already succeeded by the time this
        runs, and a failure to surface a proposal must not lose the entities. The
        failure is logged at WARNING rather than swallowed, because a queue that
        silently stops filling is exactly the defect this method exists to fix.
        """
        metadata = getattr(result, "metadata", None) or {}
        proposals = metadata.get("proposed_extensions") or []
        if shared_ontologies:
            proposals = [
                p for p in proposals
                if not isinstance(p, dict)
                or _find_definition(str(p.get("type_name") or ""), shared_ontologies) is None
            ]

        try:
            if proposals:
                added = await notebook_schema_repo.merge_pending_extensions(
                    notebook_id, proposals
                )
                logger.info(
                    "pass1 proposals queued for {nb}: {added} new of {total} proposed",
                    nb=notebook_id,
                    added=added,
                    total=len(proposals),
                )
        except Exception as e:
            logger.warning(f"could not queue pass1 proposals for {notebook_id}: {e}")

        try:
            coverage = await self._rolling_coverage(notebook_id, pass1_repo)
            if coverage is not None:
                await notebook_schema_repo.set_coverage_pct(notebook_id, coverage)
        except Exception as e:
            logger.warning(f"could not update coverage for {notebook_id}: {e}")

        await self._emit_soft_nudge(metadata, notebook_id, source_id)

    async def _emit_soft_nudge(
        self, metadata: Dict[str, Any], notebook_id: str, source_id: str
    ) -> None:
        """Write the soft-nudge decision where the banner can read it (PC.1b/W1).

        `_decide_soft_nudge` has run on every document since B.1e and its verdict
        lands in `metadata["soft_nudge"]`. Meanwhile B.3c built the other half:
        the `notebook_event` table, its repository, a router filtering on
        `_KNOWN_NUDGE_EVENT_TYPES`, and `SchemaSoftNudge.tsx` polling it. The two
        halves had never met, and `notebook_event` held ZERO rows.

        Note what the defect was NOT. `workflow.py` reads
        `merged, _decision = await run_multi_schema(...)`, and an earlier version
        of this docstring blamed that underscore. A review showed otherwise: the
        verdict is already written into `merged.metadata["soft_nudge"]` and this
        method reads exactly that key. Nothing was dropped in transit — the state
        was carried faithfully and no writer acted on it, which is a better
        illustration of the phase's thesis than the version I first wrote.

        Measured on the five sources carrying Pass-1 rows before this landed: all
        five would have fired a banner (coverage 0.45 / 0.48 / 0.55 / 0.58 ->
        `schema_mismatch`, 0.85 -> `extension_suggested`). The enum's own
        docstring says its values "round-trip through `notebook_event.type`
        without manual conversion", so the design intent was always this write.

        Two deliberate choices:

        * **NONE writes nothing.** A schema that fits is not news, and an event
          the banner would ignore is still a row a curator has to dismiss.
        * **An unread event of the same type for the same notebook suppresses a
          duplicate.** Re-extracting a notebook's documents one after another
          would otherwise queue one banner per document saying the same thing.
          The source that triggered it is in the payload, so the first event
          still names a concrete document.

        Best-effort like its two siblings above: extraction has already succeeded,
        and failing to raise a banner must not lose the entities.
        """
        decision = str(metadata.get("soft_nudge") or "")
        if not decision or decision == "none":
            return
        try:
            from surrealdb_service.repositories.notebook_event import (
                NotebookEventRepository,
            )

            repo = self._notebook_event_repo or NotebookEventRepository()
            existing = await repo.list_unread(notebook_id, event_type=decision)
            if existing:
                logger.debug(
                    "soft nudge {d} already pending for {nb}; not duplicating",
                    d=decision,
                    nb=notebook_id,
                )
                return
            await repo.record(
                notebook_id,
                decision,
                {
                    "source_id": source_id,
                    "best_coverage": metadata.get("best_coverage"),
                    "schemas_attempted": metadata.get("schemas_attempted") or [],
                },
            )
            logger.info(
                "soft nudge {d} raised for {nb} (source={src})",
                d=decision,
                nb=notebook_id,
                src=source_id,
            )
        except Exception as e:
            logger.warning(f"could not raise soft nudge for {notebook_id}: {e}")

    # How many `pass1_results` rows the coverage average reads. Pass 1 writes one
    # row per applied schema per extraction, so ~2-3 rows per document; 1000 is
    # roughly 300-500 documents including re-extractions. Stated as a number
    # rather than left at the repository default of 100, which a review measured
    # would silently start truncating at ~57 documents — and truncate a source
    # PARTIALLY, giving a wrong average rather than a stale one.
    _COVERAGE_WINDOW = 1000

    # Group key for `pass1_results` rows written before PC.1 stamped a run id.
    # They are one run per source, not one run each, so a legacy source still
    # reports its best schema rather than whichever schema happens to be newest.
    _LEGACY_RUN = "__pre_pc1__"

    # Confidence stamped on a schema recovered from the notebook's history.
    # Above `MIN_APPLICABLE_CONFIDENCE` so it is actually applied, and below the
    # 0.85 a curator's explicit `base_ontology` carries, because history is
    # weaker evidence than a declaration. It never competes with a content match
    # — this path runs only when detection returned nothing at all.
    _NOTEBOOK_FALLBACK_CONFIDENCE = 0.5

    @staticmethod
    async def _notebook_fallback_schemas(
        notebook_id: str,
        pass1_repo: "Pass1ResultRepository",
        candidates: List["Ontology"],
    ) -> List[Tuple["Ontology", float]]:
        """The schema this notebook's other documents turned out to be.

        Consulted ONLY when per-document detection found nothing, so it can
        never overrule a document that spoke for itself. Ranked by how often a
        schema was attempted, ties broken by mean coverage — frequency is the
        notebook's verdict, coverage is how well that verdict worked.

        **Frequency dominates absolutely**, so a schema attempted eight times at
        coverage 0.0 beats one attempted three times at 0.9 — and `coverage_pct`
        is precisely the measurement of whether that schema fit. A review raised
        it and it is a decision, not an oversight: a low-coverage schema a
        notebook keeps landing on is still what that notebook is about, while the
        alternative ranking lets one lucky document redefine it — which is what
        the "per document, history only as fallback" rule exists to prevent.
        Bounded by firing only when detection found nothing and by returning a
        single schema. Revisit it if a notebook is seen converging on a schema
        that measures near zero.

        Note what the history is made of: rows written by the blind first-chunk
        detector this same change replaces. It self-corrects as documents are
        re-extracted.

        Returns at most one schema. A fallback that guessed three would be
        asserting more about the document than the evidence supports, and every
        extra schema is another Pass-1 and Pass-2 LLM call.

        Returns an empty list when the notebook has no history yet, when its
        history names schemas no longer in the registry, or when the repository
        read fails — in each case the caller drops to the legacy path exactly as
        it did before this existed.
        """
        try:
            rows = await pass1_repo.list_by_notebook(
                notebook_id, limit=EntityExtractionService._COVERAGE_WINDOW
            )
        except Exception as e:
            logger.warning(f"notebook fallback: could not read pass1 history ({e})")
            return []
        if not rows:
            return []

        counts: Dict[str, int] = {}
        coverage: Dict[str, List[float]] = {}
        for row in rows:
            name = str(getattr(row, "schema_attempted", "") or "")
            if not name:
                continue
            counts[name] = counts.get(name, 0) + 1
            coverage.setdefault(name, []).append(
                float(getattr(row, "coverage_pct", 0.0) or 0.0)
            )
        if not counts:
            return []

        by_name = {
            (ont.metadata.name if ont.metadata else ""): ont for ont in candidates
        }
        ranked = sorted(
            counts,
            key=lambda n: (counts[n], sum(coverage[n]) / len(coverage[n])),
            reverse=True,
        )
        for name in ranked:
            ontology = by_name.get(name)
            if ontology is not None:
                return [
                    (
                        ontology,
                        EntityExtractionService._NOTEBOOK_FALLBACK_CONFIDENCE,
                    )
                ]
        return []

    @staticmethod
    async def _rolling_coverage(
        notebook_id: str, pass1_repo: "Pass1ResultRepository"
    ) -> Optional[float]:
        """Mean over SOURCES of each source's best coverage IN ITS NEWEST RUN.

        The field is documented as "rolling-average coverage across all sources
        processed in this notebook", so it is recomputed from `pass1_results`
        rather than accumulated incrementally.

        `pass1_results` is append-only — re-extracting a source adds rows, it does
        not replace them — and one extraction writes one row PER APPLIED SCHEMA.
        So a source's current measurement is the set of rows sharing its newest
        ``run_id``, and the summary per source is the max across the schemas in
        that run: Pass 1 runs once per applied schema and a notebook's coverage is
        how well its best-fitting schema did, not the average of a good fit and a
        bad one.

        Two earlier versions of this rule were wrong in the same direction, both
        caught by review, so the reasoning is spelled out. A plain max over all of
        a source's rows spans re-runs, so coverage could only ever rise. Grouping
        on the newest row per ``(source, schema_attempted)`` fixes re-extraction
        under an UNCHANGED schema set and still cannot fall when the set changes —
        the abandoned schema's pair is never superseded, only aged — which is
        precisely the flow the soft-nudge exists to drive: low coverage, curator
        edits the schema, re-extract, number should move. Measured: after a run
        under `scholarly` scoring 0.30, the pair rule still reported 0.90 from an
        abandoned `deals` row. Only a run boundary expresses "these rows are the
        current measurement and those are history", so PC.1 stamps one
        (`pass1_metadata["run_id"]`, written by the orchestrator).

        **Rows written before PC.1 carry no ``run_id``.** They are grouped under a
        single synthetic per-source run so they behave as they did before rather
        than each becoming a run of its own, and any real run supersedes them:
        once a source has one stamped row, its unstamped history is dropped.

        A source deleted from the notebook keeps contributing until its rows age
        out of the window — `pass1_results` has no cascade and this method does
        not check liveness. Recorded as a known limit rather than fixed here: the
        fix is a delete-time cleanup or a liveness join, which is a change to
        somebody else's write path.

        Returns ``None`` when there is nothing to average, so the caller writes
        nothing rather than writing 0.0 and claiming a measurement nobody made.
        """
        rows = await pass1_repo.list_by_notebook(
            notebook_id, limit=EntityExtractionService._COVERAGE_WINDOW
        )
        if not rows:
            return None

        # Rows arrive newest-first, so the first run_id seen for a source is that
        # source's current run. `_LEGACY_RUN` groups the pre-PC.1 tail.
        current_run: Dict[str, str] = {}
        best_per_source: Dict[str, float] = {}
        for row in rows:
            source = str(getattr(row, "source", "") or "")
            if not source:
                continue
            meta = getattr(row, "pass1_metadata", None) or {}
            run = str(meta.get("run_id") or "") or EntityExtractionService._LEGACY_RUN
            seen = current_run.setdefault(source, run)
            if run != seen:
                continue  # an older run of this source
            value = float(getattr(row, "coverage_pct", 0.0) or 0.0)
            best_per_source[source] = max(best_per_source.get(source, 0.0), value)

        if not best_per_source:
            return None
        return sum(best_per_source.values()) / len(best_per_source)

    async def _concept_alignment_deps(
        self, f_config: "FilteringConfig"
    ) -> Dict[str, Any]:
        """Collaborators Stage 15 needs, or ``None`` for each that cannot be built.

        Track N.4d.4 / D-N4-8. The stage is reachable only if something actually
        hands it a graph to query, an ontology to resolve types against, a judge,
        and a place to record gaps — the previous tier shipped behind a config
        default nothing set and never ran in production.

        Returns all four keys always, so the caller wires whatever is available
        and the workflow's own DEGRADED warning reports the rest. Nothing here
        raises: a failure to build one collaborator degrades that tier, it does
        not fail the extraction.
        """
        deps: Dict[str, Any] = {
            "entity_repo": None,
            "schemas": None,
            "gap_recorder": None,
            "llm_caller": None,
            "gap_ontology_name": None,
        }
        if not f_config.concept_alignment.enabled:
            return deps

        try:
            from app_main.dependencies import get_entity_repo

            deps["entity_repo"] = get_entity_repo()
        except Exception as e:
            logger.warning(f"concept alignment: no entity repo ({e})")

        # ALL the ontologies this run applied — not just the first.
        # `detect_applicable_schemas(top_k=3)` means an applied set holds three,
        # and a type declared in the second or third would otherwise fail
        # `resolve_canonical_type`, produce a code that licenses no gap, and make
        # the loop under-fire for two thirds of the applied vocabulary.
        #
        # None on the single-schema path and on `run_filtering_only`, where no
        # set is detected at all: there no canonical type resolves, so every
        # verdict carries a code that licenses no gap and the loop is inert.
        # Stated rather than left to be rediscovered.
        if self._applicable_schemas:
            deps["schemas"] = list(self._applicable_schemas)
        deps["gap_ontology_name"] = self._gap_ontology_name

        try:
            from ontology_manager import get_ontology_manager

            deps["gap_recorder"] = get_ontology_manager().evolution
        except Exception as e:
            logger.warning(f"concept alignment: no gap recorder ({e})")

        if f_config.concept_alignment.judge_enabled:
            try:
                deps["llm_caller"] = await self._make_routed_caller()
            except Exception as e:
                logger.warning(f"concept alignment: no judge caller ({e})")

        # `gap_ontology_name` is absent for a notebook with no configured base
        # ontology, which is the Regio-Deal corpus's state today. That is not a
        # wiring failure — the aligner falls back to the applied schema's own
        # name — so it is not reported as one.
        missing = [
            name
            for name, value in deps.items()
            if value is None and name != "gap_ontology_name"
        ]
        if missing:
            # The workflow warns about its own DEGRADED tiers, but only about the
            # ones it was handed. This says which could not be BUILT, which is a
            # different fact and the one an operator can act on.
            logger.warning(
                "Concept alignment enabled but these collaborators could not be "
                "wired: {missing}",
                missing=missing,
            )
        return deps

    async def _typing_schemas_or_all(self) -> Optional[List["Ontology"]]:
        """This run's typing vocabulary, or every ontology when no detection ran.

        PC.4. `_run_multi_schema` sets `_typing_schemas`; the single-schema path
        and the re-filter do not detect, and type against every ontology. A load
        that RAISES degrades to the alias/enum path (None). An unreachable
        database does not raise: the pool waits for it (connection.py), as every
        other query in the run would.
        """
        if self._typing_schemas is not None:
            return self._typing_schemas
        try:
            return typing_vocabulary(None, await load_typing_ontologies())
        except Exception as e:
            logger.warning(f"typing vocabulary: could not load the ontologies ({e})")
            return None

    def _typing_schemas_for(
        self,
        applicable_schemas: List[Tuple["Ontology", float]],
        candidate_ontologies: List["Ontology"],
    ) -> List["Ontology"]:
        """The typing vocabulary for this run: the registry's active set.

        Track PC.4: the selection first, then every other ontology, so the type
        is the same under every selection. OE.5: nothing per notebook is
        projected over it any more — the ontologies are shared by every
        notebook, and a curator's decision reaches them through the editor.
        """
        applied = [o for o, _c in applicable_schemas]
        return typing_vocabulary(applied, candidate_ontologies)

    async def _run_multi_schema(
        self,
        workflow: "ExtractionWorkflow",
        source_id: str,
        notebook_id: str,
        chunks: List[Dict[str, Any]],
        config: "ExtractionConfig",
    ) -> "ExtractionResult":
        """Detect applicable schemas and invoke the B.1e orchestrator.

        Sequence:

        1. Load the :class:`NotebookSchema` row (if any). Enforce the
           ``review_required`` gate before doing any LLM work.
        2. Discover the candidate ontologies via the registry and score
           them with :func:`detect_applicable_schemas`. When no schema
           clears the applicability floor we fall back to a single-pass
           run against the configured ``ontology_name`` — the
           orchestrator with an empty applicable list would return zero
           entities, and the legacy path is the right safety net.
        3. Hand off to ``ExtractionWorkflow.extract(mode="multi", ...)``.

        The notebook-schema repo is lazily constructed when the caller
        didn't inject one, so production wiring stays a single ``__init__``
        argument away.
        """
        # Lazy-construct the repos so DI is optional. Production wires
        # them via ``get_entity_extraction_service``; tests inject fakes.
        from ontology_manager import get_ontology_manager

        notebook_schema_repo = self._notebook_schema_repo or NotebookSchemaRepository()
        pass1_repo = self._pass1_repo or Pass1ResultRepository()

        notebook_schema = await notebook_schema_repo.get_by_notebook(notebook_id)

        # PC.1: create the row HERE, before schema detection, because otherwise a
        # notebook can never get one. Measured on the live corpus, the cycle is
        # closed: no row -> no base ontology forced -> content scoring clears the
        # floor for nothing -> the fallback below returns early -> Pass 1 never
        # runs -> no proposals -> nothing creates the row. The comment on that
        # fallback already calls this "the common case: a notebook with no
        # configured schema"; what it did not say is that the case is permanent.
        #
        # Creating it before `_apply_notebook_schema_default` means the base is
        # forced on THIS run, so the first document a notebook ever sees gets a
        # Pass-1 pass instead of the legacy path.
        #
        # The row is created with NO declared base ontology, and that is the
        # whole point of it. Two attempts guessed a value here — first this
        # run's `config.ontology_name` ("general", a per-request parameter
        # nobody sets), then `DEFAULT_BASE_ONTOLOGY` ("scholarly", a constant
        # chosen for the TTL-export READ path) — and both had the same defect:
        # whatever lands in this field is forced into the applied set of every
        # later extraction in the notebook, at confidence 0.85, ahead of what
        # the document itself says. Measured on the project's corpus,
        # `scholarly` is detected for ZERO of fourteen sources and all 17
        # `pass1_results` rows ran against `policy_themes`, so forcing it would
        # have extracted Dutch Regio-Deal convenanten against Article, Author
        # and PreprintServer.
        #
        # The schema is established PER DOCUMENT — that is what detection is
        # for — and only if that genuinely fails does the notebook's own history
        # decide (see the fallback further down). An empty base is invisible to
        # the read paths, which all resolve `base_ontology or
        # DEFAULT_BASE_ONTOLOGY`, and is skipped by
        # `_apply_notebook_schema_default`, so nothing is forced until a curator
        # chooses one in the Schema tab. What the row buys is the queue and the
        # coverage field: both writers refuse when no row exists, which is the
        # cycle G0b described.
        if notebook_schema is None:
            try:
                created = await notebook_schema_repo.ensure_row(
                    notebook_id, NO_DECLARED_BASE
                )
                if created:
                    notebook_schema = await notebook_schema_repo.get_by_notebook(
                        notebook_id
                    )
            except Exception as e:
                logger.warning(
                    f"could not create notebook_schema for {notebook_id}: {e}"
                )

        # Review gate — if the user has paused processing for this
        # notebook, do not run extraction. The caller (handler) treats
        # this exception as "park the job in PAUSED_FOR_REVIEW".
        if (
            notebook_schema is not None
            and notebook_schema.review_required
            and not any(
                _counts_toward_review(ext)
                for ext in notebook_schema.accepted_extensions
            )
        ):
            raise SchemaReviewPendingError(
                notebook_id=notebook_id,
                source_id=source_id,
                pending_count=len(notebook_schema.pending_extensions),
            )

        # Document-type hint comes from the source's metadata bag —
        # ``parser_engine_used``, ``document_type``, etc. live there.
        source = await self._source_repo.get(source_id)
        document_type: Optional[str] = None
        if source is not None and getattr(source, "metadata", None):
            document_type = source.metadata.get("document_type")

        # Sample text for the applicability scorer.
        #
        # This used to be the first chunk, capped at 2000 characters, described
        # as "a cheap signal". Measured on the project's own corpus it is not a
        # signal at all: the first chunk of a parsed PDF is a title or heading
        # fragment with a MEDIAN LENGTH OF 66 CHARACTERS (shortest 15), so the
        # scorer was matching a document's vocabulary against its cover page.
        # Detection fired for 2 of 14 sources. Against a sample spread over the
        # body it fires for 13 of 14, and the winner is `policy_themes` at the
        # 0.90 cap for nine of them — which is what all 17 existing
        # `pass1_results` rows actually ran against.
        #
        # That mattered far beyond the scorer. A document with no detected
        # schema takes the legacy single-schema path below, where Pass 1 never
        # runs, so the curator queue this phase exists to fill stayed empty for
        # reasons that had nothing to do with the queue. The forced notebook
        # base was papering over a blind detector.
        sample_text = _applicability_sample(chunks)

        # Discover candidate ontologies. ``list_ontologies`` returns
        # names; we resolve them through the registry. Missing ontology
        # entries (e.g. ``general`` declared but file not registered)
        # are silently skipped.
        manager = get_ontology_manager()
        ontology_names = await manager.list_ontologies()
        candidate_ontologies = []
        for name in ontology_names:
            ontology = await manager.get_ontology(name)
            if ontology is not None:
                candidate_ontologies.append(ontology)

        applicable_schemas = await detect_applicable_schemas(
            document_type=document_type,
            document_text=sample_text,
            ontologies=candidate_ontologies,
            top_k=3,
        )

        # L.4: a per-notebook ``notebook_schema`` default beats auto-detection.
        # When the notebook has a configured ``base_ontology`` (empty today for
        # the Regio-Deal corpus, but set via the seed/endpoint), force that base
        # plus its affinity bundle — config is an explicit operator choice and must not be overruled by a low
        # keyword score. ``applicable_schemas`` is the auto-detected set; the
        # forced schemas are merged in (additive — auto-detected schemas are
        # kept so a notebook default never *loses* a schema the document clearly
        # warrants).
        if notebook_schema is not None and notebook_schema.base_ontology:
            applicable_schemas = self._apply_notebook_schema_default(
                applicable_schemas,
                candidate_ontologies,
                notebook_schema,
            )

        self._typing_schemas = self._typing_schemas_for(
            applicable_schemas, candidate_ontologies
        )

        # L.1: stash the applied ontologies (without the confidence scores) so
        # the persist step can bridge ontology labels -> canonical types and
        # preserve the rich type. ``detect_applicable_schemas`` returns
        # ``(ontology, confidence)`` tuples — keep only the ontologies.
        self._applicable_schemas = [
            ontology for ontology, _conf in applicable_schemas
        ]
        # N.4d.4 B1: the name every ontology GAP this run records is filed
        # under. It must be the notebook's DECLARED vocabulary, never a member
        # of the applied set: `detect_applicable_schemas` ranks by per-document
        # content overlap, so `applicable_schemas[0]` changes with the document,
        # and gaps are keyed on `(entity_text, ontology_name)`. Naming them after
        # a per-document winner splits one concept across two rows at frequency
        # 1 instead of accumulating one at frequency 2 — exactly the failure the
        # cross-document `source_id` plumbing exists to make visible.
        self._gap_ontology_name = (
            notebook_schema.base_ontology
            if notebook_schema is not None and notebook_schema.base_ontology
            else None
        )

        if not applicable_schemas:
            # Nothing cleared the floor for THIS document. Before dropping to
            # the legacy path — where Pass 1 never runs and the curator queue
            # gets nothing — ask the notebook what its other documents turned
            # out to be. A notebook of Regio-Deal convenanten whose fifteenth
            # document scores below the floor is still a notebook of Regio-Deal
            # convenanten. On this corpus that is one document of fourteen, a
            # journal article whose vocabulary overlaps no bundled ontology.
            #
            # This is a fallback, not a declaration: it is consulted only when
            # per-document detection fails, it never overrides a document that
            # detected something, and it reads evidence the notebook already
            # produced rather than a constant somebody picked.
            fallback = await self._notebook_fallback_schemas(
                notebook_id, pass1_repo, candidate_ontologies
            )
            if fallback:
                logger.info(
                    "multi_schema: no detection for source={src}, falling back "
                    "to the notebook's own history: {names}",
                    src=source_id,
                    names=[o.metadata.name for o, _c in fallback],
                )
                self._typing_schemas = self._typing_schemas_for(
                    fallback, candidate_ontologies
                )
                applicable_schemas = fallback
                self._applicable_schemas = [o for o, _c in applicable_schemas]

        if not applicable_schemas:
            # No schema cleared the floor — fall back to the configured
            # default ontology via the legacy single-schema path. This
            # mirrors the orchestrator's own "no schemas applicable"
            # behaviour but gives the caller *some* entities instead of
            # an empty result.
            logger.info(
                "multi_schema: no applicable schemas for source={src} "
                "notebook={nb}, falling back to single-schema path",
                src=source_id,
                nb=notebook_id,
            )
            # B.8c fix: the passed-in ``workflow`` was built bare (no extractor),
            # so its single-mode path would use a caller-less LLMExtractor and
            # silently return 0 entities. Rebuild with the wired extractor so the
            # fallback actually extracts (the common case: a notebook with no
            # configured schema).
            fallback_workflow = await self._build_single_schema_workflow(config)
            return await fallback_workflow.extract(chunks)

        # Build the production LLM caller once per run so Pass-1 and
        # Pass-2 share a single bound ``LanguageModel``. Failures here
        # are logged but non-fatal — Pass-1/Pass-2 fall through to
        # their lazy-default empty-result paths if no caller arrives.
        llm_caller = None
        try:
            llm_caller = await self._make_routed_caller()
        except Exception as e:
            logger.warning(
                f"multi_schema: failed to wire LLM caller ({e}); "
                "Pass-1/Pass-2 will run with their lazy defaults."
            )

        result = await workflow.extract(
            chunks=chunks,
            mode="multi",
            applicable_schemas=applicable_schemas,
            source_id=source_id,
            notebook_id=notebook_id,
            pass1_repo=pass1_repo,
            llm_caller=llm_caller,
        )
        await self._record_pass1_outcome(
            result, notebook_id, notebook_schema_repo, pass1_repo, source_id,
            shared_ontologies=candidate_ontologies,
        )
        return result

    async def run_extraction(
        self,
        source_id: str,
        ontology_name: str = "general",
        extractor_type: str = "llm",
        config_overrides: Dict[str, Any] | None = None,
        run_filtering: bool = True,
        filtering_config: Optional[FilteringConfig] = None,
        notebook_id: Optional[str] = None,
        multi_schema_enabled: bool = True,
        run_id: Optional[str] = None,
        fragmenten: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """
        Run entity extraction and optional filtering for a source.

        Stap ``entity_extraction`` (F2-15): ``fragmenten`` zijn die van de
        bronversie en de run van de stap, en ``run_id`` gaat mee naar de
        modelpoort, zodat ``processing_run.models`` gevuld wordt (ADR 0003,
        0007). Zonder ``fragmenten`` de huidige versie en run van de bron.

        1. Fetch chunks via SourceRepository.
        2. Build ExtractionConfig and ExtractionWorkflow.
        3. Run extraction.
        4. Optionally run FilteringWorkflow for dedup/enrichment.
        5. Persist raw results to ``extraction_result`` table.
        6. Persist filtered entities to KG tables (entity, relation).
        7. Return summary dict.

        Args:
            notebook_id: When provided AND ``multi_schema_enabled`` is
                ``True``, routes through the B.1e multi-schema
                orchestrator (Pass-1 schema validation + Pass-2 typed
                extraction + merge). Required for the notebook-schema
                review-gating logic. When ``None``, the legacy
                single-schema path runs unchanged.
            multi_schema_enabled: Ops kill-switch. ``False`` forces the
                legacy single-schema path even when ``notebook_id`` is
                set — used as the emergency rollback if the orchestrator
                misbehaves in production.

        Raises:
            SchemaReviewPendingError: Multi-schema path only; raised
                when the notebook's ``review_required`` flag is true
                and no accepted extensions exist yet. The handler maps
                this to ``PAUSED_FOR_REVIEW`` job status; the API
                returns 409 Conflict.
        """
        logger.info(f"Starting entity extraction for source: {source_id}")

        # 1. Fetch chunks
        # F2-04 (ADR 0003): alleen de huidige versie en run, zodat oude en
        # nieuwe tekst niet allebei naar het model gaan. De run van een
        # vermelding leidt de database af uit haar fragment (migratie 90).
        self._run_id = run_id
        chunks = (
            fragmenten if fragmenten is not None
            else await self._source_repo.get_chunks(source_id, huidig=True)
        )
        if not chunks:
            logger.warning(f"No chunks found for source {source_id}")
            # AC#3: every run_extraction call writes exactly ONE
            # extraction.complete metric, including the zero-chunk
            # early-return. Without this we'd lose the "we tried but
            # got nothing" signal that's useful for dashboards.
            await record_metric(
                "extraction.complete",
                {
                    "entity_count": 0,
                    "relation_count": 0,
                    "avg_confidence": 0.0,
                    "multi_schema": False,
                    "no_chunks": True,
                },
                source=source_id,
                notebook=notebook_id,
            )
            return {
                "source_id": source_id,
                "entity_count": 0,
                "relation_count": 0,
            }

        # 2. Convert to workflow format — include structural metadata for
        #    section-aware entity extraction (stap 2D).
        chunk_dicts = []
        for c in chunks:
            if not c.text:
                continue
            d: Dict[str, Any] = {"text": c.text, "id": str(c.id)}
            # Carry document structure through to extraction
            d["section_path"] = c.section_path or []
            d["section_level"] = c.section_level
            d["physical_page"] = c.physical_page
            d["element_type"] = c.element_type
            d["source_id"] = source_id
            if c.section_path:
                d["section_heading"] = c.section_path[-1]
            chunk_dicts.append(d)

        # 2b. De classificatie van de fragmenten gaat mee naar de modelpoort.
        self._classificatie = classificatie_van(
            getattr(c, "classification", None) for c in chunks
        )
        # L.1: clear any prior run's applied schemas — only a multi-schema run
        # populates this (below). A single-schema run leaves it None so the
        # persist bridge degrades to the alias/enum path.
        self._applicable_schemas = None
        self._typing_schemas = None
        self._gap_ontology_name = None

        # 3. Build config and workflow
        config_kwargs: Dict[str, Any] = {
            "ontology_name": ontology_name,
            "extractor_type": extractor_type,
        }
        if config_overrides:
            config_kwargs.update(config_overrides)
        config = ExtractionConfig(**config_kwargs)

        # 4. Run extraction — branch on multi-schema vs single-schema.
        use_multi_schema = (
            multi_schema_enabled
            and notebook_id is not None
            and extractor_type == "llm"
        )
        if use_multi_schema:
            # Multi-schema path: workflow has no extractor bound — the
            # orchestrator routes via Pass-1/Pass-2 with the injected
            # caller from ``_run_multi_schema``.
            workflow = ExtractionWorkflow(config)
            result = await self._run_multi_schema(
                workflow=workflow,
                source_id=source_id,
                notebook_id=notebook_id,
                chunks=chunk_dicts,
                config=config,
            )
        else:
            # Legacy single-schema path — hit when ``notebook_id`` is omitted
            # (CLI / tests) or when ops disables multi-schema via the flag. The
            # extractor is wired with the extraction-model caller via the shared
            # builder (B.1f/B.8c) so production callers actually hit the LLM.
            workflow = await self._build_single_schema_workflow(config)
            result = await workflow.extract(chunk_dicts)

        # Store extractor_type in metadata
        if not hasattr(result, "metadata") or result.metadata is None:
            result.metadata = {}
        result.metadata["extractor_type"] = extractor_type

        # 4b. Embed entity texts for semantic dedup
        if result.entities:
            await self._embed_entities(result)

        # 5. Optionally run filtering/deduplication
        filtered_entities = result.entities
        filtered_relations = result.relations
        merge_groups = None
        filtering_stats = {}

        # PC.1b / W3: bound before the filtering guard, not inside it. The
        # summary below reports what was WRITTEN, and filtering is skipped
        # entirely when `run_filtering` is false or the extraction found nothing
        # — in which case nothing was written and `None` is the honest answer.
        persist_result: Any = None
        if run_filtering and (result.entities or result.relations):
            filtered = None
            all_relations: List[Dict[str, Any]] = []
            try:
                if filtering_config:
                    f_config = filtering_config
                else:
                    # Default config: string dedup + fuzzy + embedding
                    from entity_filtering.config import (
                        ConceptAlignmentConfig,
                        EmbeddingDedupConfig,
                        FuzzyDedupConfig,
                        KGResolutionConfig,
                    )

                    f_config = FilteringConfig(
                        dedup_enabled=True,
                        fuzzy_dedup=FuzzyDedupConfig(
                            enabled=True,
                            algorithm="levenshtein",
                            similarity_threshold=0.85,
                        ),
                        embedding_dedup=EmbeddingDedupConfig(
                            enabled=True,
                            similarity_threshold=0.90,
                        ),
                        # D6 measured stage 13 and it goes OFF. 538 of 1,892
                        # relations were its output, stamped `llm` like extracted
                        # edges; about one in six is a relation the text states,
                        # and 528 of 538 join two entities that already share an
                        # extracted neighbour, typically the deal. It comes back
                        # only as a proposer with its own provenance
                        # (phase-D6-edge-prediction.md;
                        # guarded by tests/test_edge_prediction_default.py).
                        edge_prediction_enabled=False,
                        # PC.3 turned stage 10 ON here and then measured it.
                        # It goes back OFF, and the reason is NOT that the stage
                        # is bad — it is that its answer has nowhere to go.
                        #
                        # `kg_entity_id`, `kg_match_type` and `kg_similarity_score`
                        # have no consumer anywhere in the tree. Persistence holds
                        # zero occurrences of `kg_` and identifies the entity by
                        # `name_key` exactly as it does with the stage off, so the
                        # verdict is written into the properties bag and never
                        # read. Enabling it consolidates nothing and costs a
                        # candidate fetch plus Levenshtein and cosine per entity.
                        #
                        # And the answer it produces is mostly the wrong SHAPE.
                        # Measured over all five active entity types: 18 merges
                        # across 531 entities, of which ~5 are defensible. Seven
                        # of the eight clear errors are not failed identity calls
                        # — they are correct RELATIONS recorded as identity.
                        # `coöperatief wonen` is narrower than `wonen`; the
                        # Staatssecretaris is an organ of IenW; VROM is the
                        # predecessor of VRO. The signal is real; the destination
                        # is missing. Stage 15's own docstring says subsumption is
                        # "currently handled nowhere", and PC.2 filed the same gap
                        # for the office-of class.
                        #
                        # It comes back when that destination exists — a relation
                        # proposal, or the curator queue a human already reads —
                        # not when a threshold is retuned. See
                        # docs/tracks/PC-pipeline-coherence/phase-PC.3-measurement.md
                        kg_resolution=KGResolutionConfig(enabled=False),
                        # N.4d.4 / D-N4-8: the stage was otherwise unreachable in
                        # every real run. An explicitly supplied filtering_config
                        # is the caller's own choice and is never overridden.
                        concept_alignment=ConceptAlignmentConfig(
                            enabled=get_concept_alignment_enabled()
                        ),
                    )

                align_deps = await self._concept_alignment_deps(f_config)
                f_workflow = FilteringWorkflow(
                    config=f_config,
                    entity_repo=align_deps["entity_repo"],
                    alignment_schemas=align_deps["schemas"],
                    gap_recorder=align_deps["gap_recorder"],
                    gap_ontology_name=align_deps["gap_ontology_name"],
                    llm_chat=self._llm_chat,
                )

                filtered = await f_workflow.process(
                    result,
                    source_id=source_id,
                    alignment_llm_caller=align_deps["llm_caller"],
                )

                merge_groups = filtered.merged_entity_groups
                all_relations = [
                    r.model_dump() for r in filtered.relations
                ] + [
                    r.model_dump() for r in filtered.predicted_edges
                ]

                filtering_stats = {
                    "entities_before": len(result.entities),
                    "entities_after": len(filtered.entities),
                    "entities_removed": len(filtered.removed_entities),
                    "merge_groups": len(merge_groups) if merge_groups else 0,
                    "predicted_edges": len(filtered.predicted_edges),
                }
                # PC.3: `kg_resolution_report` gets its reader. PC.1b filed it as
                # derived state carried faithfully to a place nothing reads, and
                # this phase's acceptance criterion needs the figure it holds —
                # "materially fewer rows" has no producer without it.
                #
                # `capped_fetches` travels with the counts on purpose. `matched`
                # says how many mentions were consolidated; the cap ratio says how
                # much of the graph was looked at to decide that. A high match
                # count from a capped look is a different claim from the same
                # count from a complete one, and only one of them supports the AC.
                kg_report = filtered.kg_resolution_report
                if kg_report:
                    filtering_stats["kg_resolution"] = {
                        "matched": kg_report.get("matched_count", 0),
                        "new": kg_report.get("new_count", 0),
                        "by_tier": dict(kg_report.get("match_type_counts", {})),
                        "candidate_fetches": kg_report.get("candidate_fetches", 0),
                        "capped_fetches": kg_report.get("capped_fetches", 0),
                        "capped_types": list(kg_report.get("capped_types", [])),
                        "candidate_cap": kg_report.get("candidate_cap"),
                    }
                # N.4d.4: the gap counters are the phase's only operator-visible
                # output. Left on `concept_alignment_report` alone they never
                # reach anyone — this dict is what lands in
                # `extraction_result.metadata["filtering"]`. `eligible` vs
                # `recorded` is the distinction that matters: `record_gap`
                # returns a gap with `id=None` when the write failed, so without
                # both numbers a run where nothing was written reads exactly like
                # one where everything was.
                alignment_report = filtered.concept_alignment_report
                if alignment_report:
                    filtering_stats["concept_alignment"] = {
                        "aligned": alignment_report.get("aligned_count", 0),
                        "judged": alignment_report.get("judged_count", 0),
                        "gap_eligible": alignment_report.get("gap_eligible", 0),
                        "gaps_recorded": alignment_report.get("gaps_recorded", 0),
                        "gaps_unrecorded": alignment_report.get("gaps_unrecorded", 0),
                        "gap_duplicates_suppressed": alignment_report.get(
                            "gap_duplicates_suppressed", 0
                        ),
                        "gap_recorder_wired": alignment_report.get(
                            "gap_recorder_wired", False
                        ),
                        # The store's STANDING totals — this run's counters say
                        # what happened, these say what has accumulated, which is
                        # what a curator acts on. `status` travels with them
                        # because three of its four values leave the totals null.
                        "gap_statistics": alignment_report.get("gap_statistics"),
                        "gap_statistics_status": alignment_report.get(
                            "gap_statistics_status"
                        ),
                    }
                result.metadata["filtering"] = filtering_stats

                logger.info(
                    f"Filtering complete for source {source_id}: "
                    f"{filtering_stats}"
                )

            except ConfigurationError:
                # PC.6: a REFUSAL is not a failure and must not be swallowed here.
                #
                # `ConfigurationError` subclasses RuntimeError, so the handler
                # below caught it and turned the refusal into a log line — with an
                # outcome worse than the state it replaced: the whole filtering
                # stage is dropped (dedup, fuzzy, embedding, edge prediction,
                # alignment) while the extraction completes and reports success.
                #
                # And it is reachable without a restart: `get_concept_alignment_enabled`
                # reads the env at call time, deliberately, so flipping
                # ENABLE_CONCEPT_ALIGNMENT on a running server produced exactly
                # that on every subsequent extraction. `run_filtering_only` has no
                # enclosing try and propagated correctly, so the two call sites
                # disagreed about whether a refusal means anything. They agree now.
                raise
            except Exception as e:
                # Filtering itself failing is non-fatal — fall through and the
                # raw extraction results are still saved below. ``filtered``
                # stays None so the persist step is skipped.
                logger.error(f"Filtering failed for source {source_id}: {e}")

            # 6. Persist filtered entities to KG.
            # B.8a: this is intentionally OUTSIDE the filtering try/except.
            # A persistence failure (e.g. a fully-failed entity batch raising
            # from persist_filtered_result) must PROPAGATE — otherwise it would
            # be masked as "Filtering failed" and the extraction would report
            # success while writing nothing to the KG. Provenance (real method
            # + resolved model) is threaded so the KG records what produced
            # each entity.
            if filtered is not None:
                persist_result = await self._persistence.persist_filtered_result(
                    source_id=source_id,
                    entities=[e.model_dump() for e in filtered.entities],
                    relations=all_relations,
                    merge_groups=merge_groups,
                    match_candidates=[c.model_dump() for c in filtered.match_candidates] if filtered.match_candidates else None,
                    extraction_method=extractor_type,
                    # Record the model that actually SERVED this run (J-Q7),
                    # falling back to the register model for the task when
                    # nothing was served.
                    # Not config.llm_model, which is "default" on the common path
                    # and would stamp None even though a model produced the rows.
                    extraction_model=await self._served_extraction_model(config),
                    # L.1 + PC.4: the TYPING vocabulary — the schemas detected
                    # for this run followed by every other ontology, with the
                    # notebook's edits (set by _run_multi_schema). No
                    # re-detection here.
                    applicable_schemas=await self._typing_schemas_or_all(),
                )

                # Q.4 R1 hook (main path): fire triage AFTER the persist, OUTSIDE
                # the filtering try/except, with its own log-and-continue guard.
                await self._run_triage(source_id, persist_result)

        # PC.1b / W3: record what was WRITTEN before the row is saved.
        #
        # It goes on the RESULT METADATA as well as the summary, and that is not
        # belt-and-braces. The summary lands in `job.result`, which nothing in the
        # repo reads — writing only there would have created a fresh producer with
        # no consumer, in the phase whose whole point is to stop doing that.
        # Building `scripts/pc1b_handoff_probe.py` is what caught it. The metadata
        # copy is persisted on `extraction_result` and the probe reads it.
        #
        # It cannot sit beside the filtering stats — those are assigned before
        # persistence has run — nor beside the summary below, because
        # `_save_result` has already happened by then.
        written = _persisted_counts(persist_result)
        if written:
            result.metadata["persisted"] = written

        # PC.4: which ontologies detection selected for this document. It was
        # recorded nowhere, so measuring detection meant replaying it with
        # today's ontologies. Read by `scripts/pc4_detection_probe.py`.
        if self._applicable_schemas:
            result.metadata["applied_ontologies"] = [
                o.metadata.name for o in self._applicable_schemas if o.metadata
            ]
        elif result.metadata.get("ontology_name"):
            # The single-schema (legacy) path: its one ontology IS the selection.
            result.metadata["applied_ontologies"] = [result.metadata["ontology_name"]]

        # 7. Persist raw extraction results
        await self._save_result(source_id, result)

        summary = {
            "source_id": source_id,
            "entity_count": result.entity_count,
            "relation_count": result.relation_count,
            **filtering_stats,
        }
        counters = _observability_counters(getattr(result, "metadata", None))
        if counters:
            summary["counters"] = counters
        # PC.1b / W3: what was WRITTEN, beside what was extracted.
        # `persist_filtered_result` has always returned these five; only
        # `persisted_entity_ids` was read, so `summary["entity_count"]` is the
        # count the LLM produced and nothing could express the difference. The
        # pipeline review measured 124 entities extracted against 117 rows in the
        # graph — a real and interesting gap that the system producing it had no
        # way to report. PC.3's own AC ("materially fewer than 117 rows, with a
        # named figure") needs this instrument.
        if written:
            summary["persisted"] = written
        logger.info(
            f"Entity extraction completed for source {source_id}: "
            f"{result.entity_count} entities, {result.relation_count} relations"
        )

        # 8. Emit one telemetry row per run_extraction (B.4 / RETRO #5).
        # ``record_metric`` is failure-tolerant — extraction never fails
        # on a telemetry hiccup. Computed here (rather than inside each
        # branch) so the multi-schema and single-schema paths share one
        # canonical event shape.
        avg_confidence = _avg_entity_confidence(result.entities)
        await record_metric(
            "extraction.complete",
            {
                "entity_count": result.entity_count,
                "relation_count": result.relation_count,
                "avg_confidence": avg_confidence,
                "multi_schema": use_multi_schema,
            },
            source=source_id,
            notebook=notebook_id,
        )

        # 9. B.5b: retry any orphans still in ``pending_reconnect`` for
        # this notebook. The fresh chunks from this source-import may
        # contain co-occurrences that B.5a could not previously confirm.
        # This is best-effort: a failure here MUST NOT fail extraction.
        if notebook_id:
            await self._retry_pending_reconnects_best_effort(
                notebook_id=notebook_id,
                chunks=chunk_dicts,
            )

        return summary

    async def _retry_pending_reconnects_best_effort(
        self,
        notebook_id: str,
        chunks: List[Dict[str, Any]],
    ) -> None:
        """Best-effort B.5b retry after extraction.

        Calls :func:`retry_pending_reconnects` for the notebook,
        swallowing any failure so a flaky orphan-connector cannot kill
        an otherwise successful extraction. Logs at WARNING when
        something goes wrong; the surrounding extraction telemetry has
        already been emitted.

        Optimisation: skip the LLM-caller build when no pending orphans
        exist for the notebook. The list query is cheap and avoids a
        second :func:`make_default_llm_caller` round-trip on every
        extraction (the hot path). This is also what keeps the existing
        ``test_invokes_default_llm_caller_factory_for_multi_path`` test
        green -- the spy asserts ``assert_awaited_once``, so we must not
        build a second caller when nothing is pending.
        """
        try:
            # Lazy imports keep the hot path (single-source extraction
            # without orphans) free of orphan-prune cost.
            from entity_filtering.resolution.orphan_prune import (
                STATUS_PENDING_RECONNECT,
                retry_pending_reconnects,
            )

            from app_main.dependencies import get_entity_repo

            entity_repo = get_entity_repo()

            # Cheap precheck -- skip the LLM-caller build (and the whole
            # retry path) when there's nothing pending. Critical for the
            # hot path because make_default_llm_caller resolves the
            # default chat model on every call.
            pending = await entity_repo.list_orphans_with_status(
                notebook_id, STATUS_PENDING_RECONNECT
            )
            if not pending:
                return

            # Try to build an LLM caller, but tolerate failure -- the
            # retry will then just record the attempt without confirming.
            llm_caller = None
            try:
                llm_caller = await self._make_routed_caller()
            except Exception as e:
                logger.warning(
                    f"B.5b retry: failed to wire LLM caller ({e}); "
                    "retry will record attempts but cannot confirm."
                )

            outcome = await retry_pending_reconnects(
                notebook_id,
                entity_repo,
                chunks,
                llm_caller=llm_caller,
            )
            if outcome.attempted:
                logger.info(
                    "B.5b retry on notebook={nb}: attempted={a} "
                    "reconnected={r} still_pending={sp}",
                    nb=notebook_id,
                    a=outcome.attempted,
                    r=outcome.reconnected,
                    sp=outcome.still_pending,
                )
        except Exception as e:
            # Best-effort: never block extraction on an orphan-prune
            # hiccup.
            logger.warning(
                f"B.5b retry_pending_reconnects failed for notebook "
                f"{notebook_id}: {e}"
            )

    async def run_filtering_only(
        self,
        source_id: str,
        filtering_config: Optional[FilteringConfig] = None,
    ) -> Dict[str, Any]:
        """Run filtering on an existing extraction result without re-extracting.

        Fetches the raw extraction_result, runs FilteringWorkflow, and
        persists filtered entities to the KG tables.
        """
        # N.4d.4: this path re-detects NO schemas, so anything left on the
        # instance belongs to a previous run of a different source. Cleared
        # FIRST, before any early return, for the same reason the L.1 comment
        # below passes `applicable_schemas=None` to persist: "so a reused service
        # instance can't leak a prior run's schemas onto a different source's
        # re-filter". An earlier draft argued the leak was unreachable because
        # the DI provider happens to build a fresh instance per call — the
        # assumption that comment had already declined to rely on.
        self._applicable_schemas = None
        self._typing_schemas = None
        self._gap_ontology_name = None
        # F2-08: geen classificatie van een eerdere run meenemen; onbekend telt als intern.
        self._classificatie = None
        self._run_id = None

        # Fetch existing extraction result
        rows = await execute_query(
            "SELECT * FROM extraction_result WHERE source_id = $source_id LIMIT 1",
            {"source_id": source_id},
        )
        if not rows:
            raise ValueError(f"No extraction result found for source {source_id}")

        row = rows[0]
        entity_dicts = row.get("entities", [])
        relation_dicts = row.get("relations", [])

        if not entity_dicts:
            return {
                "source_id": source_id,
                "entities_before": 0,
                "entities_after": 0,
                "entities_removed": 0,
                "merge_groups": 0,
                "predicted_edges": 0,
            }

        # Reconstruct ExtractionResult for the FilteringWorkflow
        from shared.models.extraction import (
            ExtractedEntity,
            ExtractedRelation,
            ExtractionResult,
        )

        extraction = ExtractionResult(
            entities=[ExtractedEntity(**e) for e in entity_dicts],
            relations=[ExtractedRelation(**r) for r in relation_dicts],
            metadata=row.get("metadata", {}),
        )

        # Embed entities for semantic dedup (if not already embedded)
        has_embeddings = any(
            e.properties.get("embedding") for e in extraction.entities
        )
        if not has_embeddings:
            await self._embed_entities(extraction)

        # Run filtering
        f_config = filtering_config or FilteringConfig()
        # The same collaborators the main path wires, so a re-filter is not
        # silently a different pipeline. With no schemas detected the stage is
        # inert here regardless (no canonical type resolves, so no verdict
        # licenses a gap); wiring it costs nothing and means a caller who DOES
        # enable it gets the same behaviour rather than a quietly degraded one.
        align_deps = await self._concept_alignment_deps(f_config)
        f_workflow = FilteringWorkflow(
            config=f_config,
            entity_repo=align_deps["entity_repo"],
            alignment_schemas=align_deps["schemas"],
            gap_recorder=align_deps["gap_recorder"],
            gap_ontology_name=align_deps["gap_ontology_name"],
            llm_chat=self._llm_chat,
        )
        filtered = await f_workflow.process(
            extraction,
            source_id=source_id,
            alignment_llm_caller=align_deps["llm_caller"],
        )

        # Persist to KG
        all_relations = [
            r.model_dump() for r in filtered.relations
        ] + [
            r.model_dump() for r in filtered.predicted_edges
        ]
        # B.8a: preserve the ORIGINAL extraction's provenance on re-filter —
        # don't rewrite every entity's method to the default "llm". The stored
        # extraction_result.metadata carries the extractor_type.
        persist_result = await self._persistence.persist_filtered_result(
            source_id=source_id,
            entities=[e.model_dump() for e in filtered.entities],
            relations=all_relations,
            merge_groups=filtered.merged_entity_groups,
            match_candidates=[c.model_dump() for c in filtered.match_candidates] if filtered.match_candidates else None,
            extraction_method=extraction.metadata.get("extractor_type", "llm"),
            # L.1: the re-filter path does not re-detect schemas. PC.4: it
            # still types against every ontology, as the extraction did —
            # typing no longer depends on the selection, so there is nothing to
            # re-detect. `_typing_schemas` was reset above, so a reused instance
            # cannot leak a prior run's vocabulary.
            applicable_schemas=await self._typing_schemas_or_all(),
        )

        # Q.4 R1 hook (re-filter path): triage fires here too (operator-confirmed,
        # R1) so a re-filter re-derives status with the same non-blocking guard.
        await self._run_triage(source_id, persist_result)

        stats = {
            "source_id": source_id,
            "entities_before": len(entity_dicts),
            "entities_after": len(filtered.entities),
            "entities_removed": len(filtered.removed_entities),
            "merge_groups": len(filtered.merged_entity_groups)
            if filtered.merged_entity_groups
            else 0,
            "predicted_edges": len(filtered.predicted_edges),
        }

        # Update extraction_result metadata with filtering stats.
        #
        # PC.1b / W3b: MERGE rather than overwrite. This dict has six keys; the
        # extraction path's version also carries a `concept_alignment` block with
        # the alignment counters, and assigning over the whole key destroyed it —
        # silently, on any re-filter. Keys this path recomputes win; keys it knows
        # nothing about survive.
        metadata = row.get("metadata", {})
        previous_filtering = metadata.get("filtering")
        if isinstance(previous_filtering, dict):
            metadata["filtering"] = {**previous_filtering, **stats}
        else:
            metadata["filtering"] = stats
        # PC.5c: this run persisted again, so `persisted` (what GET /sources/{id}
        # shows as the source's relations) must describe this run, not the first.
        written = _persisted_counts(persist_result)
        if written:
            metadata["persisted"] = written
        await execute_query(
            "UPDATE extraction_result SET metadata = $metadata "
            "WHERE source_id = $source_id",
            {"source_id": source_id, "metadata": metadata},
        )

        logger.info(f"Filtering-only completed for source {source_id}: {stats}")
        return stats

    async def _save_result(self, source_id: str, result) -> None:
        """Persist extraction result to SurrealDB.

        ``entities``/``relations`` and their counts are the RAW extraction,
        before filtering — this row is the re-filter cache
        (``run_filtering_only`` reads it). What reached the graph is
        ``metadata.persisted``; a reader that shows a graph count must use that
        (PC.5c: ``GET /sources/{id}`` did not).
        """
        try:
            await execute_query(
                "DELETE FROM extraction_result WHERE source_id = $source_id",
                {"source_id": source_id},
            )
            await execute_query(
                "CREATE extraction_result SET "
                "source_id = $source_id, "
                "entities = $entities, "
                "relations = $relations, "
                "metadata = $metadata, "
                "entity_count = $entity_count, "
                "relation_count = $relation_count, "
                "created = time::now()",
                {
                    "source_id": source_id,
                    "entities": [e.model_dump() for e in result.entities],
                    "relations": [r.model_dump() for r in result.relations],
                    "metadata": result.metadata,
                    "entity_count": result.entity_count,
                    "relation_count": result.relation_count,
                },
            )
            logger.info(
                f"Saved extraction result for source {source_id}"
            )
        except Exception as e:
            # B.8d: by this point the entities + relations are ALREADY persisted
            # to the KG (entity/relation tables). The extraction_result record is
            # only a secondary cache for the re-filter path (run_filtering_only).
            # A failure here — notably the surrealdb async-ws client raising
            # ``KeyError(<request-uuid>)`` likely triggered by the large serialized entities/
            # relations payload — must NOT fail the whole extraction job, or a
            # successful extraction (hundreds of persisted entities) gets marked
            # "failed" and dead-lettered. Log and continue; the re-filter cache
            # is simply absent (re-extraction regenerates it).
            logger.warning(
                f"Could not save raw extraction_result for {source_id} "
                f"(entities already persisted; re-filter cache skipped): {e}"
            )
