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

from typing import Any, Dict, Iterable, List, Optional

from loguru import logger
from shared.services.metrics import record_metric
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories import (
    NotebookSchemaRepository,
    Pass1ResultRepository,
    SourceRepository,
)

from job_queue import JobPausedForReviewError
from ontology_extraction.config import ExtractionConfig
from ontology_extraction.multi_schema_orchestrator import detect_applicable_schemas
from ontology_extraction.workflow import ExtractionWorkflow

from entity_filtering.config import FilteringConfig
from entity_filtering.workflow import FilteringWorkflow

from app_main.services.entity_persistence_service import EntityPersistenceService


def _is_resume_sentinel(extension: Dict[str, Any]) -> bool:
    """Return ``True`` for a B.3c resume-sentinel extension entry.

    The sentinel is appended by ``POST /extraction/resume`` to satisfy
    the review-gate predicate (``review_required AND accepted_extensions
    empty``). It carries no ontology content — only the marker fields
    ``type_name="_resumed_without_extensions"`` and
    ``is_resume_sentinel=True``.

    Any consumer that forwards ``accepted_extensions`` into a downstream
    artifact (LLM prompt, TTL export, JSON response) MUST filter
    sentinels first — otherwise the LLM gets instructed to treat
    ``_resumed_without_extensions`` as a first-class entity type, which
    pollutes Pass-2 output. This helper centralises the predicate so
    every filter site agrees on the marker shape.
    """
    return bool(extension.get("is_resume_sentinel"))


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


async def resolve_default_model_id(
    default_field: str = "default_chat_model",
    model_id: Optional[str] = None,
) -> Optional[str]:
    """Resolve the model id for a use case — the single source of truth for
    model-selection precedence (used by both the LLM-caller factory and entity
    provenance, so they can never drift):

    explicit ``model_id`` override → the requested per-function default (e.g.
    ``default_extraction_model``) → ``default_chat_model`` as the universal
    fallback. Returns ``None`` if nothing is configured.
    """
    from llm_manager import get_model_manager

    from app_main.dependencies import get_default_models_repo

    mm = get_model_manager()
    defaults = mm.get_defaults()
    if not defaults or not defaults.default_chat_model:
        defaults = await get_default_models_repo().get()
        mm.set_defaults(defaults)

    if not defaults:
        return model_id
    return (
        model_id
        or getattr(defaults, default_field, None)
        or defaults.default_chat_model
    )


async def make_default_llm_caller(
    model_id: Optional[str] = None,
    *,
    default_field: str = "default_chat_model",
):
    """Build an async ``LLMCaller`` backed by :class:`ModelManager`.

    ``default_field`` selects which configured default to resolve (e.g.
    ``default_extraction_model`` for the KG-extraction path), independent of
    the chat model. Resolution precedence is owned by
    :func:`resolve_default_model_id`.

    Phase B.1f fixes the long-standing LLMExtractor "silent empty"
    bug. The old code wrote ``from llm_manager.manager import LLMManager``
    and ``manager.generate(...)`` — both symbols had been renamed
    (``LLMManager`` → :class:`ModelManager`) and replaced
    (``.generate`` → ``ModelInstance.achat_complete``). The
    consequence in production: ``ImportError`` caught silently,
    extractor returned empty results, no extraction ever happened.

    This factory closes the loop. It:

    1. Looks up the configured default chat model via
       :class:`shared.models.DefaultModels`.
    2. Resolves the :class:`Model` row from the repository.
    3. Instantiates the esperanto ``LanguageModel`` via
       ``ModelManager.get_model_from_config``.
    4. Returns an async ``(system, user, model) -> str`` callable
       that dispatches via ``LanguageModel.achat_complete``.

    Args:
        model_id: Override the default chat model id. ``None`` means
            "use the configured default".

    Returns:
        An async callable matching the
        :type:`ontology_extraction.pass2_typed_extraction.LLMCaller`
        protocol (``async (system_prompt, user_prompt, model) -> str``).
    """
    # Local imports to keep module-import cheap (these chains are
    # heavy: esperanto loads provider SDKs lazily).
    from esperanto import LanguageModel
    from llm_manager import get_model_manager

    from app_main.dependencies import get_model_repo

    mm = get_model_manager()

    resolved_id = await resolve_default_model_id(default_field, model_id)
    if not resolved_id:
        raise RuntimeError(
            f"No model configured — set DefaultModels.{default_field} or "
            "default_chat_model, or pass model_id explicitly."
        )

    model_record = await get_model_repo().get(resolved_id)
    if not model_record:
        raise RuntimeError(
            f"Model '{resolved_id}' not found in database."
        )

    instance = mm.get_model_from_config(model_record)
    if not isinstance(instance, LanguageModel):
        raise TypeError(
            f"Model '{resolved_id}' is not a LanguageModel "
            f"(got {type(instance).__name__})."
        )

    async def _caller(system_prompt: str, user_prompt: str, _model: str) -> str:
        # The injected ``model`` arg is for telemetry/parity with
        # Pass-1/Pass-2 callers; the LanguageModel itself is already
        # bound. We honour it by logging when the caller asks for a
        # model id that differs from the bound instance — a sign the
        # caller wants per-call model overrides that we don't yet
        # support.
        if _model and _model not in ("default", model_record.id):
            logger.warning(
                f"LLMExtractor caller requested model={_model!r} but "
                f"this caller is bound to {model_record.id!r}"
            )
        response = await instance.achat_complete(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
        )
        # esperanto ``ChatCompletion`` shape: ``response.choices[0].message.content``
        # Defensive extraction — return empty string if the shape changes
        # rather than raising, because the extractor's JSON parser
        # tolerates empty/invalid output.
        try:
            return response.choices[0].message.content or ""
        except (AttributeError, IndexError) as e:
            logger.error(
                f"Unexpected esperanto ChatCompletion shape: {e}; "
                f"response={response!r}"
            )
            return ""

    return _caller


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
    ):
        self._source_repo = source_repo
        # B.1f wiring: the multi-schema orchestrator needs both repos.
        # The legacy single-schema path doesn't, so they default to
        # ``None`` and are lazily constructed when needed. Tests can
        # inject fakes to assert on the multi-schema branch.
        self._notebook_schema_repo = notebook_schema_repo
        self._pass1_repo = pass1_repo
        self._persistence = EntityPersistenceService()

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
            vectors = await embed_svc.embedding_model.aembed(texts)

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
                llm_caller = await make_default_llm_caller(
                    model_id=config.llm_model
                    if config.llm_model != "default"
                    else None,
                    default_field="default_extraction_model",
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

        # Review gate — if the user has paused processing for this
        # notebook, do not run extraction. The caller (handler) treats
        # this exception as "park the job in PAUSED_FOR_REVIEW".
        if (
            notebook_schema is not None
            and notebook_schema.review_required
            and not notebook_schema.accepted_extensions
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

        # Sample text for the applicability scorer — first chunk's
        # content is a cheap signal. The orchestrator's Pass-1 step
        # builds a richer sample independently.
        sample_text: Optional[str] = None
        for chunk in chunks:
            text = chunk.get("text", "")
            if text:
                sample_text = text[:2000]
                break

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

        # Build the per-schema accepted-extensions map from the
        # notebook's accepted extensions. Each extension dict carries
        # an optional ``schema_name`` field; ones without it are
        # broadcast to every schema (conservative default).
        #
        # B.3c: resume-sentinel entries are infrastructure markers, not
        # real entity types. If we forward them, ``_run_pass2`` will
        # render them into the LLM prompt as first-class extensions
        # (see ``_format_accepted_extensions``). Filter at this seam so
        # the prompt builder never sees them; ``_format_accepted_extensions``
        # also filters defensively (defence-in-depth) in case a future
        # caller bypasses this service.
        accepted_by_schema: Dict[str, List[Dict[str, Any]]] = {}
        if notebook_schema is not None:
            for ext in notebook_schema.accepted_extensions:
                if _is_resume_sentinel(ext):
                    continue
                schema_name = ext.get("schema_name")
                if schema_name:
                    accepted_by_schema.setdefault(schema_name, []).append(ext)
                else:
                    for ontology, _conf in applicable_schemas:
                        name = ontology.metadata.name if ontology.metadata else ""
                        if name:
                            accepted_by_schema.setdefault(name, []).append(ext)

        # Build the production LLM caller once per run so Pass-1 and
        # Pass-2 share a single bound ``LanguageModel``. Failures here
        # are logged but non-fatal — Pass-1/Pass-2 fall through to
        # their lazy-default empty-result paths if no caller arrives.
        llm_caller = None
        try:
            llm_caller = await make_default_llm_caller(
                default_field="default_extraction_model"
            )
        except Exception as e:
            logger.warning(
                f"multi_schema: failed to wire LLM caller ({e}); "
                "Pass-1/Pass-2 will run with their lazy defaults."
            )

        return await workflow.extract(
            chunks=chunks,
            mode="multi",
            applicable_schemas=applicable_schemas,
            source_id=source_id,
            notebook_id=notebook_id,
            pass1_repo=pass1_repo,
            accepted_extensions_by_schema=accepted_by_schema or None,
            llm_caller=llm_caller,
        )

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
    ) -> Dict[str, Any]:
        """
        Run entity extraction and optional filtering for a source.

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
        chunks = await self._source_repo.get_chunks(source_id)
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
            # (CLI / tests), when ops disables multi-schema via the flag, or
            # when the caller selects ``extractor_type="langextract"``. The
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

        if run_filtering and (result.entities or result.relations):
            filtered = None
            all_relations: List[Dict[str, Any]] = []
            try:
                if filtering_config:
                    f_config = filtering_config
                else:
                    # Default config: string dedup + fuzzy + embedding
                    from entity_filtering.config import (
                        EmbeddingDedupConfig,
                        FuzzyDedupConfig,
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
                        edge_prediction_enabled=True,
                    )
                f_workflow = FilteringWorkflow(config=f_config)

                filtered = await f_workflow.process(result)

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
                result.metadata["filtering"] = filtering_stats

                logger.info(
                    f"Filtering complete for source {source_id}: "
                    f"{filtering_stats}"
                )

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
                await self._persistence.persist_filtered_result(
                    source_id=source_id,
                    entities=[e.model_dump() for e in filtered.entities],
                    relations=all_relations,
                    merge_groups=merge_groups,
                    match_candidates=[c.model_dump() for c in filtered.match_candidates] if filtered.match_candidates else None,
                    extraction_method=extractor_type,
                    # Record the model actually used (resolved via the same
                    # precedence as the LLM caller) — not config.llm_model,
                    # which is "default" on the common path and would stamp
                    # None even though the extraction model produced the rows.
                    extraction_model=await resolve_default_model_id(
                        "default_extraction_model",
                        config.llm_model if config.llm_model != "default" else None,
                    ),
                )

        # 7. Persist raw extraction results
        await self._save_result(source_id, result)

        summary = {
            "source_id": source_id,
            "entity_count": result.entity_count,
            "relation_count": result.relation_count,
            **filtering_stats,
        }
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
                llm_caller = await make_default_llm_caller(
                    default_field="default_extraction_model"
                )
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
        f_workflow = FilteringWorkflow(config=f_config)
        filtered = await f_workflow.process(extraction)

        # Persist to KG
        all_relations = [
            r.model_dump() for r in filtered.relations
        ] + [
            r.model_dump() for r in filtered.predicted_edges
        ]
        # B.8a: preserve the ORIGINAL extraction's provenance on re-filter —
        # don't rewrite every entity's method to the default "llm". The stored
        # extraction_result.metadata carries the extractor_type.
        await self._persistence.persist_filtered_result(
            source_id=source_id,
            entities=[e.model_dump() for e in filtered.entities],
            relations=all_relations,
            merge_groups=filtered.merged_entity_groups,
            match_candidates=[c.model_dump() for c in filtered.match_candidates] if filtered.match_candidates else None,
            extraction_method=extraction.metadata.get("extractor_type", "llm"),
        )

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

        # Update extraction_result metadata with filtering stats
        metadata = row.get("metadata", {})
        metadata["filtering"] = stats
        await execute_query(
            "UPDATE extraction_result SET metadata = $metadata "
            "WHERE source_id = $source_id",
            {"source_id": source_id, "metadata": metadata},
        )

        logger.info(f"Filtering-only completed for source {source_id}: {stats}")
        return stats

    async def _save_result(self, source_id: str, result) -> None:
        """Persist extraction result to SurrealDB."""
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
