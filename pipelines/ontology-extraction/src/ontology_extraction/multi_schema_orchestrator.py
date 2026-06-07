"""Multi-schema orchestrator (Phase B.1e).

Ties Pass-1 (B.1c) and Pass-2 (B.1d) into a single pipeline that:

1. Selects the top-K (default 3) ontologies most applicable to a
   source via :func:`detect_applicable_schemas`.
2. Runs Pass-1 sequentially against each candidate, persists the
   ``pass1_results`` row, and aggregates proposed extensions across
   schemas (deduped by ``type_name``).
3. Decides whether to emit a soft-nudge for the user (silent /
   ``EXTENSION_SUGGESTED`` / ``SCHEMA_MISMATCH``) based on the best
   coverage achieved across all Pass-1 attempts.
4. Runs Pass-2 once per applicable schema, then merges the per-schema
   ``ExtractionResult``s into a single result with multi-type tagging
   on entities (``type_tags`` / ``primary_type``) and max-confidence
   relation dedup.

Design rules (FEATURE_ROADMAP §621-639 + plan §Phase B.1e):

- **Sequential**, not parallel: a single LLM caller is the shared
  resource and we want deterministic ordering of the Pass-2 results so
  ``primary_type`` is reproducible. Parallelism is an optimisation
  that would change merge semantics if two passes finished in
  different orders on rerun.
- **In-process merge, no LLM**: the merge is purely a string-keyed
  aggregation. The "smart" merge work (LLM-arbitrated type
  precedence) belongs in B.4, not here.
- **DI-friendly**: the LLM caller, ``pass1_repo``, and
  ``notebook_schema_repo`` are all injected. Tests pass fakes; B.1f
  wires the real services.
- **Single import point for name normalization** (Q9 future-proofing).
  We import :func:`shared.utils.name_normalizer.normalize_entity_name`
  exactly once — when Q9 swaps the body, every caller picks up the
  improvement.
- **Single source of truth for thresholds**: the soft-nudge cutoffs
  and the applicability floor are re-exported from
  :mod:`shared.config` so the future B.3c UI slider reads the same
  values the backend enforces (RETRO lesson #2).
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Union

from loguru import logger
from ontology_manager.document_mapper import get_ontology_for_document_type
from ontology_manager.schema import Ontology
from shared.config import (
    MIN_APPLICABLE_CONFIDENCE,
    SOFT_NUDGE_COVERAGE_HIGH,
    SOFT_NUDGE_COVERAGE_LOW,
)
from shared.models.extraction import (
    ExtractedEntity,
    ExtractedRelation,
    ExtractionResult,
)
from shared.models.notebook_schema import Pass1Result
from shared.utils.name_normalizer import normalize_entity_name

from ontology_extraction.pass1_schema_validation import (
    Pass1Output,
    Pass1SchemaValidator,
)
from ontology_extraction.pass2_typed_extraction import LLMCaller, run_pass2

# Re-export the shared thresholds so callers can do
# ``from ontology_extraction.multi_schema_orchestrator import
# SOFT_NUDGE_COVERAGE_HIGH`` without reaching into the shared package.
# The module-level rebinding keeps the source-of-truth in
# ``shared.config`` while preserving local import ergonomics.
__all__ = [
    "MIN_APPLICABLE_CONFIDENCE",
    "PASS1_TOKEN_BUDGET",
    "PASS2_SAMPLE_CHAR_BUDGET",
    "SOFT_NUDGE_COVERAGE_HIGH",
    "SOFT_NUDGE_COVERAGE_LOW",
    "MergedEntity",
    "Pass1ResultRepoProtocol",
    "SoftNudgeDecision",
    "detect_applicable_schemas",
    "run_multi_schema",
]

# Approximate character budget for the text sample we feed Pass-1.
# 1500 tokens × ~4 chars/token = 6000 chars. Pass-1's own token-budget
# guard catches a runaway prompt, so this is just a friendly upper
# bound at sample-selection time.
PASS2_SAMPLE_CHAR_BUDGET = 6000

# Mirrors Pass-1's ``TOKEN_BUDGET_TARGET`` for clarity when the
# orchestrator logs sampling decisions. Kept as an int rather than a
# float; Pass-1 imports its own copy from its module so this constant
# is purely descriptive.
PASS1_TOKEN_BUDGET = 1500


class SoftNudgeDecision(str, Enum):
    """Outcome of the soft-nudge coverage-threshold check.

    Returned alongside the merged ``ExtractionResult`` so the caller
    (B.1f / B.3c) can decide what to surface in the UI. Values are
    strings to keep DB persistence trivial — they round-trip through
    ``notebook_event.type`` without manual conversion.

    Members:
        NONE: Best Pass-1 coverage exceeded ``SOFT_NUDGE_COVERAGE_HIGH``
            — schema fits the source well, no UI prompt.
        EXTENSION_SUGGESTED: Best coverage between LOW and HIGH —
            propose the accumulated extension list to the user.
        SCHEMA_MISMATCH: Best coverage below LOW — the user may want
            to switch base ontology entirely.
    """

    NONE = "none"
    EXTENSION_SUGGESTED = "extension_suggested"
    SCHEMA_MISMATCH = "schema_mismatch"


class Pass1ResultRepoProtocol:
    """Structural protocol the orchestrator expects from a pass1 repo.

    Concrete callers pass
    :class:`surrealdb_service.repositories.notebook_schema.Pass1ResultRepository`.
    Tests pass a fake whose ``record`` method captures calls. Keeping
    this as a *protocol* (rather than an ABC) avoids forcing the test
    fake to inherit from a SurrealDB-coupled class.

    The protocol is not used by ``isinstance`` checks — it exists to
    document the contract and give type checkers something to bite
    on. ``run_multi_schema``'s ``pass1_repo`` parameter is typed as
    ``Any`` so duck typing keeps working.
    """

    async def record(self, result: Pass1Result) -> str:
        ...  # pragma: no cover - documentation only


class MergedEntity:
    """In-process aggregation of one surface form across passes.

    Lives inside ``run_multi_schema`` only — it's deliberately not a
    Pydantic model because it never crosses a serialization boundary
    (the final form is :class:`ExtractedEntity`). Keeping it plain
    Python keeps the merge loop allocation-cheap and makes the test
    fixtures trivial.

    The ``best_confidence`` field records the maximum entity-level
    confidence seen across passes; whichever pass posted that
    confidence wins ``primary_type`` and supplies the canonical
    surface ``text`` (so casing variations resolve to the most
    confidently-extracted form).
    """

    __slots__ = (
        "best_confidence",
        "best_label",
        "best_text",
        "first_entity",
        "type_tags",
    )

    def __init__(self, entity: ExtractedEntity) -> None:
        # Seed with the first occurrence — subsequent passes overwrite
        # via ``add`` when they beat the current best confidence.
        self.first_entity = entity
        self.best_text = entity.text
        self.best_label = entity.label
        self.best_confidence = float(entity.confidence)
        self.type_tags: List[str] = [entity.label]

    def add(self, entity: ExtractedEntity) -> None:
        """Fold another pass's view of the same entity into the merge.

        Tag list accumulates labels in insertion order (no dedup —
        the same schema can't appear twice in the orchestrator's
        sequential loop, so duplicates would be a programmer bug
        worth surfacing). ``primary_type`` is recomputed whenever a
        higher-confidence pass arrives. Ties keep the first-seen
        label, mirroring B.1c-r2's timestamp-recency precedent (the
        first pass *is* the most-recent at the moment of tie).
        """
        if entity.label not in self.type_tags:
            # Dedup defensively — a malformed test fixture could pass
            # the same label twice and we'd rather keep the tag list
            # tidy than expose internal book-keeping.
            self.type_tags.append(entity.label)
        ent_conf = float(entity.confidence)
        if ent_conf > self.best_confidence:
            self.best_confidence = ent_conf
            self.best_label = entity.label
            self.best_text = entity.text


# Callable shape: ``(document_type: Optional[str]) -> str``.
# Allows tests to inject a stub mapper without monkey-patching the
# ontology_manager module. ``None`` falls back to the real one.
DocumentTypeMapper = Callable[[Optional[str]], str]

# Async or sync caller for the Pass-1 LLM seam.
LLMCallerAlias = LLMCaller


async def detect_applicable_schemas(
    document_type: Optional[str],
    document_text: Optional[str],
    ontologies: List[Ontology],
    top_k: int = 3,
    mapper: Optional[DocumentTypeMapper] = None,
) -> List[Tuple[Ontology, float]]:
    """Score each ontology by applicability to a source and return the top-K.

    Combines two signals:

    1. **Document-type mapping** (high precision). When
       ``document_type`` maps to one of the known ontology names via
       ``ontology_manager.document_mapper.get_ontology_for_document_type``,
       that ontology gets the floor confidence ``0.92`` (matches the
       AC #4 single-schema-input contract).
    2. **Keyword overlap** (broad recall). For every ontology, count
       how many of its entity-type names appear case-insensitively in
       ``document_text``. The score is ``min(0.9, matches / type_count)``
       capped so the keyword path never beats the document-type
       mapping when both fire.

    The two signals are combined per ontology by ``max(...)`` so an
    ontology that wins either path lands in the shortlist.

    Args:
        document_type: ``Source.metadata['document_type']`` or
            similar. ``None`` skips the mapper signal entirely.
        document_text: A sample of the source's text. ``None`` skips
            the keyword path entirely (the document-type mapper still
            runs).
        ontologies: All ontologies the caller wants considered.
        top_k: Maximum number of candidates to return. Defaults to 3
            per the plan §Phase B.1e.
        mapper: Optional override for the document-type mapper, used
            in tests so we don't need to monkey-patch the
            ontology_manager module.

    Returns:
        A list of ``(ontology, confidence)`` tuples sorted by
        confidence descending, filtered to those scoring at least
        ``MIN_APPLICABLE_CONFIDENCE``. Empty list when nothing applies.
    """
    mapper_fn = mapper or get_ontology_for_document_type
    preferred_name = mapper_fn(document_type) if document_type else None

    text_lower = (document_text or "").lower()

    scored: List[Tuple[Ontology, float]] = []
    for ontology in ontologies:
        name = ontology.metadata.name if ontology.metadata else ""

        # Signal 1: document-type mapping. ``0.92`` mirrors the
        # single-schema-input snapshot value (AC #4). When the mapper
        # returns a name, the orchestrator's intent is "this is the
        # right ontology", and we want it to dominate the keyword
        # signal even if a much larger ontology happens to spam more
        # keyword matches.
        doc_score = 0.92 if preferred_name and name == preferred_name else 0.0

        # Signal 2: keyword overlap. Iterate entity-type names — they
        # are short identifiers that read like keywords. Lowercased
        # substring match is intentionally lenient; the goal is recall
        # so a generic ontology shows up as the 3rd-best candidate.
        if text_lower and ontology.entity_types:
            type_names = list(ontology.entity_types.keys())
            matches = sum(
                1
                for type_name in type_names
                if type_name.lower() in text_lower
            )
            # Normalise by type-count so a 100-type ontology doesn't
            # auto-win on raw match count. Cap at 0.9 so the keyword
            # path can never beat the document-type signal.
            kw_score = min(0.9, matches / max(1, len(type_names)))
        else:
            kw_score = 0.0

        final_score = max(doc_score, kw_score)
        if final_score >= MIN_APPLICABLE_CONFIDENCE:
            scored.append((ontology, final_score))

    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:top_k]


def _decide_soft_nudge(best_coverage: float) -> SoftNudgeDecision:
    """Translate the best Pass-1 coverage into a soft-nudge action.

    Threshold edges:

    - coverage > HIGH (0.95) → NONE
    - LOW ≤ coverage ≤ HIGH → EXTENSION_SUGGESTED
    - coverage < LOW (0.80) → SCHEMA_MISMATCH

    Inclusive at LOW so 0.80 itself is still EXTENSION_SUGGESTED —
    callers tuning the threshold to exactly 0.80 expect that value to
    fall into the suggestion bucket, not the harsher mismatch bucket.

    Inclusive-on-the-bottom keeps the symmetric assertion in the test
    suite easy to read: ``0.85 → EXTENSION_SUGGESTED`` and
    ``0.75 → SCHEMA_MISMATCH`` round-trip without thinking about edge
    semantics.
    """
    if best_coverage > SOFT_NUDGE_COVERAGE_HIGH:
        return SoftNudgeDecision.NONE
    if best_coverage >= SOFT_NUDGE_COVERAGE_LOW:
        return SoftNudgeDecision.EXTENSION_SUGGESTED
    return SoftNudgeDecision.SCHEMA_MISMATCH


def _sample_text_for_pass1(chunks: List[Dict[str, Any]]) -> str:
    """Concatenate chunk text up to ``PASS2_SAMPLE_CHAR_BUDGET`` chars.

    Pass-1 wants a ~1500-token sample of the source. We approximate by
    joining chunks (largest contribution to smallest) until we hit a
    char budget. Pass-1's own token-budget guard catches the boundary
    case where the resulting sample is still too big after a runaway
    chunk merge.

    Pure helper so the orchestrator stays linear and testable.
    """
    parts: List[str] = []
    running = 0
    for chunk in chunks:
        text = str(chunk.get("text", "") or "").strip()
        if not text:
            continue
        # Leave room for the join separator.
        remaining = PASS2_SAMPLE_CHAR_BUDGET - running - 2
        if remaining <= 0:
            break
        if len(text) > remaining:
            parts.append(text[:remaining])
            break
        parts.append(text)
        running += len(text) + 2
    return "\n\n".join(parts)


def _dedupe_extensions(
    extensions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Dedup a flat extension list by ``type_name``, keeping first-seen.

    The plan calls for accumulating extensions across passes; the same
    ``type_name`` showing up twice is the common case (a missing type
    is missing for *every* schema). First-seen wins so the order
    matches the schema-iteration order — useful when reviewing in the
    UI because the higher-applicability schema's extensions appear
    first.
    """
    seen: Dict[str, Dict[str, Any]] = {}
    for ext in extensions:
        name = ext.get("type_name")
        if not name:
            continue
        if name not in seen:
            seen[name] = ext
    return list(seen.values())


async def run_multi_schema(
    source_id: str,
    notebook_id: str,
    chunks: List[Dict[str, Any]],
    applicable_schemas: List[Tuple[Ontology, float]],
    llm_caller: Optional[LLMCallerAlias] = None,
    pass1_repo: Optional[Any] = None,
    accepted_extensions_by_schema: Optional[Dict[str, List[Dict[str, Any]]]] = None,
    model: str = "default",
) -> Tuple[ExtractionResult, SoftNudgeDecision]:
    """Run the full multi-schema Pass-1 + Pass-2 + merge pipeline.

    Algorithm summary (plan §Phase B.1e):

    1. For each ``(schema, applicability_confidence)`` in
       ``applicable_schemas``:

        a. Sample chunks up to ``PASS2_SAMPLE_CHAR_BUDGET`` chars.
        b. Run :class:`Pass1SchemaValidator` against the sample.
        c. Persist a ``pass1_results`` row via ``pass1_repo.record``.
        d. Accumulate the pass's ``proposed_extensions`` into a
           deduped flat list (so the UI sees each missing type once).

    2. Decide a :class:`SoftNudgeDecision` from the *best* coverage
       seen across all attempts.

    3. For each schema sequentially: run :func:`run_pass2` with the
       caller-supplied accepted extensions for *that* schema (or an
       empty list — extensions become accepted via B.3c approval, the
       orchestrator doesn't auto-accept them).

    4. Merge the per-schema ``ExtractionResult`` instances:

        - Entities: keyed by ``normalize_entity_name(text)``. Each
          entity contributes a ``type_tag``; the pass with highest
          per-entity confidence supplies ``primary_type`` and the
          canonical ``text`` field.
        - Relations: keyed by
          ``(normalize(source), normalize(target), relation_type)``.
          Max-confidence wins; properties from the winner are kept.

    Args:
        source_id: Record ID of the source under extraction (e.g.
            ``"source:abc"``). Forwarded to ``pass1_repo`` so the
            ``pass1_results`` rows can be filtered per source.
        notebook_id: Record ID of the owning notebook. Same forwarding.
        chunks: List of ``{"text": str, "id": Optional[str]}`` dicts.
        applicable_schemas: Output of :func:`detect_applicable_schemas`.
            Empty list short-circuits to an empty result + ``NONE``
            decision (no schemas to run against).
        llm_caller: Injected LLM caller passed through to both
            ``Pass1SchemaValidator`` and ``run_pass2``. ``None``
            falls through to their respective lazy defaults (CLI
            only — production must inject).
        pass1_repo: Object exposing ``async record(Pass1Result) -> str``.
            When ``None`` the orchestrator logs a WARNING and skips
            persistence (test mode).
        accepted_extensions_by_schema: Optional dict keyed by ontology
            name, mapping to the extensions the user has already
            approved for that schema. Missing keys default to ``[]``.
        model: Model id forwarded to both Pass-1 and Pass-2.

    Returns:
        ``(merged_result, decision)`` where ``merged_result`` is an
        :class:`ExtractionResult` with multi-tagged entities and the
        ``metadata`` bag populated with per-schema bookkeeping, and
        ``decision`` is the :class:`SoftNudgeDecision` for the UI.
    """
    accepted_map = accepted_extensions_by_schema or {}

    # Short-circuit: no schemas applicable → empty result, NONE nudge.
    # Matches the "no chunks" short-circuit in ``run_pass2`` —
    # zero-work paths must be cheap and side-effect-free.
    if not applicable_schemas:
        logger.info(
            "multi_schema_run skipped: no applicable schemas "
            "(source={src}, notebook={nb})",
            src=source_id,
            nb=notebook_id,
        )
        return (
            ExtractionResult(
                metadata={
                    "source_id": source_id,
                    "notebook_id": notebook_id,
                    "schemas_attempted": [],
                    "best_coverage": 0.0,
                    "soft_nudge": SoftNudgeDecision.NONE.value,
                }
            ),
            SoftNudgeDecision.NONE,
        )

    # ---------------------------------------------------------------
    # Phase 1: Pass-1 against each candidate, accumulate extensions.
    # ---------------------------------------------------------------
    sample = _sample_text_for_pass1(chunks)
    validator = Pass1SchemaValidator(llm_caller=llm_caller, default_model=model)

    pass1_outputs: List[Tuple[Ontology, float, Pass1Output]] = []
    accumulated_extensions: List[Dict[str, Any]] = []
    best_coverage = 0.0
    schemas_attempted: List[str] = []

    for ontology, applicability_conf in applicable_schemas:
        schema_name = (
            ontology.metadata.name if ontology.metadata else "unknown"
        )
        schemas_attempted.append(schema_name)

        logger.info(
            "pass1_attempt source={src} schema={name} applicability={conf:.2f}",
            src=source_id,
            name=schema_name,
            conf=applicability_conf,
        )

        # Use ``model_dump`` so Pass-1 receives the dict-shape it
        # expects. The validator handles both list-shape and dict-shape
        # but ``model_dump`` is the canonical path.
        ontology_dict = ontology.model_dump()
        try:
            pass1_out = await validator.run(sample, ontology_dict, model=model)
        except Exception as e:
            # Pass-1 raises only ``TokenBudgetExceeded`` and
            # ``Pass1ParseError``. Either case is a partial failure
            # for the orchestrator — log it, skip the schema, keep
            # going so the user gets *some* extraction even if one
            # candidate is broken.
            logger.warning(
                "pass1_failed source={src} schema={name}: {err}",
                src=source_id,
                name=schema_name,
                err=str(e),
            )
            continue

        pass1_outputs.append((ontology, applicability_conf, pass1_out))
        if pass1_out.coverage_pct > best_coverage:
            best_coverage = pass1_out.coverage_pct

        # Accumulate cumulative extension list. Deduped at end of loop
        # so the per-schema view stays intact in ``pass1_results``.
        accumulated_extensions.extend(pass1_out.proposed_extensions)

        # Persist the Pass-1 attempt. Failure here MUST NOT abort the
        # run — the extraction itself can still succeed.
        if pass1_repo is None:
            logger.warning(
                "multi_schema_run: pass1_repo not provided, "
                "skipping persistence for schema={name}",
                name=schema_name,
            )
        else:
            pass1_row = Pass1Result(
                source=source_id,
                notebook=notebook_id,
                schema_attempted=schema_name,
                **pass1_out.model_dump(),
            )
            try:
                await pass1_repo.record(pass1_row)
            except Exception as e:
                logger.exception(
                    "pass1_record_failed source={src} schema={name}: {err}",
                    src=source_id,
                    name=schema_name,
                    err=str(e),
                )

    deduped_extensions = _dedupe_extensions(accumulated_extensions)
    decision = _decide_soft_nudge(best_coverage)

    logger.info(
        "multi_schema_pass1_complete source={src} best_cov={cov:.2f} "
        "decision={dec} extensions={ext}",
        src=source_id,
        cov=best_coverage,
        dec=decision.value,
        ext=len(deduped_extensions),
    )

    # ---------------------------------------------------------------
    # Phase 2: Pass-2 against each schema sequentially.
    # ---------------------------------------------------------------
    per_schema_results: List[Tuple[str, ExtractionResult]] = []
    for ontology, _conf in applicable_schemas:
        schema_name = (
            ontology.metadata.name if ontology.metadata else "unknown"
        )
        accepted_for_schema = accepted_map.get(schema_name, [])
        try:
            pass2_result = await run_pass2(
                chunks=chunks,
                ontology=ontology,
                accepted_extensions=accepted_for_schema,
                llm_caller=llm_caller,
                model=model,
            )
        except Exception as e:
            logger.exception(
                "pass2_failed source={src} schema={name}: {err}",
                src=source_id,
                name=schema_name,
                err=str(e),
            )
            continue
        per_schema_results.append((schema_name, pass2_result))

    # ---------------------------------------------------------------
    # Phase 3: Merge across schemas.
    # ---------------------------------------------------------------
    merged = _merge_results(per_schema_results)
    merged.metadata.update(
        {
            "source_id": source_id,
            "notebook_id": notebook_id,
            "schemas_attempted": schemas_attempted,
            "best_coverage": best_coverage,
            "soft_nudge": decision.value,
            "proposed_extensions": deduped_extensions,
        }
    )
    return merged, decision


def _merge_results(
    per_schema_results: List[Tuple[str, ExtractionResult]],
) -> ExtractionResult:
    """Merge Pass-2 results across multiple schemas.

    Single-schema input is a degenerate case: the merge step writes
    no ``type_tags`` and leaves ``primary_type=None``, so the output
    is byte-identical to the input result. This satisfies AC #4
    (single-schema-input regression snapshot) without a special
    code path.

    Pure function so the test suite can drive it directly without
    spinning up the full ``run_multi_schema`` pipeline.
    """
    # Single-schema short-circuit: return the input untouched (no
    # per-pass merging is meaningful with one pass).
    if len(per_schema_results) <= 1:
        if not per_schema_results:
            return ExtractionResult()
        # Defensive copy so the original input is untouched.
        return per_schema_results[0][1].model_copy(deep=True)

    # ------------------------------------------------------------------
    # Entity merge.
    # ------------------------------------------------------------------
    merged_entities: Dict[str, MergedEntity] = {}
    # Track insertion order so the merged output is deterministic
    # (matters for snapshot tests and for the UI showing earlier
    # high-applicability schemas first).
    entity_insertion_order: List[str] = []

    for _schema_name, result in per_schema_results:
        for entity in result.entities:
            key = normalize_entity_name(entity.text)
            if not key:
                # Defensive: an entity with no normalisable text is
                # noise from a bad LLM response — drop it.
                continue
            if key not in merged_entities:
                merged_entities[key] = MergedEntity(entity)
                entity_insertion_order.append(key)
            else:
                merged_entities[key].add(entity)

    final_entities: List[ExtractedEntity] = []
    for key in entity_insertion_order:
        merged = merged_entities[key]
        # Use the highest-confidence entity as the basis so source
        # grounding and extraction_context carry through (whichever
        # pass was most sure about this entity gets to provide the
        # provenance).
        basis = merged.first_entity
        # Find the entity instance that hit ``best_confidence`` —
        # iterate per-schema again because we did NOT store the
        # actual instance, only the scalar conf and label. Cheap
        # because typical merges are 2–3 passes.
        for _schema, result in per_schema_results:
            for ent in result.entities:
                if (
                    normalize_entity_name(ent.text) == key
                    and float(ent.confidence) == merged.best_confidence
                ):
                    basis = ent
                    break
            else:
                continue
            break

        final_entities.append(
            ExtractedEntity(
                text=merged.best_text,
                label=merged.best_label,
                properties=dict(basis.properties),
                confidence=merged.best_confidence,
                source_chunk_id=basis.source_chunk_id,
                source_grounding=basis.source_grounding,
                extraction_context=basis.extraction_context,
                type_tags=list(merged.type_tags),
                primary_type=merged.best_label,
            )
        )

    # ------------------------------------------------------------------
    # Relation merge: dedup on (norm(src), norm(tgt), type).
    # ------------------------------------------------------------------
    rel_key_t = Tuple[str, str, str]
    relation_best: Dict[rel_key_t, ExtractedRelation] = {}
    relation_insertion_order: List[rel_key_t] = []

    for _schema_name, result in per_schema_results:
        for relation in result.relations:
            key = (
                normalize_entity_name(relation.source_entity),
                normalize_entity_name(relation.target_entity),
                relation.relation_type,
            )
            if not key[0] or not key[1] or not key[2]:
                continue
            existing = relation_best.get(key)
            if existing is None:
                relation_best[key] = relation
                relation_insertion_order.append(key)
            elif float(relation.confidence) > float(existing.confidence):
                relation_best[key] = relation

    final_relations: List[ExtractedRelation] = [
        relation_best[k] for k in relation_insertion_order
    ]

    # ------------------------------------------------------------------
    # B.4 fix (B.1f scope): rewrite each surviving relation's endpoint
    # text to the canonical merged-entity surface form. Without this
    # step the relation merger and entity merger can pick different
    # pass-winners (max-confidence picks the surface form from
    # different passes), leaving relations whose ``source_entity`` /
    # ``target_entity`` no longer matches any entity's ``text`` field.
    # Downstream KG persistence (``entity_persistence_service``) keys
    # entities by text and would drop these relations.
    #
    # Rule: for each relation endpoint, look up the merged entity by
    # the same ``normalize_entity_name`` key the entity merge used; if
    # found, copy the merged entity's ``text`` over the relation's
    # endpoint. If not found, leave it as-is (it'll be filtered out
    # downstream — better than synthesising an entity).
    # ------------------------------------------------------------------
    canonical_text_by_key: Dict[str, str] = {
        key: merged_entities[key].best_text for key in entity_insertion_order
    }
    relinked_relations: List[ExtractedRelation] = []
    for relation in final_relations:
        src_key = normalize_entity_name(relation.source_entity)
        tgt_key = normalize_entity_name(relation.target_entity)
        canon_src = canonical_text_by_key.get(src_key)
        canon_tgt = canonical_text_by_key.get(tgt_key)
        # Re-emit through model_copy so the original input dicts in
        # ``per_schema_results`` stay untouched (pure function
        # contract). Skip the copy if neither endpoint actually
        # changed — saves an allocation in the common case where the
        # relation's pass also won the entity tie.
        if (
            (canon_src is not None and canon_src != relation.source_entity)
            or (canon_tgt is not None and canon_tgt != relation.target_entity)
        ):
            relinked_relations.append(
                relation.model_copy(
                    update={
                        "source_entity": canon_src
                        if canon_src is not None
                        else relation.source_entity,
                        "target_entity": canon_tgt
                        if canon_tgt is not None
                        else relation.target_entity,
                    }
                )
            )
        else:
            relinked_relations.append(relation)

    schema_names = [name for name, _result in per_schema_results]
    return ExtractionResult(
        entities=final_entities,
        relations=relinked_relations,
        metadata={
            "merged_from_schemas": schema_names,
            "schema_count": len(per_schema_results),
            "total_entities": len(final_entities),
            "total_relations": len(relinked_relations),
        },
    )
