"""The one declarative source-ingestion pipeline (Track PL.4).

Before PL.4 the source auto-chain was scattered across the job handlers: each
handler, after doing its own work, both wrote ``source.processing_stage`` AND
fired the ad-hoc enqueue of the next stage inline (``handle_process_source``
enqueued ``embed_source``; ``_handle_embed_source`` enqueued ``run_entities``
plus the toggle-gated ``run_summaries``; ``handle_entity_extract`` ran the
inline mentions refresh). PL.4 consolidates that wiring into ONE place — a
single ordered ``SOURCE_PIPELINE`` definition + the ``advance_source`` driver —
so the chain is declarative, testable, and resumable from a source's current
stage. The handlers become THIN: after a stage's work they call
``advance_source`` and the pipeline dispatches the next action.

This is a PURE CONSOLIDATION of PL.1–PL.3: the stages, the guards, the
best-effort posture, the schema-review gate, and the INSIGHTS toggle behave
EXACTLY as before. In particular:

* the spine is INGEST(parse) -> EMBED -> EXTRACT -> GRAPH(mentions); each step's
  produced ``processing_stage`` (``ingested`` / ``embedded`` /
  ``extracted`` / ``graphed`` then ``complete``) is unchanged;
* the EXTRACT stage is gated: an unreviewed schema parks it at
  ``awaiting_schema_review`` and reraises (the worker -> ``PAUSED_FOR_REVIEW``).
  The gate handling stays IN the extract handler because it must reraise the
  ``SchemaReviewPendingError`` for the worker — ``advance_source`` is only
  reached on the success path;
* INSIGHTS is a PARALLEL best-effort branch off EMBED (``depends_on=EMBED``,
  ``auto=True``, ``gate="auto_insights"``), behind the per-notebook
  ``auto_insights`` toggle (default ON). It does NOT set ``processing_stage`` —
  it is enrichment, not part of the KG spine — so it never gates ``complete``;
* the chunk-count guard is preserved: a zero-chunk EMBED chains nothing (no
  no-op ``run_entities`` / ``run_summaries`` jobs), mirroring the
  ``chunk_count > 0`` guard on the parse -> embed link.

``complete`` with zero edges is a legitimate terminal state: a source whose
entities share no df>1 concept with the rest of the corpus produces 0
``mentions`` edges (R.6 noise normalization drops df==1 singletons), yet its
extraction succeeded and its graph "refreshed to empty" — so it correctly
reaches ``graphed`` -> ``complete``. ``complete`` means "the KG chain ran", not
"the KG is non-empty".

The driver is best-effort + idempotent: dispatching the same stage twice does
not double-run, because each downstream action is itself idempotent (embed
deletes-then-writes; extract dedups per ``(in,out,type)``; the mentions refresh
clears-then-relates) and a re-enqueue of an already-done stage just re-runs a
safe idempotent job.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Awaitable, Callable, Dict, List, Optional

from loguru import logger


class StageName(str, Enum):
    """The named stages of the source pipeline (distinct from the persisted
    ``processing_stage`` VALUES — a stage can produce a value, and the gate/
    parallel stages map onto the same value space)."""

    INGEST = "ingest"
    EMBED = "embed"
    EXTRACT = "extract"
    GRAPH = "graph"
    INSIGHTS = "insights"
    REFERENCES = "references"


@dataclass(frozen=True)
class PipelineStage:
    """One declarative stage of the source pipeline.

    Attributes:
        name: the stage identifier.
        produces: the ``source.processing_stage`` value this stage's COMPLETION
            records (``None`` for INSIGHTS — a parallel enrichment that does not
            advance the spine).
        auto: whether this stage is auto-chained by ``advance_source`` (all the
            current stages are auto; the field documents the contract and lets a
            future stage be on-demand without code changes).
        gate: a named gate that can pause/skip the stage. ``"schema_review"``
            for EXTRACT (parks at ``awaiting_schema_review``); ``"auto_insights"``
            for INSIGHTS (per-notebook toggle). ``None`` = no gate.
        depends_on: the stage whose COMPLETION triggers this one. The spine is a
            chain (each depends on the prior); INSIGHTS depends on EMBED in
            PARALLEL with EXTRACT.
        enqueue_command: the job command to enqueue (``None`` for an inline
            stage). GRAPH runs inline (a cheap pure projection, not worth a job).
        parallel: ``True`` for a side branch that does not advance the spine
            stage (INSIGHTS). ``advance_source`` dispatches it ALONGSIDE the
            spine successor, never AS the spine successor.
    """

    name: StageName
    produces: Optional[str]
    auto: bool
    depends_on: Optional[StageName]
    enqueue_command: Optional[str] = None
    gate: Optional[str] = None
    parallel: bool = False


# ---------------------------------------------------------------------------
# THE declarative definition. Ordered; each stage names its dependency, the
# stage value it produces, its gate, and how it runs (enqueue a job vs inline).
# ---------------------------------------------------------------------------

SOURCE_PIPELINE: List[PipelineStage] = [
    # parse -> chunks persisted. The DOCUMENT_PARSE handler is the entry point;
    # it produces ``ingested`` and (when chunks exist) enqueues EMBED.
    PipelineStage(
        name=StageName.INGEST,
        produces="ingested",
        auto=True,
        depends_on=None,
        enqueue_command=None,  # the parse work itself is the DOCUMENT_PARSE job
    ),
    # per-chunk + aggregate vectors. Enqueued off a successful INGEST with chunks.
    PipelineStage(
        name=StageName.EMBED,
        produces="embedded",
        auto=True,
        depends_on=StageName.INGEST,
        enqueue_command="embed_source",
    ),
    # ontology-guided entity/relation extraction. Enqueued off a successful EMBED
    # with chunks; gated by schema review (parks at awaiting_schema_review).
    PipelineStage(
        name=StageName.EXTRACT,
        produces="extracted",
        auto=True,
        depends_on=StageName.EMBED,
        enqueue_command="run_entities",
        gate="schema_review",
    ),
    # document<->entity mentions projection. Runs INLINE after EXTRACT (cheap, no
    # LLM); produces ``graphed`` then the terminal ``complete``.
    PipelineStage(
        name=StageName.GRAPH,
        produces="graphed",
        auto=True,
        depends_on=StageName.EXTRACT,
        enqueue_command=None,  # inline refresh, not a job
    ),
    # PARALLEL enrichment off EMBED, behind the per-notebook auto_insights toggle.
    # Does NOT set processing_stage; never gates ``complete``.
    PipelineStage(
        name=StageName.INSIGHTS,
        produces=None,
        auto=True,
        depends_on=StageName.EMBED,
        enqueue_command="run_summaries",
        gate="auto_insights",
        parallel=True,
    ),
    # Track V.5: PARALLEL reference-extraction pass off INGEST, behind the global
    # ``references`` gate (``ENABLE_REFERENCE_EXTRACTION`` env, DEFAULT OFF). It
    # depends on INGEST (it needs the persisted chunks, not embeddings) and does
    # NOT set processing_stage — it is enrichment, so it never gates ``complete``.
    # With the gate OFF (the default) ``advance_source`` off ``ingested`` enqueues
    # NOTHING new — the un-flagged ingest path is byte-for-byte the pre-V.5 path.
    PipelineStage(
        name=StageName.REFERENCES,
        produces=None,
        auto=True,
        depends_on=StageName.INGEST,
        enqueue_command="extract_references",
        gate="references",
        parallel=True,
    ),
]


# Map a persisted ``processing_stage`` value -> the StageName whose completion it
# records. Lets ``advance_source`` read a source's stage and find "where we are".
_VALUE_TO_STAGE: Dict[str, StageName] = {
    s.produces: s.name for s in SOURCE_PIPELINE if s.produces is not None
}

# The terminal stage value (set right after ``graphed``; the KG spine is done).
COMPLETE = "complete"

# Gate / failure stage values that are NOT a spine stage and from which
# ``advance_source`` does not auto-progress (they need an explicit resume — a
# schema review, or a re-run after a failure).
_TERMINAL_OR_PARKED = {"awaiting_schema_review", "failed", COMPLETE}


def _spine_successor(stage: StageName) -> Optional[PipelineStage]:
    """The single spine stage that ``depends_on`` ``stage`` and is NOT parallel."""
    for s in SOURCE_PIPELINE:
        if s.depends_on == stage and not s.parallel:
            return s
    return None


def _parallel_branches(stage: StageName) -> List[PipelineStage]:
    """The parallel (non-spine) stages that ``depends_on`` ``stage`` (INSIGHTS)."""
    return [s for s in SOURCE_PIPELINE if s.depends_on == stage and s.parallel]


@dataclass
class AdvanceResult:
    """What ``advance_source`` dispatched, for assertion/observability.

    ``enqueued`` maps a stage name -> the enqueued job id (or ``None`` if the
    best-effort enqueue failed / was skipped). ``inline_ran`` lists the inline
    stages that ran (GRAPH). ``skipped`` records gate/guard skips with a reason.
    """

    from_value: str
    enqueued: Dict[str, Optional[str]] = field(default_factory=dict)
    inline_ran: List[str] = field(default_factory=list)
    skipped: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The driver.
# ---------------------------------------------------------------------------


async def advance_source(
    source_id: str,
    *,
    embedded_chunks: Optional[int] = None,
) -> AdvanceResult:
    """Read ``source.processing_stage`` and dispatch the next allowed stage(s).

    This is the ONE place the auto-chain wiring lives (PL.4). The handlers call
    it after a stage's work completes; it reads where the source is and fires the
    successor action(s) per ``SOURCE_PIPELINE`` — respecting ``auto``,
    ``depends_on``, the gates, and the chunk-count guard. Best-effort: every
    enqueue / inline run is guarded so a queue hiccup or a projection failure
    never propagates (the producing work is already persisted).

    Args:
        source_id: the source to advance.
        embedded_chunks: the EMBED stage's chunk count, passed through from the
            embed handler so the EMBED -> {EXTRACT, INSIGHTS} fan-out keeps the
            PL.3 ``embedded_chunks > 0`` guard. ``None`` (the caller did not
            supply it, e.g. advancing off INGEST/EXTRACT) means "no guard here".

    Returns:
        An :class:`AdvanceResult` describing what was dispatched.
    """
    current = await _read_stage(source_id)
    result = AdvanceResult(from_value=current)

    if current in _TERMINAL_OR_PARKED:
        # Nothing to auto-advance: complete (spine done), awaiting_schema_review
        # (needs a review), or failed (needs an explicit re-run).
        result.skipped["*"] = f"stage '{current}' is terminal/parked"
        return result

    stage_name = _VALUE_TO_STAGE.get(current)
    if stage_name is None:
        logger.warning(
            f"advance_source: source {source_id} has unknown "
            f"processing_stage '{current}'; nothing to advance"
        )
        result.skipped["*"] = f"unknown stage '{current}'"
        return result

    # The chunk-count guard applies to the EMBED fan-out: a zero-chunk embed has
    # no graph to build and nothing to summarize (PL.3 folded PL.2 minor). It
    # applies ONLY when advancing off EMBED with a supplied count.
    chunk_guard_blocks = (
        stage_name is StageName.EMBED
        and embedded_chunks is not None
        and embedded_chunks <= 0
    )

    # --- the spine successor (enqueue a job, or run inline) ---
    successor = _spine_successor(stage_name)
    if successor is not None and successor.auto:
        if chunk_guard_blocks:
            logger.info(
                f"advance_source: skipping {successor.name.value} chain for "
                f"source {source_id}: 0 embedded chunks (nothing to graph)"
            )
            result.skipped[successor.name.value] = "0 embedded chunks"
        elif successor.enqueue_command is not None:
            job_id = await _best_effort_enqueue(
                source_id, successor.enqueue_command, successor.name.value
            )
            result.enqueued[successor.name.value] = job_id
        else:
            # Inline stage (GRAPH): run the projection, then advance the stage.
            ran = await _run_graph_inline(source_id)
            if ran:
                result.inline_ran.append(successor.name.value)

    # --- parallel branches (INSIGHTS off EMBED) ---
    for branch in _parallel_branches(stage_name):
        if not branch.auto:
            continue
        if chunk_guard_blocks:
            result.skipped[branch.name.value] = "0 embedded chunks"
            continue
        if branch.gate == "auto_insights":
            job_id = await _maybe_chain_insights(source_id)
            result.enqueued[branch.name.value] = job_id
        elif branch.gate == "references":
            job_id = await _maybe_chain_references(source_id)
            result.enqueued[branch.name.value] = job_id

    return result


# ---------------------------------------------------------------------------
# Stage-action implementations (the bodies the scattered handlers used to hold).
# All best-effort: the producing work is already persisted.
# ---------------------------------------------------------------------------


async def _read_stage(source_id: str) -> str:
    """Read the source's current ``processing_stage`` (default ``ingested``)."""
    try:
        from app_main.dependencies import get_source_repo

        stage = await get_source_repo().get_processing_stage(source_id)
        return stage or "ingested"
    except Exception as e:  # noqa: BLE001 — read is best-effort
        logger.warning(
            f"advance_source: could not read processing_stage for source "
            f"{source_id} (assuming 'ingested'): {e}"
        )
        return "ingested"


async def set_processing_stage(source_id: str, stage: str) -> None:
    """Best-effort write of ``source.processing_stage``.

    A stage is a STATUS reflection of an already-completed pipeline step, so a
    failure to persist it must NEVER fail that step — log and continue. Lives in
    the pipeline module (PL.4) so both the thin handlers and the driver write
    through one funnel.
    """
    try:
        from app_main.dependencies import get_source_repo

        await get_source_repo().set_processing_stage(source_id, stage)
    except Exception as stage_err:  # noqa: BLE001 — status write is best-effort
        logger.error(
            f"Failed to set processing_stage={stage} for source "
            f"{source_id} (the pipeline step itself succeeded): {stage_err}"
        )


async def _best_effort_enqueue(
    source_id: str, command_name: str, stage_label: str
) -> Optional[str]:
    """Enqueue ``command_name`` for ``source_id``; never raise (best-effort).

    Mirrors the exact posture of the scattered PL.1–PL.3 enqueues: a queue hiccup
    must not fail the (already-persisted) producing stage; the operator / a
    re-run can recover. Returns the job id or ``None`` on failure.
    """
    try:
        from app_main.services.command_service import CommandService

        job_id = await CommandService.submit_command_job(
            "open_notebook",
            command_name,
            {"source_id": source_id},
        )
        logger.info(
            f"Auto-enqueued {command_name} job {job_id} for source {source_id} "
            f"({stage_label} stage)"
        )
        return job_id
    except Exception as enqueue_err:  # noqa: BLE001 — never fail the producer
        logger.error(
            f"Failed to auto-enqueue {command_name} for source {source_id} "
            f"({stage_label} stage; the producing step succeeded, a re-run "
            f"recovers): {enqueue_err}"
        )
        return None


async def _run_graph_inline(source_id: str) -> bool:
    """Run the source-scoped ``mentions`` refresh inline, then advance the stage.

    GRAPH is the KG chain's final link: refresh the document<->entity graph for
    THIS source (the cheap, no-LLM projection — see
    ``MentionsProjectionService.refresh_source``, whose full-corpus-projection /
    scoped-write invariant is preserved unchanged), then advance ``graphed`` ->
    ``complete``. Best-effort: the entities are already persisted, so a refresh
    hiccup must NOT fail extraction — it logs and the stage stays ``extracted``
    (resumable: a re-extract or a global regenerate recovers the graph).

    Returns ``True`` if the refresh ran and the stage advanced to complete.
    """
    try:
        from app_main.dependencies import get_mentions_projection_service

        service = get_mentions_projection_service()
        refresh = await service.refresh_source(source_id)
        logger.info(
            f"PL.4 mentions refreshed for source {source_id}: "
            f"cleared={refresh.cleared} created={refresh.created} "
            f"failed={refresh.failed}"
        )
    except Exception as graph_err:  # noqa: BLE001 — graph refresh is best-effort
        logger.error(
            f"Failed to refresh mentions for source {source_id} (extraction "
            f"succeeded; stage stays 'extracted', a re-run / regenerate "
            f"recovers): {graph_err}"
        )
        return False

    # The graph materialized (possibly to zero edges — a legitimate ``complete``
    # with no shared-concept neighbours) -> graphed, then the terminal complete.
    await set_processing_stage(source_id, "graphed")
    await set_processing_stage(source_id, COMPLETE)
    return True


async def _maybe_chain_insights(source_id: str) -> Optional[str]:
    """Best-effort, toggle-gated INSIGHTS enqueue (PL.3 logic, relocated).

    Resolves the source's owning notebook and reads its ``auto_insights`` toggle
    (default ON). When ON, enqueues ``run_summaries`` (``INSIGHT_EXTRACT``,
    producing ``source_insight`` rows); when OFF, skips. Best-effort: a
    toggle-read error defaults to ON, and a queue hiccup logs and returns
    ``None`` — neither fails the already-persisted embed. INSIGHTS is a parallel
    enrichment: it does NOT touch ``processing_stage``.

    Returns the enqueued command id, or ``None`` when skipped/failed.
    """
    try:
        from app_main.dependencies import get_notebook_repo, get_source_repo

        notebook_id = await get_source_repo().get_notebook_id(source_id)
        if notebook_id is not None:
            enabled = await get_notebook_repo().get_auto_insights(notebook_id)
            if not enabled:
                logger.info(
                    f"Skipping auto-insights for source {source_id}: notebook "
                    f"{notebook_id} has auto_insights=off"
                )
                return None
        # notebook_id None (unlinked source) -> default ON, run insights.
    except Exception as toggle_err:  # noqa: BLE001 — toggle read defaults ON
        logger.warning(
            f"Could not resolve auto_insights toggle for source {source_id} "
            f"(defaulting ON): {toggle_err}"
        )

    return await _best_effort_enqueue(source_id, "run_summaries", "insights")


async def _maybe_chain_references(source_id: str) -> Optional[str]:
    """Best-effort, config-gated V.5 reference-extraction enqueue.

    Consults the global ``ENABLE_REFERENCE_EXTRACTION`` flag (default OFF — the
    safe value, keeping the un-flagged ingest path byte-for-byte identical to
    pre-V.5). When ON, enqueues ``extract_references`` (``REFERENCE_EXTRACT``,
    the whole-corpus reference pass that feeds U.3's ``cites`` materialization);
    when OFF, skips with no enqueue. Best-effort: a queue hiccup logs and returns
    ``None`` — never fails the already-persisted ingest. REFERENCES is a parallel
    enrichment: it does NOT touch ``processing_stage``.

    Returns the enqueued command id, or ``None`` when gated off / failed.
    """
    from app_main.config import get_reference_extraction_enabled

    if not get_reference_extraction_enabled():
        logger.debug(
            f"Skipping reference extraction for source {source_id}: "
            f"ENABLE_REFERENCE_EXTRACTION is off"
        )
        return None

    return await _best_effort_enqueue(
        source_id, "extract_references", "references"
    )
