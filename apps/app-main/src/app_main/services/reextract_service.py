"""Re-extraction orchestration service (Phase B.3d).

When a user accepts a schema edit (rename / merge / split / accept / etc.)
the existing entity-extraction results for sources in the notebook are
now stale — they reference the OLD type names / structure. The B.3d
flow surfaces a banner (``ReextractPromptBanner``) that lets the user
re-run extraction for some or all affected sources.

This service is the server-side counterpart of the banner. It owns:

* **Job enqueueing** — translating a list of source ids into one
  ``ENTITY_EXTRACT`` job per source via the existing ``job-queue``
  package (`CommandService.submit_command_job` is the canonical
  dispatch path so the worker sees the same payload shape as a
  user-triggered "Run entities" click).

* **Idempotency / dedup** — re-clicking ``[Re-extract all]`` with the
  same source ids must not stack a second job per source while the
  first is still queued/running. Dedup happens by checking the most
  recent job per ``source_id`` + ``job_type=entity_extract`` and
  skipping when status is ``QUEUED``, ``PROCESSING``, or ``RETRYING``.

  Completed / failed / cancelled jobs are NOT considered duplicates —
  re-extracting after a previous run finished is the normal user flow.

Design notes
============

* **The service does not own its own ``JobService`` instance.** Job
  dispatch uses the existing ``CommandService`` module-level singleton
  so the worker, queue, and registry are shared with the rest of the
  app. Tests inject a fake ``CommandService`` (any object exposing
  ``submit_command_job`` + ``_repository.get_by_source``) to keep unit
  coverage hermetic.

* **The service does NOT touch the schema row.** The "schema_changed"
  event is emitted by ``SchemaEditService``; the affected-source list
  is computed by ``reextract_candidates`` (currently: all sources in
  the notebook — see B.3d plan).

* **Payload shape mirrors ``CommandService.submit_command_job(...,
  "run_entities", ...)``** so the worker handler in ``handlers.py``
  consumes the same payload regardless of whether the job came from
  the UI "Run entities" button or this re-extract flow. The handler
  re-resolves the owning notebook from ``source_id``, so we do not
  need to pass ``notebook_id`` through the payload.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Protocol

from loguru import logger


# ---------------------------------------------------------------------------
# Job statuses considered "already in-flight" for dedup purposes.
# ---------------------------------------------------------------------------
#
# We deliberately use the string values (not the enum) so the protocol below
# remains importable without the heavy ``shared.types.enums`` chain in tests
# that fake the dependencies. The values mirror :class:`shared.types.enums.JobStatus`.

_INFLIGHT_STATUSES = frozenset(
    {"queued", "processing", "retrying", "paused_for_review"}
)
# M4 (attempt 2) — ``paused_for_review`` is treated as in-flight on
# purpose. Stacking a fresh re-extract on top of a paused-for-review
# job would orphan the paused entry: the worker picks up the new
# QUEUED job, the paused entry can't be resumed via the normal
# ``/extraction/resume`` path once a parallel job runs against the
# same source, and the user ends up with two competing entries. The
# caller (banner / API) must resume the paused review via the
# existing ``POST /extraction/resume`` endpoint first, then re-extract.


@dataclass(frozen=True)
class ReextractResult:
    """Outcome of an :meth:`ReextractService.enqueue_reextract_jobs` call.

    * ``jobs_enqueued`` is the number of NEW jobs submitted on this
      call. Dedup-skipped sources do not bump the count.
    * ``source_ids`` echoes the input list verbatim so the caller can
      correlate which sources the user asked for (useful for the
      banner's per-source processing UI). Skipped vs enqueued split
      lives in ``skipped_source_ids`` and ``enqueued_source_ids``.
    """

    jobs_enqueued: int
    source_ids: List[str]
    enqueued_source_ids: List[str]
    skipped_source_ids: List[str]


class _CommandServiceProtocol(Protocol):
    """Structural protocol matching the slice of ``CommandService`` we use.

    Tests pass an object exposing the same shape. Keeping the protocol
    explicit avoids importing the real ``CommandService`` at unit-test
    time (which would pull in the SurrealDB-backed JobRepository
    singleton).
    """

    async def submit_command_job(  # noqa: D401 — mirrors the staticmethod
        self,
        module_name: str,
        command_name: str,
        command_args: dict,
        context: Optional[dict] = None,
    ) -> str: ...


class _JobRepositoryProtocol(Protocol):
    """Structural protocol for the ``get_by_source`` slice of JobRepository."""

    async def get_by_source(self, source_id: str): ...


class ReextractService:
    """Service orchestrating bulk re-extraction job enqueueing.

    Stateless — collaborators are injected. The single public method,
    :meth:`enqueue_reextract_jobs`, is the only contract callers depend
    on.
    """

    def __init__(
        self,
        command_service: _CommandServiceProtocol,
        job_repo: _JobRepositoryProtocol,
    ):
        self._command_service = command_service
        self._job_repo = job_repo

    async def enqueue_reextract_jobs(
        self,
        notebook_id: str,
        source_ids: Iterable[str],
    ) -> ReextractResult:
        """Enqueue one ENTITY_EXTRACT job per source, with dedup.

        Args:
            notebook_id: Owning notebook record id. Carried for logging
                + future propagation (currently the worker re-resolves
                the notebook from the source — see ``handlers.py``).
            source_ids: Iterable of source record ids the user asked
                to re-extract. Order is preserved in the result lists
                for stable UI rendering.

        Returns:
            :class:`ReextractResult` with the enqueued / skipped split.
            ``jobs_enqueued`` equals ``len(enqueued_source_ids)``.

        Notes:
            * Empty ``source_ids`` → zero jobs, zero skips, empty lists.
              No error. The router uses an empty 200 to signal "user
              hit Re-extract selected with nothing selected".
            * A failure to submit ONE source does not abort the batch.
              We log and continue so a transient DB hiccup on source N
              does not stop sources N+1..M from being re-extracted.
            * Dedup is per-source: a source whose latest job is in
              QUEUED/PROCESSING/RETRYING/PAUSED_FOR_REVIEW is skipped
              silently. The caller can compare ``enqueued_source_ids``
              to ``source_ids`` to surface which ones were already
              running. PAUSED_FOR_REVIEW is treated as in-flight
              because resuming the paused review via the existing
              ``POST /extraction/resume`` endpoint is a prerequisite
              for a clean re-extract.
        """
        ordered: List[str] = list(source_ids)
        enqueued: List[str] = []
        skipped: List[str] = []

        for source_id in ordered:
            if await self._has_inflight_extract_job(source_id):
                logger.info(
                    "Skipping re-extract for source {} (notebook {}): "
                    "existing entity_extract job is still in-flight",
                    source_id,
                    notebook_id,
                )
                skipped.append(source_id)
                continue

            try:
                job_id = await self._command_service.submit_command_job(
                    "open_notebook",
                    "run_entities",
                    {
                        "source_id": source_id,
                        # multi_schema_enabled mirrors the default the
                        # UI's "Run entities" button uses — the worker
                        # picks up the notebook schema if there is one.
                        "multi_schema_enabled": True,
                    },
                )
                logger.info(
                    "Enqueued re-extract job {} for source {} (notebook {})",
                    job_id,
                    source_id,
                    notebook_id,
                )
                enqueued.append(source_id)
            except Exception as e:
                logger.error(
                    "Failed to enqueue re-extract job for source {} "
                    "(notebook {}): {}",
                    source_id,
                    notebook_id,
                    e,
                )
                # Treat as skipped for the caller's purpose — the source
                # was not enqueued, so the UI should not show "processing"
                # for it.
                skipped.append(source_id)

        return ReextractResult(
            jobs_enqueued=len(enqueued),
            source_ids=ordered,
            enqueued_source_ids=enqueued,
            skipped_source_ids=skipped,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _has_inflight_extract_job(self, source_id: str) -> bool:
        """Return True iff the source has an in-flight ENTITY_EXTRACT job.

        "In-flight" means status ∈ {QUEUED, PROCESSING, RETRYING,
        PAUSED_FOR_REVIEW}. We treat completed / failed / cancelled
        jobs as eligible for re-extraction — the user explicitly asked
        for it and the new schema means even a "successful" past
        extraction is stale. PAUSED_FOR_REVIEW is in the in-flight set
        because the user must resume the paused review (see M4 fix
        comment above) before a fresh re-extract can stack safely.

        ``JobRepository.get_by_source`` returns ALL jobs for the source
        in created-DESC order; we filter to entity_extract on the way
        through to avoid leaking other job types into the decision.
        """
        try:
            jobs = await self._job_repo.get_by_source(source_id)
        except Exception as e:
            # If we can't query the queue, fall through to "no dedup" —
            # we'd rather risk a duplicate job than refuse to re-extract
            # at all on transient DB errors.
            logger.warning(
                "Could not query existing jobs for source {} during "
                "re-extract dedup: {}; proceeding without dedup",
                source_id,
                e,
            )
            return False

        for job in jobs:
            job_type = _job_attr(job, "job_type")
            status = _job_attr(job, "status")
            # job_type / status may be plain strings or enum-valued —
            # normalise to the underlying value.
            if hasattr(job_type, "value"):
                job_type = job_type.value
            if hasattr(status, "value"):
                status = status.value
            if job_type != "entity_extract":
                continue
            if status in _INFLIGHT_STATUSES:
                return True
        return False


def _job_attr(job, name: str):
    """Read an attribute that may live on a Pydantic model or a dict.

    Tests sometimes pass dicts; production code always passes ``Job``
    Pydantic instances. Supporting both keeps the unit tests free of
    boilerplate.
    """
    if isinstance(job, dict):
        return job.get(name)
    return getattr(job, name, None)
