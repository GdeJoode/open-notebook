"""The Werkbak (WB.1): what needs the owner, across all notebooks, in one read.

``GET /api/attention`` answers "where do I need to be" when the app is opened
again, whatever arrived meanwhile through the UI, the API, MCP or the watcher.
Each group reads through the service or repository its own page reads, so its
count is that page's count; nothing here keeps state of its own. Every group
and item carries the frontend route where it is resolved.

Groups run concurrently, each under its own deadline: a slow or failing group
comes back with ``error`` set and the others still answer.

Consumers: the Werkbak page (``frontend/src/app/(dashboard)/page.tsx``) and
the sidebar badges (``frontend/src/components/layout/AppSidebar.tsx``).
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import quote

from job_queue.repository import JobRepository
from loguru import logger
from pydantic import BaseModel, Field
from shared.models.jobs import Job
from shared.types.enums import JobStatus, JobType, ProcessingStage
from surrealdb_service.repositories.notebook_schema import NotebookSchemaRepository
from surrealdb_service.repositories.source import SourceRepository

from app_main.services.notebook_service import NotebookService
from app_main.services.paused_extraction import list_paused_jobs_for_notebook

# A source short of ``complete`` with no job queued or running for it counts as
# stopped once its last write is this old. The pipeline enqueues the next stage
# in the same step that writes the stage, and a source whose job is still
# queued or running is excluded outright, so the only legitimate quiet window
# is the gap between those two writes (seconds). 15 minutes leaves a wide
# margin for a slow worker restart, while a source that stopped (live: the
# regio-deal convenants parked at ``embedded``) shows the same session.
STALLED_AFTER_MINUTES = 15

# Items listed per group; the count covers them all.
ITEM_LIMIT = 5

# Per-group deadline. The slowest group, merge candidates, takes ~0.7 s on the
# live graph (WB.0); a group past this is reported as timed out.
GROUP_TIMEOUT_S = 5.0

SHORT_OF_COMPLETE = [
    ProcessingStage.INGESTED.value,
    ProcessingStage.EMBEDDED.value,
    ProcessingStage.EXTRACTED.value,
    ProcessingStage.GRAPHED.value,
]
UNFINISHED = [JobStatus.QUEUED, JobStatus.PROCESSING, JobStatus.RETRYING]
FINISHED = [JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED]

# The retype panel lives in the ontology editor; it retypes under every active
# ontology, whichever one it is opened on.
RETYPE_LINK = "/ontologies?name=schema_core&tab=edit&retype=1"


class AttentionItem(BaseModel):
    id: str
    label: str
    link: str
    detail: Optional[str] = None
    at: Optional[str] = None


class AttentionGroup(BaseModel):
    """One Werkbak card: a count, the top items, and where to resolve them.

    ``nav`` is the sidebar entry the group belongs to (its badge sums the
    groups with that ``nav``). ``needs_action`` is false for information that
    asks nothing of the owner (jobs in progress, arrivals); those groups are
    left out of ``total`` and the badges.
    """

    key: str
    title: str
    count: int = 0
    action: str = ""
    link: str = ""
    nav: Optional[str] = None
    needs_action: bool = True
    detail: Optional[str] = None
    items: List[AttentionItem] = Field(default_factory=list)
    error: Optional[str] = None


class AttentionReport(BaseModel):
    """``generated_at`` is the server's time of the read: a client that marks
    the report as seen stores it and sends it back as ``since``."""

    generated_at: str
    since: Optional[str] = None
    total: int = 0
    groups: List[AttentionGroup] = Field(default_factory=list)
    arrived: List[AttentionGroup] = Field(default_factory=list)


def _q(record_id: str) -> str:
    return quote(str(record_id), safe="")


def source_link(source_id: str) -> str:
    return f"/sources/{_q(source_id)}"


def job_link(job: Job) -> str:
    """Where a job is followed up: its source, its notebook, else its page."""
    if job.source_id:
        return source_link(job.source_id)
    notebook_id = (job.payload or {}).get("notebook_id")
    if notebook_id:
        return f"/notebooks/{_q(notebook_id)}"
    if job.job_type is JobType.BATCH_PROCESS:
        return "/podcasts"
    if job.job_type is JobType.EMBEDDING_GENERATE:
        return "/settings"
    return "/notebooks"


def _plural(n: int, one: str, many: Optional[str] = None) -> str:
    return one if n == 1 else (many or f"{one}s")


def _job_label(job: Job) -> str:
    return str(job.job_type.value).replace("_", " ").capitalize()


def _job_item(job: Job, detail: Optional[str], at: Any) -> AttentionItem:
    return AttentionItem(
        id=str(job.id),
        label=_job_label(job),
        link=job_link(job),
        detail=detail,
        at=_iso(at),
    )


def _source_item(row: Dict[str, Any], detail: str, link: str, at: Any) -> AttentionItem:
    return AttentionItem(
        id=str(row["id"]),
        label=row.get("title") or str(row["id"]),
        link=link,
        detail=detail,
        at=_iso(at),
    )


def _iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value) if value else None


async def _shared(read: "asyncio.Future[Any]") -> Any:
    """Await a read several groups share. A group's deadline cancels that
    group, not the read, so the other groups still get it."""
    return await asyncio.shield(read)


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


class AttentionService:
    """Builds the Werkbak report from the services behind each review page."""

    def __init__(
        self,
        source_repo: Optional[SourceRepository] = None,
        job_repo: Optional[JobRepository] = None,
        notebook_service: Optional[NotebookService] = None,
        schema_repo: Optional[NotebookSchemaRepository] = None,
        stalled_after_minutes: int = STALLED_AFTER_MINUTES,
        group_timeout_s: float = GROUP_TIMEOUT_S,
    ):
        self.source_repo = source_repo or SourceRepository()
        self.job_repo = job_repo or JobRepository()
        if notebook_service is None:
            from app_main.dependencies import get_notebook_service

            notebook_service = get_notebook_service()
        self.notebook_service = notebook_service
        self.schema_repo = schema_repo or NotebookSchemaRepository()
        self.stalled_after_minutes = stalled_after_minutes
        self.group_timeout_s = group_timeout_s

    async def report(self, since: Optional[datetime] = None) -> AttentionReport:
        now = datetime.now(timezone.utc)
        # Read once, shared by the groups that need them (through ``_shared``).
        unfinished = asyncio.ensure_future(self._unfinished_jobs())
        notebooks = asyncio.ensure_future(self._notebooks())

        specs: List[tuple] = [
            (AttentionGroup(key="sources_failed", title="Failed sources", nav="/sources"),
             self._sources_failed),
            (AttentionGroup(key="sources_awaiting_review",
                            title="Sources waiting for a schema review", nav="/sources"),
             self._sources_awaiting_review),
            (AttentionGroup(key="sources_stalled", title="Sources that stopped",
                            nav="/sources"),
             lambda g: self._sources_stalled(g, now, unfinished)),
            (AttentionGroup(key="extraction_paused", title="Paused extraction",
                            nav="/notebooks"),
             lambda g: self._extraction_paused(g, notebooks)),
            (AttentionGroup(key="schema_proposals", title="Schema proposals",
                            nav="/notebooks"),
             lambda g: self._schema_proposals(g, notebooks)),
            (AttentionGroup(key="triage", title="Triage queue", nav="/knowledge-graph",
                            link="/knowledge-graph/triage"),
             self._triage),
            (AttentionGroup(key="merge_candidates", title="Merge candidates",
                            nav="/knowledge-graph", link="/knowledge-graph/resolution"),
             self._merge_candidates),
            (AttentionGroup(key="retype", title="Retype pending", nav="/ontologies",
                            link=RETYPE_LINK),
             self._retype),
            (AttentionGroup(key="jobs_failed", title="Failed jobs"), self._jobs_failed),
            (AttentionGroup(key="jobs_running", title="Jobs in progress",
                            needs_action=False),
             lambda g: self._jobs_running(g, unfinished)),
        ]
        arrived_specs: List[tuple] = []
        if since is not None:
            since = _utc(since)
            arrived_specs = [
                (AttentionGroup(key="new_sources", title="New sources", nav="/sources",
                                needs_action=False, link="/sources"),
                 lambda g: self._new_sources(g, since)),
                (AttentionGroup(key="finished_jobs", title="Finished jobs",
                                needs_action=False),
                 lambda g: self._finished_jobs(g, since)),
            ]

        try:
            await asyncio.gather(
                *(self._run(g, fn) for g, fn in specs + arrived_specs)
            )
        finally:
            for task in (unfinished, notebooks):
                if not task.done():
                    task.cancel()
                elif not task.cancelled():
                    task.exception()  # retrieved: a failure is on its groups

        groups = [g for g, _ in specs]
        return AttentionReport(
            generated_at=now.isoformat(),
            since=since.isoformat() if since else None,
            total=sum(g.count for g in groups if g.needs_action and not g.error),
            groups=groups,
            arrived=[g for g, _ in arrived_specs],
        )

    async def _run(
        self, group: AttentionGroup, fill: Callable[[AttentionGroup], Awaitable[None]]
    ) -> None:
        try:
            await asyncio.wait_for(fill(group), timeout=self.group_timeout_s)
        except asyncio.TimeoutError:
            self._fail(group, f"timed out after {self.group_timeout_s:g} s")
        except Exception as exc:
            logger.warning("attention group {k} failed: {e}", k=group.key, e=exc)
            self._fail(group, str(exc) or type(exc).__name__)

    @staticmethod
    def _fail(group: AttentionGroup, error: str) -> None:
        group.error = error
        group.count = 0
        group.items = []
        group.detail = None

    @staticmethod
    def _finish(group: AttentionGroup, action: str) -> None:
        """Set the action; a group without a page of its own links to its top item."""
        group.action = action if group.count else ""
        if not group.link and group.items:
            group.link = group.items[0].link

    # --- shared reads -------------------------------------------------------

    async def _unfinished_jobs(self) -> List[Job]:
        jobs: List[Job] = []
        for status in UNFINISHED:
            jobs += await self.job_repo.list_jobs(status=status, limit=10000)
        return sorted(jobs, key=lambda j: str(j.created or ""), reverse=True)

    async def _notebooks(self) -> List[Dict[str, Any]]:
        return await self.notebook_service.get_all_with_counts(
            order_by="name asc", limit=1000
        )

    # --- sources ------------------------------------------------------------

    async def _sources_failed(self, group: AttentionGroup) -> None:
        stage = ProcessingStage.FAILED.value
        rows = await self.source_repo.list_by_stage([stage])
        group.count = len(rows)
        group.link = f"/sources?stage={stage}"
        group.items = [
            _source_item(r, "Failed", source_link(r["id"]), r.get("updated"))
            for r in rows[:ITEM_LIMIT]
        ]
        self._finish(group, f"Review {group.count} failed {_plural(group.count, 'source')}")

    async def _sources_awaiting_review(self, group: AttentionGroup) -> None:
        stage = ProcessingStage.AWAITING_SCHEMA_REVIEW.value
        rows = await self.source_repo.list_by_stage([stage])
        group.count = len(rows)
        group.link = f"/sources?stage={stage}"
        top = rows[:ITEM_LIMIT]
        # The proposal that gates a source is accepted on the Schema tab of the
        # notebook its extraction anchored on (``get_notebook_id``).
        notebook_ids = await asyncio.gather(
            *(self.source_repo.get_notebook_id(str(r["id"])) for r in top)
        )
        group.items = [
            _source_item(
                r,
                "Waiting for a schema review",
                f"/notebooks/{_q(nb)}/schema" if nb else source_link(r["id"]),
                r.get("updated"),
            )
            for r, nb in zip(top, notebook_ids)
        ]
        self._finish(
            group, f"Review {group.count} gated {_plural(group.count, 'source')}"
        )

    async def _sources_stalled(
        self, group: AttentionGroup, now: datetime, unfinished: "asyncio.Future[List[Job]]"
    ) -> None:
        cutoff = now - timedelta(minutes=self.stalled_after_minutes)
        rows = await self.source_repo.list_by_stage(SHORT_OF_COMPLETE, updated_before=cutoff)
        busy = {j.source_id for j in await _shared(unfinished) if j.source_id}
        rows = [r for r in rows if str(r["id"]) not in busy]
        group.count = len(rows)
        group.link = "/sources?" + "&".join(f"stage={s}" for s in SHORT_OF_COMPLETE)
        group.detail = (
            f"Short of complete, with no job running, for over "
            f"{self.stalled_after_minutes} minutes"
        )
        group.items = [
            _source_item(
                r,
                f"Stopped at {r.get('processing_stage') or ProcessingStage.INGESTED.value}",
                source_link(r["id"]),
                r.get("updated"),
            )
            for r in rows[:ITEM_LIMIT]
        ]
        self._finish(group, f"Check {group.count} stopped {_plural(group.count, 'source')}")

    # --- notebooks: the Pass-1 gate ------------------------------------------

    async def _extraction_paused(
        self, group: AttentionGroup, notebooks: "asyncio.Future[List[Dict[str, Any]]]"
    ) -> None:
        # The per-notebook read filters this global list; when it is empty,
        # every notebook's is too.
        paused = await self.job_repo.list_jobs(
            status=JobStatus.PAUSED_FOR_REVIEW, job_type=JobType.ENTITY_EXTRACT, limit=1
        )
        items: List[tuple] = []
        if paused:
            nbs = await _shared(notebooks)
            per_notebook = await asyncio.gather(
                *(list_paused_jobs_for_notebook(str(nb["id"]), self.source_repo)
                  for nb in nbs)
            )
            items = [(nb, len(jobs)) for nb, jobs in zip(nbs, per_notebook) if jobs]
        items.sort(key=lambda x: -x[1])
        group.count = sum(n for _, n in items)
        group.items = [
            AttentionItem(
                id=str(nb["id"]),
                label=nb.get("name") or str(nb["id"]),
                link=f"/notebooks/{_q(nb['id'])}",
                detail=f"{n} paused {_plural(n, 'extraction')}",
            )
            for nb, n in items[:ITEM_LIMIT]
        ]
        self._finish(
            group, f"Resume {group.count} paused {_plural(group.count, 'extraction')}"
        )

    async def _schema_proposals(
        self, group: AttentionGroup, notebooks: "asyncio.Future[List[Dict[str, Any]]]"
    ) -> None:
        nbs = await _shared(notebooks)
        schemas = await asyncio.gather(
            *(self.schema_repo.get_by_notebook(str(nb["id"])) for nb in nbs)
        )
        items = [
            (nb, s.pending_extensions)
            for nb, s in zip(nbs, schemas)
            if s is not None and s.pending_extensions
        ]
        items.sort(key=lambda x: -len(x[1]))
        group.count = sum(len(p) for _, p in items)
        group.items = []
        for nb, pending in items[:ITEM_LIMIT]:
            names = [str(p.get("type_name") or "(unknown)") for p in pending]
            shown = ", ".join(names[:3]) + (", ..." if len(names) > 3 else "")
            group.items.append(
                AttentionItem(
                    id=str(nb["id"]),
                    label=nb.get("name") or str(nb["id"]),
                    link=f"/notebooks/{_q(nb['id'])}/schema",
                    detail=f"{len(pending)} {_plural(len(pending), 'proposal')}: {shown}",
                )
            )
        self._finish(group, f"Review {group.count} {_plural(group.count, 'proposal')}")

    # --- the knowledge graph ------------------------------------------------

    async def _triage(self, group: AttentionGroup) -> None:
        from app_main.dependencies import get_triage_queue_repo

        rows = await get_triage_queue_repo().list_open()
        group.count = len(rows)
        group.items = [
            AttentionItem(
                id=r.id,
                label=r.name,
                link=group.link,
                detail=f"{r.type} · {r.reason}",
            )
            for r in rows[:ITEM_LIMIT]
        ]
        self._finish(group, f"Triage {group.count} {_plural(group.count, 'entity', 'entities')}")

    async def _merge_candidates(self, group: AttentionGroup) -> None:
        from app_main.services.entity_resolution import CandidateDedupService

        report = await CandidateDedupService().propose_candidates()
        group.count = report.auto_merge_count + report.review_count
        group.detail = f"{report.auto_merge_count} auto, {report.review_count} to review"
        group.items = [
            AttentionItem(
                id=f"{c.id_a}|{c.id_b}",
                label=f"{c.name_a} / {c.name_b}",
                link=group.link,
                detail=f"{c.band} · {c.score:.2f}",
            )
            for c in (list(report.auto_merge) + list(report.review))[:ITEM_LIMIT]
        ]
        self._finish(
            group, f"Review {group.count} merge {_plural(group.count, 'candidate')}"
        )

    async def _retype(self, group: AttentionGroup) -> None:
        from app_main.dependencies import get_entity_retype_service

        plan = await get_entity_retype_service().dry_run()
        counts = plan["counts"]
        # What an apply writes; the retype panel counts the same.
        group.count = counts["changes"] + counts["flag_only"]
        if counts["collisions"]:
            group.detail = (
                f"{counts['collisions']} {_plural(counts['collisions'], 'collision')} "
                "left for a merge"
            )
        rows = plan["changes"] + plan["flag_only"]
        group.items = [
            AttentionItem(
                id=str(r["id"]),
                label=str(r.get("name") or r["id"]),
                link=group.link,
                detail=f"{r.get('old')} → {r.get('new')}",
            )
            for r in rows[:ITEM_LIMIT]
        ]
        self._finish(group, f"Retype {group.count} {_plural(group.count, 'entity', 'entities')}")

    # --- jobs ---------------------------------------------------------------

    async def _jobs_failed(self, group: AttentionGroup) -> None:
        jobs = await self.job_repo.list_unresolved_failures()
        group.count = len(jobs)
        group.detail = "Not re-run since"
        group.items = [
            _job_item(j, (j.error_message or "Failed")[:160], j.completed_at or j.created)
            for j in jobs[:ITEM_LIMIT]
        ]
        self._finish(group, f"Check {group.count} failed {_plural(group.count, 'job')}")

    async def _jobs_running(
        self, group: AttentionGroup, unfinished: "asyncio.Future[List[Job]]"
    ) -> None:
        jobs = await _shared(unfinished)
        group.count = len(jobs)
        group.items = [_job_item(j, j.status.value, j.created) for j in jobs[:ITEM_LIMIT]]
        self._finish(group, f"{group.count} {_plural(group.count, 'job')} in progress")

    # --- arrived since the last visit -----------------------------------------

    async def _new_sources(self, group: AttentionGroup, since: datetime) -> None:
        rows = await self.source_repo.list_created_since(since)
        group.count = len(rows)
        group.items = [
            _source_item(
                r,
                f"At {r.get('processing_stage') or ProcessingStage.INGESTED.value}",
                source_link(r["id"]),
                r.get("created"),
            )
            for r in rows[:ITEM_LIMIT]
        ]
        self._finish(group, f"{group.count} new {_plural(group.count, 'source')}")

    async def _finished_jobs(self, group: AttentionGroup, since: datetime) -> None:
        jobs = await self.job_repo.list_finished_since(since, FINISHED)
        group.count = len(jobs)
        failed = sum(1 for j in jobs if j.status is JobStatus.FAILED)
        if failed:
            group.detail = f"{failed} failed"
        # Failures first: they are what the owner acts on.
        jobs = sorted(jobs, key=lambda j: j.status is not JobStatus.FAILED)
        group.items = [
            _job_item(
                j,
                j.status.value
                + (f": {j.error_message[:140]}" if j.error_message else ""),
                j.completed_at,
            )
            for j in jobs[:ITEM_LIMIT]
        ]
        self._finish(group, f"{group.count} finished {_plural(group.count, 'job')}")
