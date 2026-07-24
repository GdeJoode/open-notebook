"""Opt-in periodic librarian audit (Track F.5).

A thin scheduler entry point over the F.1 audit: enumerate the notebooks that
opted in (``notebook.librarian_enabled``) and enqueue one ``run_librarian_audit``
job per notebook on the existing command/job seam. The consumer
(:func:`app_main.handlers.handle_librarian_audit`) re-runs
:meth:`AuditService.run_all` and writes a fresh ``audit_findings`` snapshot.

No new daemon: :meth:`enqueue_due` is a plain function an external periodic
trigger (cron / the worker's scheduling surface) calls on its interval. Opt-in is
default-OFF, so this is a no-op for every existing notebook until an operator
turns it on. Enqueue is best-effort per notebook — a submit failure for one
notebook is logged and does not stop the others.
"""

from __future__ import annotations

from typing import List, Optional

from loguru import logger
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.repositories import NotebookRepository

from app_main.services.command_service import CommandService


class LibrarianService:
    """Enqueue the periodic audit for every opt-in notebook (F.5)."""

    def __init__(
        self,
        notebook_repo: Optional[NotebookRepository] = None,
        config: Optional[SurrealDBConfig] = None,
    ) -> None:
        self.notebook_repo = notebook_repo or NotebookRepository(config=config)
        self.config = config

    async def enqueue_due(self) -> List[str]:
        """Enqueue one ``run_librarian_audit`` job per opt-in notebook.

        Returns the notebook ids that were successfully enqueued (may be shorter
        than the opt-in set if a submit failed). A disabled/empty set enqueues
        nothing and returns ``[]``.
        """
        notebook_ids = await self.notebook_repo.list_librarian_notebook_ids()
        enqueued: List[str] = []
        for notebook_id in notebook_ids:
            try:
                await CommandService.submit_command_job(
                    module_name="open_notebook.commands",
                    command_name="run_librarian_audit",
                    command_args={"notebook_id": notebook_id},
                )
                enqueued.append(notebook_id)
            except Exception as exc:  # noqa: BLE001 — one failure ≠ abort the rest
                logger.error(
                    "librarian: failed to enqueue audit for {nb}: {e}",
                    nb=notebook_id,
                    e=exc,
                )
        logger.info(
            "librarian: enqueued {n}/{total} notebook audits",
            n=len(enqueued),
            total=len(notebook_ids),
        )
        return enqueued


__all__ = ["LibrarianService"]
