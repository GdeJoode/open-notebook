"""Inbox file-watcher — auto-ingest dropped files (Track G.5).

An opt-in, always-on watcher on conventional inbox folders. It:

* debounces a burst of writes to one file (a slow copy) into a single ingest;
* scans a pre-existing backlog once at startup;
* routes by extension (document / audio / url; unknown → ignored + logged);
* ingests through the SAME ``process_source`` chain the API uses — the dropped
  file is copied into the canonical uploads folder (reusing the hardened
  ``generate_unique_filename`` sanitizer) and a ``process_source`` job is
  enqueued at the ``submit_command_job`` seam;
* moves the inbox original to ``_processed/`` on terminal success or
  ``_errors/`` otherwise, so a file is ingested exactly once (the moved-out
  original is never re-seen by a later scan).

OFF by default (``FILE_WATCHER_ENABLED``): the lifespan never starts it, so an
existing deployment sees zero behaviour change.

The watchdog Observer runs in its own thread; its synchronous callbacks are
bridged onto the app event loop via ``call_soon_threadsafe`` so all ingest work
runs as normal asyncio tasks. The core methods (``_route``, ``_resolve_notebook``,
``_ingest``, ``_ingest_and_move``, ``scan_on_startup``) are dependency-injected
and unit-testable without a running Observer.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Awaitable, Callable, Dict, Optional, Set

from loguru import logger

# Terminal-state subdirectories created inside each inbox; also the two folders a
# backlog scan must skip so an already-ingested file is never re-processed.
_PROCESSED_DIR = "_processed"
_ERRORS_DIR = "_errors"
_SKIP_DIRS = {_PROCESSED_DIR, _ERRORS_DIR}

_DOC_EXTS = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".markdown", ".html", ".htm",
    ".rtf", ".odt", ".pptx", ".ppt", ".epub", ".csv", ".json", ".xml",
}
_AUDIO_EXTS = {".mp3", ".m4a", ".wav", ".ogg", ".flac", ".aac", ".opus"}
_URL_EXTS = {".url", ".webloc"}


class InboxWatcher:
    """Watch inbox folders and auto-ingest dropped files (G.5)."""

    def __init__(
        self,
        *,
        inbox_paths,
        default_notebook_id: Optional[str] = None,
        debounce_seconds: int = 3,
        uploads_folder: Optional[str] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        source_service_factory: Optional[Callable] = None,
        notebook_service_factory: Optional[Callable] = None,
        submit_job: Optional[Callable[..., Awaitable[str]]] = None,
        get_status: Optional[Callable[..., Awaitable[dict]]] = None,
        poll_interval: float = 2.0,
        terminal_timeout: float = 1800.0,
        max_concurrent_ingest: int = 4,
    ) -> None:
        self._roots = [Path(p) for p in inbox_paths]
        self._default_notebook_id = default_notebook_id
        self._debounce = max(0, int(debounce_seconds))
        self._loop = loop
        self._poll_interval = poll_interval
        self._terminal_timeout = terminal_timeout
        # Bound the backlog-scan fan-out so a large inbox doesn't stampede the
        # job queue/DB with create()+enqueue+pollers all at once (MINOR-1).
        self._max_concurrent = max(1, int(max_concurrent_ingest))
        self._scan_sem: Optional[asyncio.Semaphore] = None

        # Injected seams (default to the real services/commands).
        self._source_service_factory = source_service_factory
        self._notebook_service_factory = notebook_service_factory
        self._submit_job = submit_job
        self._get_status = get_status
        self._uploads_folder = uploads_folder

        self._pending: Dict[str, asyncio.TimerHandle] = {}
        self._inflight: Set[str] = set()
        self._observer = None

    # -- lazy real-dependency resolution (kept out of __init__ so importing the
    #    module never pulls the API/service graph until the watcher runs) ------

    def _source_service(self):
        if self._source_service_factory:
            return self._source_service_factory()
        from app_main.dependencies import get_source_service

        return get_source_service()

    async def _submit(self, *args, **kwargs) -> str:
        if self._submit_job:
            return await self._submit_job(*args, **kwargs)
        from app_main.services.command_service import CommandService

        return await CommandService.submit_command_job(*args, **kwargs)

    async def _status(self, command_id: str) -> dict:
        if self._get_status:
            return await self._get_status(command_id)
        from app_main.services.command_service import CommandService

        return await CommandService.get_command_status(command_id)

    def _uploads(self) -> str:
        if self._uploads_folder:
            return self._uploads_folder
        from app_main.config import UPLOADS_FOLDER

        return UPLOADS_FOLDER

    # -- routing ------------------------------------------------------------

    @staticmethod
    def _route(path) -> Optional[str]:
        """Map a file to an ingest kind by extension, or None if unsupported."""
        ext = Path(path).suffix.lower()
        if ext in _DOC_EXTS:
            return "document"
        if ext in _AUDIO_EXTS:
            return "audio"
        if ext in _URL_EXTS:
            return "url"
        return None

    @staticmethod
    def _is_in_skip_dir(path) -> bool:
        return any(part in _SKIP_DIRS for part in Path(path).parts)

    def _resolve_notebook(self, path) -> Optional[str]:
        """Route a file to its notebook.

        A per-notebook inbox has the shape ``.../notebook:<id>/inbox/<file>`` —
        the directory that CONTAINS the ``inbox`` folder, when its name is a
        ``notebook:`` record id, owns the file. Anything else (a global inbox)
        falls back to the configured default notebook.
        """
        for parent in Path(path).resolve().parents:
            if parent.name == "inbox":
                owner = parent.parent.name
                if owner.startswith("notebook:"):
                    return owner
                break
        return self._default_notebook_id

    # -- ingest -------------------------------------------------------------

    @staticmethod
    def _read_url_target(path) -> Optional[str]:
        """Extract the target URL from a ``.url`` (INI) or ``.webloc`` file."""
        try:
            text = Path(path).read_text(encoding="utf-8", errors="ignore")
        except OSError:
            return None
        for line in text.splitlines():
            s = line.strip()
            if s.lower().startswith("url="):  # .url INI: URL=https://...
                return s[4:].strip()
            if "<string>" in s and "://" in s:  # .webloc plist: <string>...</string>
                inner = s.split("<string>", 1)[1].split("</string>", 1)[0]
                if "://" in inner:
                    return inner.strip()
        return None

    def _stage_upload(self, path) -> str:
        """Copy the dropped file into the canonical uploads folder.

        Reuses the hardened ``generate_unique_filename`` (basename sanitize +
        containment) so the source's stored ``file_path`` is stable in uploads,
        independent of the later inbox move to ``_processed``/``_errors``.

        Config constraint (MINOR-2): an inbox root must NOT contain the uploads
        folder. The default layout (``<DATA>/inbox`` + ``<DATA>/uploads``) is
        safe (siblings); but setting ``INBOX_PATHS=<DATA>`` would make each staged
        copy land inside the watched tree and be re-ingested. Keep inboxes and
        uploads disjoint.
        """
        from app_main.api.routers.sources_upload import generate_unique_filename

        dest = generate_unique_filename(Path(path).name, self._uploads())
        shutil.copy2(str(path), dest)
        return dest

    async def _ingest(self, path) -> Optional[str]:
        """Create the source + enqueue process_source. Returns the job id.

        Returns None for an unsupported extension (caller leaves it in place).
        Raises on any ingest error so the caller routes the file to ``_errors``.
        """
        p = Path(path)
        route = self._route(p)
        if route is None:
            logger.info("inbox: ignoring unsupported file {p}", p=str(p))
            return None

        notebook_id = self._resolve_notebook(p)
        svc = self._source_service()
        source = await svc.create(
            {"title": p.name, "topics": [], "private": False}
        )
        if notebook_id:
            await svc.add_to_notebook(source.id, notebook_id)

        content_state: Dict[str, object] = {"private": False}
        if route == "url":
            url = self._read_url_target(p)
            if not url:
                raise ValueError(f"no URL target in {p}")
            content_state["url"] = url
        else:
            content_state["file_path"] = self._stage_upload(p)
            content_state["delete_source"] = False

        command_id = await self._submit(
            "open_notebook",
            "process_source",
            {
                "source_id": str(source.id),
                "content_state": content_state,
                "notebook_ids": [notebook_id] if notebook_id else [],
                "processing_overrides": None,
                "private": False,
            },
        )
        logger.info(
            "inbox: enqueued {kind} ingest for {p} -> job {j} (notebook={nb})",
            kind=route,
            p=str(p),
            j=command_id,
            nb=notebook_id,
        )
        return command_id

    async def _await_terminal(self, command_id: str) -> bool:
        """Poll until the job reaches a terminal state. True only on success."""
        waited = 0.0
        while waited < self._terminal_timeout:
            status = await self._status(command_id)
            state = str(status.get("status", "")).lower()
            if state == "completed":
                return True
            if state in ("failed", "cancelled"):
                return False
            if state == "paused_for_review":
                logger.warning(
                    "inbox: job {j} paused for schema review; routing file to "
                    "_errors (source + job persist independently)",
                    j=command_id,
                )
                return False
            await asyncio.sleep(self._poll_interval)
            waited += self._poll_interval
        logger.warning("inbox: job {j} did not finish within timeout", j=command_id)
        return False

    def _move(self, path, subdir: str) -> None:
        """Move the inbox original into ``<inbox-dir>/<subdir>/`` (unique name).

        Best-effort: a move failure (dest on another filesystem, permissions) is
        logged but never raised — a raise here would escape ``_ingest_and_move``
        as an unretrieved task exception AND strand the file re-ingestable.
        """
        p = Path(path)
        if not p.exists():
            return
        try:
            dest_dir = p.parent / subdir
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / p.name
            i = 1
            while dest.exists():
                dest = dest_dir / f"{p.stem} ({i}){p.suffix}"
                i += 1
            shutil.move(str(p), str(dest))
            logger.info("inbox: moved {p} -> {d}", p=str(p), d=str(dest))
        except Exception as e:  # noqa: BLE001 — never strand via a raising move
            logger.error("inbox: could not move {p} to {s}: {e}", p=str(p), s=subdir, e=e)

    async def _ingest_and_move(self, path) -> None:
        """Full lifecycle for one file: ingest → await terminal → move.

        The ingest AND the terminal-wait are both inside the guard: a transient
        error in ``_await_terminal`` (e.g. get_command_status raises) must still
        route the file to ``_errors`` — otherwise it is left re-ingestable and a
        later scan creates a DUPLICATE source (the enqueue already happened).
        """
        p = Path(path)
        key = str(p.resolve())
        if key in self._inflight:
            return  # already being processed — debounce/scan overlap guard
        if not p.exists() or self._is_in_skip_dir(p):
            return
        self._inflight.add(key)
        try:
            try:
                command_id = await self._ingest(p)
                if command_id is None:
                    return  # unsupported extension — leave in place (AC2)
                ok = await self._await_terminal(command_id)
            except Exception as e:  # noqa: BLE001 — a bad file must not kill the watcher
                logger.error("inbox: ingest failed for {p}: {e}", p=str(p), e=e)
                self._move(p, _ERRORS_DIR)
                return
            self._move(p, _PROCESSED_DIR if ok else _ERRORS_DIR)
        finally:
            self._inflight.discard(key)

    # -- backlog + debounce -------------------------------------------------

    def scan_on_startup(self) -> None:
        """Enqueue every supported backlog file present before the watcher ran.

        Skips ``_processed``/``_errors`` so an already-ingested file is never
        re-processed (idempotent, AC4). Must be called on the event loop.
        """
        if self._scan_sem is None:
            self._scan_sem = asyncio.Semaphore(self._max_concurrent)
        for root in self._roots:
            if not root.exists():
                continue
            for f in root.rglob("*"):
                if f.is_file() and not self._is_in_skip_dir(f):
                    asyncio.ensure_future(self._bounded_ingest(f))

    async def _bounded_ingest(self, path) -> None:
        """Backlog ingest bounded by the scan semaphore (MINOR-1)."""
        assert self._scan_sem is not None
        async with self._scan_sem:
            await self._ingest_and_move(path)

    def _schedule(self, path) -> None:
        """(loop thread) Debounce: (re)arm a one-shot timer for this path."""
        key = str(path)
        existing = self._pending.pop(key, None)
        if existing is not None:
            existing.cancel()
        self._pending[key] = self._loop.call_later(
            self._debounce, self._fire, path
        )

    def _fire(self, path) -> None:
        """(loop thread) Debounce elapsed with no newer event → ingest."""
        self._pending.pop(str(path), None)
        asyncio.ensure_future(self._ingest_and_move(path))

    def _on_event(self, path) -> None:
        """(watchdog thread) Bridge a filesystem event onto the loop."""
        if self._is_in_skip_dir(path) or self._loop is None:
            return
        self._loop.call_soon_threadsafe(self._schedule, path)

    # -- watchdog wiring ----------------------------------------------------

    def start(self) -> None:
        """Start the Observer + scan the backlog. Idempotent."""
        if self._observer is not None:
            return
        if self._loop is None:
            self._loop = asyncio.get_event_loop()

        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        watcher = self

        class _Handler(FileSystemEventHandler):
            def on_created(self, event):
                if not event.is_directory:
                    watcher._on_event(event.src_path)

            def on_modified(self, event):
                if not event.is_directory:
                    watcher._on_event(event.src_path)

            def on_moved(self, event):
                dest = getattr(event, "dest_path", None)
                if not event.is_directory and dest:
                    watcher._on_event(dest)

        self._observer = Observer()
        for root in self._roots:
            root.mkdir(parents=True, exist_ok=True)
            self._observer.schedule(_Handler(), str(root), recursive=True)
        self._observer.start()
        logger.info(
            "inbox watcher started on {roots} (debounce={d}s, default_notebook={nb})",
            roots=[str(r) for r in self._roots],
            d=self._debounce,
            nb=self._default_notebook_id,
        )
        self.scan_on_startup()

    def stop(self) -> None:
        """Stop the Observer and cancel pending debounce timers."""
        for timer in self._pending.values():
            timer.cancel()
        self._pending.clear()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None
            logger.info("inbox watcher stopped")


def build_watcher_from_config() -> Optional[InboxWatcher]:
    """Construct the watcher from env config, or None when disabled (G-D7)."""
    from app_main.config import (
        get_file_watcher_enabled,
        get_inbox_debounce_seconds,
        get_inbox_default_notebook_id,
        get_inbox_paths,
    )

    if not get_file_watcher_enabled():
        return None
    return InboxWatcher(
        inbox_paths=get_inbox_paths(),
        default_notebook_id=get_inbox_default_notebook_id(),
        debounce_seconds=get_inbox_debounce_seconds(),
    )


__all__ = ["InboxWatcher", "build_watcher_from_config"]
