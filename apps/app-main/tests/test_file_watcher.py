"""Unit tests for the G.5 inbox file-watcher.

Exercise the core logic with injected seams (no watchdog Observer, no DB, no real
ingest): extension routing, per-notebook vs default resolution, the enqueue at
the submit_command_job seam, move-to-_processed/_errors on mocked terminal
states, backlog scan (skipping _processed/_errors), debounce single-enqueue, and
the disabled (default) no-op.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app_main.services.agents.file_watcher import (
    InboxWatcher,
    build_watcher_from_config,
)


def _svc():
    svc = AsyncMock()
    svc.create = AsyncMock(return_value=SimpleNamespace(id="source:x"))
    svc.add_to_notebook = AsyncMock(return_value=True)
    return svc


def _watcher(tmp_path, *, submit=None, status=None, svc=None, default_nb=None):
    return InboxWatcher(
        inbox_paths=[str(tmp_path / "inbox")],
        default_notebook_id=default_nb,
        debounce_seconds=1,
        uploads_folder=str(tmp_path / "uploads"),
        source_service_factory=lambda: svc or _svc(),
        submit_job=submit or AsyncMock(return_value="job:1"),
        get_status=status or AsyncMock(return_value={"status": "completed"}),
        poll_interval=0.001,
        terminal_timeout=1.0,
    )


# -- routing ----------------------------------------------------------------


def test_route_maps_extensions():
    assert InboxWatcher._route("a.pdf") == "document"
    assert InboxWatcher._route("a.DOCX") == "document"
    assert InboxWatcher._route("talk.mp3") == "audio"
    assert InboxWatcher._route("x.m4a") == "audio"
    assert InboxWatcher._route("bookmark.url") == "url"
    assert InboxWatcher._route("s.webloc") == "url"
    assert InboxWatcher._route("archive.zip") is None
    assert InboxWatcher._route("noext") is None


def test_resolve_notebook_per_notebook_inbox(tmp_path):
    w = _watcher(tmp_path, default_nb="notebook:default")
    per = tmp_path / "base" / "notebook:abc123" / "inbox" / "f.pdf"
    assert w._resolve_notebook(per) == "notebook:abc123"


def test_resolve_notebook_global_inbox_falls_back(tmp_path):
    w = _watcher(tmp_path, default_nb="notebook:default")
    glob = tmp_path / "inbox" / "f.pdf"
    assert w._resolve_notebook(glob) == "notebook:default"


# -- ingest (enqueue seam) --------------------------------------------------


@pytest.mark.asyncio
async def test_ingest_stages_file_and_enqueues_process_source(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)
    f = inbox / "paper.pdf"
    f.write_bytes(b"%PDF-1.4 x")
    submit = AsyncMock(return_value="job:42")
    svc = _svc()
    w = _watcher(tmp_path, submit=submit, svc=svc, default_nb="notebook:d")

    job_id = await w._ingest(f)

    assert job_id == "job:42"
    # exactly one process_source enqueue at the seam
    submit.assert_awaited_once()
    module, command, args = submit.await_args.args
    assert module == "open_notebook" and command == "process_source"
    assert args["source_id"] == "source:x"
    assert args["notebook_ids"] == ["notebook:d"]
    # the file was STAGED into uploads (stable path), not the inbox original
    staged = args["content_state"]["file_path"]
    assert staged.endswith("paper.pdf")
    assert (tmp_path / "uploads") == __import__("pathlib").Path(staged).parent
    svc.add_to_notebook.assert_awaited_once()


@pytest.mark.asyncio
async def test_ingest_unsupported_returns_none_no_enqueue(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)
    f = inbox / "archive.zip"
    f.write_bytes(b"PK")
    submit = AsyncMock(return_value="job:1")
    w = _watcher(tmp_path, submit=submit)

    assert await w._ingest(f) is None
    submit.assert_not_awaited()


@pytest.mark.asyncio
async def test_ingest_url_file_extracts_target(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)
    f = inbox / "link.url"
    f.write_text("[InternetShortcut]\nURL=https://example.com/doc\n")
    submit = AsyncMock(return_value="job:u")
    w = _watcher(tmp_path, submit=submit)

    await w._ingest(f)
    args = submit.await_args.args[2]
    assert args["content_state"]["url"] == "https://example.com/doc"
    assert "file_path" not in args["content_state"]


# -- ingest_and_move (terminal → move) --------------------------------------


@pytest.mark.asyncio
async def test_move_to_processed_on_success(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)
    f = inbox / "doc.pdf"
    f.write_bytes(b"%PDF x")
    w = _watcher(tmp_path, status=AsyncMock(return_value={"status": "completed"}))

    await w._ingest_and_move(f)

    assert not f.exists()
    assert (inbox / "_processed" / "doc.pdf").exists()


@pytest.mark.asyncio
async def test_move_to_errors_on_failure(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)
    f = inbox / "doc.pdf"
    f.write_bytes(b"%PDF x")
    w = _watcher(tmp_path, status=AsyncMock(return_value={"status": "failed"}))

    await w._ingest_and_move(f)

    assert not f.exists()
    assert (inbox / "_errors" / "doc.pdf").exists()


@pytest.mark.asyncio
async def test_unsupported_file_left_in_place(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)
    f = inbox / "archive.zip"
    f.write_bytes(b"PK")
    w = _watcher(tmp_path)

    await w._ingest_and_move(f)

    assert f.exists()  # not moved, not enqueued (AC2)
    assert not (inbox / "_processed").exists()


@pytest.mark.asyncio
async def test_inflight_guard_prevents_double_ingest(tmp_path):
    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)
    f = inbox / "doc.pdf"
    f.write_bytes(b"%PDF x")
    submit = AsyncMock(return_value="job:1")
    w = _watcher(tmp_path, submit=submit)
    w._inflight.add(str(f.resolve()))  # simulate an in-flight ingest

    await w._ingest_and_move(f)

    submit.assert_not_awaited()
    assert f.exists()


# -- backlog scan -----------------------------------------------------------


@pytest.mark.asyncio
async def test_scan_on_startup_enqueues_backlog_skips_processed(tmp_path):
    inbox = tmp_path / "inbox"
    (inbox / "_processed").mkdir(parents=True)
    (inbox / "_errors").mkdir(parents=True)
    (inbox / "new.pdf").write_bytes(b"%PDF x")
    (inbox / "_processed" / "old.pdf").write_bytes(b"%PDF x")
    (inbox / "_errors" / "bad.pdf").write_bytes(b"%PDF x")

    seen = []

    async def fake_move(path):
        seen.append(str(path))

    w = _watcher(tmp_path)
    w._ingest_and_move = fake_move  # capture what the scan schedules

    w.scan_on_startup()
    # let the scheduled tasks run
    import asyncio

    await asyncio.sleep(0.05)

    assert any(s.endswith("new.pdf") for s in seen)
    assert not any("_processed" in s or "_errors" in s for s in seen)


# -- debounce ---------------------------------------------------------------


@pytest.mark.asyncio
async def test_debounce_coalesces_burst_to_single_ingest(tmp_path):
    import asyncio

    inbox = tmp_path / "inbox"
    inbox.mkdir(parents=True)
    f = inbox / "doc.pdf"
    f.write_bytes(b"%PDF x")

    calls = []

    async def fake_move(path):
        calls.append(str(path))

    w = InboxWatcher(
        inbox_paths=[str(inbox)],
        debounce_seconds=0,  # fire on next loop tick
        uploads_folder=str(tmp_path / "uploads"),
        loop=asyncio.get_event_loop(),
    )
    w._ingest_and_move = fake_move

    # A burst of events for the same path arms/re-arms one timer.
    w._schedule(str(f))
    w._schedule(str(f))
    w._schedule(str(f))
    assert len(w._pending) == 1  # only one pending timer for the path

    await asyncio.sleep(0.05)
    assert calls == [str(f)]  # coalesced to a single ingest


# -- disabled (default) -----------------------------------------------------


def test_build_watcher_disabled_by_default(monkeypatch):
    monkeypatch.delenv("FILE_WATCHER_ENABLED", raising=False)
    assert build_watcher_from_config() is None


def test_build_watcher_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("FILE_WATCHER_ENABLED", "true")
    monkeypatch.setenv("INBOX_PATHS", str(tmp_path / "inbox"))
    w = build_watcher_from_config()
    assert w is not None
    assert w._roots[0].name == "inbox"
