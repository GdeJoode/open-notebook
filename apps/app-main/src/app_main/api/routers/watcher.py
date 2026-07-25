"""Inbox file-watcher status endpoint (Track G.6).

A session-authed (under ``/api``, shared-password gated) read-only view of the
watcher for the settings UI: whether it is enabled, which paths it watches, the
default notebook, and a recent-activity readout derived directly from the
``_processed``/``_errors`` folders the watcher moves files into (no new table —
the filesystem IS the record).

The watcher's configuration is environment-managed (``FILE_WATCHER_ENABLED`` etc.,
read at startup), so this endpoint is read-only: it reflects the running config,
it does not mutate it.
"""

from pathlib import Path

from fastapi import APIRouter

router = APIRouter(prefix="/watcher", tags=["watcher"])

_PROCESSED = "_processed"
_ERRORS = "_errors"
_RECENT_LIMIT = 25


def _scan_recent(paths, limit: int = _RECENT_LIMIT) -> list[dict]:
    """Most-recent processed/errored files across the watched inboxes.

    Reads the mtime of each file in every inbox's ``_processed``/``_errors``
    folder, newest first. Best-effort: an unreadable folder is skipped, never
    raised — a status read must never 500.
    """
    entries: list[dict] = []
    for root in paths:
        for sub, status in ((_PROCESSED, "processed"), (_ERRORS, "errored")):
            d = Path(root) / sub
            if not d.is_dir():
                continue
            try:
                for f in d.iterdir():
                    if not f.is_file():
                        continue
                    try:
                        mtime = f.stat().st_mtime
                    except OSError:
                        continue
                    entries.append(
                        {"name": f.name, "status": status, "mtime": mtime}
                    )
            except OSError:
                continue
    entries.sort(key=lambda e: e["mtime"], reverse=True)
    from datetime import datetime, timezone

    return [
        {
            "name": e["name"],
            "status": e["status"],
            "when": datetime.fromtimestamp(e["mtime"], tz=timezone.utc).isoformat(),
        }
        for e in entries[:limit]
    ]


@router.get("/status")
async def watcher_status() -> dict:
    """The watcher's current (env-managed) config + recent activity."""
    from app_main.config import (
        get_file_watcher_enabled,
        get_inbox_debounce_seconds,
        get_inbox_default_notebook_id,
        get_inbox_paths,
    )

    paths = get_inbox_paths()
    return {
        "enabled": get_file_watcher_enabled(),
        "paths": paths,
        "default_notebook_id": get_inbox_default_notebook_id(),
        "debounce_seconds": get_inbox_debounce_seconds(),
        "recent": _scan_recent(paths),
    }
