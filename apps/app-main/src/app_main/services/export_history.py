"""Read back the ``export.*`` metrics the exporters write (Track PC.5c).

Every exporter records one ``metrics`` row per run — ``export.jsonl``,
``export.networkx``, ``export.obsidian``, ``export.okf`` — with counts, the
duration and, on failure, an ``error``. Nothing read them: PC.6 measured the
writes and found no consumer. This is the consumer; the export UIs show the
newest row as their "last export" line.

Counts only, as written (Q-D-8): no entity ids travel through here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query

EXPORT_FORMATS = ("jsonl", "networkx", "obsidian", "okf")


def _notebook_keys(notebook_id: str) -> List[str]:
    """Both spellings of the id: the exporters store what their caller passed."""
    bare = notebook_id.split(":", 1)[1] if notebook_id.startswith("notebook:") else notebook_id
    return [f"notebook:{bare}", bare]


def _entry(row: Dict[str, Any]) -> Dict[str, Any]:
    payload = row.get("payload") or {}
    created = row.get("created_at")
    return {
        "format": str(row.get("event_type", "")).removeprefix("export."),
        "created_at": created.isoformat() if hasattr(created, "isoformat") else str(created),
        "error": payload.get("error"),
        "entities_written": int(payload.get("entities_written") or 0),
        "relations_written": int(payload.get("relations_written") or 0),
        "files_written": int(payload.get("files_written") or 0),
        "bytes_written": int(payload.get("bytes_written") or 0),
        "duration_ms": payload.get("duration_ms"),
        "mode": payload.get("mode"),
    }


async def read_export_history(
    notebook_id: str,
    formats: Optional[Sequence[str]] = None,
    limit: int = 10,
    config: Optional[SurrealDBConfig] = None,
) -> List[Dict[str, Any]]:
    """The notebook's export runs, newest first."""
    rows = await execute_query(
        "SELECT event_type, payload, created_at FROM metrics "
        "WHERE event_type IN $types AND notebook IN $nb "
        "ORDER BY created_at DESC LIMIT $limit;",
        {
            "types": [f"export.{f}" for f in (formats or EXPORT_FORMATS)],
            "nb": _notebook_keys(notebook_id),
            "limit": limit,
        },
        config,
    )
    return [_entry(r) for r in rows or []]


__all__ = ["EXPORT_FORMATS", "read_export_history"]
