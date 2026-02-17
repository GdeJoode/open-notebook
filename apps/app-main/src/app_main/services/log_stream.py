"""
In-memory log streaming service for source processing.

Provides per-source log buffers that can be consumed via SSE.
Persists logs to JSONL files for historical retrieval.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Dict, List, Optional

from app_main.config import DATA_FOLDER

# Lazy-initialized directory for persisted processing logs
PROCESSING_LOGS_DIR = os.path.join(DATA_FOLDER, "processing_logs")


def _ensure_logs_dir() -> str:
    os.makedirs(PROCESSING_LOGS_DIR, exist_ok=True)
    return PROCESSING_LOGS_DIR


def _safe_source_id(source_id: str) -> str:
    """Sanitize source_id for use as a filename."""
    return source_id.replace(":", "_").replace("/", "_")


@dataclass
class LogEntry:
    """A single log entry."""

    timestamp: float
    level: str
    message: str

    def to_sse(self) -> str:
        return f"data: {self.level}|{self.message}\n\n"

    def to_dict(self) -> dict:
        return {"timestamp": self.timestamp, "level": self.level, "message": self.message}


@dataclass
class _SourceBuffer:
    """Internal buffer for a single source's logs."""

    entries: List[LogEntry] = field(default_factory=list)
    subscribers: List[asyncio.Queue[Optional[LogEntry]]] = field(
        default_factory=list
    )
    closed: bool = False


class LogStreamService:
    """Manages per-source log buffers with pub/sub."""

    def __init__(self) -> None:
        self._buffers: Dict[str, _SourceBuffer] = {}

    def _get_buffer(self, source_id: str) -> _SourceBuffer:
        if source_id not in self._buffers:
            self._buffers[source_id] = _SourceBuffer()
        return self._buffers[source_id]

    def _persist_entry(self, source_id: str, entry: LogEntry) -> None:
        """Append a log entry as a JSON line to the persisted log file."""
        try:
            logs_dir = _ensure_logs_dir()
            safe_id = _safe_source_id(source_id)
            log_path = Path(logs_dir) / f"{safe_id}.jsonl"
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception:
            # Never break processing on I/O errors
            pass

    def emit(self, source_id: str, message: str, level: str = "INFO") -> None:
        """Emit a log message for a source."""
        buf = self._get_buffer(source_id)
        entry = LogEntry(timestamp=time.time(), level=level, message=message)
        buf.entries.append(entry)
        for q in buf.subscribers:
            q.put_nowait(entry)
        self._persist_entry(source_id, entry)

    async def subscribe(
        self, source_id: str, after: int = 0
    ) -> AsyncIterator[LogEntry]:
        """Yield log entries as they arrive. Ends when close() is called.

        Args:
            after: Skip the first ``after`` buffered entries (used by
                   reconnecting clients that already received them).
        """
        buf = self._get_buffer(source_id)
        q: asyncio.Queue[Optional[LogEntry]] = asyncio.Queue()

        # Snapshot count BEFORE adding subscriber so entries that arrive
        # during replay go only to the queue (no duplicates).
        replay_count = len(buf.entries)
        buf.subscribers.append(q)

        # Replay buffered entries (skip ones the client already has)
        for entry in buf.entries[:replay_count]:
            if after > 0:
                after -= 1
                continue
            yield entry

        # Stream new entries from the queue
        try:
            while True:
                entry = await q.get()
                if entry is None:
                    break
                yield entry
        finally:
            if q in buf.subscribers:
                buf.subscribers.remove(q)

    def close(self, source_id: str) -> None:
        """Signal that processing is done for this source."""
        buf = self._buffers.get(source_id)
        if buf is None:
            return
        buf.closed = True
        for q in buf.subscribers:
            q.put_nowait(None)

    def get_entries(self, source_id: str) -> List[LogEntry]:
        """Get all log entries for a source (for saving to file)."""
        buf = self._buffers.get(source_id)
        return list(buf.entries) if buf else []

    def get_persisted_logs(self, source_id: str) -> List[LogEntry]:
        """Read persisted JSONL log file for a source."""
        safe_id = _safe_source_id(source_id)
        log_path = Path(PROCESSING_LOGS_DIR) / f"{safe_id}.jsonl"
        if not log_path.exists():
            return []
        entries: List[LogEntry] = []
        try:
            with open(log_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    entries.append(LogEntry(
                        timestamp=data.get("timestamp", 0),
                        level=data.get("level", "INFO"),
                        message=data.get("message", ""),
                    ))
        except Exception:
            pass
        return entries

    def cleanup(self, source_id: str) -> None:
        """Remove buffer after processing is complete."""
        self._buffers.pop(source_id, None)


# Module-level singleton
_log_stream = LogStreamService()


def get_log_stream() -> LogStreamService:
    return _log_stream
