"""Always-on telemetry sink for extraction-confidence-bearing events.

RETRO #5 flagged that Track A's auto-fallback path emitted decisions only
to Loguru, leaving no queryable record for offline tuning or correlating
UI quality regressions with parser decisions. This module centralizes the
write so every confidence-bearing pipeline step routes through one place
and we can iterate on the schema or transport without a callsite migration.

Design choices
--------------

* **Always-on, never-blocking.** Telemetry must not crash the caller; every
  failure path swallows the exception with a WARNING log. Extraction
  pipelines run unattended and a flaky DB write is no reason to lose a
  user's source.
* **Env-flagged opt-out.** Setting ``OPEN_NOTEBOOK_DISABLE_METRICS=1``
  short-circuits the function before any I/O. Useful for unit tests of the
  callers themselves and for users who want a fully-quiet offline mode.
  No PII is captured — only counts, scores, decisions, and timestamps —
  so the opt-out is not a privacy switch but an operational one.
* **Lazy persistence import.** ``surrealdb_service`` is heavy
  (testcontainers, asyncpg-style pool, …) and ``shared`` is the pure-model
  layer in the workspace dependency graph. Importing ``execute_query``
  inside the function keeps ``shared``'s static deps untouched and avoids
  inverting the layering.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from loguru import logger

_DISABLE_FLAG = "OPEN_NOTEBOOK_DISABLE_METRICS"


def _metrics_disabled() -> bool:
    """Return True when the operator has disabled metric writes.

    Checked on every call (not module-init) so tests can toggle the flag
    with ``monkeypatch.setenv`` and get the expected behaviour without
    importer cache games.
    """
    return os.environ.get(_DISABLE_FLAG, "").strip() == "1"


async def record_metric(
    event_type: str,
    payload: Dict[str, Any],
    source: Optional[str] = None,
    notebook: Optional[str] = None,
) -> None:
    """Append a single telemetry row to the ``metrics`` table.

    Parameters
    ----------
    event_type:
        Canonical event discriminator (e.g. ``extraction.complete``,
        ``extraction.auto_fallback``). Convention: dotted lowercase.
    payload:
        Free-form dict of counters/scores/decisions. Stored as a
        ``FLEXIBLE`` object so callers may extend keys without a
        migration. Coerced to ``{}`` when None to satisfy the schema
        default.
    source, notebook:
        Optional context keys; persisted as ``option<string>`` so events
        outside a source/notebook context (e.g. system-level diagnostics)
        still land cleanly.

    Returns
    -------
    None.
        The function never raises. Failures are logged at WARNING and
        swallowed so that production extraction pipelines never crash on
        a telemetry hiccup.
    """
    if _metrics_disabled():
        # Quiet on purpose — emitting a log here would be noisy on every
        # call. Tests assert the no-write behaviour directly.
        return

    if not event_type or not isinstance(event_type, str):
        logger.warning(
            "record_metric: event_type must be a non-empty string; got "
            f"{event_type!r}. Skipping write."
        )
        return

    safe_payload: Dict[str, Any] = payload if isinstance(payload, dict) else {}

    try:
        # Lazy import: ``shared`` is the pure-model layer in the workspace
        # graph; ``surrealdb-service`` is the persistence layer. Importing
        # inside the function preserves that layering.
        from surrealdb_service.connection import execute_query

        await execute_query(
            "CREATE metrics SET "
            "event_type = $event_type, "
            "payload = $payload, "
            "source = $source, "
            "notebook = $notebook;",
            {
                "event_type": event_type,
                "payload": safe_payload,
                "source": source,
                "notebook": notebook,
            },
        )
    except Exception as exc:  # pragma: no cover — covered by unit test
        # Never crash the caller. Log enough to debug offline but stay
        # within Loguru's default formatter (no traceback noise on the
        # happy-degraded path).
        logger.warning(
            f"record_metric: failed to write event_type={event_type!r}: {exc}"
        )


__all__ = ["record_metric"]
