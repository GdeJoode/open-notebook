"""Durable chunk-edit audit log (Track I.H2, core slice).

The inspect-workspace chunk mutator (merge/split) records each structural edit to
loguru only; this service writes the DURABLE, append-only ``chunk_edit`` row that
the mutator's docstring defers to I.H2. It is deliberately decoupled from the
mutation transaction: the row is appended AFTER the mutation has committed, and a
failed audit write is swallowed (logged, never raised) so an observability problem
can never roll back or mask a chunk edit the operator already performed.

``before``/``after`` are stored as JSON strings (see migration 74 for why): the two
ops have different arity, and a JSON string keeps the row shape uniform without a
SCHEMAFULL object. Each entry is a compact chunk summary — enough to render a
history timeline and diff, and (later) to seed the deferred snapshot/restore half.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from loguru import logger
from shared.models import Chunk
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ensure_record_id, execute_query


def _summarize(chunk: Chunk) -> Dict[str, Any]:
    """A compact, JSON-serializable snapshot of a chunk for the audit trail."""
    return {
        "id": str(chunk.id),
        "order": chunk.order,
        "chunk_type": chunk.chunk_type,
        "text": chunk.text,
    }


def _encode(chunks: Optional[Sequence[Chunk]]) -> Optional[str]:
    if chunks is None:
        return None
    return json.dumps([_summarize(c) for c in chunks], ensure_ascii=False)


class ChunkAuditService:
    """Append and read the ``chunk_edit`` audit log (Track I.H2)."""

    def __init__(self, config: Optional[SurrealDBConfig] = None) -> None:
        self.config = config

    async def record(
        self,
        *,
        source_id: str,
        op: str,
        before: Optional[Sequence[Chunk]] = None,
        after: Optional[Sequence[Chunk]] = None,
        chunk_id: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> None:
        """Append one ``chunk_edit`` row; fail-soft.

        Args:
            source_id: The source whose chunks were edited (stored directly so the
                per-source history survives a chunk deleted by a merge).
            op: ``"merge"`` or ``"split"``.
            before: The chunk state(s) prior to the edit (merge: ``[keep, drop]``;
                split: ``[original]``).
            after: The chunk state(s) after the edit (merge: ``[merged]``; split:
                ``[original, new]``).
            chunk_id: The primary/surviving chunk id, when there is one.
            actor: The operator who made the edit, when known (the merge/split
                endpoints carry no auth identity yet, so this is usually ``None``).
        """
        try:
            data: Dict[str, Any] = {
                "source": ensure_record_id(source_id),
                "chunk": ensure_record_id(chunk_id) if chunk_id else None,
                "op": op,
                "before": _encode(before),
                "after": _encode(after),
                "actor": actor,
            }
            await execute_query(
                "CREATE chunk_edit CONTENT $data", {"data": data}, self.config
            )
        except Exception as exc:  # noqa: BLE001 — audit must never break a commit
            logger.warning(
                "chunk_edit audit write failed (op={op} source={sid}): {exc}",
                op=op,
                sid=source_id,
                exc=exc,
            )

    async def history(self, source_id: str) -> List[Dict[str, Any]]:
        """Return the source's chunk-edit rows, most recent first."""
        rows = await execute_query(
            "SELECT * FROM chunk_edit WHERE source = $source ORDER BY ts DESC",
            {"source": ensure_record_id(source_id)},
            self.config,
        )
        return rows or []


__all__ = ["ChunkAuditService"]
