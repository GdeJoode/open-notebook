"""Repository for the ``reference_entity`` vocabulary table (Phase K.4).

``reference_entity`` (migration 41) caches validated entries from external
vocabularies (TOOI, Crossref, ...). The K.4 providers populate it; the
vocabulary reconciler queries it to link a persisted ``entity`` to its stable
external identifier(s).

Key invariant (migration 41's UNIQUE index ``idx_ref_name_source`` on
``(canonical_name, source_vocabulary)``): ``upsert`` is keyed on that pair so a
re-run of ``refresh`` never creates duplicate rows (AC6 idempotency). ``upsert``
does a Python-side pre-fetch on that key to decide CREATE-vs-UPDATE, then lets
the server merge the payload via SurrealQL ``UPDATE $id MERGE $data`` (the field
set is replaced/extended server-side, not reconstructed in Python).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from loguru import logger

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query


class ReferenceEntityRepository:
    """CRUD + lookup over the ``reference_entity`` vocabulary table."""

    table_name = "reference_entity"

    def __init__(self, config: Optional[SurrealDBConfig] = None) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Upsert (idempotent on (canonical_name, source_vocabulary))
    # ------------------------------------------------------------------

    async def upsert(self, record: Dict[str, Any]) -> str:
        """Create or refresh one reference entry, returning its record id.

        Lookup is by ``(canonical_name, source_vocabulary)``. On a hit the row is
        UPDATEd (external_uri / external_id / aliases / properties refreshed,
        ``last_validated`` re-stamped); otherwise a new row is CREATEd. Idempotent:
        repeated upserts of the same logical record do not multiply rows.
        """
        canonical_name = record["canonical_name"]
        source = record["source_vocabulary"]

        existing = await execute_query(
            "SELECT id FROM reference_entity "
            "WHERE canonical_name = $cn AND source_vocabulary = $src LIMIT 1",
            {"cn": canonical_name, "src": source},
            self.config,
        )

        payload: Dict[str, Any] = {
            "canonical_name": canonical_name,
            "entity_type": record.get("entity_type", "organization"),
            "source_vocabulary": source,
            "external_uri": record.get("external_uri"),
            "external_id": record.get("external_id"),
            "aliases": list(record.get("aliases") or []),
            "properties": dict(record.get("properties") or {}),
            "parent_name": record.get("parent_name"),
            "last_validated": datetime.now(timezone.utc),
        }

        if existing:
            rid = str(existing[0]["id"])
            rows = await execute_query(
                "UPDATE $id MERGE $data RETURN AFTER",
                {"id": _thing(rid), "data": payload},
                self.config,
            )
            return str(rows[0]["id"]) if rows else rid

        rows = await execute_query(
            "CREATE reference_entity CONTENT $data RETURN AFTER",
            {"data": payload},
            self.config,
        )
        return str(rows[0]["id"])

    async def bulk_load(self, records: List[Dict[str, Any]]) -> int:
        """Upsert a batch of reference entries, returning the count loaded.

        Used by the providers' ``refresh()``. Each record is upserted on the
        unique key so the whole operation is idempotent.
        """
        loaded = 0
        for record in records:
            try:
                await self.upsert(record)
                loaded += 1
            except Exception as exc:  # one bad row must not abort the batch
                logger.error(
                    "reference_entity bulk_load skipped {cn!r}: {e}",
                    cn=record.get("canonical_name"),
                    e=exc,
                )
        return loaded

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    async def lookup_by_name(
        self, name: str, source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return reference rows whose ``canonical_name`` equals ``name``.

        Optionally constrained to one ``source_vocabulary``. Returns ``[]`` on no
        hit or any DB error (callers treat the lookup as a no-match).
        """
        query = "SELECT * FROM reference_entity WHERE canonical_name = $name"
        params: Dict[str, Any] = {"name": name}
        if source is not None:
            query += " AND source_vocabulary = $src"
            params["src"] = source
        try:
            return await execute_query(query, params, self.config)
        except Exception as exc:
            logger.warning("reference_entity lookup_by_name failed: {e}", e=exc)
            return []

    async def lookup_by_alias(
        self, alias: str, source: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return reference rows that carry ``alias`` in their ``aliases`` array."""
        query = "SELECT * FROM reference_entity WHERE $alias IN aliases"
        params: Dict[str, Any] = {"alias": alias}
        if source is not None:
            query += " AND source_vocabulary = $src"
            params["src"] = source
        try:
            return await execute_query(query, params, self.config)
        except Exception as exc:
            logger.warning("reference_entity lookup_by_alias failed: {e}", e=exc)
            return []

    async def count(self, source: Optional[str] = None) -> int:
        """Count reference rows, optionally for one vocabulary."""
        query = "SELECT count() AS c FROM reference_entity"
        params: Dict[str, Any] = {}
        if source is not None:
            query += " WHERE source_vocabulary = $src"
            params["src"] = source
        query += " GROUP ALL"
        try:
            rows = await execute_query(query, params, self.config)
            return int(rows[0]["c"]) if rows else 0
        except Exception as exc:
            logger.warning("reference_entity count failed: {e}", e=exc)
            return 0

    async def last_validated(self, source: str) -> Optional[str]:
        """Most recent ``last_validated`` for a vocabulary, or None if unloaded."""
        try:
            rows = await execute_query(
                "SELECT last_validated FROM reference_entity "
                "WHERE source_vocabulary = $src "
                "ORDER BY last_validated DESC LIMIT 1",
                {"src": source},
                self.config,
            )
        except Exception as exc:
            logger.warning("reference_entity last_validated failed: {e}", e=exc)
            return None
        if not rows:
            return None
        return str(rows[0].get("last_validated"))


def _thing(record_id: str) -> Any:
    """Coerce a ``table:id`` string to a SurrealDB record-id binding."""
    from surrealdb_service.connection import ensure_record_id

    return ensure_record_id(record_id)


__all__ = ["ReferenceEntityRepository"]
