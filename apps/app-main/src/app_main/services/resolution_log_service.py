"""
Service for querying and updating the resolution_log table.

The resolution log records every entity-resolution decision the entity
filtering pipeline made (matches, candidates, manual reviews). This service
exposes it for the review UI.
"""

from typing import Any

from surrealdb_service.connection import ensure_record_id, execute_query


class MergeRecordImmutable(Exception):
    """The row records an applied merge; its status is not a decision to change."""


class ResolutionLogService:
    """Service for querying and updating the resolution_log table."""

    async def list_candidates(
        self,
        limit: int = 50,
        offset: int = 0,
        status: str | None = None,
        source_id: str | None = None,
        match_method: str | None = None,
        match_only: bool | None = None,
    ) -> tuple[list[dict[str, Any]], int]:
        """List resolution log entries with filters.

        Returns:
            (items, total_count)
        """
        where_clauses = []
        params: dict[str, Any] = {"limit": limit, "offset": offset}

        if status:
            where_clauses.append("status = $status")
            params["status"] = status
        if source_id:
            where_clauses.append("source_id = $source_id")
            params["source_id"] = source_id
        if match_method:
            where_clauses.append("match_method = $match_method")
            params["match_method"] = match_method
        if match_only is not None:
            where_clauses.append("match = $match_only")
            params["match_only"] = match_only

        where = f" WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        items_result = await execute_query(
            f"SELECT * FROM resolution_log{where} ORDER BY match_timestamp DESC LIMIT $limit START $offset",
            params,
        )
        count_result = await execute_query(
            f"SELECT count() AS total FROM resolution_log{where} GROUP ALL",
            params,
        )

        # `execute_query` returns the rows themselves. This read `[0]` and
        # `[0][0]` as if it returned one list per statement, which raised
        # KeyError on the first real row — unnoticed while the table had no
        # writer (PC.5b gave it one).
        items = list(items_result or [])
        total = int(count_result[0].get("total", 0)) if count_result else 0

        return items, total

    async def get_candidate(self, candidate_id: str) -> dict[str, Any] | None:
        """Get a single resolution log entry; ``None`` for a malformed id too."""
        try:
            rid = ensure_record_id(candidate_id)
        except ValueError:
            return None
        result = await execute_query("SELECT * FROM $id", {"id": rid})
        return result[0] if result else None

    async def update_status(
        self, candidate_id: str, status: str, *, author: str
    ) -> dict[str, Any] | None:
        """Accept or reject a resolution log entry.

        ``author`` (F2-06) is the person who decided; it replaces the pipeline
        as author of the row.

        Raises :class:`MergeRecordImmutable` for the record of an applied merge
        (``winner_id`` set, PC.5b): rejecting it would not undo the merge.
        """
        row = await self.get_candidate(candidate_id)
        if not row:
            return None
        if row.get("winner_id"):
            raise MergeRecordImmutable(candidate_id)
        result = await execute_query(
            "UPDATE $id SET status = $status, author = $author, reviewed_at = time::now()"
            " RETURN AFTER",
            {"id": ensure_record_id(candidate_id), "status": status, "author": author},
        )
        return result[0] if result else None

    async def bulk_update_status(
        self, candidate_ids: list[str], status: str, *, author: str
    ) -> int:
        """Bulk accept/reject resolution log entries."""
        updated = 0
        for cid in candidate_ids:
            try:
                result = await self.update_status(cid, status, author=author)
            except MergeRecordImmutable:
                continue  # a merge record is not a decision to revisit
            if result:
                updated += 1
        return updated

    async def get_methods(self) -> list[str]:
        """Get distinct match methods."""
        result = await execute_query(
            "SELECT match_method FROM resolution_log GROUP BY match_method",
            {},
        )
        return [r.get("match_method", "") for r in result or []]

    async def get_stats(self) -> dict[str, Any]:
        """Get resolution log summary statistics."""
        result = await execute_query(
            """SELECT
                count() AS total,
                math::sum(IF match THEN 1 ELSE 0 END) AS matches,
                math::sum(IF status = 'pending' THEN 1 ELSE 0 END) AS pending,
                math::sum(IF status = 'accepted' THEN 1 ELSE 0 END) AS accepted,
                math::sum(IF status = 'rejected' THEN 1 ELSE 0 END) AS rejected,
                math::sum(IF status = 'auto_accepted' THEN 1 ELSE 0 END) AS auto_accepted
            FROM resolution_log GROUP ALL""",
            {},
        )
        if result:
            return result[0]
        return {"total": 0, "matches": 0, "pending": 0, "accepted": 0, "rejected": 0, "auto_accepted": 0}
