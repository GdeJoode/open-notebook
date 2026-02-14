"""
Repository for summary storage and retrieval.

Manages the `summary` table in SurrealDB for persisting
summarization results from the summarization pipeline.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query


class SummaryRepository:
    """Repository for summary CRUD operations."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        self.config = config

    async def list_summaries(
        self, source_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List summaries, optionally filtered by source_id."""
        if source_id:
            query = (
                "SELECT id, source_id, strategy, document_summary, "
                "num_layers, num_input_chunks, array::len(summary_nodes) AS node_count, "
                "created FROM summary WHERE source_id = $source_id ORDER BY created DESC"
            )
            params = {"source_id": source_id}
        else:
            query = (
                "SELECT id, source_id, strategy, document_summary, "
                "num_layers, num_input_chunks, array::len(summary_nodes) AS node_count, "
                "created FROM summary ORDER BY created DESC"
            )
            params = {}

        try:
            return await execute_query(query, params, self.config)
        except Exception as e:
            logger.error(f"Failed to list summaries: {e}")
            return []

    async def get_summary(self, summary_id: str) -> Optional[Dict[str, Any]]:
        """Get a single summary by ID with full content."""
        try:
            result = await execute_query(
                "SELECT * FROM summary WHERE id = $id",
                {"id": summary_id},
                self.config,
            )
            if result:
                return result[0]
            return None
        except Exception as e:
            logger.error(f"Failed to get summary '{summary_id}': {e}")
            return None

    async def create_summary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new summary record."""
        try:
            result = await execute_query(
                "CREATE summary CONTENT $data",
                {"data": data},
                self.config,
            )
            if result:
                return result[0]
            return data
        except Exception as e:
            logger.error(f"Failed to create summary: {e}")
            raise

    async def delete_summary(self, summary_id: str) -> bool:
        """Delete a summary by ID."""
        try:
            result = await execute_query(
                "DELETE summary WHERE id = $id RETURN BEFORE",
                {"id": summary_id},
                self.config,
            )
            return bool(result)
        except Exception as e:
            logger.error(f"Failed to delete summary '{summary_id}': {e}")
            return False
