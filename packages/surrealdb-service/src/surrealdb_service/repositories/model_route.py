"""Repository for the ``model_route`` table (Track J.1).

Backs the privacy-aware ordered provider chain. The schema lives in
``migrations/51.surrealql`` and the Pydantic mirror is
:class:`shared.models.llm.ModelRoute`.

Singleton-per-task semantics are enforced by the UNIQUE index on
``model_route.task``. :meth:`upsert` therefore queries by task first and
chooses UPDATE vs CREATE, mirroring :class:`NotebookSchemaRepository` — a
blind CREATE would raise on the second call for the same task.
"""

from typing import Any, Dict, List, Optional

from loguru import logger

from shared.models.llm import ModelRoute
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import execute_query
from surrealdb_service.repositories.base import BaseRepository


class ModelRouteRepository(BaseRepository[ModelRoute]):
    """Repository for :class:`ModelRoute` rows — one per LLM task."""

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        super().__init__(ModelRoute, config)

    async def get_by_task(self, task: str) -> Optional[ModelRoute]:
        """Fetch the route row for a task, or ``None`` if none exists.

        Args:
            task: One of ``entity_extraction`` | ``summarization`` | ``chat``.

        Returns:
            The :class:`ModelRoute` row or ``None`` when no row has been
            configured yet (the resolver then falls back to its hard-coded
            default chain).
        """
        try:
            result = await execute_query(
                "SELECT * FROM model_route WHERE task = $task LIMIT 1;",
                {"task": task},
                self.config,
            )
            if result:
                return ModelRoute(**result[0])
            return None
        except Exception as e:
            logger.error(f"Failed to fetch model_route for task {task!r}: {e}")
            return None

    async def list_all(self) -> List[ModelRoute]:
        """Return every configured route row (used by the J.5 config UI)."""
        try:
            rows = await execute_query(
                "SELECT * FROM model_route;",
                config=self.config,
            )
            return [ModelRoute(**row) for row in rows]
        except Exception as e:
            logger.error(f"Failed to list model_route rows: {e}")
            return []

    async def upsert(self, route: ModelRoute) -> str:
        """Create or rewrite the route row for ``route.task``.

        The UNIQUE index on ``task`` means a blind CREATE would raise on the
        second call. Query-by-task first, then UPDATE the existing row or
        CREATE a new one.

        Args:
            route: The :class:`ModelRoute` to persist. ``task`` MUST be set;
                ``id`` is ignored (the existing row id is preferred).

        Returns:
            The record id of the persisted row (e.g. ``"model_route:abc"``).

        Raises:
            RuntimeError: On unexpected SurrealDB errors.
        """
        data: Dict[str, Any] = route.model_dump(
            exclude={"id", "created", "updated"},
            exclude_none=False,
        )

        try:
            existing = await execute_query(
                "SELECT id FROM model_route WHERE task = $task LIMIT 1;",
                {"task": route.task},
                self.config,
            )

            if existing:
                existing_id = existing[0]["id"]
                result = await execute_query(
                    "UPDATE type::thing($id) MERGE $data RETURN AFTER;",
                    {"id": existing_id, "data": data},
                    self.config,
                )
                if not result:
                    raise RuntimeError(
                        f"UPDATE model_route {existing_id} returned no rows"
                    )
                return str(result[0]["id"])

            result = await execute_query(
                "CREATE model_route CONTENT $data RETURN AFTER;",
                {"data": data},
                self.config,
            )
            if not result:
                raise RuntimeError("CREATE model_route returned no rows")
            return str(result[0]["id"])
        except Exception as e:
            logger.exception(
                f"Failed to upsert model_route for task {route.task!r}: {e}"
            )
            raise
