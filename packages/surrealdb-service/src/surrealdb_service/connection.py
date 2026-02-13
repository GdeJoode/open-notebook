"""
SurrealDB connection management.
"""

from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from loguru import logger
from surrealdb import AsyncSurreal, RecordID

from surrealdb_service.config import SurrealDBConfig, get_config


def parse_record_ids(obj: Any) -> Any:
    """Recursively parse and convert RecordIDs into strings."""
    if isinstance(obj, dict):
        return {k: parse_record_ids(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [parse_record_ids(item) for item in obj]
    elif isinstance(obj, RecordID):
        return str(obj)
    return obj


def ensure_record_id(value: Union[str, RecordID]) -> RecordID:
    """Ensure a value is a RecordID."""
    if isinstance(value, RecordID):
        return value
    return RecordID.parse(value)


@asynccontextmanager
async def db_connection(
    config: Optional[SurrealDBConfig] = None,
) -> AsyncGenerator[AsyncSurreal, None]:
    """
    Create a database connection context manager.

    Args:
        config: Optional configuration. Uses global config if not provided.

    Yields:
        AsyncSurreal connection.
    """
    if config is None:
        config = get_config()

    db = AsyncSurreal(config.url)
    try:
        await db.signin(
            {
                "username": config.username,
                "password": config.password,
            }
        )
        await db.use(config.namespace, config.database)
        yield db
    finally:
        await db.close()


class ConnectionPool:
    """
    Simple connection pool for SurrealDB.

    Note: This is a basic implementation. For production, consider using
    a more sophisticated connection pool or the built-in connection management.
    """

    def __init__(self, config: Optional[SurrealDBConfig] = None):
        self.config = config or get_config()
        self._connections: List[AsyncSurreal] = []

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[AsyncSurreal, None]:
        """Acquire a connection from the pool."""
        # For now, just create a new connection each time
        # A full implementation would reuse connections
        async with db_connection(self.config) as conn:
            yield conn


# Global pool instance
_pool: Optional[ConnectionPool] = None


def get_pool() -> ConnectionPool:
    """Get the global connection pool."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool()
    return _pool


async def execute_query(
    query: str,
    params: Optional[Dict[str, Any]] = None,
    config: Optional[SurrealDBConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Execute a SurrealQL query.

    Args:
        query: The SurrealQL query string.
        params: Optional query parameters.
        config: Optional database configuration.

    Returns:
        List of result dictionaries.

    Raises:
        RuntimeError: If query execution fails.
    """
    async with db_connection(config) as connection:
        try:
            result = parse_record_ids(await connection.query(query, params))
            if isinstance(result, str):
                raise RuntimeError(result)
            return result
        except Exception as e:
            logger.error(f"Query failed: {query[:200]} params: {params}")
            logger.exception(e)
            raise
