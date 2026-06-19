"""
SurrealDB connection management.
"""

import asyncio
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
    Async connection pool for SurrealDB using asyncio.Queue.

    Reuses connections across requests to avoid the overhead of
    opening and closing a connection per query.
    """

    def __init__(
        self,
        config: Optional[SurrealDBConfig] = None,
        max_size: int = 10,
    ):
        self.config = config or get_config()
        self._max_size = max_size
        self._pool: asyncio.Queue[AsyncSurreal] = asyncio.Queue(maxsize=max_size)
        self._size = 0
        self._lock = asyncio.Lock()

    async def _create_connection(self) -> AsyncSurreal:
        """Create and authenticate a new SurrealDB connection."""
        db = AsyncSurreal(self.config.url)
        await db.signin(
            {
                "username": self.config.username,
                "password": self.config.password,
            }
        )
        await db.use(self.config.namespace, self.config.database)
        return db

    async def _health_check(self, conn: AsyncSurreal) -> bool:
        """Check if a connection is still alive."""
        try:
            await conn.query("SELECT 1")
            return True
        except Exception:
            return False

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[AsyncSurreal, None]:
        """Acquire a connection from the pool, creating one if needed."""
        conn: Optional[AsyncSurreal] = None

        # Try to reuse an existing connection
        while not self._pool.empty():
            try:
                candidate = self._pool.get_nowait()
            except asyncio.QueueEmpty:
                break
            if await self._health_check(candidate):
                conn = candidate
                break
            else:
                # Dead connection — discard and decrement size
                async with self._lock:
                    self._size -= 1
                try:
                    await candidate.close()
                except Exception:
                    pass

        # Create a new connection if none available
        if conn is None:
            async with self._lock:
                if self._size < self._max_size:
                    self._size += 1
                    create_new = True
                else:
                    create_new = False

            if create_new:
                conn = await self._create_connection()
            else:
                # Pool exhausted — wait for one to be returned
                conn = await self._pool.get()
                if not await self._health_check(conn):
                    try:
                        await conn.close()
                    except Exception:
                        pass
                    conn = await self._create_connection()

        try:
            yield conn
        finally:
            # Return connection to pool
            try:
                self._pool.put_nowait(conn)
            except asyncio.QueueFull:
                # Pool is full — close the excess connection
                async with self._lock:
                    self._size -= 1
                try:
                    await conn.close()
                except Exception:
                    pass

    async def close_all(self) -> None:
        """Close all pooled connections. Call during app shutdown."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                await conn.close()
            except Exception:
                pass
        async with self._lock:
            self._size = 0
        logger.info("Connection pool closed")


# Global pool instance
_pool: Optional[ConnectionPool] = None


def get_pool(config: Optional[SurrealDBConfig] = None) -> ConnectionPool:
    """Get the global connection pool."""
    global _pool
    if _pool is None:
        _pool = ConnectionPool(config=config)
    return _pool


async def close_pool() -> None:
    """Shut down the global connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close_all()
        _pool = None


async def execute_query(
    query: str,
    params: Optional[Dict[str, Any]] = None,
    config: Optional[SurrealDBConfig] = None,
) -> List[Dict[str, Any]]:
    """
    Execute a SurrealQL query using the connection pool.

    Args:
        query: The SurrealQL query string.
        params: Optional query parameters.
        config: Optional database configuration.

    Returns:
        List of result dictionaries.

    Raises:
        RuntimeError: If query execution fails.
    """
    pool = get_pool(config)
    async with pool.acquire() as connection:
        try:
            result = parse_record_ids(await connection.query(query, params))
            if isinstance(result, str):
                raise RuntimeError(result)
            return result
        except Exception as e:
            logger.error(f"Query failed: {query[:200]} params: {params}")
            logger.exception(e)
            raise


def _check_transaction_response(response: Any) -> List[Any]:
    """Validate a ``query_raw`` response, raising on any statement error.

    SurrealDB returns one entry per statement, each ``{"status": "OK"|"ERR",
    "result": ...}``. A failed statement inside a ``BEGIN ... COMMIT`` block
    rolls the whole transaction back, but the plain ``query()`` seam only
    returns the *first* statement's result and so can swallow a later
    statement's error — making a rolled-back transaction look like a silent
    no-op. This inspects every statement and raises ``RuntimeError`` if any
    reports ``ERR`` (or the response carries a top-level error), so a
    rolled-back transaction always surfaces as an exception.

    Returns the per-statement results on success.
    """
    if isinstance(response, dict) and response.get("error") is not None:
        raise RuntimeError(str(response["error"]))
    statements = (
        response.get("result", []) if isinstance(response, dict) else []
    )
    errors = [
        s.get("result")
        for s in statements
        if isinstance(s, dict) and s.get("status") == "ERR"
    ]
    if errors:
        raise RuntimeError(f"SurrealQL transaction failed: {errors}")
    return [
        s.get("result") for s in statements if isinstance(s, dict)
    ]


async def execute_transaction(
    query: str,
    params: Optional[Dict[str, Any]] = None,
    config: Optional[SurrealDBConfig] = None,
) -> List[Any]:
    """Execute a multi-statement SurrealQL transaction, raising on any error.

    Use this instead of :func:`execute_query` for ``BEGIN ... COMMIT`` blocks:
    it runs the block through ``query_raw`` and inspects every statement's
    status (see :func:`_check_transaction_response`), so a statement that
    fails mid-transaction propagates as an exception rather than being
    silently swallowed.
    """
    pool = get_pool(config)
    async with pool.acquire() as connection:
        try:
            response = await connection.query_raw(query, params or {})
        except Exception as e:
            logger.error(
                f"Transaction failed: {query[:200]} params: {params}"
            )
            logger.exception(e)
            raise
    return parse_record_ids(_check_transaction_response(response))
