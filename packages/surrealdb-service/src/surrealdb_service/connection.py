"""
SurrealDB connection management.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from loguru import logger
from surrealdb import AsyncSurreal, RecordID
from websockets.exceptions import WebSocketException

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


class DatabaseUnavailableError(RuntimeError):
    """No usable SurrealDB connection within the pool's time bounds.

    Raised instead of waiting forever, so a request fails (503) and the next
    one can succeed once the database answers again.
    """


# Errors after which a connection's socket state is unknown. A statement the
# server rejected arrives as a plain ``Exception`` from the SDK and leaves the
# socket usable, so it is not listed here.
_BROKEN_CONNECTION_ERRORS = (
    asyncio.CancelledError,
    asyncio.TimeoutError,
    OSError,
    WebSocketException,
)


def _current_task_cancelling() -> bool:
    """True when the running task itself has been asked to cancel.

    ``Task.cancelling`` exists from Python 3.11; without it every
    ``CancelledError`` is taken to be ours, which is the old behaviour.
    """
    task = asyncio.current_task()
    cancelling = getattr(task, "cancelling", None)
    return cancelling is None or cancelling() > 0


class ConnectionPool:
    """
    Async connection pool for SurrealDB using asyncio.Queue.

    Reuses connections across requests to avoid the overhead of
    opening and closing a connection per query.

    A semaphore holds one permit per connection that exists or is being
    created, so a failed or cancelled sign-in hands its permit back instead of
    leaking a slot. Every wait is bounded: sign-in (``connect_timeout``), the
    liveness ping on reuse (``ping_timeout``), and the whole checkout,
    including the wait for a free slot (``acquire_timeout``).
    """

    def __init__(
        self,
        config: Optional[SurrealDBConfig] = None,
        max_size: int = 10,
        connect_timeout: Optional[float] = None,
        ping_timeout: float = 5.0,
        acquire_timeout: float = 30.0,
    ):
        self.config = config or get_config()
        self._max_size = max_size
        self._pool: asyncio.Queue[AsyncSurreal] = asyncio.Queue(maxsize=max_size)
        self._slots = asyncio.Semaphore(max_size)
        self._connect_timeout = (
            connect_timeout
            if connect_timeout is not None
            else float(self.config.connection_timeout)
        )
        self._ping_timeout = ping_timeout
        self._acquire_timeout = acquire_timeout

    async def _create_connection(self) -> AsyncSurreal:
        """Create and authenticate a new SurrealDB connection."""
        db = AsyncSurreal(self.config.url)
        try:
            await asyncio.wait_for(self._authenticate(db), self._connect_timeout)
        except asyncio.TimeoutError as e:
            await self._close_quietly(db)
            raise DatabaseUnavailableError(
                f"SurrealDB sign-in did not finish within {self._connect_timeout}s"
            ) from e
        except BaseException:
            await self._close_quietly(db)
            raise
        return db

    async def _authenticate(self, db: AsyncSurreal) -> None:
        await db.signin(
            {
                "username": self.config.username,
                "password": self.config.password,
            }
        )
        await db.use(self.config.namespace, self.config.database)

    async def _health_check(self, conn: AsyncSurreal) -> bool:
        """Check if a connection is still alive."""
        try:
            await asyncio.wait_for(conn.query("SELECT 1"), self._ping_timeout)
            return True
        except asyncio.CancelledError:
            if _current_task_cancelling():
                raise
            # The SDK cancels pending requests when the server closes the
            # socket; nobody cancelled us, so the connection is dead.
            return False
        except Exception:
            return False

    @staticmethod
    async def _close_quietly(conn: AsyncSurreal) -> None:
        try:
            await asyncio.wait_for(conn.close(), 2.0)
        except Exception:
            pass

    async def _checkout(self) -> AsyncSurreal:
        """Checkout within one deadline: the slot wait, pings and sign-in."""
        try:
            return await asyncio.wait_for(
                self._checkout_unbounded(), self._acquire_timeout
            )
        except asyncio.TimeoutError as e:
            raise DatabaseUnavailableError(
                f"No SurrealDB connection within {self._acquire_timeout}s "
                f"(pool of {self._max_size})"
            ) from e

    async def _checkout_unbounded(self) -> AsyncSurreal:
        """Take a slot, then an idle healthy connection or a new one."""
        await self._slots.acquire()
        try:
            while not self._pool.empty():
                candidate = self._pool.get_nowait()
                try:
                    healthy = await self._health_check(candidate)
                except BaseException:
                    # Cancelled mid-ping: the candidate is off the queue and
                    # nobody else will close it.
                    await self._close_quietly(candidate)
                    raise
                if healthy:
                    return candidate
                await self._close_quietly(candidate)
            return await self._create_connection()
        except BaseException:
            self._slots.release()
            raise

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[AsyncSurreal, None]:
        """Acquire a connection from the pool, creating one if needed."""
        conn = await self._checkout()
        broken = False
        try:
            yield conn
        except _BROKEN_CONNECTION_ERRORS:
            broken = True
            raise
        finally:
            if not broken:
                self._pool.put_nowait(conn)
            self._slots.release()
            if broken:
                await self._close_quietly(conn)

    async def close_all(self) -> None:
        """Close all pooled connections. Call during app shutdown."""
        while not self._pool.empty():
            try:
                conn = self._pool.get_nowait()
                await conn.close()
            except Exception:
                pass
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
