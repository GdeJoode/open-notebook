"""
SurrealDB connection management.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Awaitable, Dict, List, Optional, TypeVar, Union

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

T = TypeVar("T")


def _receiving(conn: AsyncSurreal) -> bool:
    """True while the SDK's receive task reads the connection's socket.

    That one task answers every request; once it has ended (the socket closed,
    or it died of an error) nothing sent on the connection is answered again,
    though the socket may still be open. The HTTP client has no such task.
    """
    if not hasattr(conn, "recv_task"):
        return True
    task = conn.recv_task
    return task is not None and not task.done()


def _discard_outcome(request: "asyncio.Future[Any]") -> None:
    if not request.cancelled():
        request.exception()


async def _request(conn: AsyncSurreal, call: Awaitable[T]) -> T:
    """Await an SDK request without letting a cancellation into the SDK.

    SDK 1.0.6 cannot take one: when the response arrives in the loop turn the
    cancel lands in, its receive task sets the result on the cancelled future,
    dies of the ``InvalidStateError`` and fails every request on the
    connection from then on, the cancelled one with a ``KeyError`` instead of
    ``CancelledError``. So the request runs in a task of its own and only the
    wait for it is cancelled. The pool then closes the connection, which ends
    the request and stops the statement on the server.

    A request that fails because the receive task ended (the socket was lost)
    raises :class:`DatabaseUnavailableError`.
    """
    request = asyncio.ensure_future(call)
    try:
        return await asyncio.shield(request)
    except asyncio.CancelledError:
        request.add_done_callback(_discard_outcome)
        raise
    except Exception as e:
        if _receiving(conn):
            raise
        raise DatabaseUnavailableError("SurrealDB connection lost") from e


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

    A connection goes back to the pool only while its SDK receive task still
    runs, and never after a cancelled or broken request. Requests go through
    :func:`_request`, so a cancellation never reaches the SDK.
    """

    def __init__(
        self,
        config: Optional[SurrealDBConfig] = None,
        # One Werkbak report checks out 13-14 connections at once (two
        # notebooks; its per-notebook groups add one each per notebook). With
        # 10, its heavy groups queued for a slot behind the others' reads and
        # ran into their deadline on a cold database, and the pages read next
        # to it waited too. A pooled connection costs one idle socket.
        max_size: int = 20,
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
        self._close_timeout = 2.0
        # Set by close_all: connections handed back afterwards are closed.
        self._closed = False
        # Closers of abandoned sign-ins (see ``_close_when_done``).
        self._closers: "set[asyncio.Task[None]]" = set()

    async def _create_connection(self) -> AsyncSurreal:
        """Create and authenticate a new SurrealDB connection."""
        db = AsyncSurreal(self.config.url)
        # Shielded for the reason given at ``_request``. A sign-in abandoned
        # half-way is still opening its socket, so it cannot be closed now:
        # ``_close_when_done`` closes it once the sign-in ends.
        auth = asyncio.ensure_future(self._authenticate(db))
        try:
            await asyncio.wait_for(asyncio.shield(auth), self._connect_timeout)
        except asyncio.TimeoutError as e:
            self._abandon(auth, db)
            raise DatabaseUnavailableError(
                f"SurrealDB sign-in did not finish within {self._connect_timeout}s"
            ) from e
        except BaseException:
            self._abandon(auth, db)
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

    def _abandon(self, auth: "asyncio.Future[None]", db: AsyncSurreal) -> None:
        closer = asyncio.ensure_future(self._close_when_done(auth, db))
        self._closers.add(closer)
        closer.add_done_callback(self._closers.discard)

    async def _close_when_done(
        self, auth: "asyncio.Future[None]", db: AsyncSurreal
    ) -> None:
        """Close an abandoned connection once its sign-in has ended.

        A sign-in that hangs is cancelled after ``connect_timeout``; the
        connection is closed right after, so the SDK's state no longer matters.
        """
        try:
            await asyncio.wait_for(auth, self._connect_timeout)
        except Exception:
            pass
        finally:
            await self._close_quietly(db)

    async def _health_check(self, conn: AsyncSurreal) -> bool:
        """Check if a connection is still alive."""
        try:
            await asyncio.wait_for(
                _request(conn, conn.query("RETURN true")), self._ping_timeout
            )
            return True
        except Exception:
            return False

    async def _close_quietly(self, conn: AsyncSurreal) -> None:
        try:
            await asyncio.wait_for(conn.close(), self._close_timeout)
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
                if not _receiving(candidate):
                    # A ping on it would wait out ``ping_timeout`` unanswered.
                    await self._close_quietly(candidate)
                    continue
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
            keep = not broken and not self._closed and _receiving(conn)
            if keep:
                self._pool.put_nowait(conn)
            self._slots.release()
            if not keep:
                await self._close_quietly(conn)

    async def close_all(self) -> None:
        """Close all pooled connections. Call during app shutdown."""
        self._closed = True
        while not self._pool.empty():
            await self._close_quietly(self._pool.get_nowait())
        # Cancelling a closer stops its wait; it still closes its connection.
        closers = list(self._closers)
        for closer in closers:
            closer.cancel()
        await asyncio.gather(*closers, return_exceptions=True)
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
            result = parse_record_ids(
                await _request(connection, connection.query(query, params))
            )
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
            response = await _request(
                connection, connection.query_raw(query, params or {})
            )
        except Exception as e:
            logger.error(
                f"Transaction failed: {query[:200]} params: {params}"
            )
            logger.exception(e)
            raise
    return parse_record_ids(_check_transaction_response(response))
