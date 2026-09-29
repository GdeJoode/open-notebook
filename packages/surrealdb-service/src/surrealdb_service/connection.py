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
    DatabaseUnavailableError,
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
    though the socket may still be open. ``recv_task`` is the SDK's private
    attribute: a client without it (the HTTP one, another SDK version) counts
    as not receiving, so it is closed after use rather than pooled blind.
    """
    task = getattr(conn, "recv_task", None)
    return task is not None and not task.done()


def _current_task_cancelling() -> bool:
    """True when the running task itself has been asked to cancel.

    ``Task.cancelling`` exists from Python 3.11; without it every
    ``CancelledError`` is taken to be ours.
    """
    task = asyncio.current_task()
    cancelling = getattr(task, "cancelling", None)
    return cancelling is None or cancelling() > 0


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
    raises :class:`DatabaseUnavailableError`. So does a request the SDK
    cancelled itself while our task was not cancelled (1.0.6 raises
    ``KeyError`` there; an SDK that cancels pending requests would not).
    """
    request = asyncio.ensure_future(call)
    try:
        return await asyncio.shield(request)
    except asyncio.CancelledError as e:
        if request.cancelled() and not _current_task_cancelling():
            raise DatabaseUnavailableError("SurrealDB connection lost") from e
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

    A semaphore holds one permit per connection checked out or being
    created. A failed sign-in hands its permit back; one its caller gave up
    on keeps it until it has ended and its connection is closed, so a
    starved database never has more sign-ins than the pool has slots. Every
    wait is bounded: a sign-in (``connect_timeout`` from its start), the
    liveness ping on reuse (``ping_timeout``), and the whole checkout,
    including the wait for a free slot (``acquire_timeout``).

    A connection goes back to the pool only while its SDK receive task still
    runs, and never after a cancelled or broken request. Requests go through
    :func:`_request`, so a cancellation never reaches the SDK.
    """

    def __init__(
        self,
        config: Optional[SurrealDBConfig] = None,
        max_size: Optional[int] = None,
        connect_timeout: Optional[float] = None,
        ping_timeout: float = 5.0,
        acquire_timeout: float = 30.0,
    ):
        self.config = config or get_config()
        if max_size is None:
            max_size = self.config.pool_size
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
        # Closer of each abandoned sign-in -> that sign-in (``_close_when_done``).
        self._closers: "Dict[asyncio.Task[None], asyncio.Future[AsyncSurreal]]" = {}

    async def _new_connection(self) -> AsyncSurreal:
        """Sign in a new connection on the permit the caller holds.

        The sign-in runs in a task of its own (shielded, for the reason given
        at ``_request``) and owns the permit until it has ended: a caller that
        stops waiting leaves it to finish or time out, after which
        ``_close_when_done`` closes its connection and hands the permit back.
        So abandoned sign-ins never outnumber the pool.
        """
        sign_in = asyncio.ensure_future(self._sign_in())
        try:
            return await asyncio.shield(sign_in)
        except asyncio.CancelledError:
            self._abandon(sign_in)
            raise
        except BaseException:
            self._slots.release()
            raise

    async def _sign_in(self) -> AsyncSurreal:
        """Open and sign in a connection within ``connect_timeout`` of the start.

        A sign-in that fails for the connection's sake (refused, lost, too
        slow) raises :class:`DatabaseUnavailableError`; its socket is closed.
        """
        db = AsyncSurreal(self.config.url)
        try:
            await asyncio.wait_for(self._authenticate(db), self._connect_timeout)
        except asyncio.TimeoutError as e:
            await self._close_quietly(db)
            raise DatabaseUnavailableError(
                f"SurrealDB sign-in did not finish within {self._connect_timeout}s"
            ) from e
        except (OSError, WebSocketException, KeyError) as e:
            # Refused, or the socket went mid-sign-in (1.0.6 raises KeyError).
            await self._close_quietly(db)
            raise DatabaseUnavailableError(f"SurrealDB sign-in failed: {e!r}") from e
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

    def _abandon(self, sign_in: "asyncio.Future[AsyncSurreal]") -> None:
        closer = asyncio.ensure_future(self._close_when_done(sign_in))
        self._closers[closer] = sign_in
        closer.add_done_callback(lambda done: self._closers.pop(done, None))

    async def _close_when_done(self, sign_in: "asyncio.Future[AsyncSurreal]") -> None:
        """Close an abandoned sign-in's connection once it has ended, then
        hand its permit back. A sign-in that failed closed its own."""
        try:
            db = await sign_in
        except BaseException:
            pass
        else:
            await self._close_quietly(db)
        finally:
            self._slots.release()

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
        except BaseException:
            self._slots.release()
            raise
        return await self._new_connection()

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
        # Ending the sign-ins lets their closers close the connections now. (A
        # closer cancelled before its first step would never run its close.)
        closers = list(self._closers.items())
        for _, sign_in in closers:
            sign_in.cancel()
        await asyncio.gather(*(c for c, _ in closers), return_exceptions=True)
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
