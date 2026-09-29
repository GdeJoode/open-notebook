"""Cancel safety of pooled connections, with the real SDK and a real SurrealDB.

The live app degraded after one Werkbak read (2026-09-29): a group deadline
cancelled a query in the same event-loop turn in which its response arrived.
SDK 1.0.6 then sets the result on the cancelled future, its receive task dies
of the ``InvalidStateError``, it clears every pending request, and the caller
gets a ``KeyError`` instead of ``CancelledError``. The pool took that for a
statement error and kept the connection; every later request on it waited
for nothing until the 5 s reuse ping gave up. A socket lost mid-query ends
the receive task the same way, so its request also sees a ``KeyError``.

These tests produce both states with the real client, then check that the
pool neither keeps nor hands out such a connection.
"""

import asyncio
import gc
import logging
import time
from typing import Any, Callable, List

import pytest
import websockets
from loguru import logger

from surrealdb_service import connection as conn_mod
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import (
    ConnectionPool,
    DatabaseUnavailableError,
    execute_query,
)

pytestmark = [pytest.mark.asyncio, pytest.mark.requires_docker]

# A local SurrealDB answers ``RETURN 1`` in milliseconds; this leaves room for
# a loaded CI box while still failing on a 5 s reuse ping.
NORMAL_S = 1.0


@pytest.fixture
async def pool(live_surrealdb: SurrealDBConfig):
    original = conn_mod._pool
    # A reuse ping far above NORMAL_S: a dead connection that is pinged
    # instead of recognised shows as a slow query.
    conn_mod._pool = ConnectionPool(config=live_surrealdb, max_size=4, ping_timeout=5.0)
    yield conn_mod._pool
    await conn_mod._pool.close_all()
    conn_mod._pool = original


@pytest.fixture
def logged(caplog) -> Callable[[], str]:
    """Everything logged so far: our loguru lines, the SDK's stdlib debug lines
    (where its receive task reports the error it died of) and asyncio's."""
    caplog.set_level(logging.DEBUG, logger="surrealdb")
    lines: List[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), level="DEBUG")

    def text() -> str:
        gc.collect()  # a task whose error nobody read reports it when collected
        return "\n".join(lines + [r.getMessage() for r in caplog.records])

    yield text
    logger.remove(sink)


async def _warm(pool: ConnectionPool) -> None:
    await asyncio.gather(*(execute_query("RETURN 1") for _ in range(pool._max_size)))


def _idle(pool: ConnectionPool) -> List[Any]:
    return list(pool._pool._queue)


async def _cancel_as_the_response_arrives(query: str = "SLEEP 150ms; RETURN 1;") -> BaseException:
    """Cancel an in-flight query in the loop turn its response is read.

    The deadline and the response land while the loop is busy (the Werkbak's
    other groups, in the live app), so both are handled in one turn: the
    response wakes the SDK's receive task first, the cancel lands before it runs.
    """
    task = asyncio.ensure_future(execute_query(query))
    await asyncio.sleep(0.05)  # checked out and sent
    asyncio.get_running_loop().call_later(0.05, task.cancel)
    time.sleep(0.4)  # a busy loop: the response and the deadline both arrive
    try:
        await task
    except BaseException as e:  # noqa: BLE001 - the type is the assertion
        return e
    raise AssertionError("the query was not cancelled")


async def _fifty_normal_queries() -> List[float]:
    took = []
    for i in range(50):
        t = time.perf_counter()
        rows = await asyncio.wait_for(execute_query(f"RETURN {i}"), 10)
        took.append(time.perf_counter() - t)
        assert rows == i
    return took


async def test_cancel_racing_the_response_leaves_the_pool_healthy(pool, logged):
    await _warm(pool)
    for _ in range(4):
        err = await _cancel_as_the_response_arrives()
        # The caller sees its own cancellation, so a deadline reads as one.
        assert isinstance(err, asyncio.CancelledError), repr(err)
        assert all(not c.recv_task.done() for c in _idle(pool))

    took = await _fifty_normal_queries()
    assert max(took) < NORMAL_S, took
    log = logged()
    assert "KeyError" not in log
    assert "Unexpected error in _recv_task" not in log


async def test_many_mid_flight_cancels_then_normal_queries(pool, logged):
    """Smoke test, not a guard: ordinary concurrent cancels hit the race only
    sometimes (it did in one run on main, not in another). The race test
    above is the guard."""
    await _warm(pool)
    for _ in range(3):
        slow = [
            asyncio.wait_for(execute_query("SLEEP 300ms; RETURN 1;"), 0.02 + 0.03 * i)
            for i in range(8)
        ]
        results = await asyncio.gather(*slow, return_exceptions=True)
        assert all(isinstance(r, asyncio.TimeoutError) for r in results), results

    took = await _fifty_normal_queries()
    assert max(took) < NORMAL_S, took
    assert "KeyError" not in logged()


async def test_socket_lost_mid_query_is_unavailable_and_not_pooled(pool):
    await _warm(pool)
    for c in _idle(pool):
        asyncio.get_running_loop().call_later(0.1, c.socket.transport.abort)

    with pytest.raises(DatabaseUnavailableError):
        await asyncio.wait_for(execute_query("SLEEP 500ms; RETURN 1;"), 5)
    # The query's connection was not handed back.
    assert len(_idle(pool)) == pool._max_size - 1
    assert await asyncio.wait_for(execute_query("RETURN 7"), 5) == 7


async def test_an_idle_connection_is_reused(pool):
    """The reuse ping must be a statement SurrealDB accepts: ``SELECT 1`` is a
    parse error on v2, so every checkout dropped every idle connection and
    signed in anew."""
    await execute_query("RETURN 1")
    idle = _idle(pool)
    for i in range(5):
        assert await execute_query(f"RETURN {i}") == i
    assert _idle(pool) == idle


async def test_a_dead_idle_connection_is_dropped_without_the_ping(pool):
    """The state the race leaves: socket open, nobody reading it. A ping on it
    is sent and never answered, so it costs the whole ping timeout."""
    await _warm(pool)
    for c in _idle(pool):
        c.recv_task.cancel()
    await asyncio.sleep(0.05)
    assert all(c.recv_task.done() and c.socket.state.name == "OPEN" for c in _idle(pool))

    t = time.perf_counter()
    assert await asyncio.wait_for(execute_query("RETURN 8"), 10) == 8
    assert time.perf_counter() - t < NORMAL_S


def _live_receive_tasks() -> int:
    """SDK receive tasks still running: each is an open, signed-in socket."""
    return sum(
        1
        for t in asyncio.all_tasks()
        if not t.done() and "_recv_task" in t.get_coro().__qualname__
    )


@pytest.fixture
def slow_handshake(monkeypatch):
    """The WebSocket handshake takes 0.3 s, as on a slow or restarting DB."""
    real = websockets.connect

    async def slow(*args: Any, **kwargs: Any) -> Any:
        await asyncio.sleep(0.3)
        return await real(*args, **kwargs)

    monkeypatch.setattr(websockets, "connect", slow)


async def test_sign_in_past_connect_timeout_leaves_no_socket(
    live_surrealdb, slow_handshake
):
    before = _live_receive_tasks()
    pool = ConnectionPool(config=live_surrealdb, connect_timeout=0.1)
    with pytest.raises(DatabaseUnavailableError):
        async with pool.acquire():
            pass
    await asyncio.sleep(1.0)  # the handshake and sign-in finish meanwhile
    assert _live_receive_tasks() == before


async def test_sign_in_cut_off_by_the_caller_leaves_no_socket(pool, slow_handshake):
    before = _live_receive_tasks()
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(execute_query("RETURN 1"), 0.1)
    await asyncio.sleep(1.0)
    assert _live_receive_tasks() == before


class _SilentServer:
    """Accepts TCP connections and never answers: a starved database. Records
    how many sockets are open at once and how long each lived."""

    def __init__(self) -> None:
        self.open = 0
        self.peak = 0
        self.lifetimes: List[float] = []

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        started = time.perf_counter()
        self.open += 1
        self.peak = max(self.peak, self.open)
        try:
            while await reader.read(4096):
                pass
        finally:
            self.open -= 1
            self.lifetimes.append(time.perf_counter() - started)
            writer.close()

    async def __aenter__(self) -> str:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        return f"ws://127.0.0.1:{port}/rpc"

    async def __aexit__(self, *exc: Any) -> None:
        self._server.close()


async def test_abandoned_sign_ins_stay_within_the_pool_size():
    """Callers that give up do not leave their sign-ins piling up on a
    database that does not answer: each keeps its pool slot until it has
    ended, and none outlives ``connect_timeout``."""
    silent = _SilentServer()
    async with silent as url:
        pool = ConnectionPool(
            config=SurrealDBConfig(url=url), max_size=2, connect_timeout=0.5
        )

        async def caller() -> None:
            async with pool.acquire():
                pass

        # Twenty callers in a second, each giving up after 50 ms.
        for _ in range(20):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(caller(), 0.05)
        await asyncio.sleep(0.6)
        assert silent.peak <= 2, silent.peak
        assert silent.open == 0

        # Callers that give up 0.3 s in: a sign-in timed from then would
        # live 0.8 s.
        for _ in range(4):
            with pytest.raises(asyncio.TimeoutError):
                await asyncio.wait_for(caller(), 0.3)
        await asyncio.sleep(0.6)
        assert silent.open == 0
        assert max(silent.lifetimes) < 0.5 + 0.15, silent.lifetimes

        # A new caller waits for a slot, then gets unavailable, not a hang.
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(caller(), 0.05)
        await pool.close_all()
        await asyncio.sleep(0.05)
        assert silent.open == 0


async def test_refused_sign_in_is_unavailable():
    pool = ConnectionPool(config=SurrealDBConfig(url="ws://127.0.0.1:9/rpc"))
    with pytest.raises(DatabaseUnavailableError):
        async with pool.acquire():
            pass


async def test_socket_lost_during_sign_in_is_unavailable(live_surrealdb, monkeypatch):
    real = websockets.connect

    async def dropped(*args: Any, **kwargs: Any) -> Any:
        ws = await real(*args, **kwargs)
        ws.transport.abort()
        return ws

    monkeypatch.setattr(websockets, "connect", dropped)
    pool = ConnectionPool(config=live_surrealdb)
    with pytest.raises(DatabaseUnavailableError):
        async with pool.acquire():
            pass
