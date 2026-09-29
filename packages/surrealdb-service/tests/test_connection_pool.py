"""ConnectionPool failure contract (WB.0).

The live API once hung for good: SurrealDB sign-ins were slow for a while, the
2 s health check cancelled pool acquires mid-sign-in, and every cancelled or
failed sign-in leaked a pool slot. Once all slots had leaked, every request
waited forever on an empty queue, long after the database was fast again.

``FakeSurreal`` follows the SDK's ``AsyncSurreal`` surface the pool uses
(``signin``/``use``/``query``/``close``) and lets a test make sign-in raise or
hang, and a query raise or hang, the way the real client does.
"""

import asyncio
from typing import Any, List, Optional

import pytest

from surrealdb_service import connection as conn_mod
from surrealdb_service.config import SurrealDBConfig
from surrealdb_service.connection import ConnectionPool, DatabaseUnavailableError

# A test that would hang on a broken pool fails at this bound instead.
GUARD = 3.0


class FakeServer:
    """What the fake clients talk to; tests flip its behaviour."""

    def __init__(self) -> None:
        self.signin_mode = "ok"  # "ok" | "raise" | "hang"
        self.query_mode = "ok"  # "ok" | "hang"
        self.clients: List["FakeSurreal"] = []


class FakeSurreal:
    def __init__(self, server: FakeServer, url: str) -> None:
        self.server = server
        self.url = url
        self.closed = False
        self.signed_in = False
        server.clients.append(self)

    async def signin(self, vars: dict) -> str:
        if self.server.signin_mode == "raise":
            raise OSError("connection refused")
        if self.server.signin_mode == "hang":
            await asyncio.Event().wait()
        self.signed_in = True
        return "token"

    async def use(self, namespace: str, database: str) -> None:
        return None

    async def query(self, query: str, vars: Optional[dict] = None) -> Any:
        if self.closed:
            raise OSError("socket closed")
        if self.server.query_mode == "hang":
            await asyncio.Event().wait()
        return [{"result": 1}]

    async def close(self) -> None:
        self.closed = True


@pytest.fixture
def server(monkeypatch) -> FakeServer:
    srv = FakeServer()
    monkeypatch.setattr(
        conn_mod, "AsyncSurreal", lambda url: FakeSurreal(srv, url)
    )
    return srv


def _pool(**kwargs) -> ConnectionPool:
    cfg = SurrealDBConfig(url="ws://fake:1/rpc")
    return ConnectionPool(config=cfg, **kwargs)


async def _use_once(pool: ConnectionPool) -> FakeSurreal:
    async with pool.acquire() as c:
        await c.query("RETURN 1")
        return c


async def test_failed_signins_do_not_leak_slots(server):
    pool = _pool(max_size=2)
    server.signin_mode = "raise"
    for _ in range(5):
        with pytest.raises(OSError):
            await asyncio.wait_for(_use_once(pool), GUARD)

    server.signin_mode = "ok"
    c = await asyncio.wait_for(_use_once(pool), GUARD)
    assert c.signed_in


async def test_cancelled_signins_do_not_leak_slots(server):
    """The production path: a caller's timeout cancels a slow sign-in."""
    pool = _pool(max_size=2)
    server.signin_mode = "hang"
    for _ in range(5):
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(_use_once(pool), 0.05)
    # The abandoned half-open clients are closed, not left dangling.
    assert all(c.closed for c in server.clients)

    server.signin_mode = "ok"
    c = await asyncio.wait_for(_use_once(pool), GUARD)
    assert c.signed_in


async def test_hanging_signin_is_bounded(server):
    pool = _pool(max_size=1, connect_timeout=0.05)
    server.signin_mode = "hang"
    with pytest.raises(DatabaseUnavailableError):
        await asyncio.wait_for(_use_once(pool), GUARD)

    server.signin_mode = "ok"
    await asyncio.wait_for(_use_once(pool), GUARD)


async def test_exhausted_pool_acquire_is_bounded(server):
    pool = _pool(max_size=1, acquire_timeout=0.05)
    async with pool.acquire():
        with pytest.raises(DatabaseUnavailableError):
            await asyncio.wait_for(_use_once(pool), GUARD)
    # The holder's connection went back and is reused.
    await asyncio.wait_for(_use_once(pool), GUARD)
    assert len(server.clients) == 1


async def test_waiter_gets_the_slot_a_broken_connection_frees(server):
    pool = _pool(max_size=1, acquire_timeout=GUARD)
    holder_in = asyncio.Event()
    release_holder = asyncio.Event()

    async def holder() -> None:
        async with pool.acquire():
            holder_in.set()
            await release_holder.wait()
            raise OSError("connection reset")

    h = asyncio.create_task(holder())
    await holder_in.wait()
    waiter = asyncio.create_task(_use_once(pool))
    await asyncio.sleep(0)
    release_holder.set()
    with pytest.raises(OSError):
        await h
    c = await asyncio.wait_for(waiter, GUARD)
    assert c is not server.clients[0]


async def test_broken_connection_is_not_returned(server):
    pool = _pool(max_size=2)
    with pytest.raises(OSError):
        async with pool.acquire() as first:
            raise OSError("connection reset")
    assert first.closed

    second = await asyncio.wait_for(_use_once(pool), GUARD)
    assert second is not first


async def test_cancelled_request_does_not_return_its_connection(server):
    pool = _pool(max_size=2)
    server.query_mode = "hang"
    first: List[FakeSurreal] = []

    async def request() -> None:
        async with pool.acquire() as c:
            first.append(c)
            await c.query("SELECT * FROM slow")

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(request(), 0.05)
    assert first[0].closed

    server.query_mode = "ok"
    second = await asyncio.wait_for(_use_once(pool), GUARD)
    assert second is not first[0]


async def test_query_error_keeps_the_connection(server):
    """A statement the server rejected leaves the socket fine: reuse it."""
    pool = _pool(max_size=2)
    with pytest.raises(Exception, match="parse error"):
        async with pool.acquire() as first:
            raise Exception("There was a problem with the database: parse error")
    assert not first.closed

    second = await asyncio.wait_for(_use_once(pool), GUARD)
    assert second is first
    assert len(server.clients) == 1


async def test_hanging_ping_on_reuse_is_bounded(server):
    pool = _pool(max_size=2, ping_timeout=0.05)
    first = await asyncio.wait_for(_use_once(pool), GUARD)

    server.query_mode = "hang"
    # Reuse pings the idle connection; the ping hangs, so the pool drops it
    # and signs in a fresh one rather than waiting on it.
    async with pool.acquire() as c:
        pass
    assert c is not first
    assert first.closed
