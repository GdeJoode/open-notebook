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
from typing import Any, Callable, List, Optional

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
        self.close_mode = "ok"  # "ok" | "hang"
        self.clients: List["FakeSurreal"] = []


class FakeSurreal:
    def __init__(self, server: FakeServer, url: str) -> None:
        self.server = server
        self.url = url
        self.closed = False
        self.signed_in = False
        # The SDK's receive task, which answers every request. It ends when
        # the socket is lost; SDK 1.0.6 then clears its request map, so a
        # request fails with a ``KeyError`` on its own id.
        self.recv_task: "asyncio.Future[None]" = (
            asyncio.get_running_loop().create_future()
        )
        server.clients.append(self)

    def lose_socket(self) -> None:
        self.recv_task.set_result(None)

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
        if self.recv_task.done():
            raise KeyError("request-id")
        if self.server.query_mode == "hang":
            await asyncio.Event().wait()
        return [{"result": 1}]

    async def close(self) -> None:
        if self.server.close_mode == "hang":
            await asyncio.Event().wait()
        self.closed = True
        if not self.recv_task.done():
            self.recv_task.cancel()


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


async def _until(condition: Callable[[], bool]) -> None:
    while not condition():
        await asyncio.sleep(0.01)


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
    pool = _pool(max_size=2, connect_timeout=0.2)
    server.signin_mode = "hang"
    for _ in range(5):
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(_use_once(pool), 0.05)
    # The abandoned half-open clients are closed once their sign-in is given
    # up on (``connect_timeout``), not left dangling.
    await asyncio.wait_for(_until(lambda: all(c.closed for c in server.clients)), GUARD)

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

    async def acquire_only() -> FakeSurreal:
        async with pool.acquire() as c:
            return c

    c = await asyncio.wait_for(acquire_only(), GUARD)
    assert c is not first
    assert first.closed


async def test_checkout_cancelled_during_reuse_ping_closes_the_candidate(server):
    pool = _pool(max_size=2, ping_timeout=GUARD)
    first = await asyncio.wait_for(_use_once(pool), GUARD)

    server.query_mode = "hang"
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(asyncio.wait_for(_use_once(pool), 0.05), GUARD)

    assert first.closed
    assert pool._pool.empty()
    # The cancellation stopped the checkout; it did not go on to sign in.
    assert len(server.clients) == 1


async def test_idle_connection_with_a_lost_socket_is_replaced(server):
    pool = _pool(max_size=2)
    first = await asyncio.wait_for(_use_once(pool), GUARD)

    first.lose_socket()
    second = await asyncio.wait_for(_use_once(pool), GUARD)

    assert second is not first
    assert first.closed


async def test_whole_checkout_shares_one_deadline_slow_sign_in(server):
    pool = _pool(max_size=1, connect_timeout=GUARD * 2, acquire_timeout=0.05)
    server.signin_mode = "hang"
    with pytest.raises(DatabaseUnavailableError):
        await asyncio.wait_for(_use_once(pool), GUARD)

    server.signin_mode = "ok"
    await asyncio.wait_for(_use_once(pool), GUARD)


async def test_whole_checkout_shares_one_deadline_slow_ping(server):
    pool = _pool(max_size=1, ping_timeout=GUARD * 2, acquire_timeout=0.05)
    first = await asyncio.wait_for(_use_once(pool), GUARD)

    server.query_mode = "hang"
    with pytest.raises(DatabaseUnavailableError):
        await asyncio.wait_for(_use_once(pool), GUARD)
    assert first.closed


async def test_close_all_bounds_each_close(server):
    pool = _pool(max_size=2)
    pool._close_timeout = 0.05
    async with pool.acquire(), pool.acquire():
        pass
    server.close_mode = "hang"

    await asyncio.wait_for(pool.close_all(), GUARD)
    assert pool._pool.empty()


async def test_connection_returned_after_close_all_is_closed(server):
    pool = _pool(max_size=2)
    async with pool.acquire() as held:
        await pool.close_all()
    assert held.closed
    assert pool._pool.empty()


async def test_socket_lost_during_a_query_is_unavailable(server):
    pool = _pool(max_size=2)
    with pytest.raises(DatabaseUnavailableError):
        async with pool.acquire() as c:
            c.lose_socket()
            await conn_mod._request(c, c.query("SELECT * FROM source"))
    assert c.closed
    await asyncio.wait_for(_use_once(pool), GUARD)


async def test_a_sign_in_that_ends_after_its_deadline_is_closed(server, monkeypatch):
    """A sign-in is shielded, so it goes on after the caller gave up on it; the
    connection it opens is closed when it ends, not pooled or left open."""
    gate = asyncio.Event()
    signin = FakeSurreal.signin

    async def slow_signin(self: FakeSurreal, vars: dict) -> str:
        await gate.wait()
        return await signin(self, vars)

    monkeypatch.setattr(FakeSurreal, "signin", slow_signin)
    pool = _pool(max_size=1, connect_timeout=GUARD)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_use_once(pool), 0.05)
    abandoned = server.clients[0]
    assert not abandoned.closed  # still signing in
    gate.set()
    await asyncio.wait_for(_until(lambda: abandoned.closed), GUARD)
    assert abandoned.signed_in and pool._pool.empty()


async def test_close_all_closes_a_sign_in_still_running(server):
    pool = _pool(max_size=1, connect_timeout=GUARD * 10)
    server.signin_mode = "hang"
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(_use_once(pool), 0.05)

    await asyncio.wait_for(pool.close_all(), GUARD)
    assert server.clients[0].closed


async def test_request_the_sdk_cancelled_is_unavailable_not_a_cancel(server):
    """Not what 1.0.6 does (it raises KeyError), but what an SDK that cancels
    pending requests on a lost socket would: nobody cancelled us."""

    async def cancelled_by_the_sdk() -> None:
        raise asyncio.CancelledError()

    pool = _pool(max_size=1)
    with pytest.raises(DatabaseUnavailableError):
        async with pool.acquire() as c:
            await conn_mod._request(c, cancelled_by_the_sdk())
    assert c.closed


async def test_client_without_a_receive_task_is_not_pooled(server):
    pool = _pool(max_size=1)
    async with pool.acquire() as first:
        del first.recv_task
    assert first.closed and pool._pool.empty()


async def test_our_cancel_stays_a_cancel_when_the_request_ends_cancelled_too():
    """Caller and request cancelled in one turn (loop shutdown cancels both):
    the caller was cancelled, so it gets CancelledError, not unavailable."""
    request = asyncio.ensure_future(asyncio.Event().wait())
    caller = asyncio.ensure_future(conn_mod._request(object(), request))
    await asyncio.sleep(0)
    request.cancel()
    caller.cancel()
    with pytest.raises(asyncio.CancelledError):
        await caller
