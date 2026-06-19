"""Tests for per-IP rate limiting (Track I, Phase I.H1).

Covers AC3 (burst over RATE_LIMIT_RPM from one IP → 429 with Retry-After)
and AC5 (the limit is per-IP, so a second IP is unaffected).

slowapi keys production buckets on `get_remote_address` (i.e. the client
host). Starlette's `TestClient` pins the client host to a single value for
every request it sends, which makes "two distinct IPs" awkward to express
through the normal client. To exercise the per-key isolation faithfully we
build a test app whose limiter keys on a request header standing in for the
remote address; the bucketing logic under test (one key trips, another does
not) is identical regardless of where the key string comes from. A separate
assertion pins the *production* limiter to `get_remote_address` so the
per-IP contract (AC5) cannot silently regress to a per-process key.
"""

from __future__ import annotations

import pytest
from app_main.api.rate_limit import limiter as production_limiter
from fastapi import FastAPI, Request, Response
from fastapi.testclient import TestClient
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address


def _ip_from_header(request: Request) -> str:
    """Stand-in key func: read the simulated client IP from a header."""
    return request.headers.get("X-Test-IP", "0.0.0.0")


def _make_client(limit_per_minute: int) -> TestClient:
    """Build an app with a header-keyed limiter and a single guarded route."""
    # No `default_limits` here: the per-route `@limit` decorator below is the
    # sole limit, so a request consumes the bucket exactly once. Combining a
    # global default with an identical per-route limit would double-count.
    test_limiter = Limiter(
        key_func=_ip_from_header,
        headers_enabled=True,
    )
    app = FastAPI()
    app.state.limiter = test_limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    app.add_middleware(SlowAPIMiddleware)

    @app.post("/sources/")
    @test_limiter.limit(lambda: f"{limit_per_minute}/minute")
    async def _upload(
        request: Request, response: Response
    ):  # noqa: ANN202 — test handler
        # `response` mirrors the production endpoint: with
        # `headers_enabled=True` slowapi injects rate-limit headers into it
        # on the success path.
        return {"ok": True}

    return TestClient(app)


def test_burst_from_one_ip_trips_429_with_retry_after():
    """More than the limit from a single IP → 429 carrying Retry-After."""
    client = _make_client(limit_per_minute=3)
    # Unique IP per test: slowapi's default in-memory store is keyed by
    # (limit, key) at process scope, so a shared IP string would let one
    # test's hits bleed into another within the same per-minute window.
    headers = {"X-Test-IP": "10.1.0.1"}

    statuses = [
        client.post("/sources/", headers=headers).status_code for _ in range(4)
    ]

    assert statuses[:3] == [200, 200, 200]
    assert statuses[3] == 429

    tripped = client.post("/sources/", headers=headers)
    assert tripped.status_code == 429
    assert "Retry-After" in tripped.headers


def test_second_ip_is_unaffected():
    """A second IP gets its own bucket — not blocked by the first IP's burst."""
    client = _make_client(limit_per_minute=2)

    ip_a = {"X-Test-IP": "10.2.0.1"}
    ip_b = {"X-Test-IP": "10.2.0.2"}

    # Exhaust IP A.
    assert client.post("/sources/", headers=ip_a).status_code == 200
    assert client.post("/sources/", headers=ip_a).status_code == 200
    assert client.post("/sources/", headers=ip_a).status_code == 429

    # IP B still has a full bucket.
    assert client.post("/sources/", headers=ip_b).status_code == 200
    assert client.post("/sources/", headers=ip_b).status_code == 200
    assert client.post("/sources/", headers=ip_b).status_code == 429


def test_production_limiter_keyed_per_ip():
    """AC5 guard: production limiter keys on remote address, not per-process."""
    assert production_limiter._key_func is get_remote_address
