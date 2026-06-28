"""Unit tests for ``RerankerHttpClient`` (transport layer, service mocked)."""

from __future__ import annotations

import httpx
import pytest

from app_main.services.reranker_http_client import (
    DEFAULT_SERVICE_URL,
    RerankerHttpClient,
    RerankerServiceError,
)


def test_service_url_defaults(monkeypatch):
    monkeypatch.delenv("RERANKER_SERVICE_URL", raising=False)
    assert RerankerHttpClient().service_url == DEFAULT_SERVICE_URL


def test_service_url_from_env(monkeypatch):
    monkeypatch.setenv("RERANKER_SERVICE_URL", "http://host:9000/")
    # Trailing slash stripped.
    assert RerankerHttpClient().service_url == "http://host:9000"


def test_explicit_url_overrides_env(monkeypatch):
    monkeypatch.setenv("RERANKER_SERVICE_URL", "http://env:1/")
    assert RerankerHttpClient("http://explicit:2").service_url == "http://explicit:2"


async def test_empty_passages_short_circuits(monkeypatch):
    # No HTTP call should happen for empty input.
    def fail(*a, **k):  # pragma: no cover
        raise AssertionError("should not call httpx for empty passages")

    monkeypatch.setattr(httpx, "AsyncClient", fail)
    out = await RerankerHttpClient().rerank("q", [])
    assert out == []


async def test_rerank_parses_results(monkeypatch):
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        captured["payload"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"results": [{"index": 1, "score": 0.9}, {"index": 0, "score": 0.2}]},
        )

    transport = httpx.MockTransport(handler)

    # Patch AsyncClient to use the mock transport.
    real_client = httpx.AsyncClient

    def make_client(*a, **k):
        k.pop("timeout", None)
        return real_client(transport=transport)

    monkeypatch.setattr(httpx, "AsyncClient", make_client)

    out = await RerankerHttpClient("http://r:8105").rerank("q", ["p0", "p1"])
    assert out == [(1, 0.9), (0, 0.2)]
    assert captured["payload"]["query"] == "q"
    assert captured["payload"]["passages"] == ["p0", "p1"]


async def test_http_error_raises_service_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "model unavailable"})

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def make_client(*a, **k):
        k.pop("timeout", None)
        return real_client(transport=transport)

    monkeypatch.setattr(httpx, "AsyncClient", make_client)

    with pytest.raises(RerankerServiceError):
        await RerankerHttpClient("http://r:8105").rerank("q", ["p"])


async def test_connect_error_raises_service_error(monkeypatch):
    def handler(request: httpx.Request) -> httpx.Response:  # pragma: no cover
        raise httpx.ConnectError("refused")

    transport = httpx.MockTransport(handler)
    real_client = httpx.AsyncClient

    def make_client(*a, **k):
        k.pop("timeout", None)
        return real_client(transport=transport)

    monkeypatch.setattr(httpx, "AsyncClient", make_client)

    with pytest.raises(RerankerServiceError):
        await RerankerHttpClient("http://r:8105").rerank("q", ["p"])
