"""Unit tests for the RePEc/CitEc resolver (V.4).

Covers the two paths that matter: the CONFIG-GATED silent skip when no access
code is set (the primary requirement — it must never raise or block), and a real
guarded lookup when a code is configured.
"""

from __future__ import annotations

from typing import Callable

import httpx
import pytest
from shared.references.repec_resolver import _REPEC_ENV_VAR, RePEcResolver
from shared.retrieval.cites_matching import ParsedReference
from shared.vocabulary.http_client import VocabularyHTTPClient

_ITEM = {
    "handle": "RePEc:nbr:nberwo:12345",
    "title": "Trade and the Location of Industry",
    "authors": [{"name": "Paul Krugman"}],
    "year": 1996,
    "series": "NBER Working Papers",
}


def _configured_resolver(
    handler: Callable[[httpx.Request], httpx.Response],
) -> RePEcResolver:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    http = VocabularyHTTPClient(
        user_agent="test", min_interval=0.0, cache_ttl=0.0, client=client
    )
    return RePEcResolver(api_key="test-code", http_client=http)


@pytest.mark.asyncio
async def test_unconfigured_is_unavailable_and_skips(monkeypatch):
    """No access code → unavailable, resolve() returns None without raising."""
    monkeypatch.delenv(_REPEC_ENV_VAR, raising=False)
    resolver = RePEcResolver()  # no api_key, env unset
    assert resolver.available is False
    result = await resolver.resolve(
        ParsedReference(raw_text="NBER Working Paper 12345", title="Trade and Location")
    )
    assert result is None


@pytest.mark.asyncio
async def test_empty_env_code_is_unavailable(monkeypatch):
    monkeypatch.setenv(_REPEC_ENV_VAR, "   ")
    resolver = RePEcResolver()
    assert resolver.available is False
    assert await resolver.resolve(ParsedReference(raw_text="x", title="y")) is None


@pytest.mark.asyncio
async def test_env_code_makes_it_available(monkeypatch):
    monkeypatch.setenv(_REPEC_ENV_VAR, "code-from-env")
    resolver = RePEcResolver()
    assert resolver.available is True


@pytest.mark.asyncio
async def test_configured_resolver_resolves_with_guard():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "citec.repec.org" in str(request.url)
        assert "code=test-code" in str(request.url)
        return httpx.Response(200, json={"results": [_ITEM]})

    resolver = _configured_resolver(handler)
    work = await resolver.resolve(
        ParsedReference(
            raw_text="Krugman, NBER WP",
            title="Trade and the Location of Industry",
            authors=("Krugman",),
        )
    )
    assert work is not None
    assert work.source == "repec"
    assert work.method == "title_author"
    assert work.external_id == "RePEc:nbr:nberwo:12345"
    assert work.year == 1996


@pytest.mark.asyncio
async def test_configured_generic_near_miss_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [_ITEM]})

    resolver = _configured_resolver(handler)
    work = await resolver.resolve(
        ParsedReference(raw_text="x", title="Some Other Paper", authors=("Nobody",))
    )
    assert work is None
