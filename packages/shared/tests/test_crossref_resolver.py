"""Unit tests for the Crossref WORK resolver (V.4) — all HTTP mocked.

Distinct from ``test_crossref_provider.py`` (the K.4 entity→DOI provider); this
covers the whole-reference → :class:`ResolvedWork` resolver.
"""

from __future__ import annotations

from typing import Callable

import httpx
import pytest
from shared.references.crossref_resolver import CrossrefResolver
from shared.retrieval.cites_matching import ParsedReference
from shared.vocabulary.http_client import VocabularyHTTPClient

_WORK = {
    "DOI": "10.1257/aer.91.5.1369",
    "title": ["The Colonial Origins of Comparative Development"],
    "author": [
        {"given": "Daron", "family": "Acemoglu"},
        {"given": "Simon", "family": "Johnson"},
        {"given": "James A.", "family": "Robinson"},
    ],
    "container-title": ["American Economic Review"],
    "issued": {"date-parts": [[2001, 12]]},
}


def _resolver(handler: Callable[[httpx.Request], httpx.Response]) -> CrossrefResolver:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    http = VocabularyHTTPClient(
        user_agent="test", min_interval=0.0, cache_ttl=0.0, client=client
    )
    return CrossrefResolver(http_client=http)


@pytest.mark.asyncio
async def test_doi_exact_resolves():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "api.crossref.org/works" in str(request.url)
        return httpx.Response(200, json={"message": _WORK})

    resolver = _resolver(handler)
    work = await resolver.resolve(
        ParsedReference(raw_text="x", doi="10.1257/aer.91.5.1369")
    )
    assert work is not None
    assert work.method == "doi"
    assert work.confidence == 1.0
    assert work.doi == "10.1257/aer.91.5.1369"
    assert work.year == 2001
    assert work.venue == "American Economic Review"
    assert work.authors[0] == "Daron Acemoglu"
    assert work.external_uri == "https://doi.org/10.1257/aer.91.5.1369"


@pytest.mark.asyncio
async def test_title_author_query_resolves():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "query.bibliographic" in str(request.url)
        return httpx.Response(200, json={"message": {"items": [_WORK]}})

    resolver = _resolver(handler)
    work = await resolver.resolve(
        ParsedReference(
            raw_text="x",
            title="The Colonial Origins of Comparative Development",
            authors=("Acemoglu, D.",),
        )
    )
    assert work is not None
    assert work.method == "title_author"
    assert work.confidence >= 0.5


@pytest.mark.asyncio
async def test_title_query_generic_near_miss_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"items": [_WORK]}})

    resolver = _resolver(handler)
    work = await resolver.resolve(
        ParsedReference(raw_text="x", title="An Introduction", authors=("Smith",))
    )
    assert work is None


@pytest.mark.asyncio
async def test_network_failure_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    resolver = _resolver(handler)
    work = await resolver.resolve(
        ParsedReference(raw_text="x", doi="10.1257/aer.91.5.1369")
    )
    assert work is None
