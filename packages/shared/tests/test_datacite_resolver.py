"""Unit tests for the DataCite work resolver (V.4) — all HTTP mocked."""

from __future__ import annotations

from typing import Callable

import httpx
import pytest
from shared.references.datacite_resolver import DataCiteResolver
from shared.retrieval.cites_matching import ParsedReference
from shared.vocabulary.http_client import VocabularyHTTPClient

_RESPONSE = {
    "data": {
        "id": "10.5281/zenodo.1234567",
        "attributes": {
            "doi": "10.5281/zenodo.1234567",
            "titles": [{"title": "A Reusable Dataset of Policy Documents"}],
            "creators": [
                {"name": "De Jonge, Anna", "givenName": "Anna", "familyName": "De Jonge"},
                {"givenName": "Bram", "familyName": "Visser"},
            ],
            "publicationYear": 2022,
            "publisher": "Zenodo",
        },
    }
}


def _resolver(handler: Callable[[httpx.Request], httpx.Response]) -> DataCiteResolver:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    http = VocabularyHTTPClient(
        user_agent="test", min_interval=0.0, cache_ttl=0.0, client=client
    )
    return DataCiteResolver(http_client=http)


@pytest.mark.asyncio
async def test_doi_resolves_dataset():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "api.datacite.org/dois" in str(request.url)
        return httpx.Response(200, json=_RESPONSE)

    resolver = _resolver(handler)
    work = await resolver.resolve(
        ParsedReference(raw_text="x", doi="10.5281/zenodo.1234567")
    )
    assert work is not None
    assert work.method == "doi"
    assert work.confidence == 1.0
    assert work.title == "A Reusable Dataset of Policy Documents"
    assert work.year == 2022
    assert work.venue == "Zenodo"
    assert work.authors == ("De Jonge, Anna", "Bram Visser")
    assert work.external_uri == "https://doi.org/10.5281/zenodo.1234567"


@pytest.mark.asyncio
async def test_no_doi_reference_is_skipped():
    called = {"hit": False}

    def handler(request: httpx.Request) -> httpx.Response:
        called["hit"] = True
        return httpx.Response(200, json=_RESPONSE)

    resolver = _resolver(handler)
    work = await resolver.resolve(ParsedReference(raw_text="x", title="No DOI here"))
    assert work is None
    assert called["hit"] is False


@pytest.mark.asyncio
async def test_doi_mismatch_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        resp = {
            "data": {
                "id": "10.9/other",
                "attributes": {
                    **_RESPONSE["data"]["attributes"],
                    "doi": "10.9/other",
                },
            }
        }
        return httpx.Response(200, json=resp)

    resolver = _resolver(handler)
    work = await resolver.resolve(
        ParsedReference(raw_text="x", doi="10.5281/zenodo.1234567")
    )
    assert work is None


@pytest.mark.asyncio
async def test_network_failure_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    resolver = _resolver(handler)
    work = await resolver.resolve(
        ParsedReference(raw_text="x", doi="10.5281/zenodo.1234567")
    )
    assert work is None
