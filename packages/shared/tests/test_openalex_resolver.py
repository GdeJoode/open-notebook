"""Unit tests for the OpenAlex work resolver (V.4) — all HTTP mocked."""

from __future__ import annotations

from typing import Callable

import httpx
import pytest
from shared.references.openalex_resolver import OpenAlexResolver
from shared.retrieval.cites_matching import ParsedReference
from shared.vocabulary.http_client import VocabularyHTTPClient

_WORK = {
    "id": "https://openalex.org/W2741809807",
    "doi": "https://doi.org/10.7717/peerj.4375",
    "display_name": "The state of OA: a large-scale analysis",
    "publication_year": 2018,
    "authorships": [
        {
            "author": {
                "display_name": "Heather Piwowar",
                "orcid": "https://orcid.org/0000-0003-1613-5981",
            },
            "institutions": [
                {"display_name": "Impactstory", "ror": "https://ror.org/0293rh119"}
            ],
        }
    ],
    "primary_location": {"source": {"display_name": "PeerJ"}},
}


def _resolver(handler: Callable[[httpx.Request], httpx.Response]) -> OpenAlexResolver:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    http = VocabularyHTTPClient(
        user_agent="test", min_interval=0.0, cache_ttl=0.0, client=client
    )
    return OpenAlexResolver(http_client=http)


@pytest.mark.asyncio
async def test_doi_path_resolves_and_maps_ids():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "api.openalex.org/works" in str(request.url)
        return httpx.Response(200, json=_WORK)

    resolver = _resolver(handler)
    work = await resolver.resolve(
        ParsedReference(raw_text="x", doi="10.7717/peerj.4375")
    )
    assert work is not None
    assert work.source == "openalex"
    assert work.method == "doi"
    assert work.confidence == 1.0
    assert work.doi == "10.7717/peerj.4375"
    assert work.year == 2018
    assert work.venue == "PeerJ"
    assert work.authors == ("Heather Piwowar",)
    assert work.author_ids == ("https://orcid.org/0000-0003-1613-5981",)
    assert work.institutions == ("Impactstory",)
    assert work.institution_ids == ("https://ror.org/0293rh119",)
    assert work.external_id == "https://openalex.org/W2741809807"


@pytest.mark.asyncio
async def test_title_path_resolves_with_guard():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "search" in str(request.url)
        return httpx.Response(200, json={"results": [_WORK]})

    resolver = _resolver(handler)
    work = await resolver.resolve(
        ParsedReference(
            raw_text="x",
            title="The state of OA: a large-scale analysis",
            authors=("Piwowar",),
        )
    )
    assert work is not None
    assert work.method == "title_author"
    assert work.confidence >= 0.5


@pytest.mark.asyncio
async def test_title_path_generic_near_miss_rejected():
    """An unrelated returned title is dropped by the precision guard."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [_WORK]})

    resolver = _resolver(handler)
    work = await resolver.resolve(
        ParsedReference(raw_text="x", title="Quantum Gravity in Loops", authors=("Rovelli",))
    )
    assert work is None


@pytest.mark.asyncio
async def test_doi_path_mismatch_rejected():
    """A returned work whose DOI is not the one asked for is not trusted."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={**_WORK, "doi": "https://doi.org/10.9/other"})

    resolver = _resolver(handler)
    work = await resolver.resolve(
        ParsedReference(raw_text="x", doi="10.7717/peerj.4375", title="x")
    )
    # DOI leg rejects the mismatch; falls to title path but title 'x' has no
    # overlap → unresolved.
    assert work is None


@pytest.mark.asyncio
async def test_network_failure_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    resolver = _resolver(handler)
    work = await resolver.resolve(ParsedReference(raw_text="x", title="Anything"))
    assert work is None
