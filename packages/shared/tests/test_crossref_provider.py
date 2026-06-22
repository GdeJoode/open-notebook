"""Unit tests for the Crossref vocabulary provider (K.4).

ALL HTTP IS MOCKED — no live network in CI. We inject an ``httpx.AsyncClient``
backed by ``httpx.MockTransport`` into the provider's HTTP client, so the tests
exercise the real request-building / parsing / scoring / fail-soft logic without
touching the network.
"""

from __future__ import annotations

from typing import Callable

import httpx
import pytest
from shared.vocabulary.crossref_provider import CrossrefProvider
from shared.vocabulary.http_client import VocabularyHTTPClient


def _provider_with_handler(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    min_score: float = 70.0,
) -> CrossrefProvider:
    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    http = VocabularyHTTPClient(
        user_agent="open-notebook/0.1 (mailto:test@example.com)",
        min_interval=0.0,
        cache_ttl=0.0,
        client=client,
    )
    return CrossrefProvider(http_client=http, min_score=min_score)


def _works_response(items: list) -> httpx.Response:
    return httpx.Response(200, json={"message": {"items": items}})


@pytest.mark.asyncio
async def test_lookup_returns_doi_for_resolvable_title():
    """AC4: a scholarly_article with a resolvable title returns a DOI URI."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert "api.crossref.org/works" in str(request.url)
        # Polite-pool User-Agent carries a contact.
        assert "mailto:" in request.headers["user-agent"]
        return _works_response(
            [
                {
                    "DOI": "10.1145/3292500.3330701",
                    "title": ["Attention Is All You Need"],
                    "score": 95.0,
                }
            ]
        )

    provider = _provider_with_handler(handler)
    matches = await provider.lookup("Attention Is All You Need", "scholarly_article")

    assert len(matches) == 1
    assert matches[0].external_uri == "https://doi.org/10.1145/3292500.3330701"
    assert matches[0].external_id == "10.1145/3292500.3330701"
    assert matches[0].source_vocabulary == "crossref"
    assert matches[0].confidence > 0.9


@pytest.mark.asyncio
async def test_non_scholarly_type_skips_network():
    """A government org is never sent to Crossref (no DOI hallucination)."""
    called = {"hit": False}

    def handler(request: httpx.Request) -> httpx.Response:
        called["hit"] = True
        return _works_response([])

    provider = _provider_with_handler(handler)
    matches = await provider.lookup("Binnenlandse Zaken", "organization")

    assert matches == []
    assert called["hit"] is False


@pytest.mark.asyncio
async def test_low_score_hit_is_dropped():
    """A weak relevance score is below the floor → not a confident match."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _works_response(
            [{"DOI": "10.1/weak", "title": ["Loosely Related"], "score": 12.0}]
        )

    provider = _provider_with_handler(handler, min_score=70.0)
    matches = await provider.lookup("Some Title", "paper")
    assert matches == []


@pytest.mark.asyncio
async def test_high_score_but_unrelated_title_is_dropped():
    """K.4 rev2 title-gate: a strong score on an unrelated title is rejected.

    Crossref relevance is query-length dependent; a short/generic query can
    score above the floor against a wholly different paper. The title-overlap
    gate must drop it so the single-candidate auto-link never links the wrong
    DOI (the recurring Track-K over-link concern).
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return _works_response(
            [
                {
                    "DOI": "10.1/unrelated",
                    "title": ["Quantum Chromodynamics on the Lattice"],
                    "score": 95.0,
                }
            ]
        )

    provider = _provider_with_handler(handler)
    matches = await provider.lookup("Graph Neural Networks", "scholarly_article")
    assert matches == [], (
        "high-score but lexically-unrelated title should be dropped by the "
        "title-overlap gate"
    )


@pytest.mark.asyncio
async def test_partial_title_overlap_survives_gate():
    """A title sharing most tokens with the query clears the overlap floor."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _works_response(
            [
                {
                    "DOI": "10.1/rel",
                    "title": ["Graph Neural Networks for Recommendation"],
                    "score": 92.0,
                }
            ]
        )

    provider = _provider_with_handler(handler)
    matches = await provider.lookup("Graph Neural Networks", "scholarly_article")
    assert len(matches) == 1
    assert matches[0].external_id == "10.1/rel"


@pytest.mark.asyncio
async def test_two_strong_hits_surface_both():
    """Two equally-strong hits both return → the reconciler will refuse to link."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _works_response(
            [
                {"DOI": "10.1/a", "title": ["Graph Neural Networks"], "score": 90.0},
                {"DOI": "10.1/b", "title": ["Graph Neural Networks"], "score": 89.0},
            ]
        )

    provider = _provider_with_handler(handler)
    matches = await provider.lookup("Graph Neural Networks", "scholarly_article")
    assert len(matches) == 2


@pytest.mark.asyncio
async def test_network_failure_returns_no_match():
    """A transport error degrades to [] — never raises (fail-soft)."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    provider = _provider_with_handler(handler)
    matches = await provider.lookup("Anything", "scholarly_article")
    assert matches == []


@pytest.mark.asyncio
async def test_http_500_returns_no_match():
    """A 5xx is fail-soft too."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    provider = _provider_with_handler(handler)
    matches = await provider.lookup("Anything", "scholarly_article")
    assert matches == []


@pytest.mark.asyncio
async def test_refresh_is_noop():
    """Crossref is on-demand; refresh upserts nothing."""
    provider = _provider_with_handler(lambda r: _works_response([]))
    assert await provider.refresh() == 0


@pytest.mark.asyncio
async def test_response_is_cached():
    """A repeated identical query hits the network once (cache)."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return _works_response(
            [{"DOI": "10.1/x", "title": ["Cached"], "score": 95.0}]
        )

    transport = httpx.MockTransport(handler)
    client = httpx.AsyncClient(transport=transport)
    http = VocabularyHTTPClient(
        user_agent="test", min_interval=0.0, cache_ttl=3600.0, client=client
    )
    provider = CrossrefProvider(http_client=http)

    await provider.lookup("Cached", "paper")
    await provider.lookup("Cached", "paper")
    assert calls["n"] == 1
