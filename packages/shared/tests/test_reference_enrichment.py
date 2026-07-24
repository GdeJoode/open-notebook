"""Unit tests for the opt-in ORCID/ROR enrichment (V.4) — all HTTP mocked."""

from __future__ import annotations

from typing import Callable

import httpx
import pytest
from shared.references.enrichment import ReferenceEnricher
from shared.references.work_resolver import ResolvedWork
from shared.vocabulary.http_client import VocabularyHTTPClient


def _client(handler: Callable[[httpx.Request], httpx.Response]) -> VocabularyHTTPClient:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return VocabularyHTTPClient(
        user_agent="test", min_interval=0.0, cache_ttl=0.0, client=client
    )


def _work(**kw) -> ResolvedWork:
    base = dict(title="A Paper", source="crossref", method="doi", confidence=1.0)
    base.update(kw)
    return ResolvedWork(**base)


@pytest.mark.asyncio
async def test_orcid_attached_when_family_name_matches():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "pub.orcid.org" in str(request.url)
        return httpx.Response(
            200,
            json={
                "expanded-result": [
                    {
                        "orcid-id": "0000-0002-1825-0097",
                        "given-names": "Josiah",
                        "family-names": "Carberry",
                    }
                ]
            },
        )

    enricher = ReferenceEnricher(
        orcid_client=_client(handler), enable_ror=False
    )
    work = _work(authors=("Josiah Carberry",))
    enriched = await enricher.enrich(work)
    assert enriched.author_ids == ("0000-0002-1825-0097",)


@pytest.mark.asyncio
async def test_orcid_rejected_when_family_name_does_not_match():
    """A returned ORCID whose family name is not in the author is not trusted."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "expanded-result": [
                    {"orcid-id": "0000-0002-9999-9999", "family-names": "Einstein"}
                ]
            },
        )

    enricher = ReferenceEnricher(orcid_client=_client(handler), enable_ror=False)
    work = _work(authors=("Josiah Carberry",))
    enriched = await enricher.enrich(work)
    assert enriched.author_ids == ()


@pytest.mark.asyncio
async def test_ror_attached_only_when_chosen():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "api.ror.org" in str(request.url)
        return httpx.Response(
            200,
            json={
                "items": [
                    {
                        "chosen": False,
                        "organization": {"id": "https://ror.org/aaa", "name": "Other"},
                    },
                    {
                        "chosen": True,
                        "organization": {
                            "id": "https://ror.org/02jx3x895",
                            "name": "Utrecht University",
                        },
                    },
                ]
            },
        )

    enricher = ReferenceEnricher(ror_client=_client(handler), enable_orcid=False)
    work = _work(institutions=("Utrecht University",))
    enriched = await enricher.enrich(work)
    assert enriched.institution_ids == ("https://ror.org/02jx3x895",)


@pytest.mark.asyncio
async def test_work_with_existing_ids_is_left_untouched():
    called = {"hit": False}

    def handler(request: httpx.Request) -> httpx.Response:
        called["hit"] = True
        return httpx.Response(200, json={})

    enricher = ReferenceEnricher(
        orcid_client=_client(handler), ror_client=_client(handler)
    )
    work = _work(authors=("X",), author_ids=("0000-0000-0000-0002",))
    enriched = await enricher.enrich(work)
    assert enriched is work  # nothing to do → same object, no network call
    assert called["hit"] is False


@pytest.mark.asyncio
async def test_enrichment_failure_leaves_work_unchanged():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    enricher = ReferenceEnricher(
        orcid_client=_client(handler), ror_client=_client(handler)
    )
    work = _work(authors=("Josiah Carberry",), institutions=("Utrecht University",))
    enriched = await enricher.enrich(work)
    assert enriched.author_ids == ()
    assert enriched.institution_ids == ()
