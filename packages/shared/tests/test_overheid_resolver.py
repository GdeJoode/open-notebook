"""Unit tests for the overheid.nl KOOP SRU resolver (V.4) — SRU XML mocked."""

from __future__ import annotations

from typing import Callable

import httpx
import pytest
from shared.references.overheid_resolver import OverheidResolver
from shared.retrieval.cites_matching import ParsedReference
from shared.vocabulary.http_client import VocabularyHTTPClient


def _sru(
    title: str,
    identifier: str,
    *,
    doc_type: str = "Kamerstuk",
    dossiernummer: str | None = None,
    date: str = "2019-09-17",
) -> str:
    """A KOOP SRU record shaped like the live API (verified 2026-07-24).

    Real records date via ``dcterms:date`` (there is no ``issued``) and carry a
    dedicated ``<dossiernummer>`` element separate from the identifier.
    """
    dossier_el = f"<dossiernummer>{dossiernummer}</dossiernummer>" if dossiernummer else ""
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<srw:searchRetrieveResponse
    xmlns:srw="http://docs.oasis-open.org/ns/search-ws/sruResponse">
  <srw:numberOfRecords>1</srw:numberOfRecords>
  <srw:records>
    <srw:record>
      <srw:recordData>
        <gzd xmlns="http://standaarden.overheid.nl/sru">
          <originalData>
            <meta>
              <owmskern xmlns:dcterms="http://purl.org/dc/terms/">
                <dcterms:title>{title}</dcterms:title>
                <dcterms:identifier>{identifier}</dcterms:identifier>
                <dcterms:type>{doc_type}</dcterms:type>
                <dcterms:date>{date}</dcterms:date>
              </owmskern>
              {dossier_el}
            </meta>
          </originalData>
          <enrichedData>
            <preferredUrl>https://zoek.officielebekendmakingen.nl/{identifier}.html</preferredUrl>
          </enrichedData>
        </gzd>
      </srw:recordData>
    </srw:record>
  </srw:records>
</srw:searchRetrieveResponse>
"""


def _resolver(handler: Callable[[httpx.Request], httpx.Response]) -> OverheidResolver:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    http = VocabularyHTTPClient(
        user_agent="test", min_interval=0.0, cache_ttl=0.0, client=client
    )
    return OverheidResolver(http_client=http)


@pytest.mark.asyncio
async def test_dossier_identifier_match_is_strong():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "repository.overheid.nl/sru" in str(request.url)
        return httpx.Response(
            200, text=_sru("Wijziging van de begrotingsstaat", "kst-35300-1")
        )

    resolver = _resolver(handler)
    work = await resolver.resolve(
        ParsedReference(raw_text="Kamerstukken II 2019/20, 35 300, nr. 1")
    )
    assert work is not None
    assert work.source == "overheid"
    assert work.method == "sru"
    assert work.confidence == 0.95
    assert work.external_id == "kst-35300-1"
    assert work.venue == "Kamerstuk"
    assert work.year == 2019
    assert work.external_uri.endswith("kst-35300-1.html")


@pytest.mark.asyncio
async def test_title_overlap_match_when_no_dossier():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, text=_sru("Nationale Omgevingsvisie", "stcrt-2020-1")
        )

    resolver = _resolver(handler)
    work = await resolver.resolve(
        ParsedReference(
            raw_text="Staatscourant: Nationale Omgevingsvisie",
            title="Nationale Omgevingsvisie",
        )
    )
    assert work is not None
    assert work.method == "sru"
    assert work.confidence >= 0.5


@pytest.mark.asyncio
async def test_unrelated_record_rejected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_sru("Iets Heel Anders", "stcrt-2020-9"))

    resolver = _resolver(handler)
    work = await resolver.resolve(
        ParsedReference(
            raw_text="Staatscourant beleidsnota klimaat",
            title="Beleidsnota Klimaat en Energie",
        )
    )
    assert work is None


@pytest.mark.asyncio
async def test_dossier_match_via_dedicated_element():
    # The record's identifier is a bijlage id that does NOT contain the dossier;
    # the match must come from the dedicated <dossiernummer> element.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=_sru(
                "Voortgangsbrief bijlage",
                "blg-1116478",
                doc_type="Bijlage",
                dossiernummer="31305",
            ),
        )

    resolver = _resolver(handler)
    work = await resolver.resolve(ParsedReference(raw_text="Kamerstuk 31305-489"))
    assert work is not None
    assert work.confidence == 0.95
    assert work.external_id == "blg-1116478"  # id kept, dossier matched separately
    assert work.year == 2019  # from dcterms:date, not issued


@pytest.mark.asyncio
async def test_query_uses_all_relation_not_phrase():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["query"] = request.url.params.get("query", "")
        return httpx.Response(200, text=_sru("X", "kst-1-1"))

    resolver = _resolver(handler)
    await resolver.resolve(
        ParsedReference(raw_text="Kamerstukken II 2019/20, 35 300, nr. 1")
    )
    # AND-of-words, not an exact-phrase match (the live recall bug).
    assert seen["query"].startswith('cql.textAndIndexes all "')


@pytest.mark.asyncio
async def test_network_failure_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    resolver = _resolver(handler)
    work = await resolver.resolve(
        ParsedReference(raw_text="Kamerstukken II 2019/20, 35 300, nr. 1")
    )
    assert work is None
