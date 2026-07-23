"""Unit tests for the arXiv work resolver (V.4) — Atom XML mocked."""

from __future__ import annotations

from typing import Callable

import httpx
import pytest
from shared.references.arxiv_resolver import ArxivResolver
from shared.retrieval.cites_matching import ParsedReference
from shared.vocabulary.http_client import VocabularyHTTPClient

_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"
      xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/1706.03762v5</id>
    <published>2017-06-12T17:57:34Z</published>
    <title>Attention Is All You Need</title>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <arxiv:doi>10.5555/3295222.3295349</arxiv:doi>
    <arxiv:journal_ref>NeurIPS 2017</arxiv:journal_ref>
  </entry>
</feed>
"""

_ERROR_ATOM = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/api/errors#incorrect_id_format</id>
    <title>Error</title>
  </entry>
</feed>
"""


def _resolver(handler: Callable[[httpx.Request], httpx.Response]) -> ArxivResolver:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    http = VocabularyHTTPClient(
        user_agent="test", min_interval=0.0, cache_ttl=0.0, client=client
    )
    return ArxivResolver(http_client=http)


@pytest.mark.asyncio
async def test_arxiv_id_resolves():
    def handler(request: httpx.Request) -> httpx.Response:
        assert "export.arxiv.org/api/query" in str(request.url)
        assert "id_list=1706.03762" in str(request.url)
        return httpx.Response(200, text=_ATOM)

    resolver = _resolver(handler)
    work = await resolver.resolve(
        ParsedReference(raw_text="Vaswani et al. arXiv:1706.03762v5")
    )
    assert work is not None
    assert work.source == "arxiv"
    assert work.method == "arxiv_id"
    assert work.confidence == 1.0
    assert work.title == "Attention Is All You Need"
    assert work.year == 2017
    assert work.authors == ("Ashish Vaswani", "Noam Shazeer")
    assert work.doi == "10.5555/3295222.3295349"
    assert work.venue == "NeurIPS 2017"
    assert work.external_id == "arXiv:1706.03762"


@pytest.mark.asyncio
async def test_no_arxiv_id_returns_none_without_call():
    called = {"hit": False}

    def handler(request: httpx.Request) -> httpx.Response:
        called["hit"] = True
        return httpx.Response(200, text=_ATOM)

    resolver = _resolver(handler)
    work = await resolver.resolve(ParsedReference(raw_text="No preprint id here"))
    assert work is None
    assert called["hit"] is False


@pytest.mark.asyncio
async def test_unknown_id_error_entry_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_ERROR_ATOM)

    resolver = _resolver(handler)
    work = await resolver.resolve(ParsedReference(raw_text="arXiv:9999.99999"))
    assert work is None


@pytest.mark.asyncio
async def test_network_failure_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503)

    resolver = _resolver(handler)
    work = await resolver.resolve(ParsedReference(raw_text="arXiv:1706.03762"))
    assert work is None
