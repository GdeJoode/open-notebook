"""Unit tests for the V.4 work-resolution cascade (routing + precision guard).

These exercise the pure cascade logic with FAKE resolvers (no HTTP): shape-first
routing, first-confident-hit stop, network-failure fall-through, the RePEc
config-gate skip, and the title/author precision guard. Per-provider HTTP parsing
is covered in the provider-specific test modules.
"""

from __future__ import annotations

from typing import List, Optional

import pytest
from shared.references.work_resolver import (
    ResolvedWork,
    ResolverCascade,
    extract_arxiv_id,
    looks_like_econ_wp,
    looks_like_nl_policy,
    passes_title_author_guard,
    title_overlap,
)
from shared.retrieval.cites_matching import ParsedReference


class FakeResolver:
    """A resolver that records calls and returns a preset result."""

    def __init__(
        self,
        name: str,
        result: Optional[ResolvedWork] = None,
        *,
        available: bool = True,
        raises: bool = False,
    ) -> None:
        self.name = name
        self._result = result
        self.available = available
        self._raises = raises
        self.calls: List[ParsedReference] = []

    async def resolve(self, ref: ParsedReference) -> Optional[ResolvedWork]:
        self.calls.append(ref)
        if self._raises:
            raise RuntimeError("boom")
        return self._result


def _work(source: str, method: str = "doi", confidence: float = 1.0) -> ResolvedWork:
    return ResolvedWork(
        title="A Work", source=source, method=method, confidence=confidence
    )


# -- shape detection --------------------------------------------------------


def test_extract_arxiv_id_from_doi():
    ref = ParsedReference(raw_text="x", doi="10.48550/arXiv.2301.01234")
    assert extract_arxiv_id(ref) == "2301.01234"


def test_extract_arxiv_id_from_text():
    ref = ParsedReference(raw_text="Vaswani et al. arXiv:1706.03762v5")
    assert extract_arxiv_id(ref) == "1706.03762"


def test_bare_number_without_arxiv_keyword_is_not_an_id():
    ref = ParsedReference(raw_text="pp. 1706.03762 of the ledger")
    assert extract_arxiv_id(ref) is None


def test_looks_like_nl_policy_marker():
    ref = ParsedReference(raw_text="Kamerstukken II 2019/20, 35 300, nr. 1")
    assert looks_like_nl_policy(ref) is True


def test_looks_like_nl_policy_negative():
    ref = ParsedReference(raw_text="Acemoglu, D. (2001). The colonial origins.")
    assert looks_like_nl_policy(ref) is False


def test_looks_like_econ_wp():
    ref = ParsedReference(raw_text="NBER Working Paper No. 12345")
    assert looks_like_econ_wp(ref) is True


# -- precision guard --------------------------------------------------------


def test_title_overlap_identical():
    assert title_overlap("Graph Neural Networks", "graph neural networks") == 1.0


def test_guard_rejects_generic_title_near_miss():
    """A generic title with no shared content stays unresolved (precision)."""
    ref = ParsedReference(raw_text="x", title="Introduction", authors=("Smith",))
    accepted, conf = passes_title_author_guard(
        ref, "Conclusion", ("Jones",)
    )
    assert accepted is False
    assert conf == 0.0


def test_guard_requires_author_confirmer_when_ref_has_authors():
    """Strong title but zero author agreement → rejected (author is a confirmer)."""
    ref = ParsedReference(
        raw_text="x", title="Cohesion Policy and Growth", authors=("Acemoglu",)
    )
    accepted, _ = passes_title_author_guard(
        ref, "Cohesion Policy and Growth", ("Robinson",)
    )
    assert accepted is False


def test_guard_accepts_strong_title_and_author():
    ref = ParsedReference(
        raw_text="x", title="Cohesion Policy and Growth", authors=("Acemoglu, D.",)
    )
    accepted, conf = passes_title_author_guard(
        ref, "Cohesion Policy and Growth", ("Daron Acemoglu",)
    )
    assert accepted is True
    assert conf >= 0.5


def test_guard_accepts_titleonly_when_no_authors_on_reference():
    """A title-only reference can resolve on a strong title alone."""
    ref = ParsedReference(raw_text="x", title="Attention Is All You Need")
    accepted, conf = passes_title_author_guard(
        ref, "Attention Is All You Need", ()
    )
    assert accepted is True
    assert conf >= 0.5


# -- cascade routing --------------------------------------------------------


@pytest.mark.asyncio
async def test_doi_routes_to_crossref_first():
    crossref = FakeResolver("crossref", _work("crossref"))
    datacite = FakeResolver("datacite", _work("datacite"))
    openalex = FakeResolver("openalex", _work("openalex"))
    cascade = ResolverCascade(
        crossref=crossref, datacite=datacite, openalex=openalex
    )
    ref = ParsedReference(raw_text="x", doi="10.1/abc")
    result = await cascade.resolve(ref)
    assert result is not None and result.source == "crossref"
    assert len(crossref.calls) == 1
    assert datacite.calls == []  # stopped at first confident hit
    assert openalex.calls == []


@pytest.mark.asyncio
async def test_doi_falls_through_crossref_to_datacite():
    crossref = FakeResolver("crossref", None)  # miss
    datacite = FakeResolver("datacite", _work("datacite"))
    openalex = FakeResolver("openalex", _work("openalex"))
    cascade = ResolverCascade(
        crossref=crossref, datacite=datacite, openalex=openalex
    )
    ref = ParsedReference(raw_text="x", doi="10.5281/zenodo.123")
    result = await cascade.resolve(ref)
    assert result is not None and result.source == "datacite"
    assert len(crossref.calls) == 1
    assert len(datacite.calls) == 1


@pytest.mark.asyncio
async def test_arxiv_id_routes_to_arxiv():
    arxiv = FakeResolver("arxiv", _work("arxiv", method="arxiv_id"))
    openalex = FakeResolver("openalex", _work("openalex"))
    cascade = ResolverCascade(arxiv=arxiv, openalex=openalex)
    ref = ParsedReference(raw_text="Foo. arXiv:2301.01234")
    result = await cascade.resolve(ref)
    assert result is not None and result.source == "arxiv"
    assert len(arxiv.calls) == 1
    assert openalex.calls == []


@pytest.mark.asyncio
async def test_nl_policy_routes_to_overheid():
    overheid = FakeResolver("overheid", _work("overheid", method="sru", confidence=0.95))
    openalex = FakeResolver("openalex", _work("openalex"))
    cascade = ResolverCascade(overheid=overheid, openalex=openalex)
    ref = ParsedReference(raw_text="Kamerstukken II 2019/20, 35 300, nr. 1")
    result = await cascade.resolve(ref)
    assert result is not None and result.source == "overheid"
    assert len(overheid.calls) == 1


@pytest.mark.asyncio
async def test_title_only_routes_to_openalex_fallback():
    openalex = FakeResolver(
        "openalex", _work("openalex", method="title_author", confidence=0.8)
    )
    crossref = FakeResolver("crossref", _work("crossref"))
    cascade = ResolverCascade(openalex=openalex, crossref=crossref)
    ref = ParsedReference(raw_text="x", title="Some Paper Title")
    result = await cascade.resolve(ref)
    assert result is not None and result.source == "openalex"
    # No DOI ⇒ crossref (DOI leg) never consulted.
    assert crossref.calls == []
    assert len(openalex.calls) == 1


@pytest.mark.asyncio
async def test_network_failure_falls_through_to_next_leg():
    crossref = FakeResolver("crossref", None, raises=True)  # network error
    datacite = FakeResolver("datacite", _work("datacite"))
    cascade = ResolverCascade(crossref=crossref, datacite=datacite)
    ref = ParsedReference(raw_text="x", doi="10.1/abc")
    result = await cascade.resolve(ref)
    assert result is not None and result.source == "datacite"


@pytest.mark.asyncio
async def test_nothing_clears_guard_returns_none():
    crossref = FakeResolver("crossref", None)
    datacite = FakeResolver("datacite", None)
    openalex = FakeResolver("openalex", None)
    cascade = ResolverCascade(
        crossref=crossref, datacite=datacite, openalex=openalex
    )
    ref = ParsedReference(raw_text="x", doi="10.1/missing")
    assert await cascade.resolve(ref) is None


# -- RePEc config-gate ------------------------------------------------------


@pytest.mark.asyncio
async def test_repec_unconfigured_is_skipped_cascade_still_resolves():
    """An unavailable RePEc leg is skipped silently; OpenAlex still wins."""
    repec = FakeResolver("repec", _work("repec"), available=False)
    openalex = FakeResolver(
        "openalex", _work("openalex", method="title_author", confidence=0.8)
    )
    cascade = ResolverCascade(repec=repec, openalex=openalex)
    ref = ParsedReference(raw_text="NBER Working Paper No. 999", title="Econ Paper")
    result = await cascade.resolve(ref)
    assert result is not None and result.source == "openalex"
    assert repec.calls == []  # never even invoked


@pytest.mark.asyncio
async def test_repec_configured_and_econ_shape_is_routed():
    repec = FakeResolver(
        "repec", _work("repec", method="title_author", confidence=0.8), available=True
    )
    openalex = FakeResolver("openalex", _work("openalex"))
    cascade = ResolverCascade(repec=repec, openalex=openalex)
    ref = ParsedReference(raw_text="IZA Discussion Paper 555", title="Labour Econ")
    result = await cascade.resolve(ref)
    assert result is not None and result.source == "repec"
    assert len(repec.calls) == 1


# -- enrichment -------------------------------------------------------------


class FakeEnricher:
    def __init__(self) -> None:
        self.calls: List[ResolvedWork] = []

    async def enrich(self, work: ResolvedWork) -> ResolvedWork:
        self.calls.append(work)
        from dataclasses import replace

        return replace(work, author_ids=("0000-0000-0000-0001",))


@pytest.mark.asyncio
async def test_enricher_is_applied_to_the_winner():
    openalex = FakeResolver(
        "openalex", _work("openalex", method="title_author", confidence=0.8)
    )
    enricher = FakeEnricher()
    cascade = ResolverCascade(openalex=openalex, enricher=enricher)
    ref = ParsedReference(raw_text="x", title="Some Paper")
    result = await cascade.resolve(ref)
    assert result is not None
    assert result.author_ids == ("0000-0000-0000-0001",)
    assert len(enricher.calls) == 1
