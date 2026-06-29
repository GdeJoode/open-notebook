"""Tests for app_main.graphs.citations (Track X.2)."""

import importlib.util
import sys
from pathlib import Path


def _load_citations():
    """Load ``app_main.graphs.citations`` without triggering the package
    ``__init__``.

    The ``app_main.graphs`` package ``__init__`` eagerly imports sibling graph
    modules (ask/source_chat/...) that depend on the optional ``ai_prompter``
    package, absent from the unit-test environment. ``citations`` has no such
    dependency, so load it directly from its path (same pattern as
    ``test_chat_routing_telemetry._load_graph_utils``).
    """
    if "app_main.graphs.citations" in sys.modules:
        return sys.modules["app_main.graphs.citations"]
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "app_main"
        / "graphs"
        / "citations.py"
    )
    spec = importlib.util.spec_from_file_location(
        "app_main.graphs.citations", path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["app_main.graphs.citations"] = module
    spec.loader.exec_module(module)
    return module


_citations = _load_citations()
citations_from_hits = _citations.citations_from_hits
format_citation_tag = _citations.format_citation_tag
hit_to_citation = _citations.hit_to_citation
merge_citations = _citations.merge_citations


def _hit(**overrides):
    base = {
        "id": "source:abc",
        "parent_id": "source:abc",
        "source": "source:abc",
        "chunk_id": "chunk:c1",
        "physical_page": 7,
        "printed_page": 8,
        "section_path": ["Chapter 1", "Section 1.2"],
        "element_type": "text",
    }
    base.update(overrides)
    return base


class TestHitToCitation:
    def test_full_provenance(self):
        c = hit_to_citation(_hit())
        assert c == {
            "source": "source:abc",
            "page": 7,
            "chunk_id": "chunk:c1",
            "section": "Chapter 1 > Section 1.2",
        }

    def test_page_less_hit_keeps_source(self):
        c = hit_to_citation(
            _hit(physical_page=None, chunk_id=None, section_path=None)
        )
        assert c["source"] == "source:abc"
        assert c["page"] is None
        assert c["chunk_id"] is None
        assert c["section"] is None

    def test_source_falls_back_to_parent_id(self):
        c = hit_to_citation(
            {"id": "source:xyz", "parent_id": "source:xyz", "physical_page": 3}
        )
        assert c["source"] == "source:xyz"
        assert c["page"] == 3

    def test_uncitable_hit_returns_none(self):
        # A note hit with no source anchor cannot be cited.
        assert hit_to_citation({"id": "note:n1", "physical_page": None}) is None

    def test_string_section_path(self):
        c = hit_to_citation(_hit(section_path="Just One"))
        assert c["section"] == "Just One"


class TestCitationsFromHits:
    def test_dedup_same_chunk(self):
        hits = [_hit(), _hit()]
        assert len(citations_from_hits(hits)) == 1

    def test_same_source_different_page_kept(self):
        hits = [_hit(physical_page=7, chunk_id="chunk:a"),
                _hit(physical_page=9, chunk_id="chunk:b")]
        cites = citations_from_hits(hits)
        assert len(cites) == 2
        assert {c["page"] for c in cites} == {7, 9}

    def test_order_preserved(self):
        hits = [_hit(source="source:1", chunk_id="chunk:1"),
                _hit(source="source:2", chunk_id="chunk:2")]
        cites = citations_from_hits(hits)
        assert [c["source"] for c in cites] == ["source:1", "source:2"]

    def test_skips_uncitable(self):
        hits = [_hit(), {"id": "note:n1"}]
        assert len(citations_from_hits(hits)) == 1


class TestFormatCitationTag:
    def test_full_tag(self):
        tag = format_citation_tag(_hit())
        assert tag == "[source: source:abc | p.7 | Chapter 1 > Section 1.2]"

    def test_page_less_tag(self):
        tag = format_citation_tag(
            _hit(physical_page=None, section_path=None, chunk_id=None)
        )
        assert tag == "[source: source:abc]"

    def test_uncitable_tag_empty(self):
        assert format_citation_tag({"id": "note:n1"}) == ""

    def test_page_zero_is_included(self):
        # physical_page is 0-indexed; page 0 must still render (not falsy-skip).
        tag = format_citation_tag(
            _hit(physical_page=0, section_path=None)
        )
        assert "p.0" in tag


class TestMergeCitations:
    def test_merge_dedups_across_groups(self):
        g1 = citations_from_hits([_hit()])
        g2 = citations_from_hits([_hit()])
        merged = merge_citations([g1, g2])
        assert len(merged) == 1

    def test_merge_keeps_distinct(self):
        g1 = citations_from_hits([_hit(chunk_id="chunk:a")])
        g2 = citations_from_hits([_hit(chunk_id="chunk:b")])
        merged = merge_citations([g1, g2])
        assert len(merged) == 2

    def test_merge_empty(self):
        assert merge_citations([]) == []
        assert merge_citations([[], []]) == []
