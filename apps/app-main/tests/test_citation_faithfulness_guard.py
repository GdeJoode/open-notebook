"""Citation faithfulness-guard tests (Track X.3).

The genuine hallucination risk is the LLM's inline ``[document_id, p.X]``
markers in the answer prose — the structured ``citations`` array is built
deterministically from the context, so it is ``⊆`` the retrieval set by
construction. These tests exercise:

- ``guard_answer_citations`` — the PRIMARY guard: a planted hallucinated marker
  (a ``(source, page)`` never retrieved, or an id absent from the context) is
  detected and droppable; a genuine marker (in the context) is kept; the answer
  text and valid markers survive (least-destructive).
- ``guard_citation_array`` — the SECONDARY safety-net: a no-op on current
  context-derived output, but it WOULD catch a planted out-of-set entry.

``citations.py`` has no heavy deps, so it loads directly from its file path
(same bypass-the-package-__init__ pattern as ``test_graph_citations``).
"""

import importlib.util
import sys
from pathlib import Path


def _load_citations():
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


_c = _load_citations()
guard_answer_citations = _c.guard_answer_citations
guard_citation_array = _c.guard_citation_array
parse_inline_markers = _c.parse_inline_markers
citations_from_hits = _c.citations_from_hits


# A retrieval set: source:doc1 retrieved at page 3, source:doc2 at page 12,
# plus a page-less insight hit (no page; cites the source/insight ids only).
HITS = [
    {
        "id": "source:doc1",
        "parent_id": "source:doc1",
        "source": "source:doc1",
        "chunk_id": "chunk:p3",
        "physical_page": 3,
        "section_path": ["Intro"],
    },
    {
        "id": "source:doc2",
        "parent_id": "source:doc2",
        "source": "source:doc2",
        "chunk_id": "chunk:p12",
        "physical_page": 12,
        "section_path": ["Methods"],
    },
    {
        "id": "source_insight:i1",
        "parent_id": "source:doc3",
        "source": "source:doc3",
        "chunk_id": None,
        "physical_page": None,
        "section_path": None,
    },
]


class TestParseInlineMarkers:
    def test_paged_marker(self):
        out = parse_inline_markers("text [source:doc1, p.3] more")
        assert out == [("source:doc1", 3, "[source:doc1, p.3]")]

    def test_page_less_marker(self):
        out = parse_inline_markers("text [note:n1] more")
        assert out == [("note:n1", None, "[note:n1]")]

    def test_multiple_markers(self):
        out = parse_inline_markers("[source:a, p.1] and [insight:b]")
        assert [m[0] for m in out] == ["source:a", "insight:b"]
        assert out[0][1] == 1 and out[1][1] is None

    def test_ignores_non_record_brackets(self):
        # Plain footnote-style brackets must not be parsed as citations.
        assert parse_inline_markers("see [1] and [see below]") == []

    def test_tolerates_whitespace(self):
        out = parse_inline_markers("[ source:doc1 , p. 7 ]")
        assert out == [("source:doc1", 7, "[ source:doc1 , p. 7 ]")]

    def test_empty_answer(self):
        assert parse_inline_markers("") == []
        assert parse_inline_markers(None) == []


class TestGuardAnswerCitations:
    def test_genuine_paged_marker_kept(self):
        """AC1: a marker whose (source, page) WAS retrieved is kept."""
        res = guard_answer_citations("Claim [source:doc1, p.3].", HITS)
        assert res["dropped_count"] == 0
        assert res["kept_count"] == 1
        assert res["kept"][0]["id"] == "source:doc1"
        assert res["text"] == "Claim [source:doc1, p.3]."

    def test_hallucinated_page_for_real_source_detected(self):
        """AC1: source:doc1 WAS retrieved, but only at p.3 — a marker citing it
        at p.99 invents a page the model never saw → hallucinated."""
        res = guard_answer_citations("Claim [source:doc1, p.99].", HITS)
        assert res["dropped_count"] == 1
        assert res["dropped"][0] == {
            "id": "source:doc1",
            "page": 99,
            "marker": "[source:doc1, p.99]",
        }
        assert res["kept_count"] == 0

    def test_hallucinated_source_detected(self):
        """A marker citing a source entirely absent from the context."""
        res = guard_answer_citations("Claim [source:ghost, p.1].", HITS)
        assert res["dropped_count"] == 1
        assert res["dropped"][0]["id"] == "source:ghost"

    def test_page_less_marker_for_context_id_kept(self):
        """A page-less [id] for a record present in the context is faithful."""
        res = guard_answer_citations("From [source_insight:i1].", HITS)
        assert res["kept_count"] == 1
        assert res["dropped_count"] == 0

    def test_page_less_marker_for_absent_id_dropped(self):
        res = guard_answer_citations("From [note:ghost].", HITS)
        assert res["dropped_count"] == 1
        assert res["dropped"][0]["id"] == "note:ghost"

    def test_mixed_genuine_and_hallucinated(self):
        """The genuine marker is kept while the hallucinated one is flagged —
        precision-first, both present in the same answer."""
        answer = "A [source:doc1, p.3] but B [source:doc2, p.50]."
        res = guard_answer_citations(answer, HITS)
        assert {d["id"] for d in res["dropped"]} == {"source:doc2"}
        assert {k["id"] for k in res["kept"]} == {"source:doc1"}

    def test_default_is_non_destructive(self):
        """AC3: by default the answer text is NEVER modified — only recorded."""
        answer = "A [source:doc1, p.99] claim that should survive verbatim."
        res = guard_answer_citations(answer, HITS)
        assert res["text"] == answer
        assert res["dropped_count"] == 1

    def test_strip_removes_only_the_bad_marker(self):
        """AC3: with strip=True only the hallucinated marker token is removed;
        the surrounding prose and the valid marker survive intact."""
        answer = "Good [source:doc1, p.3] and bad [source:doc2, p.50] here."
        res = guard_answer_citations(answer, HITS, strip=True)
        assert "[source:doc2, p.50]" not in res["text"]
        assert "[source:doc1, p.3]" in res["text"]
        assert "Good" in res["text"] and "and bad" in res["text"]
        assert "here." in res["text"]

    def test_no_markers_no_corruption(self):
        """AC3: an answer with no markers returns unchanged, never raises."""
        res = guard_answer_citations("Just prose, no citations.", HITS)
        assert res["text"] == "Just prose, no citations."
        assert res["kept_count"] == 0 and res["dropped_count"] == 0

    def test_empty_answer_no_500(self):
        """AC3: empty / None answer is handled without raising."""
        assert guard_answer_citations("", HITS)["text"] == ""
        assert guard_answer_citations(None, HITS)["text"] is None

    def test_empty_retrieval_set_flags_everything(self):
        res = guard_answer_citations("[source:doc1, p.3]", [])
        assert res["dropped_count"] == 1

    def test_page_zero_marker(self):
        """physical_page is 0-indexed; a marker citing p.0 of a retrieved p.0
        chunk is faithful (not falsy-rejected)."""
        hits = [{"source": "source:z", "physical_page": 0, "chunk_id": "chunk:z"}]
        res = guard_answer_citations("[source:z, p.0]", hits)
        assert res["kept_count"] == 1


class TestGuardCitationArray:
    def test_no_op_on_context_derived_array(self):
        """AC2: citations built from the very hits passed as the retrieval set
        pass through unchanged — proves the safety-net is a no-op today."""
        citations = citations_from_hits(HITS)
        res = guard_citation_array(citations, HITS)
        assert res["dropped_count"] == 0
        assert res["citations"] == citations

    def test_catches_planted_out_of_set_chunk(self):
        """AC2: a planted citation bearing a chunk_id NOT in the retrieval set
        (a future regression where citations stop being context-derived) is
        caught and filtered."""
        citations = citations_from_hits(HITS)
        planted = {
            "source": "source:doc1",
            "page": 3,
            "chunk_id": "chunk:NOT_RETRIEVED",
            "section": None,
        }
        res = guard_citation_array(citations + [planted], HITS)
        assert res["dropped_count"] == 1
        assert res["dropped"][0]["chunk_id"] == "chunk:NOT_RETRIEVED"
        assert planted not in res["citations"]

    def test_page_less_source_citation_always_kept(self):
        """A chunk_id-less (source-level) citation makes no checkable chunk
        claim and is never dropped, even if the source were odd."""
        citations = [
            {"source": "source:any", "page": None, "chunk_id": None, "section": None}
        ]
        res = guard_citation_array(citations, HITS)
        assert res["dropped_count"] == 0
