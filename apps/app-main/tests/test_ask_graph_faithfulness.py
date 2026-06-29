"""Ask-graph faithfulness-guard integration tests (Track X.3).

Exercises the REAL ``ask`` nodes with the X.3 guard wired in:

- ``provide_answer``: a planted HALLUCINATED inline marker in the LLM's answer
  prose is detected and flagged (logged), while a GENUINE marker survives — and
  the answer text is NOT corrupted (non-destructive flagging).
- ``write_final_answer``: the array safety-net catches a planted out-of-set
  citation when the retrieval hits are threaded, and is a no-op on genuine
  context-derived citations.

Retrieval + LLM are mocked; the node wiring, the guard call, and the citation
derivation are real. Module loaded via the file-path spec with ``ai_prompter``
stubbed, mirroring ``test_ask_graph_citations``.
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _install_ai_prompter_stub():
    if "ai_prompter" in sys.modules:
        return sys.modules["ai_prompter"]
    mod = types.ModuleType("ai_prompter")

    class _Prompter:
        last_render_data = None

        def __init__(self, prompt_template=None, parser=None):
            self.prompt_template = prompt_template
            self.parser = parser

        def render(self, data=None):
            _Prompter.last_render_data = data
            return f"PROMPT[{self.prompt_template}]"

    mod.Prompter = _Prompter
    sys.modules["ai_prompter"] = mod
    return mod


_GRAPHS_DIR = Path(__file__).resolve().parents[1] / "src" / "app_main" / "graphs"


def _load_submodule(name: str):
    dotted = f"app_main.graphs.{name}"
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = importlib.util.spec_from_file_location(dotted, _GRAPHS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


def _stub_graphs_package():
    if "app_main.graphs" in sys.modules:
        return
    pkg = types.ModuleType("app_main.graphs")
    pkg.__path__ = [str(_GRAPHS_DIR)]
    sys.modules["app_main.graphs"] = pkg


def _load_ask():
    if "app_main.graphs.ask" in sys.modules:
        return sys.modules["app_main.graphs.ask"]
    _install_ai_prompter_stub()
    _stub_graphs_package()
    _load_submodule("utils")
    _load_submodule("citations")
    return _load_submodule("ask")


ask = _load_ask()


SEEDED_HITS = [
    {
        "id": "source:doc1",
        "parent_id": "source:doc1",
        "source": "source:doc1",
        "chunk_id": "chunk:p3",
        "physical_page": 3,
        "section_path": ["Intro"],
        "similarity": 0.9,
    },
    {
        "id": "source:doc2",
        "parent_id": "source:doc2",
        "source": "source:doc2",
        "chunk_id": "chunk:p12",
        "physical_page": 12,
        "section_path": ["Methods"],
        "similarity": 0.8,
    },
]


def _mock_model(content: str):
    model = MagicMock()
    model.ainvoke = AsyncMock(return_value=MagicMock(content=content))
    return model


async def _run_provide_answer(answer_text: str):
    retrieval = MagicMock()
    retrieval.vector_search = AsyncMock(return_value=SEEDED_HITS)
    with patch.object(
        ask, "provision_langchain_model",
        AsyncMock(return_value=_mock_model(answer_text)),
    ), patch(
        "retrieval.service.RetrievalService", return_value=retrieval
    ), patch(
        "app_main.dependencies.get_embedding_service",
        AsyncMock(return_value=MagicMock(embedding_model=MagicMock())),
    ):
        return await ask.provide_answer(
            {"question": "q", "term": "t", "instructions": "i"},
            {"configurable": {}},
        )


async def _run_write_final_answer(final_text: str, *, retrieval_hits=SEEDED_HITS,
                                  citation_groups=None):
    """Drive ``write_final_answer`` with a given final-answer text and the
    accumulated retrieval set, capturing the real marker-guard verdict.

    Returns ``(node_output, guard_result)``.
    """
    if citation_groups is None:
        citation_groups = [ask.citations_from_hits(SEEDED_HITS)]
    captured = {}
    real_guard = ask.guard_answer_citations

    def _spy(text, hits, **kw):
        res = real_guard(text, hits, **kw)
        captured["res"] = res
        return res

    with patch.object(
        ask, "provision_langchain_model",
        AsyncMock(return_value=_mock_model(final_text)),
    ), patch.object(ask, "guard_answer_citations", side_effect=_spy):
        out = await ask.write_final_answer(
            {
                "question": "q",
                "strategy": MagicMock(),
                "answers": ["a"],
                "citation_groups": citation_groups,
                "retrieval_hits": retrieval_hits,
            },
            {"configurable": {}},
        )
    return out, captured.get("res")


class TestProvideAnswerThreadsHits:
    """``provide_answer`` no longer guards markers (the intermediate sub-answers
    are never surfaced); it threads ``retrieval_hits`` so the final node can
    guard the user-visible answer."""

    @pytest.mark.asyncio
    async def test_sub_answer_text_untouched_and_hits_threaded(self):
        out = await _run_provide_answer("Sub [source:doc1, p.99] answer.")
        assert out["answers"][0] == "Sub [source:doc1, p.99] answer."
        assert out["retrieval_hits"] == SEEDED_HITS


class TestWriteFinalAnswerFaithfulness:
    """The PRIMARY X.3 guard: the FINAL, user-visible ask answer is validated."""

    @pytest.mark.asyncio
    async def test_hallucinated_final_marker_flagged_genuine_kept(self):
        """AC (final-answer): a planted marker citing source:doc1 at p.99 (only
        p.3 was retrieved) in the FINAL answer is flagged; [source:doc2, p.12]
        (in the retrieval set) is kept. Answer text stays byte-identical."""
        final = "Claim A [source:doc1, p.99] and claim B [source:doc2, p.12]."
        out, res = await _run_write_final_answer(final)
        assert res["dropped_count"] == 1
        assert res["dropped"][0]["marker"] == "[source:doc1, p.99]"
        assert {k["id"] for k in res["kept"]} == {"source:doc2"}
        # Non-destructive — the user-visible answer survives verbatim.
        assert out["final_answer"] == final

    @pytest.mark.asyncio
    async def test_all_genuine_final_markers_no_drop(self):
        final = "A [source:doc1, p.3] and B [source:doc2, p.12]."
        out, res = await _run_write_final_answer(final)
        assert res["dropped_count"] == 0
        assert out["final_answer"] == final

    @pytest.mark.asyncio
    async def test_no_marker_final_answer_not_corrupted(self):
        final = "A synthesized final answer with no citation markers at all."
        out, res = await _run_write_final_answer(final)
        assert res["dropped_count"] == 0 and res["kept_count"] == 0
        assert out["final_answer"] == final

    @pytest.mark.asyncio
    async def test_final_guard_validates_against_union_of_hits(self):
        """The membership set is the accumulated union of sub-answer hits — a
        marker citing a source only retrieved in another sub-answer is kept."""
        final = "From [source:doc2, p.12]."
        # doc2 came from a different sub-answer; both are in retrieval_hits.
        out, res = await _run_write_final_answer(final, retrieval_hits=SEEDED_HITS)
        assert res["dropped_count"] == 0
        assert out["final_answer"] == final


class TestWriteFinalAnswerArraySafetyNet:
    @pytest.mark.asyncio
    async def test_array_safety_net_drops_planted_out_of_set(self):
        """AC2 (catch path): with retrieval_hits threaded, a planted citation
        bearing a chunk_id NOT in the retrieval set is dropped from the final
        emitted citations."""
        genuine = ask.citations_from_hits(SEEDED_HITS)
        planted = {
            "source": "source:doc1",
            "page": 3,
            "chunk_id": "chunk:NOT_RETRIEVED",
            "section": None,
        }
        with patch.object(
            ask, "provision_langchain_model",
            AsyncMock(return_value=_mock_model("FINAL")),
        ):
            out = await ask.write_final_answer(
                {
                    "question": "q",
                    "strategy": MagicMock(),
                    "answers": ["a"],
                    "citation_groups": [genuine + [planted]],
                    "retrieval_hits": SEEDED_HITS,
                },
                {"configurable": {}},
            )

        chunk_ids = {c["chunk_id"] for c in out["citations"]}
        assert "chunk:NOT_RETRIEVED" not in chunk_ids
        assert {"chunk:p3", "chunk:p12"} <= chunk_ids
        assert out["final_answer"] == "FINAL"

    @pytest.mark.asyncio
    async def test_array_safety_net_no_op_on_genuine(self):
        """AC2 (no-op path): genuine context-derived citations pass through
        unchanged when checked against their own retrieval set."""
        genuine = ask.citations_from_hits(SEEDED_HITS)
        with patch.object(
            ask, "provision_langchain_model",
            AsyncMock(return_value=_mock_model("FINAL")),
        ):
            out = await ask.write_final_answer(
                {
                    "question": "q",
                    "strategy": MagicMock(),
                    "answers": ["a"],
                    "citation_groups": [genuine],
                    "retrieval_hits": SEEDED_HITS,
                },
                {"configurable": {}},
            )
        assert {(c["source"], c["page"]) for c in out["citations"]} == {
            ("source:doc1", 3),
            ("source:doc2", 12),
        }
