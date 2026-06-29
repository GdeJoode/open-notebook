"""Ask-graph citation tests (Track X.2).

Exercises the real ``ask`` LangGraph nodes (``provide_answer`` →
``write_final_answer``) against a seeded multi-page source, asserting the
emitted ``citations`` match the chunks' provenance and that the no-page path
degrades gracefully. Retrieval and the LLM are mocked; the graph wiring,
provenance threading, prompt-injection and citation derivation are real.

``ask.py`` imports the optional ``ai_prompter`` package (absent in the unit-test
env), so the module is loaded via a file-path spec with ``ai_prompter`` stubbed
in ``sys.modules`` first — same bypass-the-package-__init__ discipline as
``test_chat_routing_telemetry``.
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


def _install_ai_prompter_stub():
    """Provide a minimal ``ai_prompter.Prompter`` so ``ask.py`` imports.

    ``Prompter(prompt_template=...).render(data=...)`` is the only surface the
    graph uses. The stub records the rendered ``data`` so a test can assert the
    provenance tags reached the prompt, and returns a deterministic string.
    """
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


_GRAPHS_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "app_main" / "graphs"
)


def _load_submodule(name: str):
    """Load ``app_main.graphs.<name>`` from its file, registered in sys.modules
    under the dotted name so sibling ``from app_main.graphs.<name> import``
    statements resolve without ever executing the package ``__init__`` (which
    eagerly imports chat/transformation and compiles a checkpointer-bound graph
    that needs a live saver)."""
    dotted = f"app_main.graphs.{name}"
    if dotted in sys.modules:
        return sys.modules[dotted]
    spec = importlib.util.spec_from_file_location(
        dotted, _GRAPHS_DIR / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


def _stub_graphs_package():
    """Put a bare ``app_main.graphs`` package object in sys.modules so the
    ``from app_main.graphs.X import`` machinery finds the parent without
    running the real (heavy) package ``__init__``."""
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
    # Pre-load the real sibling deps ask.py imports, so its ``from
    # app_main.graphs.{utils,citations} import`` resolve from sys.modules.
    _load_submodule("utils")
    _load_submodule("citations")
    return _load_submodule("ask")


ask = _load_ask()
Prompter = sys.modules["ai_prompter"].Prompter


# --- Seeded source with multi-page chunks --------------------------------

SEEDED_HITS = [
    {
        "id": "source:doc1",
        "parent_id": "source:doc1",
        "source": "source:doc1",
        "chunk_id": "chunk:p3",
        "physical_page": 3,
        "printed_page": 4,
        "section_path": ["Intro", "Background"],
        "element_type": "text",
        "similarity": 0.91,
    },
    {
        "id": "source:doc2",
        "parent_id": "source:doc2",
        "source": "source:doc2",
        "chunk_id": "chunk:p12",
        "physical_page": 12,
        "printed_page": None,
        "section_path": ["Methods"],
        "element_type": "text",
        "similarity": 0.82,
    },
]


def _mock_model(content: str):
    model = MagicMock()
    model.ainvoke = AsyncMock(return_value=MagicMock(content=content))
    return model


class TestProvideAnswerCitations:
    @pytest.mark.asyncio
    async def test_citations_match_chunk_provenance(self):
        """AC1: each citation's page == the cited chunk's physical_page and
        source == the real source id; hydration is requested (hydrate=True)."""
        retrieval = MagicMock()
        retrieval.vector_search = AsyncMock(return_value=SEEDED_HITS)

        with patch.object(
            ask, "provision_langchain_model",
            AsyncMock(return_value=_mock_model("answer [source:doc1, p.3]")),
        ), patch(
            "retrieval.service.RetrievalService", return_value=retrieval
        ), patch(
            "app_main.dependencies.get_embedding_service",
            AsyncMock(return_value=MagicMock(embedding_model=MagicMock())),
        ):
            out = await ask.provide_answer(
                {"question": "q", "term": "t", "instructions": "i"},
                {"configurable": {}},
            )

        # hydration opt-in
        assert retrieval.vector_search.call_args.kwargs.get("hydrate") is True

        groups = out["citation_groups"]
        assert len(groups) == 1
        cites = groups[0]
        by_source = {c["source"]: c for c in cites}
        assert by_source["source:doc1"]["page"] == 3
        assert by_source["source:doc1"]["chunk_id"] == "chunk:p3"
        assert by_source["source:doc1"]["section"] == "Intro > Background"
        assert by_source["source:doc2"]["page"] == 12

    @pytest.mark.asyncio
    async def test_provenance_tags_injected_into_prompt(self):
        """The context fed to the LLM carries [source | p.X | section] tags so
        the model can attribute claims."""
        retrieval = MagicMock()
        retrieval.vector_search = AsyncMock(return_value=SEEDED_HITS)

        with patch.object(
            ask, "provision_langchain_model",
            AsyncMock(return_value=_mock_model("ans")),
        ), patch(
            "retrieval.service.RetrievalService", return_value=retrieval
        ), patch(
            "app_main.dependencies.get_embedding_service",
            AsyncMock(return_value=MagicMock(embedding_model=MagicMock())),
        ):
            await ask.provide_answer(
                {"question": "q", "term": "t", "instructions": "i"},
                {"configurable": {}},
            )

        rendered = Prompter.last_render_data["results"]
        assert "[source: source:doc1 | p.3 | Intro > Background]" in rendered
        assert "[source: source:doc2 | p.12 | Methods]" in rendered

    @pytest.mark.asyncio
    async def test_no_page_source_graceful(self):
        """AC3: an insight/text/audio hit with no page yields a citation with
        page=None (source-level) and never raises / 500s."""
        no_page_hits = [
            {
                "id": "source_insight:i1",
                "parent_id": "source:doc3",
                "source": "source:doc3",
                "chunk_id": None,
                "physical_page": None,
                "section_path": None,
                "element_type": None,
                "relevance": 2.0,
            },
        ]
        retrieval = MagicMock()
        retrieval.vector_search = AsyncMock(return_value=no_page_hits)

        with patch.object(
            ask, "provision_langchain_model",
            AsyncMock(return_value=_mock_model("ans [source:doc3]")),
        ), patch(
            "retrieval.service.RetrievalService", return_value=retrieval
        ), patch(
            "app_main.dependencies.get_embedding_service",
            AsyncMock(return_value=MagicMock(embedding_model=MagicMock())),
        ):
            out = await ask.provide_answer(
                {"question": "q", "term": "t", "instructions": "i"},
                {"configurable": {}},
            )

        cites = out["citation_groups"][0]
        assert cites == [
            {
                "source": "source:doc3",
                "page": None,
                "chunk_id": None,
                "section": None,
            }
        ]

    @pytest.mark.asyncio
    async def test_empty_results_emit_empty_citations(self):
        retrieval = MagicMock()
        retrieval.vector_search = AsyncMock(return_value=[])

        with patch(
            "retrieval.service.RetrievalService", return_value=retrieval
        ), patch(
            "app_main.dependencies.get_embedding_service",
            AsyncMock(return_value=MagicMock(embedding_model=MagicMock())),
        ):
            out = await ask.provide_answer(
                {"question": "q", "term": "t", "instructions": "i"},
                {"configurable": {}},
            )

        assert out["answers"] == []
        assert out["citation_groups"] == []


class TestWriteFinalAnswerCitations:
    @pytest.mark.asyncio
    async def test_merges_citation_groups(self):
        """AC1/AC3: final answer carries the merged, de-duplicated citation set;
        the answer text is still produced (additive)."""
        g1 = ask.citations_from_hits([SEEDED_HITS[0]])
        g2 = ask.citations_from_hits([SEEDED_HITS[1], SEEDED_HITS[0]])  # overlap

        with patch.object(
            ask, "provision_langchain_model",
            AsyncMock(return_value=_mock_model("FINAL ANSWER")),
        ):
            out = await ask.write_final_answer(
                {
                    "question": "q",
                    "strategy": MagicMock(),
                    "answers": ["a1", "a2"],
                    "citation_groups": [g1, g2],
                },
                {"configurable": {}},
            )

        assert out["final_answer"] == "FINAL ANSWER"
        cites = out["citations"]
        # doc1 appears in both groups -> de-duplicated to one.
        keys = {(c["source"], c["page"]) for c in cites}
        assert keys == {("source:doc1", 3), ("source:doc2", 12)}

    @pytest.mark.asyncio
    async def test_no_citation_groups_yields_empty(self):
        with patch.object(
            ask, "provision_langchain_model",
            AsyncMock(return_value=_mock_model("FINAL")),
        ):
            out = await ask.write_final_answer(
                {
                    "question": "q",
                    "strategy": MagicMock(),
                    "answers": [],
                    "citation_groups": [],
                },
                {"configurable": {}},
            )

        assert out["final_answer"] == "FINAL"
        assert out["citations"] == []
