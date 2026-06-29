"""source_chat-graph citation tests (Track X.2).

Exercises the real ``call_model_with_source_context`` node: it must fetch
chunk-level context, inject page-cited passages into the prompt, and emit
chunk-level citations (source + page/section), not just the source title.
ContextService and the LLM are mocked; the formatting + citation derivation are
real.

``source_chat.py`` imports the optional ``ai_prompter`` package (absent in the
unit-test env) and compiles a checkpointer-bound graph at import; both are
handled via the file-path loader with ``ai_prompter`` stubbed, mirroring
``test_ask_graph_citations``.
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


_GRAPHS_DIR = (
    Path(__file__).resolve().parents[1] / "src" / "app_main" / "graphs"
)


def _install_ai_prompter_stub():
    if "ai_prompter" in sys.modules:
        return sys.modules["ai_prompter"]
    mod = types.ModuleType("ai_prompter")

    class _Prompter:
        last_render_data = None

        def __init__(self, prompt_template=None, parser=None):
            self.prompt_template = prompt_template

        def render(self, data=None):
            _Prompter.last_render_data = data
            return f"PROMPT[{self.prompt_template}]"

    mod.Prompter = _Prompter
    sys.modules["ai_prompter"] = mod
    return mod


def _stub_graphs_package():
    if "app_main.graphs" in sys.modules:
        return
    pkg = types.ModuleType("app_main.graphs")
    pkg.__path__ = [str(_GRAPHS_DIR)]
    sys.modules["app_main.graphs"] = pkg


def _load_submodule(name: str):
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


def _stub_sqlite_saver():
    """Stub ``SqliteSaver.from_conn_string`` to return ``None``.

    The real call returns a ``_GeneratorContextManager`` which the
    module-level ``.compile(checkpointer=memory)`` rejects under this langgraph
    version — a pre-existing import-time concern unrelated to Track X.2. We only
    need the module importable to test the node function; ``None`` is a valid
    checkpointer for ``compile``.
    """
    import langgraph.checkpoint.sqlite as sqlite_mod

    sqlite_mod.SqliteSaver.from_conn_string = staticmethod(lambda *a, **k: None)


def _load_source_chat():
    if "app_main.graphs.source_chat" in sys.modules:
        return sys.modules["app_main.graphs.source_chat"]
    _install_ai_prompter_stub()
    _stub_graphs_package()
    _stub_sqlite_saver()
    _load_submodule("utils")
    _load_submodule("citations")
    return _load_submodule("source_chat")


source_chat = _load_source_chat()
Prompter = sys.modules["ai_prompter"].Prompter


def _ctx_data(source_id="source:doc1", title="Doc", full_text="Full text."):
    return {
        "sources": [
            {"id": source_id, "title": title, "full_text": full_text}
        ],
        "insights": [],
        "notes": [],
        "total_tokens": 10,
        "total_items": 1,
        "metadata": {"source_count": 1, "note_count": 0, "insight_count": 0},
    }


CHUNK_CONTEXT = [
    {
        "chunk_id": "chunk:a",
        "source": "source:doc1",
        "text": "Passage on page 3.",
        "physical_page": 3,
        "printed_page": 4,
        "section_path": ["Intro", "Background"],
        "element_type": "text",
        "order": 0,
    },
    {
        "chunk_id": "chunk:b",
        "source": "source:doc1",
        "text": "Passage on page 9.",
        "physical_page": 9,
        "printed_page": None,
        "section_path": ["Methods"],
        "element_type": "text",
        "order": 1,
    },
]


def _mock_context_service(ctx_data, chunk_context):
    svc = MagicMock()
    svc.build_source_context = AsyncMock(return_value=ctx_data)
    svc.build_source_chunks = AsyncMock(return_value=chunk_context)
    return svc


def _mock_model(content="answer"):
    model = MagicMock()
    ai = MagicMock()
    ai.content = content
    model.ainvoke = AsyncMock(return_value=ai)
    return model


class TestSourceChatCitations:
    @pytest.mark.asyncio
    async def test_chunk_level_citations_emitted(self):
        """AC2: citations carry chunk-level provenance (source + page/section),
        not just the source title."""
        svc = _mock_context_service(_ctx_data(), CHUNK_CONTEXT)

        with patch.object(
            source_chat, "get_context_service", return_value=svc
        ), patch.object(
            source_chat, "provision_langchain_model",
            AsyncMock(return_value=_mock_model()),
        ):
            out = await source_chat.call_model_with_source_context(
                {"source_id": "source:doc1", "messages": []},
                {"configurable": {}},
            )

        cites = out["citations"]
        by_chunk = {c["chunk_id"]: c for c in cites}
        assert by_chunk["chunk:a"]["page"] == 3
        assert by_chunk["chunk:a"]["section"] == "Intro > Background"
        assert by_chunk["chunk:b"]["page"] == 9
        assert all(c["source"] == "source:doc1" for c in cites)

    @pytest.mark.asyncio
    async def test_page_tags_injected_into_prompt(self):
        svc = _mock_context_service(_ctx_data(), CHUNK_CONTEXT)

        with patch.object(
            source_chat, "get_context_service", return_value=svc
        ), patch.object(
            source_chat, "provision_langchain_model",
            AsyncMock(return_value=_mock_model()),
        ):
            await source_chat.call_model_with_source_context(
                {"source_id": "source:doc1", "messages": []},
                {"configurable": {}},
            )

        context = Prompter.last_render_data["context"]
        assert "SOURCE PASSAGES" in context
        assert "[source: source:doc1 | p.3 | Intro > Background]" in context
        assert "[source: source:doc1 | p.9 | Methods]" in context

    @pytest.mark.asyncio
    async def test_no_chunks_falls_back_to_source_level_citation(self):
        """AC3: a source with no chunks (audio/plain text) still yields a
        source-level citation (page=None) and never raises."""
        svc = _mock_context_service(_ctx_data(), [])

        with patch.object(
            source_chat, "get_context_service", return_value=svc
        ), patch.object(
            source_chat, "provision_langchain_model",
            AsyncMock(return_value=_mock_model()),
        ):
            out = await source_chat.call_model_with_source_context(
                {"source_id": "source:doc1", "messages": []},
                {"configurable": {}},
            )

        assert out["citations"] == [
            {
                "source": "source:doc1",
                "page": None,
                "chunk_id": None,
                "section": None,
            }
        ]
        # Source-level full_text is still present in the prompt (fallback).
        assert "SOURCE CONTENT" in Prompter.last_render_data["context"]

    @pytest.mark.asyncio
    async def test_answer_text_still_produced(self):
        """AC3: existing behavior intact — the AI message is returned and
        citations are purely additive."""
        svc = _mock_context_service(_ctx_data(), CHUNK_CONTEXT)

        with patch.object(
            source_chat, "get_context_service", return_value=svc
        ), patch.object(
            source_chat, "provision_langchain_model",
            AsyncMock(return_value=_mock_model("the answer")),
        ):
            out = await source_chat.call_model_with_source_context(
                {"source_id": "source:doc1", "messages": []},
                {"configurable": {}},
            )

        assert out["messages"].content == "the answer"
        assert "context_indicators" in out
        assert out["source"].id == "source:doc1"
