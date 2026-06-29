"""source_chat faithfulness-guard integration tests (Track X.3).

Exercises the REAL ``call_model_with_source_context`` node with the X.3 guard:

- a planted HALLUCINATED inline marker (a page the source's chunks were never
  retrieved at) is flagged, while a genuine page-cited marker survives;
- the answer text is NOT corrupted (non-destructive);
- the structured ``citations`` array is NOT narrowed by prose parsing — every
  validated context chunk's provenance is preserved (recall-first; the X.2
  minor #1 decision). The array safety-net is a no-op on this context-derived set.

ContextService + LLM are mocked; the node wiring + guard call are real. Loaded
via the file-path spec with ``ai_prompter`` + ``SqliteSaver`` stubbed, mirroring
``test_source_chat_citations``.
"""

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


_GRAPHS_DIR = Path(__file__).resolve().parents[1] / "src" / "app_main" / "graphs"


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
    spec = importlib.util.spec_from_file_location(dotted, _GRAPHS_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[dotted] = module
    spec.loader.exec_module(module)
    return module


def _stub_sqlite_saver():
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


def _ctx_data(source_id="source:doc1"):
    return {
        "sources": [{"id": source_id, "title": "Doc", "full_text": "Full text."}],
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
        "section_path": ["Intro"],
        "element_type": "text",
        "order": 0,
    },
    {
        "chunk_id": "chunk:b",
        "source": "source:doc1",
        "text": "Passage on page 9.",
        "physical_page": 9,
        "section_path": ["Methods"],
        "element_type": "text",
        "order": 1,
    },
]


def _mock_context_service(chunk_context):
    svc = MagicMock()
    svc.build_source_context = AsyncMock(return_value=_ctx_data())
    svc.build_source_chunks = AsyncMock(return_value=chunk_context)
    return svc


def _mock_model(content):
    model = MagicMock()
    ai = MagicMock()
    ai.content = content
    model.ainvoke = AsyncMock(return_value=ai)
    return model


async def _run(answer_text, chunk_context=CHUNK_CONTEXT):
    svc = _mock_context_service(chunk_context)
    captured = {}
    real_guard = source_chat.guard_answer_citations

    def _spy(text, hits, **kw):
        res = real_guard(text, hits, **kw)
        captured["res"] = res
        return res

    with patch.object(source_chat, "get_context_service", return_value=svc), \
        patch.object(
            source_chat, "provision_langchain_model",
            AsyncMock(return_value=_mock_model(answer_text)),
        ), \
        patch.object(source_chat, "guard_answer_citations", side_effect=_spy):
        out = await source_chat.call_model_with_source_context(
            {"source_id": "source:doc1", "messages": []},
            {"configurable": {}},
        )
    return out, captured.get("res")


class TestSourceChatFaithfulness:
    @pytest.mark.asyncio
    async def test_hallucinated_page_flagged_genuine_kept(self):
        """AC1: a marker citing source:doc1 at p.77 (chunks were retrieved at
        p.3 / p.9 only) is flagged; [source:doc1, p.3] is kept."""
        answer = "Genuine [source:doc1, p.3] but invented [source:doc1, p.77]."
        out, res = await _run(answer)
        assert res["dropped_count"] == 1
        assert res["dropped"][0]["marker"] == "[source:doc1, p.77]"
        assert res["kept"][0]["page"] == 3
        # AC3: answer text survives verbatim (non-destructive).
        assert out["messages"].content == answer

    @pytest.mark.asyncio
    async def test_page_less_source_marker_kept(self):
        """A page-less [source:doc1] is faithful (the source is in context)."""
        out, res = await _run("Per the source [source:doc1].")
        assert res["dropped_count"] == 0
        assert res["kept"][0]["id"] == "source:doc1"

    @pytest.mark.asyncio
    async def test_citations_array_not_narrowed_by_prose(self):
        """X.2 minor #1 decision: the structured citations are NOT narrowed to
        only the chunks the model wrote markers for. Even when the answer cites
        only p.3, BOTH context chunks' provenance survives in the array
        (recall-first; prose parsing is too fragile to drop valid provenance)."""
        out, _ = await _run("Only references [source:doc1, p.3].")
        pages = {c["page"] for c in out["citations"]}
        assert pages == {3, 9}  # both context chunks kept, not just the cited p.3
        chunk_ids = {c["chunk_id"] for c in out["citations"]}
        assert chunk_ids == {"chunk:a", "chunk:b"}
