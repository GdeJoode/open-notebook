"""§1.2 hard-guardrail canary: embeddings stay LOCAL and FIXED (J.3, F2-08).

Embeddings never go through the Track J route resolver, and since F2-08 they go
through the model gate only: ``ModelGate`` to ``/v1/embeddings`` of llama-swap,
with the embedding model from the register (``mxbai-embed-large``, 1 024
dimensions, ADR 0008) and the classification of the input in the audit row.

Assertions:
  (a) the embedding service source imports nothing from ``model_routing``;
  (b) ``LLMTask`` has exactly the three routed members — no EMBEDDING/PARSING;
  (c) calling ``resolve_route`` with a string ``"embedding"`` raises;
  (d) F2-08: the app's embedding service embeds through the gate to llama-swap,
      with the register's local embedding model; no model library is imported.
"""

from __future__ import annotations

import inspect

import pytest
from app_main.services.model_routing.route_resolver import (
    LLMTask,
    PrivacyMode,
    resolve_route,
)

# ---------------------------------------------------------------------------
# (a) The embedding service source must not import model_routing
# ---------------------------------------------------------------------------


def test_embedding_service_does_not_import_model_routing():
    """The EmbeddingService module source contains no ``model_routing`` ref.

    Reads the actual source of the class the app resolves via
    ``get_embedding_service`` (``embeddings.service.EmbeddingService``) and
    asserts the string ``model_routing`` does not appear anywhere — imports,
    type hints, or comments. A cloud-routed embedding is the exact failure this
    guards against.
    """
    from embeddings.service import EmbeddingService

    source = inspect.getsource(inspect.getmodule(EmbeddingService))
    assert "model_routing" not in source, (
        "embeddings.service imports/references model_routing — embeddings must "
        "stay local + fixed (I.G 768-dim pin, J plan §1.2)"
    )


def test_embedding_service_module_imports_clean():
    """Belt-and-suspenders: no imported name in the module resolves into the
    ``app_main.services.model_routing`` package."""
    from embeddings import service as embedding_service_module

    for name, value in vars(embedding_service_module).items():
        module_name = getattr(value, "__module__", "") or ""
        assert "model_routing" not in module_name, (
            f"{name} in embeddings.service comes from {module_name} — "
            "model_routing must not leak into the embedding path"
        )


# ---------------------------------------------------------------------------
# (b) LLMTask has exactly the three routed members
# ---------------------------------------------------------------------------


def test_llmtask_has_no_embedding_or_parsing_member():
    """LLMTask is exactly {ENTITY_EXTRACTION, SUMMARIZATION, CHAT}."""
    members = {m.name for m in LLMTask}
    assert members == {"ENTITY_EXTRACTION", "SUMMARIZATION", "CHAT"}, (
        f"LLMTask membership drifted to {members}; embeddings/parsing must "
        "never become a routed task"
    )
    assert not hasattr(LLMTask, "EMBEDDING")
    assert not hasattr(LLMTask, "PARSING")


def test_llmtask_rejects_embedding_value():
    """Constructing an LLMTask from 'embedding' / 'parsing' raises."""
    with pytest.raises(ValueError):
        LLMTask("embedding")
    with pytest.raises(ValueError):
        LLMTask("parsing")


# ---------------------------------------------------------------------------
# (c) resolve_route('embedding') raises instead of silently routing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_resolve_route_rejects_embedding_string():
    """``resolve_route("embedding", ...)`` raises (ValueError/KeyError).

    The coercion happens before any repo/DI access, so this never silently
    builds an embedding route — the call fails loudly at the public surface.
    """
    with pytest.raises((ValueError, KeyError)):
        await resolve_route("embedding", PrivacyMode.CLOUD)


@pytest.mark.asyncio
async def test_resolve_route_rejects_parsing_string():
    with pytest.raises((ValueError, KeyError)):
        await resolve_route("parsing", PrivacyMode.PRIVATE)


# ---------------------------------------------------------------------------
# (d) F2-08: embeddings go through the model gate to llama-swap
# ---------------------------------------------------------------------------


def test_the_register_embedding_model_is_local_behind_llama_swap():
    """The shipped register: one embedding model, local, via llama-swap."""
    from llm_manager.register import lees_register

    model = lees_register().voor_taak("embedding")
    assert model is not None
    assert (model.naam, model.aanroepwijze, model.locatie, model.toegestaan_intern) == (
        "embed", "llama-swap", "lokaal", "ja")
    assert "mxbai-embed-large" in model.versie


def test_the_embedding_path_imports_no_model_library():
    import ast

    from embeddings import service as embedding_service_module
    from embeddings.providers import poort_provider

    for module in (embedding_service_module, poort_provider):
        boom = ast.parse(inspect.getsource(module))
        modules = {
            (k.module or "") for k in ast.walk(boom) if isinstance(k, ast.ImportFrom)
        } | {a.name for k in ast.walk(boom) if isinstance(k, ast.Import) for a in k.names}
        assert not {m for m in modules if m.split(".")[0] in ("esperanto", "openai", "ollama")}


@pytest.mark.asyncio
async def test_the_embedding_service_embeds_through_the_gate():
    """``get_embedding_service`` embeds via ModelGate to llama-swap /v1/embeddings."""
    import json

    import httpx
    from embeddings.providers import PoortEmbedder
    from llm_manager import aanroep
    from llm_manager.providers.factory import LlamaSwapClient
    from llm_manager.register import lees_register

    from app_main.dependencies import get_embedding_service

    verzoeken = []
    regels = []

    def llamaswap(verzoek):
        verzoeken.append(verzoek)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [0.0] * 1024}]})

    class _Audit:
        async def schrijf(self, regel):
            regels.append(regel)

    aanroep.configureer(_Audit(), register=lees_register(),
                        client=LlamaSwapClient(transport=httpx.MockTransport(llamaswap)))
    try:
        svc = await get_embedding_service()
        assert isinstance(svc.embedding_model, PoortEmbedder)
        [vector] = await svc.embedding_model.aembed(["fragment"], classificatie="intern")
    finally:
        aanroep.reset()

    assert len(vector) == 1024
    assert [str(v.url) for v in verzoeken] == ["http://llama-swap:8080/v1/embeddings"]
    assert json.loads(verzoeken[0].content) == {"model": "embed", "input": ["fragment"]}
    [regel] = regels
    assert (regel.task, regel.classification, regel.location) == ("embedding", "intern", "lokaal")
