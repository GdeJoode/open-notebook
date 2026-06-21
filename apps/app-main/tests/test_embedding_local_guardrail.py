"""§1.2 hard-guardrail canary: embeddings + parsing stay LOCAL and FIXED (J.3).

The Track J route resolver must NEVER touch the embedding or parsing paths.
Cloud embeddings would change the vector space/dimensionality and silently
corrupt the I.G 768-dim HNSW pin. This file is the falsifiable canary the J
plan §1.2 requires; any change that wires ``model_routing`` into embeddings, or
adds an EMBEDDING/PARSING member to :class:`LLMTask`, breaks one of these
assertions.

Three assertions (J.3 AC6):
  (a) the embedding service source imports nothing from ``model_routing``;
  (b) ``LLMTask`` has exactly the three routed members — no EMBEDDING/PARSING;
  (c) calling ``resolve_route`` with a string ``"embedding"`` raises
      (ValueError/KeyError) rather than silently routing.
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
