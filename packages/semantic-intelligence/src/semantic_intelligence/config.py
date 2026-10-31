"""
Configuration + embedding helpers for the semantic intelligence layer.

SurrealDB plumbing was deduplicated against
``surrealdb_service.connection`` — use ``execute_query`` from there
instead of the previously-here ``execute`` / ``get_db``.

Embeddings go through the model gate to llama-swap (F2-08, ADR 0006): the
process must have configured ``llm_manager.aanroep`` first (the fork does so at
startup); without it there is no model call.

Usage:
    from semantic_intelligence.config import embed_text
    from surrealdb_service.connection import execute_query

    result = await execute_query(
        "SELECT * FROM entity WHERE type = $type", {"type": "PERSON"}
    )
    vec = await embed_text("query text", classificatie="onbekend")
"""

import os
from typing import List, Optional


# ---------------------------------------------------------------------------
# Shared filesystem
# ---------------------------------------------------------------------------

SHARED_DATA_DIR = os.getenv("SHARED_DATA_DIR", "/data")


# ---------------------------------------------------------------------------
# Embedding helpers (via de modelpoort)
# ---------------------------------------------------------------------------


async def embed_texts(
    texts: List[str], *, classificatie: Optional[str]
) -> List[List[float]]:
    """Embeddings via ``ModelGate`` naar llama-swap (F2-08).

    ``classificatie`` is die van de invoer (verplicht); ``onbekend`` telt als
    intern. Het model komt uit het register (taak ``embedding``).
    """
    from llm_manager.aanroep import get_aanroeper

    return await get_aanroeper().embed(teksten=list(texts), classificatie=classificatie)


async def embed_text(text: str, *, classificatie: Optional[str]) -> List[float]:
    """Generate embedding for a single text."""
    result = await embed_texts([text], classificatie=classificatie)
    return result[0] if result else []
