"""
Ollama configuration + embedding helpers for the semantic intelligence layer.

SurrealDB plumbing was deduplicated against
``surrealdb_service.connection`` — use ``execute_query`` from there
instead of the previously-here ``execute`` / ``get_db``.

Reads from environment variables with sensible defaults that work
both inside Docker (container names) and locally (localhost).

Usage:
    from semantic_intelligence.config import OLLAMA_URL, embed_text
    from surrealdb_service.connection import execute_query

    result = await execute_query(
        "SELECT * FROM entity WHERE type = $type", {"type": "PERSON"}
    )
    vec = await embed_text("query text")
"""

import os
from typing import List


# ---------------------------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = os.getenv("OLLAMA_URL", os.getenv("OLLAMA_API_BASE", "http://localhost:11434"))
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b-instruct-q4_0")


# ---------------------------------------------------------------------------
# Shared filesystem
# ---------------------------------------------------------------------------

SHARED_DATA_DIR = os.getenv("SHARED_DATA_DIR", "/data")


# ---------------------------------------------------------------------------
# Ollama embedding helpers
# ---------------------------------------------------------------------------


async def embed_texts(texts: List[str], model: str = "") -> List[List[float]]:
    """Generate embeddings for a list of texts via Ollama.

    Args:
        texts: Texts to embed.
        model: Ollama model name. Defaults to OLLAMA_EMBED_MODEL.

    Returns:
        List of embedding vectors (list of floats).
    """
    import httpx

    model = model or OLLAMA_EMBED_MODEL
    embeddings = []

    async with httpx.AsyncClient(timeout=120) as client:
        for text in texts:
            resp = await client.post(
                f"{OLLAMA_URL}/api/embed",
                json={"model": model, "input": text},
            )
            resp.raise_for_status()
            data = resp.json()
            # Ollama /api/embed returns {"embeddings": [[...]]}
            embs = data.get("embeddings", [[]])
            emb = embs[0] if embs else []
            embeddings.append(emb)

    return embeddings


async def embed_text(text: str, model: str = "") -> List[float]:
    """Generate embedding for a single text."""
    result = await embed_texts([text], model)
    return result[0] if result else []
