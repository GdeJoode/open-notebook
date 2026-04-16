"""
Shared configuration for the semantic intelligence layer.

All modules use this for connecting to SurrealDB and Ollama.
Reads from environment variables with sensible defaults that work
both inside Docker (container names) and locally (localhost).

Usage:
    from semantic_layer.config import get_db, execute, OLLAMA_URL

    # Async context manager for SurrealDB
    async with get_db() as db:
        result = await db.query("SELECT * FROM entity LIMIT 10")

    # Or use the convenience function
    result = await execute("SELECT * FROM entity WHERE type = $type", {"type": "PERSON"})
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, List, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# SurrealDB configuration
# ---------------------------------------------------------------------------

SURREALDB_URL = os.getenv("SURREALDB_URL", "ws://localhost:8000/rpc")
SURREALDB_NS = os.getenv("SURREALDB_NS", os.getenv("SURREAL_NAMESPACE", "open_notebook"))
SURREALDB_DB = os.getenv("SURREALDB_DB", os.getenv("SURREAL_DATABASE", "open_notebook"))
SURREALDB_USER = os.getenv("SURREALDB_USER", os.getenv("SURREAL_USER", "root"))
SURREALDB_PASS = os.getenv("SURREALDB_PASS", os.getenv("SURREAL_PASSWORD", "root"))

# ---------------------------------------------------------------------------
# Ollama configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = os.getenv("OLLAMA_URL", os.getenv("OLLAMA_API_BASE", "http://localhost:11434"))
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL", "llama3.1:8b-instruct-q4_0")

# ---------------------------------------------------------------------------
# Shared filesystem
# ---------------------------------------------------------------------------

SHARED_DATA_DIR = os.getenv("SHARED_DATA_DIR", "/data")

# ---------------------------------------------------------------------------
# Database helpers
# ---------------------------------------------------------------------------


@asynccontextmanager
async def get_db() -> AsyncGenerator:
    """Async context manager for a SurrealDB connection.

    Handles sign-in, namespace/database selection, and cleanup.

    Usage:
        async with get_db() as db:
            result = await db.query("SELECT * FROM entity")
    """
    from surrealdb import AsyncSurreal

    db = AsyncSurreal(SURREALDB_URL)
    try:
        await db.signin({"username": SURREALDB_USER, "password": SURREALDB_PASS})
        await db.use(SURREALDB_NS, SURREALDB_DB)
        yield db
    finally:
        await db.close()


async def execute(
    query: str,
    params: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Execute a SurrealQL query and return results.

    Convenience wrapper around get_db() for one-shot queries.
    For multiple queries in sequence, use get_db() directly.
    """
    async with get_db() as db:
        result = await db.query(query, params)
        # SurrealDB returns a list of result sets (one per statement)
        if isinstance(result, list):
            # Flatten: return the result of the last statement
            return result[-1] if result else []
        return result


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
            emb = data.get("embeddings", [[]])[0]
            embeddings.append(emb)

    return embeddings


async def embed_text(text: str, model: str = "") -> List[float]:
    """Generate embedding for a single text."""
    result = await embed_texts([text], model)
    return result[0] if result else []
