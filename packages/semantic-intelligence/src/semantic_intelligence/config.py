"""
Shared configuration for the semantic intelligence layer.

All modules use this for connecting to SurrealDB and Ollama.
Reads from environment variables with sensible defaults that work
both inside Docker (container names) and locally (localhost).

Usage:
    from semantic_intelligence.config import get_db, execute, OLLAMA_URL

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
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "mxbai-embed-large")
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


def _parse_ids(obj: Any) -> Any:
    """Recursively convert RecordID objects to strings."""
    from surrealdb import RecordID

    if isinstance(obj, dict):
        return {k: _parse_ids(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_parse_ids(item) for item in obj]
    elif isinstance(obj, RecordID):
        return str(obj)
    return obj


async def execute(
    query: str,
    params: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Execute a SurrealQL query and return results.

    Convenience wrapper around get_db() for one-shot queries.
    RecordID objects are automatically converted to strings.
    """
    async with get_db() as db:
        result = await db.query(query, params)
        result = _parse_ids(result)
        # Normalize: always return a list of dicts
        if isinstance(result, list):
            # SurrealDB may return nested lists (one per statement)
            # or a flat list of dicts
            if result and isinstance(result[0], list):
                # Nested: take the last statement's results
                return result[-1] if result else []
            return result
        elif isinstance(result, dict):
            return [result]
        return []


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
