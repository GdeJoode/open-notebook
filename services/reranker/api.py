"""
Cross-encoder reranker service.

Standalone HTTP API that re-scores (query, passage) pairs with a local
cross-encoder. Used by ``app-main`` to rerank the top-N hits from hybrid
search before returning them. Keeping the model here keeps the heavy
``torch``/``sentence-transformers`` dependencies out of ``app-main``.

The model is loaded once, lazily, on the first ``/rerank`` call (and warmed
at startup) so the ~2 GB ``bge-reranker-v2-m3`` download/load cost is paid
once. The model is configurable via ``RERANKER_MODEL`` — the default is
multilingual (the corpus is Dutch); swap for a smaller/faster variant to
tune latency.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from loguru import logger
from pydantic import BaseModel, Field


@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Warm the model at startup unless explicitly skipped.

    Set ``RERANKER_SKIP_WARMUP=1`` to defer loading to the first request
    (useful in tests / fast boots). Failures here are non-fatal: the service
    still answers ``/health`` and retries the load on the first ``/rerank``.
    """
    if os.getenv("RERANKER_SKIP_WARMUP", "").lower() in ("1", "true", "yes"):
        logger.info("[reranker] Skipping model warmup (RERANKER_SKIP_WARMUP set)")
    else:
        try:
            get_model()
        except Exception as e:  # pragma: no cover
            logger.warning(
                f"[reranker] Warmup failed (will retry on first call): {e}"
            )
    yield


app = FastAPI(
    title="Cross-Encoder Reranker Service",
    description="Re-scores (query, passage) pairs with a local cross-encoder",
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Multilingual default — the corpus is Dutch, so an English-leaning reranker
# (e.g. mxbai-rerank) would degrade quality. Keep configurable per the plan's
# open decision on model size vs latency.
DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
MODEL_NAME = os.getenv("RERANKER_MODEL", DEFAULT_MODEL)
MODEL_DEVICE = os.getenv("RERANKER_DEVICE") or None  # None → CrossEncoder auto


# Lazy-loaded model (heavy: do it once).
_model = None


def get_model():
    """Get or lazily load the cross-encoder model.

    Imported and constructed lazily so the module is importable (and the
    health endpoint usable) without paying the model-load cost, and so the
    unit tests can mock the ``CrossEncoder`` without the 2 GB download.
    """
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder

        logger.info(f"[reranker] Loading cross-encoder: {MODEL_NAME}")
        t0 = time.time()
        kwargs = {}
        if MODEL_DEVICE:
            kwargs["device"] = MODEL_DEVICE
        _model = CrossEncoder(MODEL_NAME, **kwargs)
        logger.info(
            f"[reranker] Model loaded in {time.time() - t0:.1f}s "
            f"(device={MODEL_DEVICE or 'auto'})"
        )
    return _model


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class RerankRequest(BaseModel):
    query: str = Field(..., description="The search query")
    passages: List[str] = Field(
        ..., description="Candidate passage texts to re-score against the query"
    )
    top_k: Optional[int] = Field(
        default=None,
        description="If set, return only the top_k highest-scoring passages",
        ge=1,
    )


class RerankResult(BaseModel):
    index: int = Field(
        ..., description="Index of the passage in the original request list"
    )
    score: float = Field(..., description="Cross-encoder relevance score")


class RerankResponse(BaseModel):
    results: List[RerankResult] = Field(
        ..., description="Passages reordered by score, highest first"
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    """Liveness probe. Reports whether the model is already loaded."""
    return {
        "status": "ok",
        "service": "reranker",
        "model": MODEL_NAME,
        "model_loaded": _model is not None,
    }


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    """Re-score passages against the query and return them ordered by score.

    Returns ``{index, score}`` pairs sorted descending by score, where
    ``index`` refers to the position in the request's ``passages`` list so
    the caller can reorder its own richer result objects.
    """
    if not req.passages:
        return RerankResponse(results=[])

    try:
        model = get_model()
    except Exception as e:  # pragma: no cover - exercised only with real deps
        logger.error(f"[reranker] Model load failed: {e}")
        raise HTTPException(status_code=503, detail=f"Model unavailable: {e}")

    pairs = [[req.query, passage] for passage in req.passages]

    t0 = time.time()
    try:
        scores = model.predict(pairs)
    except Exception as e:  # pragma: no cover - exercised only with real deps
        logger.error(f"[reranker] predict() failed: {e}")
        raise HTTPException(status_code=500, detail=f"Rerank failed: {e}")

    scored = [
        RerankResult(index=i, score=float(score))
        for i, score in enumerate(scores)
    ]
    scored.sort(key=lambda r: r.score, reverse=True)

    if req.top_k is not None:
        scored = scored[: req.top_k]

    logger.info(
        f"[reranker] Reranked {len(req.passages)} passages in "
        f"{time.time() - t0:.3f}s"
    )
    return RerankResponse(results=scored)
