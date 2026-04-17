"""
Proxy router for Docker microservices.

Forwards requests from the frontend (via main app API on :5055) to the
standalone Docker services (extraction :8103, summarization :8101).
Adds auth context and privacy routing.

All endpoints are prefixed with /services.
"""

import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query
from loguru import logger
from pydantic import BaseModel, Field

router = APIRouter(prefix="/services", tags=["services"])

# Service URLs — configurable via env vars
EXTRACTION_URL = os.getenv("EXTRACTION_SERVICE_URL", "http://localhost:8103")
SUMMARIZATION_URL = os.getenv("SUMMARIZATION_SERVICE_URL", "http://localhost:8101")

PROXY_TIMEOUT = int(os.getenv("SERVICE_PROXY_TIMEOUT", "600"))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _proxy_get(service_url: str, path: str, params: Optional[Dict] = None) -> Any:
    """Forward GET request to a Docker service."""
    try:
        async with httpx.AsyncClient(timeout=PROXY_TIMEOUT) as client:
            resp = await client.get(f"{service_url}{path}", params=params)
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=f"Service unavailable at {service_url}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def _proxy_post(service_url: str, path: str, body: Any = None, params: Optional[Dict] = None) -> Any:
    """Forward POST request to a Docker service."""
    try:
        async with httpx.AsyncClient(timeout=PROXY_TIMEOUT) as client:
            resp = await client.post(
                f"{service_url}{path}",
                json=body.model_dump() if hasattr(body, "model_dump") else body,
                params=params,
            )
            resp.raise_for_status()
            return resp.json()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=e.response.status_code, detail=str(e))
    except httpx.ConnectError:
        raise HTTPException(status_code=503, detail=f"Service unavailable at {service_url}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Service health
# ---------------------------------------------------------------------------


@router.get("/health")
async def services_health():
    """Check health of all Docker services."""
    services = {
        "extraction": EXTRACTION_URL,
        "summarization": SUMMARIZATION_URL,
    }
    status = {}
    for name, url in services.items():
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{url}/health")
                status[name] = resp.json() if resp.status_code == 200 else {"status": "error"}
        except Exception:
            status[name] = {"status": "unreachable"}
    return status


# ---------------------------------------------------------------------------
# Model routing
# ---------------------------------------------------------------------------


@router.get("/routing")
async def get_routing():
    """Get current model routing configuration."""
    from shared.model_routing import get_routing_summary
    return get_routing_summary()


# ---------------------------------------------------------------------------
# Extraction proxy
# ---------------------------------------------------------------------------


class ExtractRequest(BaseModel):
    file_path: str
    privacy: str = Field(default="internal")
    write_to_db: bool = Field(default=True)
    model_override: Optional[Dict[str, Any]] = None
    override: Optional[Dict[str, Any]] = None


@router.post("/extract")
async def extract(req: ExtractRequest):
    """Extract entities and relations from a document."""
    return await _proxy_post(EXTRACTION_URL, "/extract", req)


@router.get("/ontologies")
async def list_ontologies():
    """List available ontologies."""
    return await _proxy_get(EXTRACTION_URL, "/ontologies")


@router.get("/entities")
async def list_entities(
    type: Optional[str] = None,
    status: str = "active",
    limit: int = 100,
    offset: int = 0,
):
    """List extracted entities with filters."""
    params = {"status": status, "limit": limit, "offset": offset}
    if type:
        params["type"] = type
    return await _proxy_get(EXTRACTION_URL, "/entities", params)


@router.get("/entities/{entity_id:path}")
async def get_entity(entity_id: str):
    """Get entity detail with relations and aliases."""
    return await _proxy_get(EXTRACTION_URL, f"/entities/{entity_id}")


@router.get("/entities/{entity_id:path}/graph")
async def get_entity_graph(entity_id: str, depth: int = 2):
    """Get local graph around an entity."""
    return await _proxy_get(EXTRACTION_URL, f"/entities/{entity_id}/graph", {"depth": depth})


# ---------------------------------------------------------------------------
# Validation proxy
# ---------------------------------------------------------------------------


@router.post("/validate")
async def validate_entities(
    entity_type: Optional[str] = None,
    use_api: bool = True,
    apply_changes: bool = True,
):
    """Validate entities against external vocabularies."""
    params = {"use_api": str(use_api).lower(), "apply_changes": str(apply_changes).lower()}
    if entity_type:
        params["entity_type"] = entity_type
    return await _proxy_post(EXTRACTION_URL, "/validate", params=params)


@router.get("/vocabularies")
async def list_vocabularies():
    """List loaded vocabularies with counts."""
    return await _proxy_get(EXTRACTION_URL, "/vocabularies")


@router.post("/vocabularies/refresh")
async def refresh_vocabularies():
    """Download/refresh external vocabularies."""
    return await _proxy_post(EXTRACTION_URL, "/vocabularies/refresh")


# ---------------------------------------------------------------------------
# Deduplication proxy
# ---------------------------------------------------------------------------


@router.post("/deduplicate")
async def deduplicate(
    entity_type: Optional[str] = None,
    merge_threshold: float = 0.85,
    review_threshold: float = 0.5,
):
    """Find duplicate entity candidates."""
    params = {"merge_threshold": merge_threshold, "review_threshold": review_threshold}
    if entity_type:
        params["entity_type"] = entity_type
    return await _proxy_post(EXTRACTION_URL, "/deduplicate", params=params)


@router.get("/duplicates")
async def get_duplicates(
    entity_type: Optional[str] = None,
    min_score: float = 0.5,
):
    """Get duplicate candidates."""
    params = {"min_score": min_score}
    if entity_type:
        params["entity_type"] = entity_type
    return await _proxy_get(EXTRACTION_URL, "/duplicates", params)


class MergeRequest(BaseModel):
    canonical_id: str
    duplicate_id: str


@router.post("/merge")
async def merge_entities(req: MergeRequest):
    """Merge a duplicate entity into a canonical entity."""
    return await _proxy_post(EXTRACTION_URL, "/merge", req)


# ---------------------------------------------------------------------------
# Summarization proxy
# ---------------------------------------------------------------------------


class SummarizeRequest(BaseModel):
    file_path: str
    preset: Optional[str] = "auto"
    system_prompt: Optional[str] = None
    output_instructions: Optional[str] = None
    llm: Optional[Dict[str, Any]] = None
    embedding: Optional[Dict[str, Any]] = None


@router.post("/summarize")
async def summarize(req: SummarizeRequest):
    """Summarize a document with preset auto-detection."""
    return await _proxy_post(SUMMARIZATION_URL, "/summarize", req)


@router.get("/presets")
async def list_presets():
    """List available summarization presets."""
    return await _proxy_get(SUMMARIZATION_URL, "/presets")
