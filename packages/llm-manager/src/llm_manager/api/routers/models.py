"""
Model management API routes.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from llm_manager.manager import get_model_manager
from llm_manager.providers import list_providers
from shared.models import DefaultModels, Model, ModelTypeStr

router = APIRouter()


class ModelCreate(BaseModel):
    """Model creation request."""

    name: str
    provider: str
    type: ModelTypeStr
    display_name: Optional[str] = None
    description: Optional[str] = None
    context_window: Optional[int] = None
    max_output_tokens: Optional[int] = None
    supports_vision: Optional[bool] = False
    supports_tools: Optional[bool] = False
    supports_streaming: Optional[bool] = True
    input_cost_per_1k: Optional[float] = None
    output_cost_per_1k: Optional[float] = None
    is_local: Optional[bool] = False
    is_enabled: Optional[bool] = True


class ModelResponse(BaseModel):
    """Model response."""

    id: Optional[str] = None
    name: str
    provider: str
    type: str
    display_name: Optional[str] = None
    description: Optional[str] = None
    context_window: Optional[int] = None
    supports_vision: Optional[bool] = None
    supports_tools: Optional[bool] = None
    is_local: Optional[bool] = None
    is_enabled: Optional[bool] = None


class DefaultModelsUpdate(BaseModel):
    """Default models update request."""

    default_chat_model: Optional[str] = None
    default_transformation_model: Optional[str] = None
    large_context_model: Optional[str] = None
    default_text_to_speech_model: Optional[str] = None
    default_speech_to_text_model: Optional[str] = None
    default_embedding_model: Optional[str] = None
    default_tools_model: Optional[str] = None


class CacheStatsResponse(BaseModel):
    """Cache statistics response."""

    total_entries: int
    valid_entries: int
    cache_enabled: bool
    ttl_seconds: int


@router.get("/types")
async def get_model_types() -> List[str]:
    """
    Get available model types.

    Returns:
        List of model type strings.
    """
    return ["language", "embedding", "speech_to_text", "text_to_speech"]


@router.post("/validate")
async def validate_model_config(data: ModelCreate) -> Dict[str, Any]:
    """
    Validate a model configuration without persisting.

    Args:
        data: Model configuration to validate.

    Returns:
        Validation result with any warnings.
    """
    warnings = []

    # Check provider support
    providers = list_providers()
    provider_names = [p.name for p in providers]

    if data.provider not in provider_names:
        warnings.append(f"Unknown provider: {data.provider}")

    provider = next((p for p in providers if p.name == data.provider), None)
    if provider:
        if data.type == "embedding" and not provider.supports_embedding:
            warnings.append(f"Provider {data.provider} may not support embeddings")
        if data.type == "speech_to_text" and not provider.supports_speech_to_text:
            warnings.append(f"Provider {data.provider} may not support speech-to-text")
        if data.type == "text_to_speech" and not provider.supports_text_to_speech:
            warnings.append(f"Provider {data.provider} may not support text-to-speech")
        if data.supports_vision and not provider.supports_vision:
            warnings.append(f"Provider {data.provider} may not support vision")
        if data.supports_tools and not provider.supports_tools:
            warnings.append(f"Provider {data.provider} may not support tools")

    return {
        "valid": len(warnings) == 0 or all("may not" in w for w in warnings),
        "warnings": warnings,
        "config": data.model_dump(),
    }


@router.get("/cache/stats", response_model=CacheStatsResponse)
async def get_cache_stats() -> CacheStatsResponse:
    """
    Get model cache statistics.

    Returns:
        Cache statistics.
    """
    manager = get_model_manager()
    stats = manager.get_cache_stats()
    return CacheStatsResponse(**stats)


@router.post("/cache/clear")
async def clear_cache() -> Dict[str, str]:
    """
    Clear the model cache.

    Returns:
        Confirmation message.
    """
    manager = get_model_manager()
    manager.clear_cache()
    return {"status": "Cache cleared"}


@router.post("/cache/invalidate/{model_id}")
async def invalidate_model_cache(model_id: str) -> Dict[str, str]:
    """
    Invalidate cache for a specific model.

    Args:
        model_id: ID of the model to invalidate.

    Returns:
        Confirmation message.
    """
    manager = get_model_manager()
    manager.invalidate_model(model_id)
    return {"status": f"Cache invalidated for model: {model_id}"}


@router.get("/defaults")
async def get_defaults() -> Dict[str, Any]:
    """
    Get current default models configuration.

    Returns:
        Default models configuration.
    """
    manager = get_model_manager()
    defaults = manager.get_defaults()
    if defaults:
        return defaults.model_dump()
    return {
        "default_chat_model": None,
        "default_transformation_model": None,
        "large_context_model": None,
        "default_text_to_speech_model": None,
        "default_speech_to_text_model": None,
        "default_embedding_model": None,
        "default_tools_model": None,
    }


@router.put("/defaults")
async def update_defaults(data: DefaultModelsUpdate) -> Dict[str, Any]:
    """
    Update default models configuration.

    Args:
        data: New default models.

    Returns:
        Updated configuration.
    """
    manager = get_model_manager()
    defaults = DefaultModels(**data.model_dump(exclude_unset=True))
    manager.set_defaults(defaults)
    manager.clear_cache()  # Clear cache when defaults change
    return defaults.model_dump()
