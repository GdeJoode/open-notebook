"""Models router - AI model management."""

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app_main.api.schemas import (
    DefaultModelsResponse,
    ModelCreate,
    ModelResponse,
    ProviderAvailabilityResponse,
)
from app_main.dependencies import get_model_service
from app_main.services.model_service import ModelService

router = APIRouter(prefix="/models", tags=["models"])


def _model_to_response(model) -> ModelResponse:
    """Convert a Model domain model to a ModelResponse schema."""
    return ModelResponse(
        id=model.id,
        name=model.name,
        provider=model.provider,
        type=model.type,
        created=str(model.created),
        updated=str(model.updated),
    )


@router.get("", response_model=list[ModelResponse])
async def list_models(
    type: Optional[str] = None,
    model_service: ModelService = Depends(get_model_service),
):
    """List all models, optionally filtered by type."""
    if type:
        models = await model_service.get_by_type(type)
    else:
        models = await model_service.get_all()
    return [_model_to_response(m) for m in models]


@router.post("", response_model=ModelResponse, status_code=201)
async def create_model(
    body: ModelCreate,
    model_service: ModelService = Depends(get_model_service),
):
    """Create a new model."""
    model = await model_service.create(body.model_dump())
    logger.info("Created model {} ({})", model.name, model.provider)
    return _model_to_response(model)


@router.delete("/{model_id}", status_code=204)
async def delete_model(
    model_id: str,
    model_service: ModelService = Depends(get_model_service),
):
    """Delete a model."""
    existing = await model_service.get(model_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Model not found")
    await model_service.delete(model_id)
    logger.info("Deleted model {}", model_id)


@router.get("/defaults", response_model=DefaultModelsResponse)
async def get_default_models(
    model_service: ModelService = Depends(get_model_service),
):
    """Get default model assignments."""
    defaults = await model_service.get_defaults()
    return DefaultModelsResponse(
        default_chat_model=defaults.default_chat_model,
        default_transformation_model=defaults.default_transformation_model,
        large_context_model=defaults.large_context_model,
        default_text_to_speech_model=defaults.default_text_to_speech_model,
        default_speech_to_text_model=defaults.default_speech_to_text_model,
        default_embedding_model=defaults.default_embedding_model,
        default_tools_model=defaults.default_tools_model,
    )


@router.put("/defaults", response_model=DefaultModelsResponse)
async def update_default_models(
    body: DefaultModelsResponse,
    model_service: ModelService = Depends(get_model_service),
):
    """Update default model assignments."""
    update_data = body.model_dump(exclude_none=True)
    defaults = await model_service.update_defaults(update_data)
    logger.info("Updated default models")
    return DefaultModelsResponse(
        default_chat_model=defaults.default_chat_model,
        default_transformation_model=defaults.default_transformation_model,
        large_context_model=defaults.large_context_model,
        default_text_to_speech_model=defaults.default_text_to_speech_model,
        default_speech_to_text_model=defaults.default_speech_to_text_model,
        default_embedding_model=defaults.default_embedding_model,
        default_tools_model=defaults.default_tools_model,
    )


@router.get("/providers", response_model=ProviderAvailabilityResponse)
async def get_providers(
    model_service: ModelService = Depends(get_model_service),
):
    """Get provider availability based on environment configuration."""
    import esperanto

    # Provider to environment variable mapping
    provider_env_keys = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "groq": "GROQ_API_KEY",
        "together": "TOGETHER_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "xai": "XAI_API_KEY",
        "elevenlabs": "ELEVENLABS_API_KEY",
        "ollama": None,  # Local, always available
    }

    available = []
    unavailable = []
    supported_types: dict[str, list[str]] = {}

    for provider, env_key in provider_env_keys.items():
        if env_key is None or os.environ.get(env_key):
            available.append(provider)
        else:
            unavailable.append(provider)

        # Determine supported model types per provider
        try:
            types = []
            for model_type in ["language", "embedding", "text_to_speech", "speech_to_text"]:
                try:
                    esperanto.AIFactory(
                        model_name="test",
                        provider=provider,
                        model_type=model_type,
                    )
                    types.append(model_type)
                except Exception:
                    pass
            if types:
                supported_types[provider] = types
        except Exception:
            pass

    return ProviderAvailabilityResponse(
        available=available,
        unavailable=unavailable,
        supported_types=supported_types,
    )
