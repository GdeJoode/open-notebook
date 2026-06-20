"""Models router - AI model management."""

import os
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger

from app_main.api.schemas import (
    DefaultModelsResponse,
    ModelCreate,
    ModelResponse,
    ModelStatusIssue,
    ModelStatusResponse,
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
        default_extraction_model=defaults.default_extraction_model,
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
        default_extraction_model=defaults.default_extraction_model,
    )


EMBEDDING_MODEL_PATTERNS = {"embed", "mxbai-embed", "nomic-embed", "all-minilm", "bge-", "e5-"}


def _is_embedding_model(name: str) -> bool:
    """Heuristic: detect embedding models by name patterns."""
    lower = name.lower().split(":")[0]
    return any(pattern in lower for pattern in EMBEDDING_MODEL_PATTERNS)


@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status(
    model_service: ModelService = Depends(get_model_service),
):
    """Check model configuration validity.

    Queries Ollama for available models, syncs discovered models into the
    database, and verifies that all default model slots are properly configured.
    """
    import httpx

    issues: list[ModelStatusIssue] = []

    # -- Sync Ollama models into the database --
    ollama_base = os.environ.get("OLLAMA_API_BASE", "http://localhost:11434")
    ollama_reachable = False
    ollama_available_names: set[str] = set()

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ollama_base}/api/tags")
            resp.raise_for_status()
            data = resp.json()
            ollama_reachable = True

            # Collect available Ollama model names (without :latest suffix)
            ollama_models_raw: list[dict] = data.get("models", [])
            for m in ollama_models_raw:
                name = m.get("name", "")
                ollama_available_names.add(name)
                if ":" in name:
                    ollama_available_names.add(name.split(":")[0])

            # Sync: register any Ollama models not already in the DB
            all_models = await model_service.get_all()
            existing_ollama = {
                m.name for m in all_models if m.provider == "ollama"
            }

            for m in ollama_models_raw:
                raw_name = m.get("name", "")
                # Normalise: strip :latest tag for the DB entry
                model_name = raw_name.split(":")[0] if raw_name.endswith(":latest") else raw_name
                if model_name in existing_ollama or raw_name in existing_ollama:
                    continue
                model_type = "embedding" if _is_embedding_model(model_name) else "language"
                await model_service.create({
                    "name": model_name,
                    "provider": "ollama",
                    "type": model_type,
                    "is_local": True,
                })
                logger.info("Auto-registered Ollama model: {} ({})", model_name, model_type)

            # Remove DB entries for Ollama models that are no longer pulled
            for m in all_models:
                if m.provider != "ollama":
                    continue
                base_name = m.name.split(":")[0] if ":" in m.name else m.name
                if m.name not in ollama_available_names and base_name not in ollama_available_names:
                    await model_service.delete(m.id)
                    logger.info("Removed stale Ollama model from DB: {}", m.name)

            # Reload models after sync
            all_models = await model_service.get_all()

    except Exception as e:
        logger.warning(f"Cannot reach Ollama at {ollama_base}: {e}")
        issues.append(ModelStatusIssue(
            type="ollama_unreachable",
            message=f"Cannot reach Ollama at {ollama_base}",
            provider="ollama",
        ))
        all_models = await model_service.get_all()

    # -- Check defaults --
    defaults = await model_service.get_defaults()
    models_by_id = {m.id: m for m in all_models}

    default_slots = {
        "default_chat_model": defaults.default_chat_model,
        "default_transformation_model": defaults.default_transformation_model,
        "large_context_model": defaults.large_context_model,
        "default_text_to_speech_model": defaults.default_text_to_speech_model,
        "default_speech_to_text_model": defaults.default_speech_to_text_model,
        "default_embedding_model": defaults.default_embedding_model,
        "default_tools_model": defaults.default_tools_model,
    }

    # At minimum, chat + embedding must be set
    critical_slots = ["default_chat_model", "default_embedding_model"]
    for slot in critical_slots:
        if not default_slots.get(slot):
            issues.append(ModelStatusIssue(
                type="no_defaults",
                message=f"No model configured for {slot.replace('_', ' ')}",
                default_slot=slot,
            ))

    # -- Check default models still exist and their providers are available --
    cloud_providers_in_defaults: dict[str, list[str]] = {}

    for slot_name, model_id in default_slots.items():
        if not model_id:
            continue
        model = models_by_id.get(model_id)
        if not model:
            continue
        if model.provider == "ollama":
            if ollama_reachable:
                base_name = model.name.split(":")[0] if ":" in model.name else model.name
                if model.name not in ollama_available_names and base_name not in ollama_available_names:
                    issues.append(ModelStatusIssue(
                        type="model_unavailable",
                        message=f"Ollama model '{model.name}' is not pulled/available",
                        model_name=model.name,
                        provider="ollama",
                        default_slot=slot_name,
                    ))
        else:
            cloud_providers_in_defaults.setdefault(model.provider, []).append(slot_name)

    # -- Check cloud provider API keys --
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
    }

    for provider, slots in cloud_providers_in_defaults.items():
        env_key = provider_env_keys.get(provider)
        if env_key and not os.environ.get(env_key):
            issues.append(ModelStatusIssue(
                type="provider_unavailable",
                message=f"API key for {provider} not configured ({env_key})",
                provider=provider,
                default_slot=slots[0],
            ))

    return ModelStatusResponse(valid=len(issues) == 0, issues=issues)


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
