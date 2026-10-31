"""Models router - AI model management.

``GET /api/models`` is een alleen-lezen weergave van het modelregister
(``config/modelregister.yaml``, F2-07, ADR 0007); de schrijfroutes voor
modellen zijn vervallen. De standaardmodellen, de status en de providers lezen
nog de modelrijen in de database, tot hun lezers zijn omgezet (F2-08, F2-09).
"""

import os

from fastapi import APIRouter, Depends
from llm_manager.register import RegisterModel, get_register
from loguru import logger

from app_main.api.schemas import (
    DefaultModelsResponse,
    ModelStatusIssue,
    ModelStatusResponse,
    ProviderAvailabilityResponse,
)
from app_main.dependencies import get_model_service
from app_main.services.model_service import ModelService
from app_main.api.agent_auth import RECHT_BEHEERDER

router = APIRouter(prefix="/models", tags=["models"], dependencies=[RECHT_BEHEERDER])


@router.get("", response_model=list[RegisterModel])
async def list_models():
    """De modellen uit het register; alleen lezen."""
    return get_register().modellen


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


@router.get("/status", response_model=ModelStatusResponse)
async def get_model_status(
    model_service: ModelService = Depends(get_model_service),
):
    """Check model configuration validity.

    F2-08: alle modellen draaien achter llama-swap en staan in het register;
    er is geen Ollama meer om te bevragen of te synchroniseren. Deze route
    controleert nog de standaardmodellen in de database en vervalt in F2-09.
    """
    issues: list[ModelStatusIssue] = []
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
            continue
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
    """Get provider availability based on environment configuration.

    F2-08: de modelsoorten per aanbieder worden niet meer bepaald door een
    model op te bouwen; ``supported_types`` blijft leeg. Vervalt in F2-09.
    """

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

    return ProviderAvailabilityResponse(
        available=available,
        unavailable=unavailable,
        supported_types=supported_types,
    )
