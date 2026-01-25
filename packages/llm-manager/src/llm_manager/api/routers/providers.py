"""
Provider information API routes.
"""

import os
from typing import Any, Dict, List

from fastapi import APIRouter
from pydantic import BaseModel

from llm_manager.providers import (
    PROVIDER_CONFIGS,
    ProviderConfig,
    get_provider_config,
    list_providers,
)

router = APIRouter()


class ProviderResponse(BaseModel):
    """Provider information response."""

    name: str
    display_name: str
    supports_language: bool
    supports_embedding: bool
    supports_speech_to_text: bool
    supports_text_to_speech: bool
    supports_vision: bool
    supports_tools: bool
    is_local: bool
    is_configured: bool


class ProviderDetailResponse(ProviderResponse):
    """Detailed provider information."""

    env_var_prefix: str
    api_key_env_var: str
    base_url_env_var: str | None
    default_base_url: str | None


@router.get("", response_model=List[ProviderResponse])
async def get_providers() -> List[ProviderResponse]:
    """
    Get list of all supported providers.

    Returns:
        List of provider information.
    """
    providers = list_providers()
    return [
        ProviderResponse(
            name=p.name,
            display_name=p.display_name,
            supports_language=p.supports_language,
            supports_embedding=p.supports_embedding,
            supports_speech_to_text=p.supports_speech_to_text,
            supports_text_to_speech=p.supports_text_to_speech,
            supports_vision=p.supports_vision,
            supports_tools=p.supports_tools,
            is_local=p.is_local,
            is_configured=bool(os.environ.get(p.api_key_env_var)),
        )
        for p in providers
    ]


@router.get("/configured", response_model=List[ProviderResponse])
async def get_configured_providers() -> List[ProviderResponse]:
    """
    Get list of providers that have API keys configured.

    Returns:
        List of configured provider information.
    """
    providers = list_providers()
    configured = [
        p for p in providers
        if os.environ.get(p.api_key_env_var) or p.is_local
    ]
    return [
        ProviderResponse(
            name=p.name,
            display_name=p.display_name,
            supports_language=p.supports_language,
            supports_embedding=p.supports_embedding,
            supports_speech_to_text=p.supports_speech_to_text,
            supports_text_to_speech=p.supports_text_to_speech,
            supports_vision=p.supports_vision,
            supports_tools=p.supports_tools,
            is_local=p.is_local,
            is_configured=True,
        )
        for p in configured
    ]


@router.get("/{provider_name}", response_model=ProviderDetailResponse)
async def get_provider(provider_name: str) -> ProviderDetailResponse:
    """
    Get detailed information about a specific provider.

    Args:
        provider_name: Provider name.

    Returns:
        Provider details.
    """
    config = get_provider_config(provider_name)
    if not config:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_name}")

    return ProviderDetailResponse(
        name=config.name,
        display_name=config.display_name,
        supports_language=config.supports_language,
        supports_embedding=config.supports_embedding,
        supports_speech_to_text=config.supports_speech_to_text,
        supports_text_to_speech=config.supports_text_to_speech,
        supports_vision=config.supports_vision,
        supports_tools=config.supports_tools,
        is_local=config.is_local,
        is_configured=bool(os.environ.get(config.api_key_env_var)),
        env_var_prefix=config.env_var_prefix,
        api_key_env_var=config.api_key_env_var,
        base_url_env_var=config.base_url_env_var,
        default_base_url=config.default_base_url,
    )


@router.get("/{provider_name}/capabilities")
async def get_provider_capabilities(provider_name: str) -> Dict[str, Any]:
    """
    Get capability summary for a provider.

    Args:
        provider_name: Provider name.

    Returns:
        Provider capabilities.
    """
    config = get_provider_config(provider_name)
    if not config:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail=f"Provider not found: {provider_name}")

    return {
        "provider": config.name,
        "capabilities": {
            "language_models": config.supports_language,
            "embedding_models": config.supports_embedding,
            "speech_to_text": config.supports_speech_to_text,
            "text_to_speech": config.supports_text_to_speech,
            "vision": config.supports_vision,
            "tool_calling": config.supports_tools,
            "local_deployment": config.is_local,
        },
        "model_types_supported": [
            t for t, supported in [
                ("language", config.supports_language),
                ("embedding", config.supports_embedding),
                ("speech_to_text", config.supports_speech_to_text),
                ("text_to_speech", config.supports_text_to_speech),
            ] if supported
        ],
    }
