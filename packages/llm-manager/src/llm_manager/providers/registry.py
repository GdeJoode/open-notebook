"""
Provider registry and configuration.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""

    name: str = Field(..., description="Provider identifier")
    display_name: str = Field(..., description="Human-readable name")
    env_var_prefix: str = Field(..., description="Environment variable prefix for API key")
    api_key_env_var: str = Field(..., description="Environment variable name for API key")
    base_url_env_var: Optional[str] = Field(None, description="Env var for custom base URL")
    supports_language: bool = Field(True, description="Supports language models")
    supports_embedding: bool = Field(True, description="Supports embedding models")
    supports_speech_to_text: bool = Field(False, description="Supports STT")
    supports_text_to_speech: bool = Field(False, description="Supports TTS")
    supports_vision: bool = Field(False, description="Supports vision models")
    supports_tools: bool = Field(False, description="Supports tool calling")
    is_local: bool = Field(False, description="Is a local provider")
    default_base_url: Optional[str] = Field(None, description="Default base URL")


# Provider configurations
PROVIDER_CONFIGS: Dict[str, ProviderConfig] = {
    "openai": ProviderConfig(
        name="openai",
        display_name="OpenAI",
        env_var_prefix="OPENAI_",
        api_key_env_var="OPENAI_API_KEY",
        base_url_env_var="OPENAI_BASE_URL",
        supports_vision=True,
        supports_tools=True,
        supports_speech_to_text=True,
        supports_text_to_speech=True,
    ),
    "anthropic": ProviderConfig(
        name="anthropic",
        display_name="Anthropic",
        env_var_prefix="ANTHROPIC_",
        api_key_env_var="ANTHROPIC_API_KEY",
        supports_vision=True,
        supports_tools=True,
        supports_speech_to_text=False,
        supports_text_to_speech=False,
    ),
    "google": ProviderConfig(
        name="google",
        display_name="Google AI",
        env_var_prefix="GOOGLE_",
        api_key_env_var="GOOGLE_API_KEY",
        supports_vision=True,
        supports_tools=True,
    ),
    "groq": ProviderConfig(
        name="groq",
        display_name="Groq",
        env_var_prefix="GROQ_",
        api_key_env_var="GROQ_API_KEY",
        supports_vision=True,
        supports_tools=True,
        supports_embedding=False,
    ),
    "mistral": ProviderConfig(
        name="mistral",
        display_name="Mistral AI",
        env_var_prefix="MISTRAL_",
        api_key_env_var="MISTRAL_API_KEY",
        supports_vision=False,
        supports_tools=True,
    ),
    "ollama": ProviderConfig(
        name="ollama",
        display_name="Ollama (Local)",
        env_var_prefix="OLLAMA_",
        api_key_env_var="OLLAMA_API_KEY",
        base_url_env_var="OLLAMA_BASE_URL",
        default_base_url="http://localhost:11434",
        is_local=True,
        supports_vision=True,
        supports_tools=True,
    ),
    "llamacpp": ProviderConfig(
        name="llamacpp",
        display_name="LlamaCpp (Local)",
        env_var_prefix="LLAMACPP_",
        api_key_env_var="LLAMACPP_API_KEY",
        base_url_env_var="LLAMACPP_BASE_URL",
        default_base_url="http://localhost:8080",
        is_local=True,
        supports_vision=False,
        supports_tools=False,
    ),
    "openrouter": ProviderConfig(
        name="openrouter",
        display_name="OpenRouter",
        env_var_prefix="OPENROUTER_",
        api_key_env_var="OPENROUTER_API_KEY",
        base_url_env_var="OPENROUTER_BASE_URL",
        default_base_url="https://openrouter.ai/api/v1",
        supports_vision=True,
        supports_tools=True,
    ),
    "together": ProviderConfig(
        name="together",
        display_name="Together AI",
        env_var_prefix="TOGETHER_",
        api_key_env_var="TOGETHER_API_KEY",
        supports_vision=False,
        supports_tools=False,
    ),
    "deepseek": ProviderConfig(
        name="deepseek",
        display_name="DeepSeek",
        env_var_prefix="DEEPSEEK_",
        api_key_env_var="DEEPSEEK_API_KEY",
        supports_vision=False,
        supports_tools=True,
    ),
}


def get_provider_config(provider_name: str) -> Optional[ProviderConfig]:
    """
    Get configuration for a provider.

    Args:
        provider_name: Provider identifier.

    Returns:
        Provider configuration or None if not found.
    """
    return PROVIDER_CONFIGS.get(provider_name.lower())


def list_providers() -> List[ProviderConfig]:
    """
    List all registered providers.

    Returns:
        List of provider configurations.
    """
    return list(PROVIDER_CONFIGS.values())


def register_provider(config: ProviderConfig) -> None:
    """
    Register a new provider.

    Args:
        config: Provider configuration.
    """
    PROVIDER_CONFIGS[config.name.lower()] = config
