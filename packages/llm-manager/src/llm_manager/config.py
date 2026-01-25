"""
Configuration for LLM manager service.
"""

from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class LLMManagerConfig(BaseSettings):
    """LLM manager configuration."""

    model_config = {
        "env_prefix": "LLM_",
        "env_file": ".env",
        "extra": "ignore",
    }

    # Default provider settings
    default_provider: str = Field(
        default="openai",
        description="Default LLM provider",
    )

    # Cache settings
    cache_enabled: bool = Field(
        default=True,
        description="Enable model instance caching",
    )
    cache_ttl_seconds: int = Field(
        default=3600,
        description="Cache TTL in seconds",
    )

    # Usage tracking
    track_usage: bool = Field(
        default=True,
        description="Track model usage and costs",
    )

    # Rate limiting
    rate_limit_enabled: bool = Field(
        default=False,
        description="Enable rate limiting",
    )
    rate_limit_requests_per_minute: int = Field(
        default=60,
        description="Rate limit: requests per minute",
    )

    # Retry settings
    max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for failed requests",
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        description="Initial delay between retries",
    )
    retry_exponential_backoff: bool = Field(
        default=True,
        description="Use exponential backoff for retries",
    )

    # Timeout settings
    request_timeout_seconds: int = Field(
        default=120,
        description="Request timeout in seconds",
    )

    # Streaming settings
    default_streaming: bool = Field(
        default=True,
        description="Enable streaming by default",
    )

    # Local model settings
    local_model_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for local models (e.g., Ollama)",
    )
    local_model_provider: Literal["ollama", "llamacpp", "vllm"] = Field(
        default="ollama",
        description="Local model provider type",
    )


# Global config instance
_config: Optional[LLMManagerConfig] = None


def get_config() -> LLMManagerConfig:
    """Get the global LLM manager configuration."""
    global _config
    if _config is None:
        _config = LLMManagerConfig()
    return _config


def set_config(config: LLMManagerConfig) -> None:
    """Set the global LLM manager configuration."""
    global _config
    _config = config
