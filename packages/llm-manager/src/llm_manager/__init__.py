"""
LLM Manager - LLM provider management, routing, and usage tracking.

This package provides a unified interface for working with different LLM providers,
model instance management, and usage tracking.
"""

from llm_manager.config import LLMManagerConfig, get_config, set_config
from llm_manager.manager import ModelManager, get_model_manager
from llm_manager.providers import (
    ModelFactory,
    ModelInstance,
    ProviderConfig,
    get_provider_config,
    list_providers,
)
from llm_manager.usage import UsageMetrics, UsageTracker

__all__ = [
    # Config
    "LLMManagerConfig",
    "get_config",
    "set_config",
    # Manager
    "ModelManager",
    "get_model_manager",
    # Providers
    "ModelFactory",
    "ModelInstance",
    "ProviderConfig",
    "get_provider_config",
    "list_providers",
    # Usage
    "UsageTracker",
    "UsageMetrics",
]
