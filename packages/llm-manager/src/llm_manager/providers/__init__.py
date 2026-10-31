"""
Providerregister (metadata, vervalt in F2-09).

De client naar llama-swap staat in :mod:`llm_manager.providers.factory` en wordt
alleen gebruikt door :mod:`llm_manager.aanroep` (F2-08).
"""

from llm_manager.providers.registry import (
    PROVIDER_CONFIGS,
    ProviderConfig,
    get_provider_config,
    list_providers,
    register_provider,
)

__all__ = [
    "ProviderConfig",
    "PROVIDER_CONFIGS",
    "get_provider_config",
    "list_providers",
    "register_provider",
]
