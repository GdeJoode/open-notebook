"""
Model manager: standaardmodellen en cache-statistiek.

Modellen opbouwen en aanroepen gebeurt hier niet meer: dat loopt alleen via
:mod:`llm_manager.aanroep` en :class:`llm_manager.gate.ModelGate` naar llama-swap
(F2-08, ADR 0006). De standaardmodellen in de database vervallen in F2-09.
"""

import time
from typing import Any, Dict, Optional

from loguru import logger

from llm_manager.config import LLMManagerConfig, get_config
from shared.models import DefaultModels


class ModelManager:
    """
    Houdt de standaardmodellen bij. De cache blijft leeg sinds F2-08: er worden
    hier geen modellen meer opgebouwd. Vervalt in F2-09.
    """

    _instance: Optional["ModelManager"] = None

    def __new__(cls, *args, **kwargs) -> "ModelManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[LLMManagerConfig] = None):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self.config = config or get_config()
        self._model_cache: Dict[str, Any] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._defaults: Optional[DefaultModels] = None
        logger.info("ModelManager initialized")

    def set_defaults(self, defaults: DefaultModels) -> None:
        """
        Set the default models configuration.

        Args:
            defaults: Default models configuration.
        """
        self._defaults = defaults
        logger.info("Default models configuration updated")

    def get_defaults(self) -> Optional[DefaultModels]:
        """
        Get the current default models configuration.

        Returns:
            Default models configuration or None.
        """
        return self._defaults

    def clear_cache(self) -> None:
        """Clear all cached model instances."""
        self._model_cache.clear()
        self._cache_timestamps.clear()
        logger.info("Model cache cleared")

    def invalidate_model(self, model_id: str) -> None:
        """
        Invalidate cache entries for a specific model.

        Args:
            model_id: ID of the model to invalidate.
        """
        keys_to_remove = [k for k in self._model_cache if k.startswith(f"{model_id}:")]
        for key in keys_to_remove:
            del self._model_cache[key]
            del self._cache_timestamps[key]
        if keys_to_remove:
            logger.info(f"Invalidated {len(keys_to_remove)} cache entries for model: {model_id}")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats.
        """
        now = time.time()
        valid_entries = sum(
            1 for key in self._cache_timestamps
            if now - self._cache_timestamps[key] < self.config.cache_ttl_seconds
        )
        return {
            "total_entries": len(self._model_cache),
            "valid_entries": valid_entries,
            "cache_enabled": self.config.cache_enabled,
            "ttl_seconds": self.config.cache_ttl_seconds,
        }

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance (for testing)."""
        cls._instance = None


# Global convenience function
def get_model_manager() -> ModelManager:
    """Get the global model manager instance."""
    return ModelManager()
