"""
Model manager for centralized model instance management.
"""

import time
from typing import Any, Dict, Optional, Union

from esperanto import EmbeddingModel, LanguageModel, SpeechToTextModel, TextToSpeechModel
from loguru import logger

from llm_manager.config import LLMManagerConfig, get_config
from llm_manager.providers.factory import ModelFactory, ModelInstance
from shared.models import DefaultModels, Model


class ModelManager:
    """
    Centralized manager for AI model instances.

    Provides:
    - Model instance caching
    - Default model resolution
    - Lazy loading of models
    - Cache invalidation

    Note: This is a standalone manager that requires model and defaults
    to be passed in. For database-backed usage, use with repositories.
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
        self._model_cache: Dict[str, ModelInstance] = {}
        self._cache_timestamps: Dict[str, float] = {}
        self._defaults: Optional[DefaultModels] = None
        logger.info("ModelManager initialized")

    def _get_cache_key(self, model_id: str, kwargs: Dict[str, Any]) -> str:
        """Generate cache key from model ID and kwargs."""
        return f"{model_id}:{str(sorted(kwargs.items()))}"

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached entry is still valid."""
        if not self.config.cache_enabled:
            return False
        if cache_key not in self._cache_timestamps:
            return False
        age = time.time() - self._cache_timestamps[cache_key]
        return age < self.config.cache_ttl_seconds

    def get_model_from_config(
        self,
        model: Model,
        **kwargs: Any,
    ) -> ModelInstance:
        """
        Get or create a model instance from a Model configuration.

        Args:
            model: Model configuration.
            **kwargs: Additional configuration options.

        Returns:
            Model instance.
        """
        if not model.id:
            # No caching for models without ID
            return ModelFactory.create_model(model, kwargs)

        cache_key = self._get_cache_key(model.id, kwargs)

        if self._is_cache_valid(cache_key):
            logger.debug(f"Cache hit for model: {model.id}")
            return self._model_cache[cache_key]

        logger.debug(f"Creating new model instance: {model.id}")
        instance = ModelFactory.create_model(model, kwargs)

        if self.config.cache_enabled:
            self._model_cache[cache_key] = instance
            self._cache_timestamps[cache_key] = time.time()

        return instance

    def create_model_direct(
        self,
        model_name: str,
        provider: str,
        model_type: str,
        **kwargs: Any,
    ) -> ModelInstance:
        """
        Create a model instance directly without database lookup.

        Args:
            model_name: Name of the model.
            provider: Provider name.
            model_type: Type of model (language, embedding, etc.).
            **kwargs: Additional configuration options.

        Returns:
            Model instance.
        """
        return ModelFactory.create_by_type(
            model_type=model_type,  # type: ignore
            model_name=model_name,
            provider=provider,
            config=kwargs,
        )

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
