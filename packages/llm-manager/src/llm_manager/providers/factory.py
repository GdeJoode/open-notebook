"""
Model factory for creating AI model instances.
"""

import os
from typing import Any, Dict, Optional, Tuple, Union

from esperanto import (
    AIFactory,
    EmbeddingModel,
    LanguageModel,
    SpeechToTextModel,
    TextToSpeechModel,
)
from loguru import logger
from shared.models import Model, ModelTypeStr

ModelInstance = Union[LanguageModel, EmbeddingModel, SpeechToTextModel, TextToSpeechModel]


def _resolve_provider_endpoint(
    provider: str, config: Dict[str, Any]
) -> Tuple[str, Dict[str, Any]]:
    """Resolve the esperanto provider name + endpoint config for ``provider``.

    Most providers pass through unchanged. Cloud providers that are
    OpenAI-compatible but have no native esperanto provider (NVIDIA NIM, Track
    J.4) are mapped to esperanto's ``openai-compatible`` LanguageModel with the
    provider's ``default_base_url`` + API key threaded into the config so a
    single ``model`` row carrying ``provider="nvidia"`` reaches the NIM
    endpoint. A caller-supplied ``base_url`` / ``api_key`` in ``config`` always
    wins over the registry defaults.
    """
    from llm_manager.providers.registry import get_provider_config

    cfg = get_provider_config(provider)
    if cfg is None or cfg.is_local:
        return provider, config

    # Providers esperanto supports natively keep their name; only those without
    # a native esperanto LLM provider but a known OpenAI-compatible endpoint are
    # remapped. NVIDIA NIM is the J.4 case.
    _OPENAI_COMPATIBLE = {"nvidia"}
    if provider.lower() not in _OPENAI_COMPATIBLE:
        return provider, config

    resolved = dict(config)
    base_url = resolved.get("base_url")
    if not base_url and cfg.base_url_env_var:
        base_url = os.environ.get(cfg.base_url_env_var)
    if not base_url:
        base_url = cfg.default_base_url
    if base_url:
        resolved["base_url"] = base_url

    if not resolved.get("api_key"):
        api_key = os.environ.get(cfg.api_key_env_var)
        if api_key:
            resolved["api_key"] = api_key

    return "openai-compatible", resolved


class ModelFactory:
    """
    Factory for creating AI model instances using Esperanto.
    """

    @staticmethod
    def create_model(
        model: Model,
        config: Optional[Dict[str, Any]] = None,
    ) -> ModelInstance:
        """
        Create a model instance from a Model configuration.

        Args:
            model: Model configuration from database.
            config: Additional configuration options.

        Returns:
            Initialized model instance.

        Raises:
            ValueError: If model type is invalid.
        """
        config = config or {}
        model_type = model.type

        if model_type == "language":
            return ModelFactory.create_language_model(
                model_name=model.name,
                provider=model.provider,
                config=config,
            )
        elif model_type == "embedding":
            return ModelFactory.create_embedding_model(
                model_name=model.name,
                provider=model.provider,
                config=config,
            )
        elif model_type == "speech_to_text":
            return ModelFactory.create_speech_to_text_model(
                model_name=model.name,
                provider=model.provider,
                config=config,
            )
        elif model_type == "text_to_speech":
            return ModelFactory.create_text_to_speech_model(
                model_name=model.name,
                provider=model.provider,
                config=config,
            )
        else:
            raise ValueError(f"Invalid model type: {model_type}")

    @staticmethod
    def create_language_model(
        model_name: str,
        provider: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> LanguageModel:
        """
        Create a language model instance.

        Args:
            model_name: Name of the model.
            provider: Provider name.
            config: Additional configuration.

        Returns:
            Language model instance.
        """
        config = config or {}
        esperanto_provider, resolved_config = _resolve_provider_endpoint(
            provider, config
        )
        logger.debug(
            f"Creating language model: {provider}/{model_name}"
            + (
                f" (via esperanto {esperanto_provider})"
                if esperanto_provider != provider
                else ""
            )
        )
        return AIFactory.create_language(
            model_name=model_name,
            provider=esperanto_provider,
            config=resolved_config,
        )

    @staticmethod
    def create_embedding_model(
        model_name: str,
        provider: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> EmbeddingModel:
        """
        Create an embedding model instance.

        Args:
            model_name: Name of the model.
            provider: Provider name.
            config: Additional configuration.

        Returns:
            Embedding model instance.
        """
        config = config or {}
        logger.debug(f"Creating embedding model: {provider}/{model_name}")
        return AIFactory.create_embedding(
            model_name=model_name,
            provider=provider,
            config=config,
        )

    @staticmethod
    def create_speech_to_text_model(
        model_name: str,
        provider: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> SpeechToTextModel:
        """
        Create a speech-to-text model instance.

        Args:
            model_name: Name of the model.
            provider: Provider name.
            config: Additional configuration.

        Returns:
            Speech-to-text model instance.
        """
        config = config or {}
        logger.debug(f"Creating STT model: {provider}/{model_name}")
        return AIFactory.create_speech_to_text(
            model_name=model_name,
            provider=provider,
            config=config,
        )

    @staticmethod
    def create_text_to_speech_model(
        model_name: str,
        provider: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> TextToSpeechModel:
        """
        Create a text-to-speech model instance.

        Args:
            model_name: Name of the model.
            provider: Provider name.
            config: Additional configuration.

        Returns:
            Text-to-speech model instance.
        """
        config = config or {}
        logger.debug(f"Creating TTS model: {provider}/{model_name}")
        return AIFactory.create_text_to_speech(
            model_name=model_name,
            provider=provider,
            config=config,
        )

    @staticmethod
    def create_by_type(
        model_type: ModelTypeStr,
        model_name: str,
        provider: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> ModelInstance:
        """
        Create a model instance by type.

        Args:
            model_type: Type of model to create.
            model_name: Name of the model.
            provider: Provider name.
            config: Additional configuration.

        Returns:
            Model instance of the specified type.

        Raises:
            ValueError: If model type is invalid.
        """
        if model_type == "language":
            return ModelFactory.create_language_model(model_name, provider, config)
        elif model_type == "embedding":
            return ModelFactory.create_embedding_model(model_name, provider, config)
        elif model_type == "speech_to_text":
            return ModelFactory.create_speech_to_text_model(model_name, provider, config)
        elif model_type == "text_to_speech":
            return ModelFactory.create_text_to_speech_model(model_name, provider, config)
        else:
            raise ValueError(f"Invalid model type: {model_type}")
