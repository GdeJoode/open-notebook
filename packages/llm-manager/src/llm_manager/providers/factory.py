"""
Model factory for creating AI model instances.
"""

from typing import Any, Dict, Optional, Union

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
        logger.debug(f"Creating language model: {provider}/{model_name}")
        return AIFactory.create_language(
            model_name=model_name,
            provider=provider,
            config=config,
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
