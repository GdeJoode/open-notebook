from esperanto import LanguageModel
from langchain_core.language_models.chat_models import BaseChatModel
from loguru import logger

from llm_manager import ModelManager


def _get_model_manager() -> ModelManager:
    """Get or create the ModelManager singleton."""
    return ModelManager()


def _token_count(content: str) -> int:
    """Count tokens in content."""
    from shared.utils import token_count

    return token_count(content)


async def provision_langchain_model(
    content, model_id, default_type, **kwargs
) -> BaseChatModel:
    """
    Returns the best model to use based on the context size and on whether
    there is a specific model being requested in Config.
    If context > 105_000, returns the large_context_model
    If model_id is specified in Config, returns that model
    Otherwise, returns the default model for the given type
    """
    tokens = _token_count(str(content))
    manager = _get_model_manager()

    if tokens > 105_000:
        logger.debug(
            f"Using large context model because the content has {tokens} tokens"
        )
        model = await manager.get_default_model("large_context", **kwargs)
    elif model_id:
        model = await manager.get_model(model_id, **kwargs)
    else:
        model = await manager.get_default_model(default_type, **kwargs)

    logger.debug(f"Using model: {model}")
    assert isinstance(model, LanguageModel), f"Model is not a LanguageModel: {model}"
    return model.to_langchain()
