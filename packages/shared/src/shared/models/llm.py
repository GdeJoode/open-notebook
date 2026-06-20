"""
LLM and AI model definitions.

Pure Pydantic models for AI models. Database persistence handled by surrealdb-service.
"""

from typing import ClassVar, Literal, Optional

from pydantic import Field

from shared.models.base import ObjectModel, RecordModel


ModelTypeStr = Literal["language", "embedding", "speech_to_text", "text_to_speech"]


class Model(ObjectModel):
    """
    AI model configuration.

    Represents a configured AI model from a provider.
    """

    table_name: ClassVar[str] = "model"

    name: str = Field(..., description="Model name/identifier from the provider")
    provider: str = Field(..., description="Provider name (openai, anthropic, etc.)")
    type: ModelTypeStr = Field(..., description="Model type")
    display_name: Optional[str] = Field(None, description="Human-readable display name")
    description: Optional[str] = Field(None, description="Model description")
    context_window: Optional[int] = Field(None, description="Context window size in tokens")
    max_output_tokens: Optional[int] = Field(None, description="Maximum output tokens")
    supports_vision: Optional[bool] = Field(False, description="Whether model supports vision/images")
    supports_tools: Optional[bool] = Field(False, description="Whether model supports tool calling")
    supports_streaming: Optional[bool] = Field(True, description="Whether model supports streaming")
    input_cost_per_1k: Optional[float] = Field(None, description="Cost per 1K input tokens in USD")
    output_cost_per_1k: Optional[float] = Field(None, description="Cost per 1K output tokens in USD")
    is_local: Optional[bool] = Field(False, description="Whether this is a local model")
    is_enabled: Optional[bool] = Field(True, description="Whether model is enabled for use")


class DefaultModels(RecordModel):
    """
    Default model configuration for each use case.

    Singleton record with default model selections.
    """

    record_id: ClassVar[str] = "open_notebook:default_models"

    default_chat_model: Optional[str] = Field(
        None, description="Default model for chat/conversation"
    )
    default_transformation_model: Optional[str] = Field(
        None, description="Default model for text transformation tasks"
    )
    large_context_model: Optional[str] = Field(
        None, description="Model for large context operations"
    )
    default_text_to_speech_model: Optional[str] = Field(
        None, description="Default text-to-speech model"
    )
    default_speech_to_text_model: Optional[str] = Field(
        None, description="Default speech-to-text model"
    )
    default_embedding_model: Optional[str] = Field(
        None, description="Default embedding model"
    )
    default_tools_model: Optional[str] = Field(
        None, description="Default model for tool-using tasks"
    )
    default_extraction_model: Optional[str] = Field(
        None,
        description=(
            "Default model for knowledge-graph entity/relation extraction. "
            "Independent of the chat model; falls back to default_chat_model "
            "when unset."
        ),
    )


class ModelUsageRecord(ObjectModel):
    """
    Record of model usage for tracking and billing.
    """

    table_name: ClassVar[str] = "model_usage"

    model_id: str = Field(..., description="ID of the model used")
    model_name: str = Field(..., description="Name of the model")
    provider: str = Field(..., description="Provider name")
    operation_type: str = Field(..., description="Type of operation (chat, embedding, etc.)")
    input_tokens: int = Field(0, description="Number of input tokens")
    output_tokens: int = Field(0, description="Number of output tokens")
    total_tokens: int = Field(0, description="Total tokens used")
    cost_usd: Optional[float] = Field(None, description="Estimated cost in USD")
    duration_ms: Optional[int] = Field(None, description="Duration in milliseconds")
    success: bool = Field(True, description="Whether the operation succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    metadata: Optional[dict] = Field(None, description="Additional metadata")
