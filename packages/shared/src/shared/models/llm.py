"""
LLM and AI model definitions.

Pure Pydantic models for AI models. Database persistence handled by surrealdb-service.
Welke modellen er zijn, staat in het modelregister (``config/modelregister.yaml``,
ADR 0007), niet in de database.
"""

from typing import ClassVar, Optional

from pydantic import Field

from shared.models.base import ObjectModel


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
    # Track J.4 routing telemetry (J-Q7). ``provider`` above is the requested
    # head; these record what actually SERVED the call after any failover.
    served_provider: Optional[str] = Field(
        None, description="Provider that actually produced the result after failover"
    )
    fallback_from: Optional[list] = Field(
        None,
        description="Providers attempted (failed/skipped) before the served one",
    )
    was_failover: bool = Field(
        False, description="True when the served provider was not the route head"
    )
