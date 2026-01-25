"""
Usage tracking API routes.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from llm_manager.usage import UsageMetrics, UsageTracker

router = APIRouter()

# Global tracker instance (should be injected in production)
_tracker: Optional[UsageTracker] = None


def get_tracker() -> UsageTracker:
    """Get or create the global usage tracker."""
    global _tracker
    if _tracker is None:
        _tracker = UsageTracker(enabled=True)
    return _tracker


def set_tracker(tracker: UsageTracker) -> None:
    """Set the global usage tracker."""
    global _tracker
    _tracker = tracker


class UsageRecordResponse(BaseModel):
    """Usage record response."""

    model_id: str
    model_name: str
    provider: str
    operation_type: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: Optional[float]
    duration_ms: Optional[int]
    success: bool
    error_message: Optional[str]


class UsageMetricsResponse(BaseModel):
    """Usage metrics response."""

    total_requests: int
    total_input_tokens: int
    total_output_tokens: int
    total_tokens: int
    total_cost_usd: float
    total_duration_ms: int
    successful_requests: int
    failed_requests: int
    by_model: Dict[str, Dict[str, Any]]
    by_provider: Dict[str, Dict[str, Any]]
    by_operation: Dict[str, Dict[str, Any]]


class CostEstimateRequest(BaseModel):
    """Cost estimate request."""

    input_tokens: int
    output_tokens: int
    input_cost_per_1k: float
    output_cost_per_1k: float


class CostEstimateResponse(BaseModel):
    """Cost estimate response."""

    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float


@router.get("/metrics", response_model=UsageMetricsResponse)
async def get_usage_metrics() -> UsageMetricsResponse:
    """
    Get aggregated usage metrics.

    Returns:
        Usage metrics summary.
    """
    tracker = get_tracker()
    metrics = tracker.get_metrics()
    return UsageMetricsResponse(
        total_requests=metrics.total_requests,
        total_input_tokens=metrics.total_input_tokens,
        total_output_tokens=metrics.total_output_tokens,
        total_tokens=metrics.total_tokens,
        total_cost_usd=metrics.total_cost_usd,
        total_duration_ms=metrics.total_duration_ms,
        successful_requests=metrics.successful_requests,
        failed_requests=metrics.failed_requests,
        by_model=metrics.by_model,
        by_provider=metrics.by_provider,
        by_operation=metrics.by_operation,
    )


@router.get("/history", response_model=List[UsageRecordResponse])
async def get_usage_history(
    limit: int = Query(100, ge=1, le=1000),
    model_id: Optional[str] = None,
    provider: Optional[str] = None,
    operation_type: Optional[str] = None,
) -> List[UsageRecordResponse]:
    """
    Get usage history with optional filtering.

    Args:
        limit: Maximum records to return.
        model_id: Filter by model ID.
        provider: Filter by provider.
        operation_type: Filter by operation type.

    Returns:
        List of usage records.
    """
    tracker = get_tracker()
    records = tracker.get_history(
        limit=limit,
        model_id=model_id,
        provider=provider,
        operation_type=operation_type,
    )
    return [
        UsageRecordResponse(
            model_id=r.model_id,
            model_name=r.model_name,
            provider=r.provider,
            operation_type=r.operation_type,
            input_tokens=r.input_tokens,
            output_tokens=r.output_tokens,
            total_tokens=r.total_tokens,
            cost_usd=r.cost_usd,
            duration_ms=r.duration_ms,
            success=r.success,
            error_message=r.error_message,
        )
        for r in records
    ]


@router.post("/estimate", response_model=CostEstimateResponse)
async def estimate_cost(data: CostEstimateRequest) -> CostEstimateResponse:
    """
    Estimate cost for given token counts.

    Args:
        data: Token counts and pricing.

    Returns:
        Cost estimate.
    """
    tracker = get_tracker()
    total = tracker.estimate_cost(
        input_tokens=data.input_tokens,
        output_tokens=data.output_tokens,
        input_cost_per_1k=data.input_cost_per_1k,
        output_cost_per_1k=data.output_cost_per_1k,
    )
    input_cost = (data.input_tokens / 1000) * data.input_cost_per_1k
    output_cost = (data.output_tokens / 1000) * data.output_cost_per_1k

    return CostEstimateResponse(
        input_tokens=data.input_tokens,
        output_tokens=data.output_tokens,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=total,
    )


@router.post("/reset")
async def reset_usage() -> Dict[str, str]:
    """
    Reset all usage tracking data.

    Returns:
        Confirmation message.
    """
    tracker = get_tracker()
    tracker.reset()
    return {"status": "Usage tracking data reset"}


@router.get("/summary/by-model")
async def get_usage_by_model() -> Dict[str, Any]:
    """
    Get usage summary grouped by model.

    Returns:
        Usage summary by model.
    """
    tracker = get_tracker()
    metrics = tracker.get_metrics()
    return {"by_model": metrics.by_model}


@router.get("/summary/by-provider")
async def get_usage_by_provider() -> Dict[str, Any]:
    """
    Get usage summary grouped by provider.

    Returns:
        Usage summary by provider.
    """
    tracker = get_tracker()
    metrics = tracker.get_metrics()
    return {"by_provider": metrics.by_provider}


@router.get("/summary/by-operation")
async def get_usage_by_operation() -> Dict[str, Any]:
    """
    Get usage summary grouped by operation type.

    Returns:
        Usage summary by operation.
    """
    tracker = get_tracker()
    metrics = tracker.get_metrics()
    return {"by_operation": metrics.by_operation}
