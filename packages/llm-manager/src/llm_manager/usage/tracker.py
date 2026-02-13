"""
Usage tracking for LLM operations.
"""

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Generator, List, Optional

from loguru import logger

from shared.models import ModelUsageRecord


@dataclass
class UsageMetrics:
    """Aggregated usage metrics."""

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_duration_ms: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    by_model: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_provider: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_operation: Dict[str, Dict[str, Any]] = field(default_factory=dict)


class UsageTracker:
    """
    Tracks LLM usage for monitoring and cost analysis.

    Provides:
    - Request tracking with tokens and cost
    - Aggregated metrics
    - Usage history
    - Cost estimation
    """

    def __init__(
        self,
        enabled: bool = True,
        max_history_size: int = 10000,
        persist_callback: Optional[Callable[[ModelUsageRecord], None]] = None,
    ):
        """
        Initialize usage tracker.

        Args:
            enabled: Whether tracking is enabled.
            max_history_size: Maximum number of records to keep in memory.
            persist_callback: Optional callback to persist records.
        """
        self.enabled = enabled
        self.max_history_size = max_history_size
        self.persist_callback = persist_callback
        self._history: List[ModelUsageRecord] = []
        self._metrics = UsageMetrics()

    def record(
        self,
        model_id: str,
        model_name: str,
        provider: str,
        operation_type: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: Optional[float] = None,
        duration_ms: Optional[int] = None,
        success: bool = True,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ModelUsageRecord]:
        """
        Record a usage event.

        Args:
            model_id: Model ID.
            model_name: Model name.
            provider: Provider name.
            operation_type: Type of operation.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            cost_usd: Cost in USD.
            duration_ms: Duration in milliseconds.
            success: Whether operation succeeded.
            error_message: Error message if failed.
            metadata: Additional metadata.

        Returns:
            Created usage record or None if disabled.
        """
        if not self.enabled:
            return None

        total_tokens = input_tokens + output_tokens

        record = ModelUsageRecord(
            model_id=model_id,
            model_name=model_name,
            provider=provider,
            operation_type=operation_type,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
            success=success,
            error_message=error_message,
            metadata=metadata,
        )

        self._add_to_history(record)
        self._update_metrics(record)

        if self.persist_callback:
            try:
                self.persist_callback(record)
            except Exception as e:
                logger.warning(f"Failed to persist usage record: {e}")

        return record

    @contextmanager
    def track_operation(
        self,
        model_id: str,
        model_name: str,
        provider: str,
        operation_type: str,
        input_tokens: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Context manager for tracking an operation.

        Args:
            model_id: Model ID.
            model_name: Model name.
            provider: Provider name.
            operation_type: Type of operation.
            input_tokens: Number of input tokens.
            metadata: Additional metadata.

        Yields:
            Context dict for adding output_tokens, cost, etc.

        Example:
            with tracker.track_operation(...) as ctx:
                result = model.generate(...)
                ctx['output_tokens'] = result.usage.output_tokens
        """
        context: Dict[str, Any] = {
            "output_tokens": 0,
            "cost_usd": None,
            "success": True,
            "error_message": None,
        }
        start_time = time.time()

        try:
            yield context
        except Exception as e:
            context["success"] = False
            context["error_message"] = str(e)
            raise
        finally:
            duration_ms = int((time.time() - start_time) * 1000)
            self.record(
                model_id=model_id,
                model_name=model_name,
                provider=provider,
                operation_type=operation_type,
                input_tokens=input_tokens,
                output_tokens=context.get("output_tokens", 0),
                cost_usd=context.get("cost_usd"),
                duration_ms=duration_ms,
                success=context.get("success", True),
                error_message=context.get("error_message"),
                metadata=metadata,
            )

    def _add_to_history(self, record: ModelUsageRecord) -> None:
        """Add record to history with size limit."""
        self._history.append(record)
        if len(self._history) > self.max_history_size:
            self._history = self._history[-self.max_history_size:]

    def _update_metrics(self, record: ModelUsageRecord) -> None:
        """Update aggregated metrics."""
        self._metrics.total_requests += 1
        self._metrics.total_input_tokens += record.input_tokens
        self._metrics.total_output_tokens += record.output_tokens
        self._metrics.total_tokens += record.total_tokens
        if record.cost_usd:
            self._metrics.total_cost_usd += record.cost_usd
        if record.duration_ms:
            self._metrics.total_duration_ms += record.duration_ms
        if record.success:
            self._metrics.successful_requests += 1
        else:
            self._metrics.failed_requests += 1

        # Update by-model metrics
        if record.model_id not in self._metrics.by_model:
            self._metrics.by_model[record.model_id] = {
                "requests": 0,
                "tokens": 0,
                "cost": 0.0,
            }
        self._metrics.by_model[record.model_id]["requests"] += 1
        self._metrics.by_model[record.model_id]["tokens"] += record.total_tokens
        if record.cost_usd:
            self._metrics.by_model[record.model_id]["cost"] += record.cost_usd

        # Update by-provider metrics
        if record.provider not in self._metrics.by_provider:
            self._metrics.by_provider[record.provider] = {
                "requests": 0,
                "tokens": 0,
                "cost": 0.0,
            }
        self._metrics.by_provider[record.provider]["requests"] += 1
        self._metrics.by_provider[record.provider]["tokens"] += record.total_tokens
        if record.cost_usd:
            self._metrics.by_provider[record.provider]["cost"] += record.cost_usd

        # Update by-operation metrics
        if record.operation_type not in self._metrics.by_operation:
            self._metrics.by_operation[record.operation_type] = {
                "requests": 0,
                "tokens": 0,
                "cost": 0.0,
            }
        self._metrics.by_operation[record.operation_type]["requests"] += 1
        self._metrics.by_operation[record.operation_type]["tokens"] += record.total_tokens
        if record.cost_usd:
            self._metrics.by_operation[record.operation_type]["cost"] += record.cost_usd

    def get_metrics(self) -> UsageMetrics:
        """Get aggregated metrics."""
        return self._metrics

    def get_history(
        self,
        limit: int = 100,
        model_id: Optional[str] = None,
        provider: Optional[str] = None,
        operation_type: Optional[str] = None,
    ) -> List[ModelUsageRecord]:
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
        records = self._history.copy()

        if model_id:
            records = [r for r in records if r.model_id == model_id]
        if provider:
            records = [r for r in records if r.provider == provider]
        if operation_type:
            records = [r for r in records if r.operation_type == operation_type]

        return records[-limit:]

    def reset(self) -> None:
        """Reset all tracking data."""
        self._history.clear()
        self._metrics = UsageMetrics()
        logger.info("Usage tracker reset")

    def estimate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        input_cost_per_1k: float,
        output_cost_per_1k: float,
    ) -> float:
        """
        Estimate cost for a given token count.

        Args:
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
            input_cost_per_1k: Cost per 1K input tokens.
            output_cost_per_1k: Cost per 1K output tokens.

        Returns:
            Estimated cost in USD.
        """
        input_cost = (input_tokens / 1000) * input_cost_per_1k
        output_cost = (output_tokens / 1000) * output_cost_per_1k
        return input_cost + output_cost
