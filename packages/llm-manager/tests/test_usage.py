"""Tests for usage tracking."""

import pytest

from llm_manager.usage import UsageMetrics, UsageTracker


class TestUsageTracker:
    """Tests for UsageTracker."""

    def test_record_usage(self, usage_tracker):
        """Test recording usage."""
        record = usage_tracker.record(
            model_id="model:123",
            model_name="gpt-4",
            provider="openai",
            operation_type="chat",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
            duration_ms=500,
        )

        assert record is not None
        assert record.model_id == "model:123"
        assert record.model_name == "gpt-4"
        assert record.total_tokens == 150

    def test_record_disabled(self):
        """Test recording when disabled."""
        tracker = UsageTracker(enabled=False)
        record = tracker.record(
            model_id="model:123",
            model_name="gpt-4",
            provider="openai",
            operation_type="chat",
        )
        assert record is None

    def test_metrics_aggregation(self, usage_tracker):
        """Test metrics aggregation."""
        # Record some usage
        usage_tracker.record(
            model_id="model:1",
            model_name="gpt-4",
            provider="openai",
            operation_type="chat",
            input_tokens=100,
            output_tokens=50,
            cost_usd=0.01,
        )
        usage_tracker.record(
            model_id="model:2",
            model_name="claude-3",
            provider="anthropic",
            operation_type="chat",
            input_tokens=200,
            output_tokens=100,
            cost_usd=0.02,
        )

        metrics = usage_tracker.get_metrics()

        assert metrics.total_requests == 2
        assert metrics.total_input_tokens == 300
        assert metrics.total_output_tokens == 150
        assert metrics.total_tokens == 450
        assert metrics.total_cost_usd == 0.03
        assert metrics.successful_requests == 2
        assert metrics.failed_requests == 0

    def test_metrics_by_model(self, usage_tracker):
        """Test metrics grouped by model."""
        usage_tracker.record(
            model_id="model:1",
            model_name="gpt-4",
            provider="openai",
            operation_type="chat",
            input_tokens=100,
            output_tokens=50,
        )
        usage_tracker.record(
            model_id="model:1",
            model_name="gpt-4",
            provider="openai",
            operation_type="chat",
            input_tokens=100,
            output_tokens=50,
        )

        metrics = usage_tracker.get_metrics()

        assert "model:1" in metrics.by_model
        assert metrics.by_model["model:1"]["requests"] == 2
        assert metrics.by_model["model:1"]["tokens"] == 300

    def test_metrics_by_provider(self, usage_tracker):
        """Test metrics grouped by provider."""
        usage_tracker.record(
            model_id="model:1",
            model_name="gpt-4",
            provider="openai",
            operation_type="chat",
            input_tokens=100,
            output_tokens=50,
        )
        usage_tracker.record(
            model_id="model:2",
            model_name="claude-3",
            provider="anthropic",
            operation_type="chat",
            input_tokens=100,
            output_tokens=50,
        )

        metrics = usage_tracker.get_metrics()

        assert "openai" in metrics.by_provider
        assert "anthropic" in metrics.by_provider
        assert metrics.by_provider["openai"]["requests"] == 1
        assert metrics.by_provider["anthropic"]["requests"] == 1

    def test_metrics_by_operation(self, usage_tracker):
        """Test metrics grouped by operation."""
        usage_tracker.record(
            model_id="model:1",
            model_name="gpt-4",
            provider="openai",
            operation_type="chat",
            input_tokens=100,
            output_tokens=50,
        )
        usage_tracker.record(
            model_id="model:2",
            model_name="text-embedding",
            provider="openai",
            operation_type="embedding",
            input_tokens=100,
            output_tokens=0,
        )

        metrics = usage_tracker.get_metrics()

        assert "chat" in metrics.by_operation
        assert "embedding" in metrics.by_operation

    def test_failed_requests(self, usage_tracker):
        """Test tracking failed requests."""
        usage_tracker.record(
            model_id="model:1",
            model_name="gpt-4",
            provider="openai",
            operation_type="chat",
            success=False,
            error_message="API error",
        )

        metrics = usage_tracker.get_metrics()

        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 1

    def test_get_history(self, usage_tracker):
        """Test getting usage history."""
        for i in range(5):
            usage_tracker.record(
                model_id=f"model:{i}",
                model_name="gpt-4",
                provider="openai",
                operation_type="chat",
                input_tokens=100,
                output_tokens=50,
            )

        history = usage_tracker.get_history(limit=3)
        assert len(history) == 3

    def test_get_history_filtered(self, usage_tracker):
        """Test getting filtered history."""
        usage_tracker.record(
            model_id="model:1",
            model_name="gpt-4",
            provider="openai",
            operation_type="chat",
        )
        usage_tracker.record(
            model_id="model:2",
            model_name="claude-3",
            provider="anthropic",
            operation_type="chat",
        )

        history = usage_tracker.get_history(provider="openai")
        assert len(history) == 1
        assert history[0].provider == "openai"

    def test_history_size_limit(self):
        """Test history size limiting."""
        tracker = UsageTracker(enabled=True, max_history_size=5)

        for i in range(10):
            tracker.record(
                model_id=f"model:{i}",
                model_name="gpt-4",
                provider="openai",
                operation_type="chat",
            )

        history = tracker.get_history(limit=100)
        assert len(history) == 5

    def test_reset(self, usage_tracker):
        """Test resetting tracker."""
        usage_tracker.record(
            model_id="model:1",
            model_name="gpt-4",
            provider="openai",
            operation_type="chat",
            input_tokens=100,
            output_tokens=50,
        )

        usage_tracker.reset()

        metrics = usage_tracker.get_metrics()
        assert metrics.total_requests == 0
        assert len(usage_tracker.get_history()) == 0

    def test_estimate_cost(self, usage_tracker):
        """Test cost estimation."""
        cost = usage_tracker.estimate_cost(
            input_tokens=1000,
            output_tokens=500,
            input_cost_per_1k=0.01,
            output_cost_per_1k=0.03,
        )

        expected = (1000 / 1000 * 0.01) + (500 / 1000 * 0.03)
        assert cost == expected

    def test_track_operation_context(self, usage_tracker):
        """Test tracking operation with context manager."""
        with usage_tracker.track_operation(
            model_id="model:1",
            model_name="gpt-4",
            provider="openai",
            operation_type="chat",
            input_tokens=100,
        ) as ctx:
            ctx["output_tokens"] = 50
            ctx["cost_usd"] = 0.01

        metrics = usage_tracker.get_metrics()
        assert metrics.total_requests == 1
        assert metrics.total_output_tokens == 50

    def test_track_operation_failure(self, usage_tracker):
        """Test tracking failed operation."""
        try:
            with usage_tracker.track_operation(
                model_id="model:1",
                model_name="gpt-4",
                provider="openai",
                operation_type="chat",
            ):
                raise ValueError("Test error")
        except ValueError:
            pass

        metrics = usage_tracker.get_metrics()
        assert metrics.failed_requests == 1
        history = usage_tracker.get_history()
        assert history[0].success is False
        assert "Test error" in history[0].error_message

    def test_persist_callback(self):
        """Test persist callback."""
        persisted = []

        def callback(record):
            persisted.append(record)

        tracker = UsageTracker(enabled=True, persist_callback=callback)
        tracker.record(
            model_id="model:1",
            model_name="gpt-4",
            provider="openai",
            operation_type="chat",
        )

        assert len(persisted) == 1
        assert persisted[0].model_id == "model:1"
