"""Pytest configuration for llm-manager tests."""

import pytest

from llm_manager.usage import UsageTracker


@pytest.fixture
def usage_tracker():
    """Create a fresh usage tracker for testing."""
    return UsageTracker(enabled=True, max_history_size=100)
