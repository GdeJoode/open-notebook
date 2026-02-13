"""Tests for embedding pipeline configuration."""

from embeddings.config import EmbeddingConfig


class TestEmbeddingConfig:

    def test_default_values(self):
        config = EmbeddingConfig()
        assert config.batch_size == 10
        assert config.max_concurrent == 5
        assert config.retry_attempts == 2
        assert config.retry_delay == 1.0
        assert config.fallback_chunk_size == 1000
        assert config.fallback_chunk_overlap == 200

    def test_custom_values(self):
        config = EmbeddingConfig(
            batch_size=20,
            max_concurrent=10,
            retry_attempts=3,
            retry_delay=0.5,
            fallback_chunk_size=500,
            fallback_chunk_overlap=100,
        )
        assert config.batch_size == 20
        assert config.max_concurrent == 10
        assert config.retry_attempts == 3
        assert config.retry_delay == 0.5
        assert config.fallback_chunk_size == 500
        assert config.fallback_chunk_overlap == 100
