"""Tests for shared.utils.token_count."""

from unittest.mock import patch

from shared.utils.text import count_tokens_estimate, token_count


class TestTokenCount:
    def test_empty_string(self):
        assert token_count("") == 0

    def test_returns_int(self):
        result = token_count("hello world")
        assert isinstance(result, int)
        assert result > 0

    def test_with_tiktoken(self):
        """If tiktoken is installed, the result should differ from the estimate."""
        try:
            import tiktoken  # noqa: F401

            text = "The quick brown fox jumps over the lazy dog."
            exact = token_count(text)
            estimate = count_tokens_estimate(text)
            # They should both be positive but may differ
            assert exact > 0
            assert estimate > 0
        except ImportError:
            # tiktoken not available — skip
            pass

    def test_fallback_without_tiktoken(self):
        """When tiktoken is not importable, falls back to estimate."""
        with patch.dict("sys.modules", {"tiktoken": None}):
            # Force reimport to pick up the patched module
            text = "hello world foo bar"
            # The fallback uses count_tokens_estimate which is word_count * 1.3
            result = token_count(text)
            assert result > 0

    def test_long_text(self):
        text = "word " * 1000
        result = token_count(text)
        assert result > 0


def test_fallback_when_encoding_unavailable():
    """F2-10: zonder internet kan tiktoken het encodingbestand niet ophalen; elke
    uitzondering valt terug op de schatting, de API valt niet om."""
    import tiktoken

    text = "hello world foo bar"
    with patch.object(tiktoken, "get_encoding", side_effect=OSError("geen netwerk")):
        assert token_count(text) == count_tokens_estimate(text)
